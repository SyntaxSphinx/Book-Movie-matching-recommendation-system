# ml_model_mpnet.py
# ---------------------------------------------------
# Uses sentence-transformers all-mpnet-base-v2 to create embeddings,
# normalizes them, saves artifacts, and provides a recommend() function.
# Works on clean_data.csv (must exist in same folder).
# ---------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_FILE = "clean_data.csv"   # input (from Student 1)
EMB_FILE = os.path.join(MODELS_DIR, "embeddings_mpnet.npy")
MODEL_FILE = os.path.join(MODELS_DIR, "mpnet_model.joblib")
META_FILE = os.path.join(MODELS_DIR, "meta.pkl")   # store dataframe subset/columns

BATCH_SIZE = 64   # adjust by RAM / CPU. Reduce if you OOM.

print("[LOAD] Loading dataset...", CSV_FILE)
df = pd.read_csv(CSV_FILE)
df = df.fillna("")   # avoid NaNs in text

texts = df["clean_summary"].astype(str).tolist()

# ---------------------------------------------------
# Encode texts in batches (to avoid OOM)
# ---------------------------------------------------
def encode_in_batches(sent_model, texts_list, batch_size=64):
    N = len(texts_list)
    dim = sent_model.get_sentence_embedding_dimension()
    embeddings = np.zeros((N, dim), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        batch = texts_list[start:end]
        emb = sent_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings[start:end] = emb
        print(f"  encoded {end}/{N}")
    return embeddings


def _load_sentence_model():
    """Load MPNet only when encoding is needed (generate embeddings / free-text query)."""
    from sentence_transformers import SentenceTransformer
    print("[LOAD] Loading sentence-transformers model: all-mpnet-base-v2 (this may take a while)...")
    sent_model = SentenceTransformer("all-mpnet-base-v2")
    joblib.dump("all-mpnet-base-v2", MODEL_FILE)
    return sent_model


if not os.path.exists(EMB_FILE):
    print(f"[ENCODE] Generating embeddings with batch_size={BATCH_SIZE} ...")
    model = _load_sentence_model()
    embeddings = encode_in_batches(model, texts, batch_size=BATCH_SIZE)
    # L2-normalize embeddings (recommended for cosine similarity)
    embeddings = normalize(embeddings, norm='l2', axis=1)
    np.save(EMB_FILE, embeddings)
    print(f"[OK] Embeddings saved to {EMB_FILE} (shape: {embeddings.shape})")
else:
    print("[LOAD] Found existing embeddings file. Loading...")
    embeddings = np.load(EMB_FILE)
    # safe-check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("[WARN] Embeddings not normalized - normalizing now.")
        embeddings = normalize(embeddings, norm='l2', axis=1)
        np.save(EMB_FILE, embeddings)

# Save minimal metadata (title, type, index) to use in recommend()
# Prefer existing meta.pkl when present so artifacts stay in sync with prior runs.
if os.path.exists(META_FILE):
    meta = joblib.load(META_FILE)
else:
    meta = df[["title", "type"]].copy()
    meta.reset_index(inplace=True)  # keep original index as 'index'
    meta.rename(columns={"index": "orig_index"}, inplace=True)
    joblib.dump(meta, META_FILE)

print("\n[OK] Embeddings ready. You can now use recommend(title).")

# ---------------------------------------------------
# Recommendation function
# Note: uses L2-normalized embeddings -> cosine similarity = dot product
# ---------------------------------------------------
def recommend(title, top_n=5, cross_type=True, use_same_type=None):
    """
    Return top_n similar items for a given title.
    - title: exact title match (case-insensitive). If not found, will try fuzzy-ish match (starts-with)
    - top_n: number of recommendations to return
    - cross_type: If True (default), returns opposite type (movie->books, book->movies).
                  If False, returns any type.
    - use_same_type: If specified ('book' or 'movie'), filters to that specific type only.
                     Overrides cross_type if set.
    """
    t = str(title).strip().lower()
    # exact match
    matches = meta[meta["title"].str.lower() == t]
    if matches.empty:
        # try startswith or contains as fallback
        matches = meta[meta["title"].str.lower().str.contains(t)]
        if matches.empty:
            return {"error": f"Title '{title}' not found."}
    idx = int(matches.iloc[0]["orig_index"])

    # Get the type of the input item
    input_row = meta[meta["orig_index"] == idx].iloc[0]
    input_type = input_row["type"]

    q_vec = embeddings[idx].reshape(1, -1)  # normalized
    # since embeddings are normalized, cosine_similarity == dot product
    sims = embeddings.dot(q_vec.reshape(-1))  # faster than sklearn for single query
    # sims is shape (N,)

    # build candidates sorted by similarity
    # Exclude the same entry (exact same index)
    candidates = []
    for i, score in enumerate(sims):
        if i == idx:
            continue
        candidates.append((i, float(score)))

    # sort descending by score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Determine target type for filtering
    target_type = None
    if use_same_type:
        # Explicit type filter provided
        target_type = use_same_type
    elif cross_type:
        # Cross-type: if input is movie, return books; if input is book, return movies
        target_type = "book" if input_type == "movie" else "movie"

    # Filter by type and build results
    results = []
    for i, score in candidates:
        row = meta[meta["orig_index"] == i].iloc[0]

        # Apply type filter if specified
        if target_type and (row["type"] != target_type):
            continue

        results.append({
            "title": row["title"],
            "type": row["type"],
            "similarity": round(score, 4)
        })
        if len(results) >= top_n:
            break

    return {
        "query_title": df.loc[idx, "title"],
        "query_type": input_type,
        "query_index": int(idx),
        "recommended_type": target_type if target_type else "any",
        "results": results,
        "model": "all-mpnet-base-v2"
    }


def recommend_by_text(text, top_n=5, target_type=None):
    """
    Recommend items from free-form text using the same embedding space.
    target_type: optional 'book' or 'movie' filter.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "Text is required."}

    sent_model = _load_sentence_model()
    q_vec = sent_model.encode([text], convert_to_numpy=True)
    q_vec = normalize(q_vec, norm="l2", axis=1)
    sims = embeddings.dot(q_vec.reshape(-1))

    candidates = [(i, float(score)) for i, score in enumerate(sims)]
    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score in candidates:
        row = meta[meta["orig_index"] == i].iloc[0]
        if target_type and row["type"] != target_type:
            continue
        results.append({
            "title": row["title"],
            "type": row["type"],
            "similarity": round(score, 4)
        })
        if len(results) >= top_n:
            break

    return {
        "input_text": text,
        "recommended_type": target_type if target_type else "any",
        "results": results,
        "model": "all-mpnet-base-v2"
    }


def search_titles(query, limit=20):
    """Simple title search over catalog metadata."""
    q = str(query or "").strip().lower()
    if not q:
        return []
    matches = meta[meta["title"].str.lower().str.contains(q, na=False)].head(limit)
    return matches[["title", "type"]].to_dict(orient="records")


# ---------------------------------------------------
# Quick test (if run as script)
# ---------------------------------------------------
if __name__ == "__main__":
    # show sample - test cross-type recommendations
    sample_title = df.iloc[0]["title"]
    print(f"\n[TEST] Sample query: '{sample_title}'\n")
    out = recommend(sample_title, top_n=5, cross_type=True)
    if "error" in out:
        print(out["error"])
    else:
        print(f"Input type: {out['query_type']}")
        print(f"Recommended type: {out['recommended_type']}")
        print(f"Top 5 Recommendations:\n")
        for i, r in enumerate(out["results"], start=1):
            print(f"{i}. {r['title']} [{r['type']}] (sim={r['similarity']})")

    # Test with a movie to show it recommends books
    print("\n" + "="*60)
    print("[TEST] Testing movie -> books recommendation")
    print("="*60)
    movie_title = "Avatar"
    out = recommend(movie_title, top_n=5, cross_type=True)
    if "error" not in out:
        print(f"\nInput: '{out['query_title']}' (type: {out['query_type']})")
        print(f"Recommended type: {out['recommended_type']}")
        print(f"\nTop 5 Book Recommendations:\n")
        for i, r in enumerate(out["results"], start=1):
            print(f"{i}. {r['title']} [{r['type']}] (sim={r['similarity']})")
