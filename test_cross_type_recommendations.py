# test_cross_type_recommendations.py
# Test that the model correctly recommends cross-type items:
# - Movies -> Books
# - Books -> Movies

import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

print("\n[TEST] Testing Cross-Type Recommendations (Movies <-> Books)\n")

# Load the mpnet model components
MODELS_DIR = "models"
EMB_FILE = os.path.join(MODELS_DIR, "embeddings_mpnet.npy")
META_FILE = os.path.join(MODELS_DIR, "meta.pkl")
CSV_FILE = "clean_data.csv"

# Load data
df = pd.read_csv(CSV_FILE)

print("[LOAD] Loading MPNet embeddings and metadata...")

if not os.path.exists(EMB_FILE):
    print(f"[ERROR] Embeddings file not found: {EMB_FILE}")
    print("Please run ml_model_mpnet.py first to generate embeddings.")
    exit(1)

embeddings = np.load(EMB_FILE)
meta = joblib.load(META_FILE)

# Ensure embeddings are normalized
norms = np.linalg.norm(embeddings, axis=1)
if not np.allclose(norms, 1.0, atol=1e-3):
    print("[WARN] Embeddings not normalized - normalizing now.")
    embeddings = normalize(embeddings, norm='l2', axis=1)

print("[OK] Models loaded successfully!\n")

# Recommendation function (updated version from ml_model_mpnet.py)
def recommend(title, top_n=5, cross_type=True, use_same_type=None):
    """
    Return top_n similar items for a given title.
    - cross_type: If True (default), returns opposite type (movie->books, book->movies)
    """
    t = str(title).strip().lower()
    matches = meta[meta["title"].str.lower() == t]
    if matches.empty:
        matches = meta[meta["title"].str.lower().str.contains(t)]
        if matches.empty:
            return {"error": f"Title '{title}' not found."}
    idx = int(matches.iloc[0]["orig_index"])
    
    input_row = meta[meta["orig_index"] == idx].iloc[0]
    input_type = input_row["type"]

    q_vec = embeddings[idx].reshape(1, -1)
    sims = embeddings.dot(q_vec.reshape(-1))

    candidates = []
    for i, score in enumerate(sims):
        if i == idx:
            continue
        candidates.append((i, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)

    target_type = None
    if use_same_type:
        target_type = use_same_type
    elif cross_type:
        target_type = "book" if input_type == "movie" else "movie"

    results = []
    for i, score in candidates:
        row = meta[meta["orig_index"] == i].iloc[0]
        
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

print("=" * 80)
print("TESTING MOVIE -> BOOK RECOMMENDATIONS")
print("=" * 80)

# Test movies that should recommend books
movie_test_cases = [
    "Avatar",
    "The Dark Knight Rises",
    "Harry Potter and the Half-Blood Prince",
    "Inception",
    "The Lord of the Rings: The Fellowship of the Ring",
]

for movie_title in movie_test_cases:
    print(f"\n[TEST] Movie: '{movie_title}'")
    print("-" * 80)
    
    result = recommend(movie_title, top_n=5, cross_type=True)
    
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        continue
    
    print(f"Input Type: {result['query_type']}")
    print(f"Recommended Type: {result['recommended_type']}")
    
    # Verify all recommendations are books
    all_books = all(r['type'] == 'book' for r in result['results'])
    if all_books:
        print(f"[OK] All recommendations are BOOKS [CORRECT]")
    else:
        print(f"[WARNING] Some recommendations are not books!")
        for r in result['results']:
            if r['type'] != 'book':
                print(f"  - {r['title']} is a {r['type']}")
    
    print(f"\nTop 5 Book Recommendations:")
    for i, rec in enumerate(result["results"], 1):
        print(f"   {i}. {rec['title'][:65]:<65} (similarity: {rec['similarity']:.4f})")

print("\n" + "=" * 80)
print("TESTING BOOK -> MOVIE RECOMMENDATIONS")
print("=" * 80)

# Test books that should recommend movies
book_test_cases = [
    "Harry Potter and the Half-Blood Prince (Harry Potter  #6)",
    "The Hitchhiker's Guide to the Galaxy",
    "The Lord of the Rings",
    "A Game of Thrones",
]

for book_title in book_test_cases:
    print(f"\n[TEST] Book: '{book_title}'")
    print("-" * 80)
    
    result = recommend(book_title, top_n=5, cross_type=True)
    
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        continue
    
    print(f"Input Type: {result['query_type']}")
    print(f"Recommended Type: {result['recommended_type']}")
    
    # Verify all recommendations are movies
    all_movies = all(r['type'] == 'movie' for r in result['results'])
    if all_movies:
        print(f"[OK] All recommendations are MOVIES [CORRECT]")
    else:
        print(f"[WARNING] Some recommendations are not movies!")
        for r in result['results']:
            if r['type'] != 'movie':
                print(f"  - {r['title']} is a {r['type']}")
    
    print(f"\nTop 5 Movie Recommendations:")
    for i, rec in enumerate(result["results"], 1):
        print(f"   {i}. {rec['title'][:65]:<65} (similarity: {rec['similarity']:.4f})")

# Test with cross_type=False to show it returns any type
print("\n" + "=" * 80)
print("TESTING WITH cross_type=False (Returns Any Type)")
print("=" * 80)

sample_title = "Avatar"
print(f"\n[TEST] '{sample_title}' with cross_type=False")
print("-" * 80)

result = recommend(sample_title, top_n=5, cross_type=False)

if "error" not in result:
    print(f"Input Type: {result['query_type']}")
    print(f"Recommended Type: {result['recommended_type']}")
    print(f"\nTop 5 Recommendations (any type):")
    for i, rec in enumerate(result["results"], 1):
        print(f"   {i}. {rec['title'][:60]:<60} [{rec['type']:5}] (similarity: {rec['similarity']:.4f})")

print("\n" + "=" * 80)
print("[OK] Cross-Type Recommendation Testing Complete!")
print("=" * 80)
print("\nSummary:")
print("  - Movies should recommend BOOKS only")
print("  - Books should recommend MOVIES only")
print("  - This is the default behavior (cross_type=True)")
print("=" * 80 + "\n")

