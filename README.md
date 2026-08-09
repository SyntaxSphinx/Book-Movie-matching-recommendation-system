# Book–Movie Matching Recommendation System

Cross-domain recommender that suggests **books from a movie** (and **movies from a book**) using semantic similarity on plot/summary text.

## How it works

1. **`preprocess.py`** — cleans and merges book + movie data into `clean_data.csv`
2. **`ml_model_mpnet.py`** — builds MPNet (`all-mpnet-base-v2`) embeddings and exposes `recommend()`
3. **`models/`** — stores precomputed `embeddings_mpnet.npy` and `meta.pkl`
4. **`app.py`** — Flask API that calls the recommender
5. **`database.py`** — optional SQLite catalog from `clean_data.csv`

By default, recommendations are **cross-type**: movie → books, book → movies.

## Project structure

```
├── app.py                          # Flask API
├── ml_model_mpnet.py               # MPNet embeddings + recommend()
├── preprocess.py                   # Data cleaning / merge
├── database.py                     # Build SQLite DB (optional)
├── test_cross_type_recommendations.py
├── clean_data.csv                  # Unified catalog
├── books.csv / tmdb_5000_movies.csv
├── models/
│   ├── embeddings_mpnet.npy
│   └── meta.pkl
└── requirements.txt
```

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

If you re-run preprocessing with spaCy:

```bash
python -m spacy download en_core_web_sm
```

## Run the API

```bash
python app.py
```

Server: `http://127.0.0.1:5000`

### Example requests

**Search titles**
```text
GET /search?query=avatar
```

**Movie → books**
```text
GET /recommend/title?title=Avatar&top_n=5&cross_type=true
```

**Book → movies**
```text
GET /recommend/title?title=The Hitchhiker's Guide to the Galaxy&top_n=5
```

**Free-text recommendation** (loads the sentence-transformers model on first use)
```bash
curl -X POST http://127.0.0.1:5000/recommend/text ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"space adventure found family\", \"top_n\": 5, \"target_type\": \"book\"}"
```

## Rebuild pipeline (optional)

Only needed if you change source CSVs or want fresh embeddings:

```bash
python preprocess.py
python ml_model_mpnet.py
python database.py
```

## Notes

- Title-based recommendations use the precomputed embeddings in `models/` (no model download required).
- Free-text recommendations need `sentence-transformers` and will download `all-mpnet-base-v2` on first use.
- `tmdb_5000_credits.csv` is not required for the main flow and is ignored by git.
