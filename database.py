
import sqlite3
import pandas as pd
import os
from datetime import datetime

CSV = "clean_data.csv"         # change if located elsewhere
DB = "book_movie.db"

def normalize_val(v):
    if pd.isna(v):
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join([str(x).strip() for x in v if x is not None])
    return str(v)

def main():
    if not os.path.exists(CSV):
        print(f"[ERROR] {CSV} not found. Put clean_data.csv in the script folder or update CSV path.")
        return

    df = pd.read_csv(CSV).fillna("")

    # Try to handle common column names
    # Map columns if names vary
    colmap = {}
    for possible in ["title", "name"]:
        if possible in df.columns:
            colmap["title"] = possible
            break
    if "title" not in colmap:
        raise SystemExit("No 'title' column found in CSV.")

    # optional columns
    def get_col(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    authors_col = get_col("authors", "author", "authors_or_directors", "director")
    genres_col  = get_col("genres", "genres_list", "genre")
    summary_col = get_col("summary", "overview", "plot", "description")
    clean_col   = get_col("clean_summary", "clean_summary")
    year_col    = get_col("year", "release_year", "published_year")
    rating_col  = get_col("rating", "avg_rating", "vote_average")
    tmdb_col    = get_col("tmdb_id", "tmdbId", "tmdb")
    openlib_col = get_col("openlibrary_id", "openlibrary", "open_library_id")
    type_col    = get_col("type", "item_type")
    poster_col  = get_col("poster_url", "poster", "image_url")  # added poster column handling

    # Create DB and table
    with sqlite3.connect(DB) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            type TEXT,
            year INTEGER,
            genres TEXT,
            summary TEXT,
            clean_summary TEXT,
            authors_or_directors TEXT,
            poster_url TEXT,
            rating REAL,
            tmdb_id TEXT,
            openlibrary_id TEXT,
            created_at TEXT
        )
        """)
        conn.commit()

        inserted = 0
        for _, row in df.iterrows():
            title = normalize_val(row.get(colmap["title"]))
            typ = normalize_val(row.get(type_col)) if type_col else ""
            year = row.get(year_col) if year_col and not pd.isna(row.get(year_col)) else None
            genres = normalize_val(row.get(genres_col)) if genres_col else ""
            summary = normalize_val(row.get(summary_col)) if summary_col else ""
            clean_summary = normalize_val(row.get(clean_col)) if clean_col else ""
            authors = normalize_val(row.get(authors_col)) if authors_col else ""
            rating = float(row.get(rating_col)) if rating_col and row.get(rating_col) != "" else None
            tmdb = normalize_val(row.get(tmdb_col)) if tmdb_col else None
            openlib = normalize_val(row.get(openlib_col)) if openlib_col else None
            poster_url = normalize_val(row.get(poster_col)) if poster_col else ""  # Get poster URL

            cur.execute("""
                INSERT INTO catalog (
                    title, type, year, genres, summary, clean_summary, authors_or_directors,
                    poster_url, rating, tmdb_id, openlibrary_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title, typ, year, genres, summary, clean_summary, authors,
                poster_url, rating, tmdb, openlib, datetime.utcnow().isoformat()
            ))
            inserted += 1

        conn.commit()

    print(f"[OK] Database '{DB}' created with {inserted} rows.")


if __name__ == "__main__":
    main()
