# preprocess.py
# ---------------------------------------------
# This script:
# 1. Loads books.csv and movies.csv
# 2. Cleans text (remove punctuation, stopwords, lemmatize)
# 3. Merges datasets into one unified file
# 4. Generates TF-IDF vectors
# 5. Saves final cleaned files
# ---------------------------------------------

import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("\n🚀 Starting preprocessing...")

# ---------------------------------------------
# STEP 1: Load Datasets
# ---------------------------------------------
print("📂 Loading CSV files...")

try:
    books = pd.read_csv("books.csv", on_bad_lines='skip')  # Skip bad lines
    movies = pd.read_csv("tmdb_5000_movies.csv", on_bad_lines='skip')
except (pd.errors.ParserError, FileNotFoundError) as e:
    print(f"❌ Error reading CSV files: {e}")
    exit(1)

print("⚠️ Note: Malformed rows were skipped. Please review the input files if necessary.")

# ---------------------------------------------
# STEP 2: Select essential columns
# ---------------------------------------------
print("🔍 Selecting useful columns...")

# Books: Create summary from available columns, use empty genres if not available
if 'genres' not in books.columns:
    books['genres'] = ''
if 'summary' not in books.columns:
    # Create a basic summary from title and authors if summary doesn't exist
    books['summary'] = books['title'].fillna('') + ' ' + books.get('authors', pd.Series('')).fillna('')

books = books[['title', 'authors', 'genres', 'summary']]

# Movies: Extract year from release_date and select columns
movies = movies[['title', 'genres', 'overview', 'release_date']].copy()
movies = movies.rename(columns={'overview': 'summary'})

# Extract year from release_date
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
movies = movies[['title', 'genres', 'summary', 'year']]

# ---------------------------------------------
# STEP 3: Add type column
# ---------------------------------------------
books['type'] = 'book'
movies['type'] = 'movie'

# ---------------------------------------------
# STEP 4: Combine datasets
# ---------------------------------------------
print("🔗 Combining datasets...")
data = pd.concat([books, movies], ignore_index=True)

# ---------------------------------------------
# STEP 5: Clean text function
# ---------------------------------------------
print("🧼 Cleaning text... (downloading NLP resources)")

nltk.download('stopwords')
stop = set(stopwords.words("english"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop]
    return " ".join(tokens)

# Apply cleaning
print("✨ Applying text cleaning...")
data["clean_summary"] = data["summary"].fillna("").apply(clean_text)

# ---------------------------------------------
# STEP 6: Remove duplicates & missing values
# ---------------------------------------------
print("🧹 Removing duplicates and null values...")

data = data.dropna(subset=['title', 'clean_summary'])
data = data.drop_duplicates(subset=['title'])

# ---------------------------------------------
# STEP 7: Save cleaned CSV
# ---------------------------------------------
print("💾 Saving cleaned data to clean_data.csv ...")
data.to_csv("clean_data.csv", index=False)

# ---------------------------------------------
# OPTIONAL: Create TF-IDF vectors
# ---------------------------------------------
print("🧠 Generating TF-IDF vectors (this may take a moment)...")

vectorizer = TfidfVectorizer(max_features=15000)
tfidf_matrix = vectorizer.fit_transform(data['clean_summary'])

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")

print("\n✅ Preprocessing Completed Successfully!")
print("📁 Output Files Created:")
print("   - clean_data.csv")
print("   - tfidf_vectorizer.pkl")
print("   - tfidf_matrix.pkl\n")
