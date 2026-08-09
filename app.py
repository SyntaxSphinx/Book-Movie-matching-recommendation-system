from flask import Flask, request, jsonify
from flask_cors import CORS

from ml_model_mpnet import recommend, recommend_by_text, search_titles

app = Flask(__name__)
CORS(app)

# ---------------------
#  BASIC ROUTES
# ---------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Book-Movie matching API running",
        "endpoints": {
            "health": "/health",
            "search": "/search?query=",
            "recommend_title": "/recommend/title?title=&top_n=5&cross_type=true",
            "recommend_text": "POST /recommend/text  {\"text\": \"...\", \"top_n\": 5, \"target_type\": \"book|movie\"}"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "OK"})

# ---------------------
#  SEARCH ENDPOINT
# ---------------------

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    limit = request.args.get("limit", 20, type=int)
    results = search_titles(query, limit=limit)

    return jsonify({
        "query": query,
        "results": results
    })

# ---------------------
#  RECOMMENDATION BY TITLE
# ---------------------

@app.route("/recommend/title", methods=["GET"])
def recommend_title():
    title = request.args.get("title", "")
    top_n = request.args.get("top_n", 5, type=int)
    cross_type = request.args.get("cross_type", "true").lower() != "false"
    use_same_type = request.args.get("use_same_type")  # optional: book | movie

    if not title.strip():
        return jsonify({"error": "Query parameter 'title' is required."}), 400

    result = recommend(
        title,
        top_n=top_n,
        cross_type=cross_type,
        use_same_type=use_same_type
    )

    status = 404 if "error" in result else 200
    return jsonify(result), status

# ---------------------
#  RECOMMENDATION BY TEXT
# ---------------------

@app.route("/recommend/text", methods=["POST"])
def recommend_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    top_n = int(data.get("top_n", 5))
    target_type = data.get("target_type")  # optional: book | movie

    result = recommend_by_text(text, top_n=top_n, target_type=target_type)
    status = 400 if "error" in result else 200
    return jsonify(result), status

# ---------------------
#  RUN SERVER
# ---------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
