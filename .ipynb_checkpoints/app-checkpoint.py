from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session handling
CORS(app, supports_credentials=True)

# Load model and data
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("qa_dataframe.pkl", "rb") as f:
    df = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Make sure templates/index.html exists

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()

    # If user expresses interest
    if user_input in ["i'm interested", "im interested", "interested"]:
        last_q = session.get("last_question", "")
        if last_q:
            match = get_best_match(last_q)
            if match is not None:
                return jsonify({
                    "response": f"{match['Full Answer']}\n\n🔗 {match['URL']}"
                })
        return jsonify({"response": "Sorry, I don’t know what you're interested in."})

    # Save question to session
    session["last_question"] = user_input

    # Regular prediction
    match = get_best_match(user_input)
    if match is not None:
        return jsonify({"response": match["Short Answer"]})
    else:
        return jsonify({"response": "Sorry, I couldn’t find an answer."})

def get_best_match(user_input):
    vec = vectorizer.transform([user_input])
    scores = cosine_similarity(vec, tfidf_matrix)
    idx = np.argmax(scores)
    if scores[0][idx] < 0.3:
        return None
    return df.iloc[idx]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var for Render
    app.run(host="0.0.0.0", port=port)
