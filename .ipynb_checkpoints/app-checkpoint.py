from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = "effort_bot_secret"
CORS(app)

# Load model/vectorizer/data
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("qa_dataframe.pkl", "rb") as f:
    df = pickle.load(f)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    if user_input.lower().strip() == "i'm interested":
        last_index = session.get("last_index")
        if last_index is not None:
            row = df.iloc[last_index]
            return jsonify({
                "response": f"{row['Full Answer']}\n🔗 Read more: {row['URL']}"
            })
        else:
            return jsonify({"response": "❗ Please ask a question first."})

    query_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    best_index = np.argmax(similarities)

    session["last_index"] = int(best_index)
    row = df.iloc[best_index]

    return jsonify({
        "response": f"{row['Short Answer']}\n👉 Type *I'm interested* for full info."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var or default to 5000
    app.run(host="0.0.0.0", port=port)
