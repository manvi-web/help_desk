from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load model and data
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("qa_dataframe.pkl", "rb") as f:
    qa_df = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Set a secure secret key

# Session config
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session/"
Session(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"].strip().lower()

    if user_input == "i'm interested":
        idx = session.get("last_index")
        if idx is not None:
            full_answer = qa_df.iloc[idx]["Full Answer"]
            url = qa_df.iloc[idx]["URL"]
            response = f"{full_answer}<br><a href='{url}' target='_blank'>{url}</a>"
            return jsonify({"response": response})
        else:
            return jsonify({"response": "Please ask a question first."})

    # Vectorize the input and find best match
    user_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(user_vec, tfidf_matrix)
    idx = np.argmax(sims)

    short_answer = qa_df.iloc[idx]["Short Answer"]
    session["last_index"] = idx  # Save index for follow-up
    print(f"[DEBUG] last_index set to: {idx}")
    
    return jsonify({"response": short_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
