from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

# Load dataset
df = pd.read_csv("effort_qa_dataset_real_content.csv")

# Combine all questions for TF-IDF
all_questions = df["Title"].astype(str).tolist()

# Create vectorizer and matrix if not already created
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_questions)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"response": "Please ask a question."})

    if user_input.lower().strip() == "i'm interested":
        last_index = session.get("last_index")
        if last_index is not None:
            full_answer = df.iloc[last_index]["Full Answer"]
            url = df.iloc[last_index]["URL"]
            return jsonify({"response": f"{full_answer}\n\nRead more: {url}"})
        else:
            return jsonify({"response": "Please ask a question first."})

    # Compute similarity
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_index = similarities.argmax()

    session["last_index"] = int(best_index)
    short_answer = df.iloc[best_index]["Short Answer"]

    return jsonify({"response": short_answer})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
