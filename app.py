from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load data and models
df = pd.read_csv("effort_qa_dataset_real_content.csv")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    last_index = data.get("last_index")

    if not user_input:
        return jsonify({"response": "Please ask a question."})

    # Handle follow-up interest
    if user_input.lower() == "i'm interested":
        if last_index is not None and str(last_index).isdigit():
            i = int(last_index)
            full_answer = df.iloc[i]["Full Answer"]
            url = df.iloc[i]["URL"]
            return jsonify({"response": f"{full_answer}\n\nRead more: {url}"})
        else:
            return jsonify({"response": "Please ask a question first."})

    # Otherwise, find best match using TF-IDF similarity
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_index = similarities.argmax()
    short_answer = df.iloc[best_index]["Short Answer"]

    return jsonify({"response": short_answer, "index": best_index})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
