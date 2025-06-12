from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pickle
import pandas as pd
app = Flask(__name__)
CORS(app)
app.secret_key = 'supersecretkey'

# Load model and data
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
qa_df = pickle.load(open('qa_dataframe.pkl', 'rb'))

@app.route('/')
def home():
    return "✅ Effort Helpdesk Chatbot API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['message']

    if user_input.lower() == "i'm interested":
        last_index = session.get("last_index", None)
        if last_index is not None:
            full_answer = qa_df.iloc[last_index]["Full Answer"]
            url = qa_df.iloc[last_index]["URL"]
            return jsonify({"response": full_answer, "url": url})
        else:
            return jsonify({"response": "❌ No previous question found. Please ask a question first.", "url": ""})

    # Find most similar question
    query_vector = vectorizer.transform([user_input])
    similarity_scores = (query_vector * tfidf_matrix.T).toarray()[0]
    best_match_index = similarity_scores.argmax()

    session["last_index"] = int(best_match_index)
    short_answer = qa_df.iloc[best_match_index]["Short Answer"]

    return jsonify({"response": short_answer, "url": ""})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
