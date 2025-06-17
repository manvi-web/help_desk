from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load trained files
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
qa_df = pickle.load(open("qa_dataframe.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message'].strip().lower()

    if user_input == "i'm interested":
        idx = session.get("last_index", None)
        if idx is None:
            return jsonify({"response": "❗ Please ask a question first."})
        
        full_answer = qa_df.iloc[idx].get('Full Answer', 'No full answer found.')
        url = qa_df.iloc[idx].get('URL', 'No URL available.')
        return jsonify({"response": f"{full_answer}\n\n🔗 More info: {url}"})

    # Regular question flow
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    idx = int(np.argmax(similarity))
    session['last_index'] = idx

    short_answer = qa_df.iloc[idx].get('Short Answer', 'No short answer found.')
    return jsonify({"response": short_answer})

# ✅ Port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
