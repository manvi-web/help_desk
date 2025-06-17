from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load the ML components
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)
with open("qa_dataframe.pkl", "rb") as f:
    qa_df = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message'].strip().lower()

    # Follow-up interaction
    if user_input == "i'm interested" and 'last_index' in session:
        idx = session['last_index']
        full_answer = qa_df.iloc[idx]['Full Answer']
        url = qa_df.iloc[idx]['URL']
        return jsonify({"response": f"{full_answer}\n\nMore info: {url}"})

    # Normal question
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    idx = similarity.argmax()
    session['last_index'] = idx

    short_answer = qa_df.iloc[idx]['Short Answer']
    return jsonify({"response": short_answer})

if __name__ == "__main__":
 
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for local
    app.run(host='0.0.0.0', port=port, debug=True)
