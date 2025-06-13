from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key'

# Load artifacts
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("qa_dataframe.pkl", "rb") as f:
    df = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json.get("message", "").strip().lower()

    if "interested" in user_input:
        last_question = session.get("last_question", "")
        if last_question and last_question in df['question'].values:
            row = df[df['question'] == last_question].iloc[0]
            return jsonify({"response": f"{row['full_answer']}\n\n🔗 More info: {row['url']}"})
        return jsonify({"response": "Sorry, I don’t have more info."})

    # TF-IDF match
    user_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, tfidf_matrix)
    best_idx = scores.argmax()
    best_score = scores[0, best_idx]

    if best_score < 0.2:
        return jsonify({"response": "Sorry, I couldn’t find an answer."})

    question = df.iloc[best_idx]['question']
    short_answer = df.iloc[best_idx]['short_answer']

    session['last_question'] = question
    return jsonify({"response": short_answer})

# Render-specific port binding
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
