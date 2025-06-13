from flask import Flask, request, render_template, session, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'

# Load model and data
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
qa_df = pickle.load(open('qa_dataframe.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"response": "Please enter a question."})

    # Check if user said "I'm interested"
    if user_input.strip().lower() == "i'm interested":
        last_question = session.get("last_question")
        if not last_question:
            return jsonify({"response": "Please ask a question first."})

        row = qa_df[qa_df['Question'] == last_question].iloc[0]
        full_answer = f"{row['Full Answer']} \n\n[Read more]({row['URL']})"
        return jsonify({"response": full_answer})

    # Search using TF-IDF
    input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vec, tfidf_matrix)
    idx = similarities.argmax()
    confidence = similarities[0, idx]

    if confidence < 0.2:
        return jsonify({"response": "Sorry, I couldn’t find an answer."})

    question = qa_df.iloc[idx]['Question']
    short_answer = qa_df.iloc[idx]['Short Answer']
    
    session['last_question'] = question
    return jsonify({"response": short_answer})

if __name__ == '__main__':
    # Bind to port 5000 and listen on all interfaces (important for deployment like Render)
    app.run(host='0.0.0.0', port=5000, debug=True)
