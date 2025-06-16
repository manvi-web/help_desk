from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load data
qa_df = pickle.load(open('qa_dataframe.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip().lower()

        # Follow-up: "I'm interested"
        if user_message == "i'm interested":
            last_index = session.get('last_question_index')
            if last_index is not None:
                full_answer = qa_df.iloc[last_index]['Full Answer']
                url = qa_df.iloc[last_index]['URL']
                return jsonify({'response': f"{full_answer}\n\n🔗 [Visit Page]({url})"})
            else:
                return jsonify({'response': "Please ask a question first."})

        # New question
        query_vec = vectorizer.transform([user_message])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        best_index = similarity.argmax()

        short_answer = qa_df.iloc[best_index]['Short Answer']
        session['last_question_index'] = int(best_index)

        return jsonify({'response': short_answer})
    except Exception as e:
        return jsonify({'response': "⚠️ Error processing your request."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
