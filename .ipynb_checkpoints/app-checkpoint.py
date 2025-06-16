from flask import Flask, request, jsonify, session
from flask_session import Session
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure session to use filesystem (works on Render)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load data and models
qa_df = pickle.load(open('qa_dataframe.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

@app.route('/')
def home():
    return "✅ Effort Helpdesk Chatbot API is running!"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip().lower()
        print("User message:", user_message)

        if user_message == "i'm interested":
            last_index = session.get('last_question_index')
            print("Last index from session:", last_index)

            if last_index is not None:
                full_answer = qa_df.iloc[last_index]['Full Answer']
                url = qa_df.iloc[last_index]['URL']
                response = f"{full_answer}\nURL: {url}"
                return jsonify({'response': response})
            else:
                return jsonify({'response': "Please ask a question first."})

        # Otherwise, it's a normal question
        query_vec = vectorizer.transform([user_message])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        best_match_index = similarity.argmax()
        print("Best match index:", best_match_index)

        short_answer = qa_df.iloc[best_match_index]['Short Answer']
        session['last_question_index'] = int(best_match_index)

        return jsonify({'response': short_answer})

    except Exception as e:
        print("Exception:", str(e))
        return jsonify({'response': "Server error. Please try again."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
