from flask import Flask, request, jsonify, session
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session

# Load vectorizer, matrix, and dataset
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
qa_df = pickle.load(open('qa_dataframe.pkl', 'rb'))  # Contains Title, Short Answer, Full Answer, URL

@app.route('/')
def home():
    return '✅ Effort Helpdesk Chatbot API is running!'

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip().lower()

    if user_message == "i'm interested":
        last_index = session.get('last_question_index')
        if last_index is not None:
            full_answer = qa_df.iloc[last_index]['Full Answer']
            url = qa_df.iloc[last_index]['URL']
            return jsonify({'response': f"{full_answer}\nURL: {url}"})
        else:
            return jsonify({'response': "Please ask a question first."})

    # Handle new question
    query_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    best_match_index = similarity.argmax()
    short_answer = qa_df.iloc[best_match_index]['Short Answer']

    # Store index in session for follow-up
    session['last_question_index'] = int(best_match_index)

    return jsonify({'response': short_answer})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
