from flask import Flask, request, render_template, jsonify, session
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
app.secret_key = 'your_secret_key'
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
qa_df = pickle.load(open('qa_dataframe.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['message']

    if user_input.lower().strip() == "i'm interested":
        last_index = session.get('last_index')
        if last_index is not None:
            full_answer = qa_df.iloc[last_index]['Full Answer']
            url = qa_df.iloc[last_index]['URL']
            return jsonify({'response': f"{full_answer}\n\n🔗 More info: {url}"})
        else:
            return jsonify({'response': "Please ask a question first."})

    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_match = int(np.argmax(similarities))
    session['last_index'] = best_match

    short_answer = qa_df.iloc[best_match]['Short Answer']
    return jsonify({'response': short_answer})
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render will inject PORT
    app.run(host='0.0.0.0', port=port)
