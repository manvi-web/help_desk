from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and vectorizer
model = pickle.load(open('chatbot_model.pkl', 'rb'))
vectorizer = pickle.load(open('chatbot_vectorizer.pkl', 'rb'))

# Load CSV
df = pd.read_csv("effort_chatbot_data.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'reply': "Please enter a question."})

    # Store last question for follow-up
    if 'last_question' not in session:
        session['last_question'] = ''

    if user_message.lower() == "i'm interested":
        last_q = session.get('last_question', '')
        full_row = df[df['question'] == last_q]
        if not full_row.empty:
            full_answer = full_row.iloc[0]['full_answer']
            url = full_row.iloc[0]['URL']
            return jsonify({'reply': f"{full_answer}<br><a href='{url}' target='_blank'>Read more</a>"})
        else:
            return jsonify({'reply': "Sorry, I couldn't find more details."})
    else:
        X = vectorizer.transform([user_message])
        prediction = model.predict(X)[0]
        session['last_question'] = prediction

        row = df[df['question'] == prediction]
        if not row.empty:
            short_ans = row.iloc[0]['short_answer']
            return jsonify({'reply': short_ans})
        else:
            return jsonify({'reply': "Sorry, I don't have an answer for that."})

if __name__ == '__main__':
    app.run(debug=True)
