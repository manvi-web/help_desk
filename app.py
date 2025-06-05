import os
from flask import Flask, request, jsonify, render_template, session
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load CSV
try:
    df = pd.read_csv('effort_chatbot_data.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=["question", "short_answer", "full_answer", "URL"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()

    interested_keywords = ["i'm interested", "im interested", "interested", "tell me more", "more info"]
    if user_input in interested_keywords:
        last_question = session.get("last_question", "")
        match = df[df['question'].str.lower() == last_question.lower()]
        if not match.empty:
            full_answer = match.iloc[0]['full_answer']
            url = match.iloc[0]['URL']
            return jsonify({"reply": f"{full_answer}<br><br>More info: <a href='{url}' target='_blank'>{url}</a>"})
        else:
            return jsonify({"reply": "Please ask a question first."})

    match = df[df['question'].str.lower().str.contains(user_input, na=False)]

    if not match.empty:
        short_answer = match.iloc[0]['short_answer']
        session['last_question'] = match.iloc[0]['question']
        return jsonify({"reply": f"{short_answer}<br><br>Reply with 'I'm interested' to know more."})
    else:
        return jsonify({"reply": "Sorry, I don’t know the answer to that."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
