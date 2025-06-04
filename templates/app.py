import os
from flask import Flask, request, jsonify, render_template, session
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session to store last question

# Load chatbot data
df = pd.read_csv('effort_chatbot_data.csv')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()

    # Check for interest follow-up
    if user_input in ["i'm interested", "im interested", "interested", "tell me more", "more info"]:
        last_question = session.get("last_question", "")
        if last_question:
            match = df[df['question'].str.lower() == last_question.lower()]
            if not match.empty:
                full_answer = match.iloc[0]['full_answer']
                url = match.iloc[0]['URL']
                return jsonify({"reply": f"{full_answer}<br><br><a href='{url}' target='_blank'>Read more</a>"})
            else:
                return jsonify({"reply": "Sorry, no more info found."})
        return jsonify({"reply": "Please ask a question first."})

    # Match user's question
    match = df[df['question'].str.lower().str.contains(user_input, na=False)]
    if not match.empty:
        short_answer = match.iloc[0]['short_answer']
        session['last_question'] = match.iloc[0]['question']
        return jsonify({"reply": f"{short_answer}<br><br>Want to know more? Type 'I'm interested'."})
    else:
        return jsonify({"reply": "Sorry, I don't know the answer to that."})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Important for Render
    app.run(host="0.0.0.0", port=port)
