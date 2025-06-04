import os
from flask import Flask, request, jsonify, render_template, session
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session support

# Load chatbot data
df = pd.read_csv('effort_chatbot_data.csv')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()

    # Handle "I'm interested" messages
    interested_keywords = ["i'm interested", "im interested", "interested", "tell me more", "more info"]
    if user_input in interested_keywords:
        last_question = session.get("last_question", "")
        if last_question:
            match = df[df['question'].str.lower() == last_question.lower()]
            if not match.empty:
                full_answer = match.iloc[0]['full_answer']
                url = match.iloc[0]['URL']
                return jsonify({"reply": f"{full_answer}<br><br>More info: <a href='{url}' target='_blank'>{url}</a>"})
            else:
                return jsonify({"reply": "Sorry, I couldn't find more info on that."})
        else:
            return jsonify({"reply": "Please ask a question first."})
    
    # Try to find matching question
    match = df[df['question'].str.lower().str.contains(user_input, na=False)]

    if not match.empty:
        short_answer = match.iloc[0]['short_answer']
        session['last_question'] = match.iloc[0]['question']
        return jsonify({"reply": f"{short_answer}<br><br>Would you like to know more? Reply with 'I'm interested'."})
    else:
        return jsonify({"reply": "Sorry, I don't know the answer to that."})

# Important for Render or any cloud platform to detect the port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT environment variable
    app.run(host="0.0.0.0", port=port)
