import os
from flask import Flask, request, jsonify, render_template, session
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

# Load CSV data
csv_path = os.path.join(os.path.dirname(__file__), 'effort_chatbot_data.csv')
df = pd.read_csv(csv_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()

    # Handle "I'm interested"
    interested_phrases = ["i'm interested", "im interested", "interested", "more info", "tell me more"]
    if user_input in interested_phrases:
        last_question = session.get("last_question", "")
        if last_question:
            match = df[df['question'].str.lower() == last_question.lower()]
            if not match.empty:
                full_answer = match.iloc[0]['full_answer']
                url = match.iloc[0]['URL']
                return jsonify({"reply": f"{full_answer}<br><br><a href='{url}' target='_blank'>Read more</a>"})
            else:
                return jsonify({"reply": "Sorry, I couldn't find more info on that."})
        return jsonify({"reply": "Please ask a question first."})

    # Normal question matching
    match = df[df['question'].str.lower().str.contains(user_input, na=False)]

    if not match.empty:
        short_answer = match.iloc[0]['short_answer']
        session['last_question'] = match.iloc[0]['question']
        return jsonify({"reply": f"{short_answer}<br><br>Type 'I'm interested' to know more."})
    else:
        return jsonify({"reply": "Sorry, I don't know the answer to that."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Required for Render
    app.run(host="0.0.0.0", port=port)
