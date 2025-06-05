from flask import Flask, render_template, request, session
import pandas as pd
import re
import os
import pickle

# Load model and vectorizer
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load CSV data
CSV_PATH = "chatbot_qa_dataset.csv"  # Make sure this name matches your actual file
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError as e:
    print(f"❌ Error loading CSV: {e}")
    exit(1)

# Flask app
app = Flask(__name__)
app.secret_key = "secret123"

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_answer = ""
    url = ""
    
    if request.method == "POST":
        user_input = request.form["user_input"]

        # If user said "I'm interested", show full answer
        if user_input.lower().strip() == "i'm interested" and "last_question" in session:
            matched_row = df[df["question"] == session["last_question"]]
            if not matched_row.empty:
                full_answer = matched_row.iloc[0]["full_answer"]
                url = matched_row.iloc[0]["URL"]
        else:
            # Predict
            X_input = vectorizer.transform([user_input])
            predicted_question = model.predict(X_input)[0]
            session["last_question"] = predicted_question

            matched_row = df[df["question"] == predicted_question]
            if not matched_row.empty:
                answer = matched_row.iloc[0]["short_answer"]

    return render_template("index.html", answer=answer, full_answer=full_answer, url=url)

# Run app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
