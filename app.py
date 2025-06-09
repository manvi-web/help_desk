from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load vectorizer and model
with open("chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load CSV file
CSV_PATH = "chatbot_qa_dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Ensure 'question' column is lowercased for reliable matching
df["question"] = df["question"].str.lower()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_info = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip().lower()

        if user_input:
            # Follow-up intent
            if user_input == "i'm interested" and "last_question" in session:
                last_q = session["last_question"]
                row = df[df["question"] == last_q]
                if not row.empty:
                    full_info = row.iloc[0]["full_answer"] + "<br><a href='{}' target='_blank'>Read more</a>".format(row.iloc[0]["URL"])
                else:
                    full_info = "Sorry, I couldn't find more details."
            else:
                # Predict closest question
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0]
                predicted_question = prediction.lower()
                session["last_question"] = predicted_question

                row = df[df["question"] == predicted_question]
                if not row.empty:
                    answer = row.iloc[0]["short_answer"]
                else:
                    answer = "Sorry, I couldn't find an answer for that."

    return render_template("index.html", answer=answer, full_info=full_info)

if __name__ == "__main__":
    # Bind to PORT for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
