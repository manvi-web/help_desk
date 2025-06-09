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

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_info = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip().lower()

        if user_input:
            if user_input == "i'm interested" and "last_question" in session:
                last_q = session["last_question"]
                row = df[df["question"].str.lower() == last_q]
                if not row.empty:
                    full_answer = row.iloc[0]["full_answer"]
                    url = row.iloc[0]["URL"]
                    full_info = f"{full_answer}<br><a href='{url}' target='_blank'>Read more</a>"
                else:
                    full_info = "Sorry, more details not found."
            else:
                # Vectorize and predict
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0]
                predicted_question = str(prediction).lower()
                session["last_question"] = predicted_question

                row = df[df["question"].str.lower() == predicted_question]
                if not row.empty:
                    answer = row.iloc[0]["short_answer"]
                else:
                    answer = "Sorry, I couldn't find an answer for that."

    return render_template("index.html", answer=answer, full_info=full_info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
