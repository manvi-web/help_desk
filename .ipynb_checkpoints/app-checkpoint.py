from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"  # You can replace this

# Load model and vectorizer
try:
    with open("chatbot_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("chatbot_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading model or vectorizer:", e)
    raise

# Load CSV
CSV_PATH = "chatbot_qa_dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_info = ""
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip().lower()

        if user_input:
            # If user asks for more info
            if user_input == "i'm interested" and "last_question" in session:
                last_q = session["last_question"]
                row = df[df["question"].str.lower() == last_q]
                if not row.empty:
                    full_info = (
                        row.iloc[0]["full_answer"]
                        + f"<br><a href='{row.iloc[0]['URL']}' target='_blank'>Read more</a>"
                    )
                else:
                    full_info = "No additional information found."
            else:
                try:
                    X = vectorizer.transform([user_input])
                    prediction = model.predict(X)[0]
                    session["last_question"] = prediction.lower()
                    row = df[df["question"].str.lower() == prediction.lower()]
                    if not row.empty:
                        answer = row.iloc[0]["short_answer"]
                    else:
                        answer = "Sorry, I couldn't find an answer."
                except Exception as e:
                    answer = f"Error processing input: {str(e)}"

    return render_template("index.html", answer=answer, full_info=full_info, user_input=user_input)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Important for Render
    app.run(host="0.0.0.0", port=port)
