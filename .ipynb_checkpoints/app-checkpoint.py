from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os
app = Flask(__name__)
app.secret_key = "supersecretkey"
df = pd.read_csv("chatbot_qa_dataset.csv")
df["question"] = df["question"].astype(str)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    full_info = ""

    if request.method == "POST":
        user_input = request.form["message"]

        if user_input.lower().strip() == "i'm interested" and "last_question" in session:
            last_q = session["last_question"]
            row = df[df["question"] == last_q].iloc[0]
            full_info = row["full_answer"] + f"<br><a href='{row['URL']}' target='_blank'>Read More</a>"
        else:
            try:
                X_test = vectorizer.transform([user_input])
                pred = model.predict(X_test)[0]
                pred = str(pred)
                row = df[df["question"] == pred].iloc[0]
                response = row["short_answer"]
                session["last_question"] = pred
            except Exception as e:
                print("Prediction error:", e)
                response = "Sorry, I couldn't find an answer for that."

    return render_template("index.html", response=response, full_info=full_info)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
