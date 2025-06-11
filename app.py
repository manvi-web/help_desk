from flask import Flask, render_template, request, session
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load data and models
df = pd.read_csv("effort_qa_dataset.csv")
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

df.fillna("", inplace=True)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        user_input = request.form["question"].strip().lower()

        # Check for interest response
        if user_input == "i'm interested" and "last_question" in session:
            last_index = session["last_question"]
            full = df.loc[last_index, "full_answer"]
            url = df.loc[last_index, "URL"]
            answer = f"<b>More Info:</b> {full}<br><br><a href='{url}' target='_blank'>Click here for full manual</a>"
        else:
            # Regular question
            question_vec = vectorizer.transform([user_input])
            corpus_vec = vectorizer.transform(df["question"])
            scores = cosine_similarity(question_vec, corpus_vec).flatten()
            top_idx = scores.argmax()

            short = df.loc[top_idx, "short_answer"]
            session["last_question"] = int(top_idx)
            answer = f"{short}<br><br><i>Would you like to know more? Type 'I'm interested'.</i>"

    return render_template("index.html", answer=answer)

# Render-compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
