from flask import Flask, request, render_template, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Port binding for Render
PORT = int(os.environ.get("PORT", 10000))

# Load dataset
df = pd.read_csv("effort_qa_dataset.csv")

# Fill NaNs
df.fillna("", inplace=True)

# Preprocess questions
questions = df["question"].astype(str).tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train model
model = LogisticRegression()
model.fit(X, df.index)

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip().lower()

        # If user previously asked a question and now says "I'm interested"
        if "last_index" in session and "interested" in user_input:
            idx = session["last_index"]
            full_answer = df.loc[idx, "full_answer"]
            url = df.loc[idx, "URL"]
            response = f"<b>Full Answer:</b><br>{full_answer}<br><br><a href='{url}' target='_blank'>Read More</a>"
            session.pop("last_index", None)

        else:
            # Predict index
            vec = vectorizer.transform([user_input])
            pred_idx = model.predict(vec)[0]

            # Save for follow-up
            session["last_index"] = int(pred_idx)

            short_answer = df.loc[pred_idx, "short_answer"]
            response = f"{short_answer}<br><br>Would you like to know more? Type <b>I'm interested</b>."

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
