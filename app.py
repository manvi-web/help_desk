from flask import Flask, request, render_template, session
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = 'secret_key'

# File paths
csv_path = "chatbot_qa_dataset.csv"
model_path = "chatbot_model.pkl"
vectorizer_path = "vectorizer.pkl"

# Load or train model
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    df = pd.read_csv(csv_path)
else:
    # Load data and train model
    df = pd.read_csv(csv_path)
    df.dropna(subset=['question', 'short_answer', 'full_answer', 'URL'], inplace=True)
    X = df['question']
    y = df.index

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_vec, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        user_input = request.form.get("question", "").strip()
        last_question = session.get("last_question")

        if user_input.lower() == "i'm interested" and last_question is not None:
            row = df.loc[last_question]
            answer = f"{row['full_answer']}<br><br><a href='{row['URL']}' target='_blank'>Read more</a>"
        else:
            X_input = vectorizer.transform([user_input])
            prediction = model.predict(X_input)[0]
            session["last_question"] = prediction
            row = df.loc[prediction]
            answer = row["short_answer"]
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
