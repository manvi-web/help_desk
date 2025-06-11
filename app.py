from flask import Flask, request, render_template, session
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load CSV and model
df = pd.read_csv("effort_qa_dataset.csv")
df['question'] = df['question'].astype(str)
df['short_answer'] = df['short_answer'].astype(str)
df['full_answer'] = df['full_answer'].astype(str)
df['URL'] = df['URL'].astype(str)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

# TF-IDF matrix for all questions
X = vectorizer.transform(df["question"])

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Check if user says "I'm interested"
        if user_input.lower() == "i'm interested" and session.get("last_index") is not None:
            i = session["last_index"]
            full_answer = df.iloc[i]["full_answer"]
            url = df.iloc[i]["URL"]
            response = f"{full_answer}<br><br><a href='{url}' target='_blank'>Read more</a>"
        else:
            # Predict
            user_vec = vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vec, X).flatten()
            best_match_index = similarities.argmax()
            best_match_score = similarities[best_match_index]

            if best_match_score < 0.3:
                response = "Sorry, I couldn’t find a relevant answer. Please try rephrasing your question."
            else:
                short_answer = df.iloc[best_match_index]["short_answer"]
                session["last_index"] = best_match_index
                response = f"{short_answer}<br><br>Would you like to know more? Type 'I'm interested'."

    return render_template("index.html", response=response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
