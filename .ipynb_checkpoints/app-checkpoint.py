from flask import Flask, request, render_template, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load dataset
df = pd.read_csv("effort_qa_dataset.csv")

# Vectorize questions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    full_answer = ""
    url = ""
    
    if request.method == "POST":
        user_input = request.form["user_input"].strip().lower()

        if user_input == "i'm interested" and "last_index" in session:
            idx = session["last_index"]
            full_answer = df.loc[idx, "full_answer"]
            url = df.loc[idx, "URL"]
            response = f"{full_answer}<br><br><a href='{url}' target='_blank'>Click here for more info</a>"
        else:
            # Find best match
            input_vector = vectorizer.transform([user_input])
            similarity = cosine_similarity(input_vector, tfidf_matrix)
            best_match_idx = similarity.argmax()

            short_answer = df.loc[best_match_idx, "short_answer"]
            response = f"{short_answer}<br><br>Would you like to know more? Type 'I'm interested'."
            session["last_index"] = best_match_idx

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
