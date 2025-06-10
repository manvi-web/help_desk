from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = "helpdesk_secret_key"

# Port binding for Render
PORT = int(os.environ.get("PORT", 10000))

# Load dataset
df = pd.read_csv("effort_qa_dataset.csv")
df.fillna("", inplace=True)

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["message"].strip()

        # If user says "I'm interested"
        if user_input.lower() in ["i'm interested", "interested", "yes"]:
            last_index = session.get("last_index")
            if last_index is not None:
                full = df.loc[last_index, "full_answer"]
                url = df.loc[last_index, "url"]
                response = f"Here's more info:\n\n{full}\n\nRead more: {url}"
            else:
                response = "Please ask a question first."
        else:
            # Compute similarity with questions
            user_vec = vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vec, X)
            best_index = similarities.argmax()

            # Save last question
            session["last_index"] = int(best_index)

            short_ans = df.loc[best_index, "short_answer"]
            response = f"{short_ans}\n\nWould you like to know more? Type 'I'm interested'."

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
