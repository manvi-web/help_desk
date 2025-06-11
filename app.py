from flask import Flask, render_template, request, session
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'secret'

# Load the dataset, model, and vectorizer
df = pd.read_csv("effort_qa_dataset.csv")
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocess questions
tfidf_matrix = vectorizer.transform(df['question'])

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Check if user typed "I'm interested"
        if "interested" in user_input.lower() and "last_question" in session:
            matched_row = session.get("last_question")
            full_answer = matched_row["full_answer"]
            url = matched_row["URL"]
            response = f"<b>More info:</b> {full_answer}<br><a href='{url}' target='_blank'>Click here to read more</a>"
            session.pop("last_question", None)
        else:
            # Compute similarity
            user_vec = vectorizer.transform([user_input])
            similarity = cosine_similarity(user_vec, tfidf_matrix)
            idx = similarity.argmax()

            # Get top match
            matched_row = df.iloc[idx]
            short_answer = matched_row["short_answer"]
            response = f"{short_answer}<br><br>Would you like to know more? Type 'I'm interested'."

            # Store last matched question
            session["last_question"] = matched_row.to_dict()

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
