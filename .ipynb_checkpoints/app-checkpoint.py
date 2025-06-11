from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load dataset
df = pd.read_csv("effort_qa_dataset.csv")

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

@app.route('/', methods=["GET", "POST"])
def index():
    response = ""
    full = ""
    url = ""
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Check if user said "I'm interested"
        if user_input.lower().strip() == "i'm interested" and "last_index" in session:
            idx = session["last_index"]
            full = df.iloc[idx]["full_answer"]
            url = df.iloc[idx]["URL"]
            response = f"{full}\n\n🔗 <a href='{url}' target='_blank'>Read more</a>"
        else:
            # Compute similarity
            user_vec = vectorizer.transform([user_input])
            sim = cosine_similarity(user_vec, X)
            idx = sim.argmax()

            session["last_index"] = idx
            short = df.iloc[idx]["short_answer"]
            response = f"{short}\n\nWould you like to know more? Type 'I'm interested'."

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
