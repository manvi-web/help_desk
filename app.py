# app.py
from flask import Flask, render_template, request, session
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model files
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
df = pickle.load(open("qa_dataframe.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        user_input = request.form["question"]
        last_q_idx = session.get("last_q_idx")

        if user_input.strip().lower() == "i'm interested" and last_q_idx is not None:
            full_ans = df.iloc[last_q_idx]["Full Answer"]
            url = df.iloc[last_q_idx]["URL"]
            answer = f"{full_ans}<br><br><a href='{url}' target='_blank'>Read more</a>"
        else:
            user_vec = vectorizer.transform([user_input])
            sims = cosine_similarity(user_vec, tfidf_matrix)
            best_match = np.argmax(sims)
            session["last_q_idx"] = int(best_match)
            answer = df.iloc[best_match]["Short Answer"] + "<br><br>Would you like to know more? Type 'I'm interested'."

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
