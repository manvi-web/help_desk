import os
from flask import Flask, request, render_template, session
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("effort_qa_dataset.csv")

# Convert all to string
df['question'] = df['question'].astype(str)
df['short_answer'] = df['short_answer'].astype(str)
df['full_answer'] = df['full_answer'].astype(str)
df['URL'] = df['URL'].astype(str)

# Prepare model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])
model = LogisticRegression()
model.fit(X, df.index)  # Target is the row index

# Flask setup
app = Flask(__name__)
app.secret_key = 'any-secret-key'  # For session

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_answer = ""
    url = ""
    
    if request.method == "POST":
        user_input = request.form["question"].strip()
        
        if user_input.lower() == "i'm interested":
            if "last_index" in session:
                idx = session["last_index"]
                full_answer = df.loc[idx, "full_answer"]
                url = df.loc[idx, "URL"]
                answer = f"{full_answer}<br><br><a href='{url}' target='_blank'>Read more</a>"
            else:
                answer = "Please ask a question first."
        else:
            user_vector = vectorizer.transform([user_input])
            pred_index = model.predict(user_vector)[0]
            session["last_index"] = int(pred_index)  # store for later
            answer = df.loc[pred_index, "short_answer"]
    
    return render_template("index.html", answer=answer)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
