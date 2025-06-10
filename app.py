from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import pickle
import os
app = Flask(__name__)
app.secret_key = 'your_secret_key'  
df = pd.read_csv("chatbot_qa_dataset.csv")
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_answer = ""
    url = ""

    if request.method == "POST":
        user_input = request.form["user_input"]

        if user_input.lower() == "i'm interested" and "last_question_index" in session:
            idx = session["last_question_index"]
            full_answer = df.loc[idx, "full_answer"]
            url = df.loc[idx, "URL"]
        else:
            # Vectorize input
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            prediction = int(prediction) 
            session["last_question_index"] = prediction

            answer = df.loc[prediction, "short_answer"]

    return render_template("index.html", answer=answer, full_answer=full_answer, url=url)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)
