from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load vectorizer, model, and label encoder
with open("chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("chatbot_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load CSV
df = pd.read_csv("chatbot_qa_dataset.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_info = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip().lower()

        if user_input == "i'm interested" and "last_index" in session:
            idx = session["last_index"]
            full = df.iloc[idx].get("full_answer", "Full answer not available.")
            url = df.iloc[idx].get("URL", "")
            full_info = f"{full}<br><a href='{url}' target='_blank'>Read more</a>" if url else full

        elif user_input:
            X = vectorizer.transform([user_input])
            pred_idx = model.predict(X)[0]

            try:
                predicted_question = le.inverse_transform([pred_idx])[0]
                match = df[df["question"].str.lower() == predicted_question.lower()]
                if not match.empty:
                    idx = match.index[0]
                    short = match.iloc[0]["short_answer"]
                    answer = f"Answer: {short}"
                    session["last_index"] = idx
                else:
                    answer = "Sorry, I couldn't find an answer for that."
            except:
                answer = "Sorry, I couldn't interpret your question."

    return render_template("index.html", answer=answer, full_info=full_info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
