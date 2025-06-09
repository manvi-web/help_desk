ffrom flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os
import traceback  # for detailed error logging

app = Flask(__name__)
app.secret_key = "your_secret_key"

try:
    with open("chatbot_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("chatbot_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    df = pd.read_csv("chatbot_qa_dataset.csv")
except Exception as e:
    print("❌ Error loading files:", e)
    traceback.print_exc()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_info = ""

    try:
        if request.method == "POST":
            user_input = request.form.get("user_input", "").strip().lower()

            if user_input == "i'm interested" and "last_index" in session:
                idx = session["last_index"]
                full_answer = df.iloc[idx].get("full_answer", "Full answer not available.")
                url = df.iloc[idx].get("URL", "")
                full_info = f"{full_answer}<br><a href='{url}' target='_blank'>Read more</a>" if url else full_answer
            elif user_input:
                X = vectorizer.transform([user_input])
                predicted_question = model.predict(X)[0]

                # Fix: if the model returns int (label), convert it to string
                if isinstance(predicted_question, (int, float)):
                    predicted_question = str(predicted_question)

                matches = df[df["question"].str.lower() == predicted_question.lower()]
                if not matches.empty:
                    idx = matches.index[0]
                    session["last_index"] = idx
                    short_answer = matches.iloc[0].get("short_answer", "No short answer found.")
                    answer = f"Answer: {short_answer}"
                else:
                    answer = "Sorry, I couldn't find an answer for that."

    except Exception as e:
        print("❌ Error during request:", e)
        traceback.print_exc()
        answer = "❌ Internal error occurred."

    return render_template("index.html", answer=answer, full_info=full_info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
