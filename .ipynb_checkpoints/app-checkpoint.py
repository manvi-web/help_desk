ffrom flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load model and vectorizer
with open("chatbot_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load CSV
CSV_PATH = "chatbot_qa_dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    full_info = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip().lower()

        if user_input:
            if user_input == "i'm interested" and "last_index" in session:
                last_index = session["last_index"]
                if 0 <= last_index < len(df):
                    full = df.iloc[last_index]["full_answer"]
                    url = df.iloc[last_index]["URL"]
                    full_info = f"{full}<br><a href='{url}' target='_blank'>Read more</a>"
            else:
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0]

                # Handle prediction index or string
                try:
                    index = int(prediction)
                except:
                    index = df[df["question"].str.lower() == prediction.lower()].index
                    index = index[0] if not index.empty else -1

                if 0 <= index < len(df):
                    answer = df.iloc[index]["short_answer"]
                    session["last_index"] = index
                else:
                    answer = "Sorry, I couldn't find an answer for that."

    return render_template("index.html", answer=answer, full_info=full_info)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
