from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os

# === CONFIGURATION ===
CSV_PATH = "/home/spoors/help_desk/effort_qa_dataset.csv"
MODEL_PATH = "chatbot_model.pkl"
VECTORIZER_PATH = "chatbot_vectorizer.pkl"

# === APP SETUP ===
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

# === DEBUG: Show working directory ===
print("Current Working Directory:", os.getcwd())

# === LOAD FILES ===
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded CSV with {len(df)} rows.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    print("✅ Model and vectorizer loaded.")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    raise

# === ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    full_answer = ""
    url = ""
    
    if request.method == 'POST':
        user_input = request.form.get('question', '').strip()

        if user_input:
            if user_input.lower() == "i'm interested" and 'last_index' in session:
                idx = session['last_index']
                full_answer = df.iloc[idx]['full_answer']
                url = df.iloc[idx]['URL']
            else:
                # Predict
                input_vector = vectorizer.transform([user_input])
                prediction = model.predict(input_vector)[0]
                
                # Find match in CSV
                match = df[df['question'].str.lower() == prediction.lower()]
                if not match.empty:
                    idx = match.index[0]
                    answer = match.iloc[0]['short_answer']
                    session['last_index'] = idx
                else:
                    answer = "Sorry, I couldn't find a relevant answer."

    return render_template("index.html", answer=answer, full_answer=full_answer, url=url)

# === RUN ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
