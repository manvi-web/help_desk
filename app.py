from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths
DATASET_PATH = 'chatbot_qa_dataset.csv'
MODEL_PATH = 'chatbot_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load or train the model
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    df = pd.read_csv(DATASET_PATH)
else:
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    df.dropna(subset=['question', 'short_answer', 'full_answer', 'URL'], inplace=True)

    # Vectorize questions
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['question'])
    y = df.index  # Using row index as the label

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Save model and vectorizer
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        user_input = request.form['user_input'].strip()
        if not user_input:
            answer = "❌ Please enter a question."
        elif user_input.lower() == "i'm interested":
            if 'last_index' in session:
                idx = session['last_index']
                full_answer = df.loc[idx, 'full_answer']
                url = df.loc[idx, 'URL']
                answer = f"{full_answer} 👉 <a href='{url}' target='_blank'>Read more</a>"
            else:
                answer = "❌ Please ask a question first."
        else:
            try:
                X_input = vectorizer.transform([user_input])
                prediction = model.predict(X_input)[0]
                short_answer = df.loc[prediction, 'short_answer']
                session['last_index'] = prediction
                answer = short_answer
            except Exception as e:
                answer = f"❌ Internal error occurred: {str(e)}"
    return render_template("index.html", answer=answer)

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
