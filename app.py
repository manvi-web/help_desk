import os
import pandas as pd
import pickle
from flask import Flask, request, render_template, session
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and vectorizer
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load dataset
df = pd.read_csv("effort_qa_dataset.csv")

# Ensure all responses are strings
df.fillna('', inplace=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ''
    if request.method == 'POST':
        message = request.form['message'].strip()

        # If user replies with interest
        if message.lower() in ["i'm interested", "interested", "yes", "i am interested"]:
            if 'last_index' in session:
                idx = session['last_index']
                full_ans = df.loc[idx, 'full_answer']
                url = df.loc[idx, 'URL']
                response = f"<b>Here's more information:</b><br>{full_ans}<br><br><a href='{url}' target='_blank'>Read more</a>"
            else:
                response = "Please ask a question first."
        else:
            # Predict answer using similarity
            vec_input = vectorizer.transform([message])
            sims = cosine_similarity(vec_input, model).flatten()
            best_idx = sims.argmax()
            best_score = sims[best_idx]

            # Threshold: ignore bad matches
            if best_score < 0.3:
                response = "Sorry, I couldn’t find anything relevant. Try rephrasing your question."
            else:
                short_ans = df.loc[best_idx, 'short_answer']
                session['last_index'] = int(best_idx)  # Store index for interest
                response = f"{short_ans}<br><br><i>Would you like to know more? Type 'I'm interested'.</i>"

    return render_template('index.html', response=response)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
