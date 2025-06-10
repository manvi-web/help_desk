import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Auto-detect file location inside help_desk/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "chatbot_qa_dataset.csv")

# Load CSV
df = pd.read_csv(csv_path)

# Check required columns
required_columns = ['question', 'short_answer', 'full_answer', 'URL']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

# Encode short answers as labels
le = LabelEncoder()
y = le.fit_transform(df['short_answer'])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model, vectorizer, label encoder
joblib.dump(model, os.path.join(BASE_DIR, "chatbot_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))

print("✅ Model, vectorizer, and label encoder saved successfully.")
