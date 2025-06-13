import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Use the cleaned CSV you now have
df = pd.read_csv("effort_qa_dataset_real_content.csv")

# Save DataFrame so the app can use it
with open("qa_dataframe.pkl", "wb") as f:
    pickle.dump(df, f)

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Title'])  # OR 'Question' if you renamed it

# Save vectorizer and matrix
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("✅ Model files saved: vectorizer.pkl, tfidf_matrix.pkl, qa_dataframe.pkl")
