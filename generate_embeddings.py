import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the cleaned dataset
df = pd.read_csv("effort_qa_dataset_cleaned.csv")

# Use the Title field for similarity matching
titles = df["Title"].astype(str).tolist()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(titles)

# Save vectorizer, matrix, and dataframe
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("qa_dataframe.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ Embedding files generated and saved.")
