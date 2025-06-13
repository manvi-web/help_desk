import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your new file
df = pd.read_csv("effort_qa_dataset_real_content.csv")

# Combine Title + Full Answer for accurate matching
df["combined"] = df["Title"] + " " + df["Full Answer"]

# TF-IDF transformation
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# Save vectorizer, matrix, and dataframe
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("qa_dataframe.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ Model training complete. Files saved!")
