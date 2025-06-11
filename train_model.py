import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load CSV
df = pd.read_csv("effort_qa_dataset.csv")

# Step 2: Rename columns to expected format
df = df.rename(columns={
    "question": "Title",
    "short_answer": "Short Answer",
    "full_answer": "Full Answer"
})

# Step 3: Validate required columns exist
required_columns = ["Title", "Short Answer", "Full Answer", "URL"]
for col in required_columns:
    if col not in df.columns:
        raise Exception(f"Missing column: {col}")

# Step 4: Combine fields for training
df["combined"] = df["Title"] + " " + df["Short Answer"] + " " + df["Full Answer"]

# Step 5: Train TF-IDF model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# Step 6: Save model and data
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("qa_dataframe.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ Training complete. Model and data saved.")
