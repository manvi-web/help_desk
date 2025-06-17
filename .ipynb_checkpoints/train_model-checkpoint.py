import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

try:
    # Step 1: Load the CSV file
    csv_file = "effort_qa_dataset.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"❌ File not found: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"✅ Loaded CSV with {len(df)} rows")

    # Step 2: Validate columns
    expected_columns = {'Title', 'Short Answer', 'Full Answer', 'URL'}
    actual_columns = set(df.columns)
    if not expected_columns.issubset(actual_columns):
        raise ValueError(f"❌ CSV is missing columns. Found: {actual_columns}")

    # Step 3: Fill missing values
    df.fillna("", inplace=True)

    # Step 4: Save qa_dataframe.pkl
    with open("qa_dataframe.pkl", "wb") as f:
        pickle.dump(df, f)
    print("✅ Saved qa_dataframe.pkl")

    # Step 5: Create TF-IDF corpus
    corpus = (df["Title"] + " " + df["Short Answer"] + " " + df["Full Answer"]).tolist()

    # Step 6: Train vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print("✅ TF-IDF vectorizer trained")

    # Step 7: Save vectorizer.pkl
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("✅ Saved vectorizer.pkl")

    # Step 8: Save tfidf_matrix.pkl
    with open("tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    print("✅ Saved tfidf_matrix.pkl")

    print("\n🎉 All files generated successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
