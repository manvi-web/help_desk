import pandas as pd

df = pd.read_csv("effort_qa_dataset.csv")
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "question": "Title",
    "short_answer": "Short Answer",
    "full_answer": "Full Answer",
    "URL": "URL"
})

df.to_csv("effort_qa_dataset_cleaned.csv", index=False)
print("CSV cleaned and saved as effort_qa_dataset_cleaned.csv")

