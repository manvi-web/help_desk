import pandas as pd

# Load your original dataset
df = pd.read_csv("effort_qa_dataset.csv")

# Optional cleanup: strip whitespaces and drop missing rows
df.columns = [col.strip().title() for col in df.columns]
df = df.rename(columns={
    'Question': 'Title',
    'Short_Answer': 'Short Answer',
    'Full_Answer': 'Full Answer',
    'Url': 'URL'
})
df = df[['Title', 'Short Answer', 'Full Answer', 'URL']]  # Ensure only needed columns
df.dropna(inplace=True)

# Save cleaned CSV
df.to_csv("effort_qa_dataset_cleaned.csv", index=False)
print("Cleaned CSV saved as effort_qa_dataset_cleaned.csv")
