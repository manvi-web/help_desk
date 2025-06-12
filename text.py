import pandas as pd

# Load CSV
df = pd.read_csv("effort_qa_dataset.csv")

# Remove common repeated phrases
repeated_phrases = [
    "Jump to main content", 
    "Reference Manual", 
    "Search", 
    "=== Main Page ==="
]

def clean_text(text):
    for phrase in repeated_phrases:
        text = text.replace(phrase, "")
    return text.strip()

df["Short Answer"] = df["Short Answer"].apply(clean_text)
df["Full Answer"] = df["Full Answer"].apply(clean_text)

# Remove rows with too short answers or empty content
df = df[df["Short Answer"].str.len() > 30]
df = df[df["Full Answer"].str.len() > 100]

# Save cleaned file
df.to_csv("effort_qa_dataset_cleaned.csv", index=False)
print("✅ Cleaned CSV saved as 'effort_qa_dataset_cleaned.csv'")
