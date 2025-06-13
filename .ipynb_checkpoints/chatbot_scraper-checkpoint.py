import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re

# === CONFIG ===
INPUT_CSV = "effort_qa_dataset_cleaned.csv"
OUTPUT_CSV = "effort_qa_dataset_real_content.csv"

# Setup Selenium in headless mode
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

# Load input CSV
df = pd.read_csv(INPUT_CSV)

# Function to clean content
def clean_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Remove navigation, headers, footers if any
    for tag in soup(['nav', 'header', 'footer', 'script', 'style']):
        tag.decompose()
    
    # Get clean text from main content
    content = soup.get_text(separator=' ', strip=True)
    # Remove repeated menu terms or common boilerplate text
    content = re.sub(r'(Jump to main content|Reference Manual|Search)+', '', content, flags=re.I)
    content = re.sub(r'\s{2,}', ' ', content)
    return content.strip()

# Iterate and scrape
full_answers = []
for i, row in df.iterrows():
    url = row['URL']
    print(f"Scraping: {url}")
    try:
        driver.get(url)
        time.sleep(3)  # wait for JS to load

        cleaned = clean_text(driver.page_source)
        full_answers.append(cleaned)
    except Exception as e:
        print(f"Failed: {url} → {e}")
        full_answers.append("")

df['Full Answer'] = full_answers

# Optional: regenerate short answer from first sentence
short_answers = [text.split('.')[0] + '.' if text else "" for text in full_answers]
df['Short Answer'] = short_answers

# Save new CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Scraped data saved to: {OUTPUT_CSV}")

driver.quit()
