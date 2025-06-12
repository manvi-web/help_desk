from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Set up Chrome browser
def setup_browser():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver

# Extract Q&A data from a topic page
def extract_topic(driver, url):
    try:
        driver.get(url)
        time.sleep(2)
        title = driver.title
        content_element = driver.find_element(By.TAG_NAME, 'body')
        full_text = content_element.text.strip()
        short_answer = full_text[:250] + '...' if len(full_text) > 250 else full_text

        return {
            "Title": title,
            "Short Answer": short_answer,
            "Full Answer": full_text,
            "URL": url
        }
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

def main():
    base_url = "https://geteffort.com/manual/"
    print("📘 Opening Effort Manual...")
    driver = setup_browser()
    driver.get(base_url)

    wait = WebDriverWait(driver, 10)

    time.sleep(5)

    # Expand all accordion/menu buttons
    try:
        expandable_buttons = driver.find_elements(By.CLASS_NAME, 'v-list-item__icon')
        print(f"📂 Found {len(expandable_buttons)} expandable items.")
        for btn in expandable_buttons:
            try:
                btn.click()
                time.sleep(0.5)
            except:
                continue
    except Exception as e:
        print(f"⚠️ Error expanding menus: {e}")

    time.sleep(3)

    # Extract all links
    links = driver.find_elements(By.TAG_NAME, 'a')
    topic_links = []
    for link in links:
        href = link.get_attribute("href")
        if href and "/manual/" in href and "https://geteffort.com/manual/" in href:
            topic_links.append(href)

    topic_links = list(set(topic_links))  # remove duplicates
    print(f"🔗 Found {len(topic_links)} topic links.")

    # Visit and extract each page
    qa_data = []
    for idx, link in enumerate(topic_links):
        print(f"🔍 Scraping ({idx+1}/{len(topic_links)}): {link}")
        data = extract_topic(driver, link)
        if data:
            qa_data.append(data)

    driver.quit()
    print("🛑 Browser closed.")

    print(f"\n📊 Total pages scraped: {len(qa_data)}")

    if qa_data:
        df = pd.DataFrame(qa_data)
        df.to_csv("effort_qa_dataset.csv", index=False)
        print("✅ CSV saved as 'effort_qa_dataset.csv'")
    else:
        print("❌ No data scraped. CSV not saved.")

if __name__ == "__main__":
    main()
