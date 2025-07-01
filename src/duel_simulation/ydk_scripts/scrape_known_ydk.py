import time
import logging
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URLS = [
    #"https://ygoprodeck.com/deck-search/?_sft_category=edison%20format%20decks&sort=Deck%20Views&offset=0",
    #"https://ygoprodeck.com/deck-search/?_sft_category=goat%20format%20decks&sort=Deck%20Views&offset=0",
    "https://ygoprodeck.com/deck-search/?_sft_category=theorycrafting%20decks&sort=Deck%20Views&offset=0",
    "https://ygoprodeck.com/deck-search/?_sft_category=fun%2Fcasual%20decks&sort=Deck%20Views&offset=0",
    "https://ygoprodeck.com/deck-search/?_sft_category=non-meta%20decks&sort=Deck%20Views&offset=0",
    "https://ygoprodeck.com/deck-search/?_sft_category=meta%20decks&sort=Deck%20Views&offset=0",
    "https://ygoprodeck.com/deck-search/?_sft_category=Tournament%20Meta%20Decks&sort=Deck%20Views&offset=0"
]

YGOPRO_BASE = "https://ygoprodeck.com"
DOWNLOAD_DIR = os.path.join(os.getcwd(), "decks/known")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=options)
    return driver

def extract_deck_links(driver, url):
    logger.info(f"Visiting {url}")
    driver.get(url)
    time.sleep(3)  # Allow time for JavaScript to render

    deck_links = []
    anchors = driver.find_elements(By.CSS_SELECTOR, "a.deck_article-card-title")
    for anchor in anchors:
        name = anchor.text.strip()
        href = anchor.get_attribute("href")
        logger.info(f"Found deck: {name} -> {href}")
        deck_links.append((name, href))

    return deck_links

def extract_ydk_content(driver, deck_url, wait_time=10, output_dir="decks/known"):
    logging.info(f"Opening deck page: {deck_url}")
    driver.get(deck_url)

    try:
        # Step 1: Wait for and scroll to the "More" dropdown button
        more_button = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.ID, "dropdownMenuButton"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", more_button)
        time.sleep(0.5)  # let page settle
        driver.execute_script("arguments[0].click();", more_button)

        # Step 2: Click the "Download YDK" option inside the dropdown
        download_button = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Download YDK')]"))
        )
        driver.execute_script("arguments[0].click();", download_button)

        # Step 3: Wait for and extract the YDK content from the <pre> tag
        pre_block = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "pre"))
        )
        ydk_content = pre_block.text

        # Step 4: Write it to file
        slug = urlparse(deck_url).path.strip("/").split("/")[-1]
        filename = f"{slug}.ydk"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(ydk_content)

        logging.info(f"✅ Saved YDK file to {filepath}")
        return filepath

    except Exception as e:
        logging.warning(f"⚠️ Could not extract ydk from {deck_url}: {e}")
        return None


def main():
    driver = get_driver()
    try:
        all_decks = []
        for base_url in BASE_URLS:
            deck_links = extract_deck_links(driver, base_url)
            for name, url in deck_links:
                ydk_file = extract_ydk_content(driver, url)
                if ydk_file:
                    all_decks.append((name, url, ydk_file))

        # Log summary
        for name, url, ydk in all_decks:
            print(f"\n{name} -> {url}\n{ydk}\n")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()