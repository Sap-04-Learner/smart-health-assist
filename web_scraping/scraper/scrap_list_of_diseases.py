import json
import time

import requests
from selectolax.parser import HTMLParser

from .config import BASE_URL, START_URL, HEADERS, OUTPUT_DIR
from .logging import logger

def is_letter_heading(text: str) -> bool:
    '''Check if text is a single uppercase letter'''
    text = text.strip()
    return len(text) == 1 and text.isupper() and text.isalpha()

def scrape_a_to_z() -> dict:
    logger.info(f"Fetching A-Z index page... {START_URL}")
    try:
        response = requests.get(START_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch page: {e}")
        return {}

    parser = HTMLParser(response.text)
    diseases_by_letter = {}  # Final structure: {"A": {"Acne": "url", ...}, ...}

    # Find all h2 tags
    headings = parser.css("h2")
    logger.info(f"Found {len(headings)} <h2> tags on page")

    total_diseases = 0

    for heading in headings:
        letter_text = heading.text().strip()
        if not is_letter_heading(letter_text):
            continue  # Skip non-letter headings (e.g., page title)

        logger.info(f"Processing letter: {letter_text}")

        # Find the next ul sibling after this h2
        current = heading.next
        ul_found = None
        while current:
            if current.tag == "ul":
                ul_found = current
                break
            current = current.next

        if not ul_found:
            logger.warning(f"No <ul> found after letter {letter_text}")
            continue

        # Extract links from li > a
        diseases_names = {}  # {name: url} for this letter
        links = ul_found.css("li a")

        for link in links:
            name = link.text().strip()
            href = link.attrs.get("href")
            if not name or not href:
                continue

            # Make absolute URL
            if href.startswith("/"):
                full_url = f"{BASE_URL}{href}"
            elif href.startswith("http"):
                full_url = href
            else:
                full_url = f"{BASE_URL}/{href.lstrip('/')}"

            diseases_names[name] = full_url

        if diseases_names:
            diseases_by_letter[letter_text] = diseases_names
            total_diseases += len(diseases_names)
            logger.info(f"  → Found {len(diseases_names)} conditions under {letter_text}")

        time.sleep(0.3)  # tiny delay

    logger.info(f"Extracted {total_diseases} diseases total across {len(diseases_by_letter)} letters.")
    return diseases_by_letter


def main(filename: str = 'a_to_z_diseases_list.json'):
    f_path = OUTPUT_DIR / filename
    logger.info(f"Starting A-Z disease scrape → will save to {f_path}")

    all_diseases = scrape_a_to_z()

    if not all_diseases:
        logger.error("No diseases extracted — check selectors or network")
        return

    try:
        with open(f_path, "w", encoding="utf-8") as f:
            json.dump(all_diseases, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved {len(all_diseases)} letters / {sum(len(v) for v in all_diseases.values())} conditions to {f_path}")
    except Exception as e:
        logger.exception(f"Failed to write JSON file: {e}")
        return

    # Summary    
    logger.info(f"Total conditions: {sum(len(v) for v in all_diseases.values())}")


# Run and Save
if __name__ == "__main__":
    main()