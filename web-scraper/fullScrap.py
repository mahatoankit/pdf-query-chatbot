import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import time
from urllib.parse import urljoin, urlparse
import logging
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited_urls = set()
        self.data = defaultdict(list)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def is_internal_link(self, url):
        """Check if the URL belongs to the same domain"""
        return urlparse(self.base_url).netloc == urlparse(url).netloc

    def clean_url(self, url):
        """Clean and normalize URL"""
        # Remove fragments
        url = url.split("#")[0]
        # Remove trailing slash
        return url.rstrip("/")

    def scrape_page(self, url):
        """Scrape a single page and its content"""
        try:
            url = self.clean_url(url)

            # Skip invalid URLs
            if not url or not url.startswith("http"):
                logging.warning(f"Skipping invalid URL: {url}")
                return

            if url in self.visited_urls:
                return

            self.visited_urls.add(url)
            logging.info(f"Scraping: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            time.sleep(2)

            soup = BeautifulSoup(response.text, "html.parser")
            self.scrape_elements(soup, url)
            self.find_internal_links(soup, url)

        except requests.Timeout:
            logging.error(f"Timeout while accessing {url}")
        except requests.ConnectionError:
            logging.error(f"Connection error for {url}")
        except requests.RequestException as e:
            logging.error(f"Error scraping {url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while scraping {url}: {str(e)}")

    def scrape_elements(self, soup, url):
        """Extract elements from the page"""
        try:
            # Headings
            for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                elements = soup.find_all(tag)
                self.data[tag].extend(
                    [
                        {"url": url, "content": elem.get_text().strip() if elem else ""}
                        for elem in elements
                    ]
                )

            # Paragraphs
            self.data["p"].extend(
                [
                    {"url": url, "content": p.get_text().strip() if p else ""}
                    for p in soup.find_all("p")
                ]
            )

            # Links
            self.data["a"].extend(
                [
                    {"url": url, "href": urljoin(url, a.get("href", ""))}
                    for a in soup.find_all("a")
                    if a.get("href")
                ]
            )

            # Images
            self.data["img"].extend(
                [
                    {"url": url, "src": urljoin(url, img.get("src", ""))}
                    for img in soup.find_all("img")
                    if img.get("src")
                ]
            )

            # Lists
            for list_tag in ["ul", "ol"]:
                lists = soup.find_all(list_tag)
                for lst in lists:
                    items = lst.find_all("li")
                    self.data[list_tag].extend(
                        [
                            {
                                "url": url,
                                "content": item.get_text().strip() if item else "",
                            }
                            for item in items
                        ]
                    )

        except Exception as e:
            logging.error(f"Error parsing elements from {url}: {str(e)}")

    def find_internal_links(self, soup, url):
        """Find and follow internal links"""
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link["href"])
            if self.is_internal_link(next_url):
                self.scrape_page(next_url)

    def save_to_csv(self):
        """Save scraped data to CSV files"""
        try:
            output_dir = "scraped_data"
            os.makedirs(output_dir, exist_ok=True)

            for tag, items in self.data.items():
                if items:
                    try:
                        df = pd.DataFrame(items)
                        filename = os.path.join(output_dir, f"scraped_{tag}_data.csv")
                        df.to_csv(filename, index=False, encoding="utf-8")
                        logging.info(f"Saved {filename} with {len(items)} entries")
                    except Exception as e:
                        logging.error(f"Error saving {tag} data: {str(e)}")

        except Exception as e:
            logging.error(f"Error in save_to_csv: {str(e)}")


def main():
    try:
        base_url = "https://sunway.edu.np/"
        scraper = WebScraper(base_url)

        logging.info("Starting web scraping...")
        scraper.scrape_page(base_url)
        scraper.save_to_csv()
        logging.info("Web scraping completed!")

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")


if __name__ == "__main__":
    main()
