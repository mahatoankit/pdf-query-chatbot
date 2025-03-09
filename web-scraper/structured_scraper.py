import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os
import re
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class StructuredWebScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()
        self.structured_data = {
            'metadata': {
                'source_url': base_url,
                'scrape_date': datetime.now().isoformat(),
                'total_pages': 0
            },
            'content': {
                'courses': [],
                'events': [],
                'news': [],
                'faculty': [],
                'programs': [],
                'general': []
            }
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()

    def extract_page_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from the page"""
        return {
            'url': url,
            'title': self.clean_text(soup.title.string) if soup.title else '',
            'last_modified': soup.find('meta', {'property': 'article:modified_time'}).get('content', '') if soup.find('meta', {'property': 'article:modified_time'}) else '',
            'description': self.clean_text(soup.find('meta', {'name': 'description'}).get('content', '')) if soup.find('meta', {'name': 'description'}) else ''
        }

    def categorize_content(self, url: str) -> str:
        """Determine content type based on URL pattern"""
        patterns = {
            'courses': r'/course|/program|/study',
            'events': r'/event|/workshop|/seminar',
            'news': r'/news|/announcement|/notice',
            'faculty': r'/faculty|/staff|/teacher',
            'programs': r'/academic|/department'
        }
        
        for category, pattern in patterns.items():
            if re.search(pattern, url.lower()):
                return category
        return 'general'

    def extract_content(self, soup: BeautifulSoup) -> Dict:
        """Extract structured content from the page"""
        content = {
            'main_content': '',
            'sections': [],
            'related_links': [],
            'images': []
        }

        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main-content'])
        if main_content:
            content['main_content'] = self.clean_text(main_content.get_text())

        # Extract sections
        for section in soup.find_all(['section', 'div'], class_=['section', 'content-section']):
            section_data = {
                'title': self.clean_text(section.find(['h1', 'h2', 'h3']).get_text()) if section.find(['h1', 'h2', 'h3']) else '',
                'content': self.clean_text(section.get_text()),
                'type': section.get('class', ['unknown'])[0]
            }
            content['sections'].append(section_data)

        # Extract images with captions
        for img in soup.find_all('img', src=True):
            img_data = {
                'src': urljoin(self.base_url, img['src']),
                'alt': img.get('alt', ''),
                'caption': self.clean_text(img.find_next('figcaption').get_text()) if img.find_next('figcaption') else ''
            }
            content['images'].append(img_data)

        return content

    def scrape_page(self, url: str) -> None:
        """Scrape a single page with structured data extraction"""
        try:
            if url in self.visited_urls or not url.startswith(self.base_url):
                return

            self.visited_urls.add(url)
            logging.info(f"Scraping: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            time.sleep(1)  # Respectful scraping delay

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract and categorize content
            category = self.categorize_content(url)
            page_data = {
                'metadata': self.extract_page_metadata(soup, url),
                'content': self.extract_content(soup)
            }
            
            self.structured_data['content'][category].append(page_data)
            self.structured_data['metadata']['total_pages'] += 1

            # Find and follow internal links
            self.find_internal_links(soup, url)

        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")

    def find_internal_links(self, soup: BeautifulSoup, url: str) -> None:
        """Find and follow internal links"""
        for link in soup.find_all('a', href=True):
            next_url = urljoin(url, link['href'])
            if next_url.startswith(self.base_url):
                self.scrape_page(next_url)

    def save_structured_data(self, output_dir: str = "structured_data") -> None:
        """Save structured data in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)

        # Save complete structured data as JSON
        json_path = os.path.join(output_dir, 'complete_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.structured_data, f, indent=2, ensure_ascii=False)

        # Save category-wise CSV files
        for category, items in self.structured_data['content'].items():
            if items:
                df = pd.json_normalize(items)
                csv_path = os.path.join(output_dir, f'{category}_data.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8')

        # Create LLM training format
        training_data = []
        for category, items in self.structured_data['content'].items():
            for item in items:
                if item['content']['main_content']:
                    training_example = {
                        'prompt': f"Summarize this {category} content: {item['metadata']['title']}",
                        'completion': item['content']['main_content']
                    }
                    training_data.append(training_example)

        # Save training data
        training_path = os.path.join(output_dir, 'llm_training_data.json')
        with open(training_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Data saved in {output_dir}")

def main():
    try:
        base_url = "https://sunway.edu.np/"
        scraper = StructuredWebScraper(base_url)
        
        logging.info("Starting structured web scraping...")
        scraper.scrape_page(base_url)
        scraper.save_structured_data()
        logging.info(f"Scraping completed! Total pages: {scraper.structured_data['metadata']['total_pages']}")

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")

if __name__ == "__main__":
    main()