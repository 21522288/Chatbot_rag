"""Utility to crawl the frontend website and extract content."""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import json
from loguru import logger
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import tempfile
import os

class WebCrawler:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.content_data: List[Dict] = []
        
        # Create a temporary directory for Chrome user data
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize Selenium WebDriver with enhanced options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--user-data-dir={self.temp_dir}")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.binary_location = "/usr/bin/google-chrome"  # Specify Chrome binary path
        
        # Initialize Chrome service with the latest driver
        service = Service(ChromeDriverManager().install())
        
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)  # Wait up to 10 seconds
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {str(e)}")
            # Clean up temp directory if driver initialization fails
            self._cleanup()
            raise
        
        # Define known routes based on App.js routes
        self.known_routes = [
            "/",
            "/doctors",
            "/services",
            "/contacts",
            "/booking",
            "/sign_in",
            "/sign_up",
            "/forgetpassword",
            "/profile",
            "/mytreatmentrecord",
            "/manager",
            "/confirm_email",
            "/bookingOnline"
        ]

    def __del__(self):
        """Clean up Selenium WebDriver and temporary directory."""
        self._cleanup()

    def _cleanup(self):
        """Helper method to clean up resources."""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
        except Exception as e:
            logger.error(f"Error closing Chrome driver: {str(e)}")
        
        try:
            # Remove temporary directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error removing temporary directory: {str(e)}")

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to our frontend domain."""
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        
        # Check if it's a valid route
        path = parsed_url.path
        if path.endswith('/'):
            path = path[:-1]
            
        # Allow known routes and their sub-routes
        is_known_route = any(
            path == route or path.startswith(f"{route}/")
            for route in self.known_routes
        )
        
        return (
            (parsed_url.netloc == parsed_base.netloc or not parsed_url.netloc) and
            not url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js')) and
            is_known_route
        )

    def wait_for_content(self):
        """Wait for dynamic content to load."""
        try:
            # Wait for common content containers
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "main")))
            time.sleep(2)  # Additional small delay for React rendering
        except Exception as e:
            logger.warning(f"Timeout waiting for content: {str(e)}")

    def extract_content(self, url: str) -> Dict:
        """Extract and structure content from rendered page."""
        logger.info(f"\n{'='*50}\nExtracting content from: {url}\n{'='*50}")
        
        try:
            # Navigate to the page
            self.driver.get(url)
            self.wait_for_content()
            
            # Get rendered HTML
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract main content
            content = {
                "url": url,
                "title": self.driver.title,
                "headings": [],
                "paragraphs": [],
                "lists": [],
                "tables": [],
                "nav_text": [],
                "buttons": [],
                "inputs": [],
                "labels": []
            }

            # Log title
            logger.info(f"\nTitle: {content['title']}")

            # Extract headings
            logger.info("\nExtracting headings:")
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = h.get_text(strip=True)
                if text:
                    content["headings"].append({
                        "level": int(h.name[1]),
                        "text": text
                    })
                    logger.info(f"  {h.name}: {text}")

            # Extract paragraphs
            logger.info("\nExtracting paragraphs:")
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if text:
                    content["paragraphs"].append(text)
                    logger.info(f"  Paragraph: {text[:100]}...")

            # Extract lists
            logger.info("\nExtracting lists:")
            for lst in soup.find_all(['ul', 'ol']):
                items = [li.get_text(strip=True) for li in lst.find_all('li')]
                if items:
                    content["lists"].append({
                        "type": lst.name,
                        "items": items
                    })
                    logger.info(f"  {lst.name.upper()} List items:")
                    for item in items:
                        logger.info(f"    - {item}")

            # Extract tables
            logger.info("\nExtracting tables:")
            for table in soup.find_all('table'):
                table_data = []
                rows = table.find_all('tr')
                
                # Get headers
                headers = []
                header_row = table.find('thead')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                    if headers:
                        logger.info(f"  Table headers: {headers}")

                # Get rows
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    if cols:
                        row_data = [col.get_text(strip=True) for col in cols]
                        table_data.append(row_data)
                        logger.info(f"  Table row: {row_data}")

                if table_data:
                    content["tables"].append({
                        "headers": headers,
                        "data": table_data
                    })

            # Extract navigation text and interactive elements
            logger.info("\nExtracting navigation and interactive elements:")
            for elem in soup.find_all(['nav', 'a', 'button', 'input', 'label']):
                text = elem.get_text(strip=True)
                if text:
                    if elem.name in ['nav', 'a']:
                        content["nav_text"].append(text)
                        logger.info(f"  {elem.name}: {text}")
                    elif elem.name == 'button':
                        content["buttons"].append(text)
                        logger.info(f"  Button: {text}")
                    elif elem.name == 'label':
                        content["labels"].append(text)
                        logger.info(f"  Label: {text}")
                    elif elem.name == 'input':
                        placeholder = elem.get('placeholder', '').strip()
                        if placeholder:
                            content["inputs"].append(placeholder)
                            logger.info(f"  Input placeholder: {placeholder}")

            # Log content summary
            logger.info(f"\nContent Summary for {url}:")
            logger.info(f"  - {len(content['headings'])} headings")
            logger.info(f"  - {len(content['paragraphs'])} paragraphs")
            logger.info(f"  - {len(content['lists'])} lists")
            logger.info(f"  - {len(content['tables'])} tables")
            logger.info(f"  - {len(content['nav_text'])} navigation elements")
            logger.info(f"  - {len(content['buttons'])} buttons")
            logger.info(f"  - {len(content['inputs'])} input fields")
            logger.info(f"  - {len(content['labels'])} labels")
            logger.info(f"{'='*50}\n")

            return content
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                "url": url,
                "title": "",
                "headings": [],
                "paragraphs": [],
                "lists": [],
                "tables": [],
                "nav_text": [],
                "buttons": [],
                "inputs": [],
                "labels": []
            }

    def crawl_page(self, url: str) -> None:
        """Crawl a single page and extract its content."""
        if url in self.visited_urls:
            logger.debug(f"Skipping already visited URL: {url}")
            return

        try:
            logger.info(f"\nCrawling page: {url}")
            
            # Extract content using Selenium
            content = self.extract_content(url)
            if any(value for key, value in content.items() if key != 'url'):
                self.content_data.append(content)
                logger.info(f"Successfully added content from: {url}")
            else:
                logger.warning(f"No content found in: {url}")
            
            self.visited_urls.add(url)
            
            # Find all links in the rendered page
            logger.info("\nFound links:")
            links = self.driver.find_elements(By.TAG_NAME, "a")
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href:
                        logger.debug(f"  Found link: {href}")
                        if self.is_valid_url(href):
                            logger.info(f"  Valid link found: {href}")
                            if href not in self.visited_urls:
                                self.crawl_page(href)
                            else:
                                logger.debug(f"  Skipping already visited: {href}")
                        else:
                            logger.debug(f"  Invalid link: {href}")
                except Exception as e:
                    logger.error(f"Error processing link: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")

    def crawl(self) -> List[Dict]:
        """Start crawling from all known routes."""
        logger.info(f"\nStarting crawl from {self.base_url}")
        logger.info("Known routes:")
        for route in self.known_routes:
            logger.info(f"  - {route}")
        
        try:
            # Crawl each known route
            for route in self.known_routes:
                url = urljoin(self.base_url, route)
                if url not in self.visited_urls:
                    self.crawl_page(url)
            
            logger.info(f"\nCrawling completed. Found {len(self.content_data)} pages")
            logger.info("Crawled URLs:")
            for url in self.visited_urls:
                logger.info(f"  - {url}")
            
            return self.content_data
        finally:
            # Clean up Selenium WebDriver
            self.driver.quit()

    def format_for_vectorstore(self) -> List[Dict[str, str]]:
        """Format crawled content for vector store ingestion."""
        documents = []
        
        for page in self.content_data:
            # Format headings
            headings_text = "\n".join([
                f"{'#' * h['level']} {h['text']}"
                for h in page["headings"]
            ])
            
            # Format paragraphs
            paragraphs_text = "\n\n".join(page["paragraphs"])
            
            # Format lists
            lists_text = ""
            for lst in page["lists"]:
                if lst["type"] == "ul":
                    lists_text += "\n" + "\n".join([f"- {item}" for item in lst["items"]]) + "\n"
                else:
                    lists_text += "\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(lst["items"])]) + "\n"
            
            # Format tables
            tables_text = ""
            for table in page["tables"]:
                if table["headers"]:
                    tables_text += "\n| " + " | ".join(table["headers"]) + " |"
                    tables_text += "\n|" + "|".join(["---" for _ in table["headers"]]) + "|"
                for row in table["data"]:
                    tables_text += "\n| " + " | ".join(row) + " |"
                tables_text += "\n"
            
            # Format interactive elements
            interactive_text = "\n".join([
                "Navigation Elements:",
                *[f"- {text}" for text in page["nav_text"]],
                "\nButtons:",
                *[f"- {text}" for text in page["buttons"]],
                "\nForm Fields:",
                *[f"- {text}" for text in page["labels"]],
                *[f"- {text}" for text in page["inputs"]]
            ])
            
            # Combine all content
            full_text = f"""# {page['title']}

{headings_text}

{paragraphs_text}

{lists_text}

{tables_text}

{interactive_text}
""".strip()
            
            # Add document with metadata
            documents.append({
                "content": full_text,
                "source": page["url"],
                "type": "webpage"
            })
        
        return documents 