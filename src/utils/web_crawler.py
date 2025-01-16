"""Utility to crawl the frontend website and extract content."""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
from loguru import logger
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
        self.content_data: List[Dict] = []
        
        # Create a temporary directory for Chrome user data
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize Selenium WebDriver with enhanced options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--user-data-dir={self.temp_dir}")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.binary_location = "/usr/bin/google-chrome"
        
        # Initialize Chrome service with the latest driver
        service = Service(ChromeDriverManager().install())
        
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {str(e)}")
            self._cleanup()
            raise
        
        # Define known routes based on App.js routes
        self.routes_to_crawl = [
            "/",
            "/doctors",
            "/services",
            "/contacts",
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
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error removing temporary directory: {str(e)}")

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to our frontend domain."""
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        
        path = parsed_url.path
        if path.endswith('/'):
            path = path[:-1]
            
        is_known_route = any(
            path == route or path.startswith(f"{route}/")
            for route in self.routes_to_crawl
        )
        
        return (
            (parsed_url.netloc == parsed_base.netloc or not parsed_url.netloc) and
            not url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js')) and
            is_known_route
        )

    def wait_for_content(self):
        """Wait for dynamic content to load."""
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "main")))
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Timeout waiting for content: {str(e)}")

    def extract_content(self, url: str) -> Dict:
        """Extract and structure content from rendered page."""
        logger.info(f"\n{'='*50}\nExtracting content from: {url}\n{'='*50}")
        
        try:
            # Navigate to the page
            self.driver.get(url)
            self.wait_for_content()
            
            # Initialize content structure
            content = {
                "url": url,
                "title": self.driver.title,
                "sections": []  # Will contain ordered content sections
            }

            # Get rendered HTML
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')

            # Process content in document order
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'p', 'span', 'ul', 'ol']):
                if element.name.startswith('h'):
                    # Extract heading
                    text = element.get_text(separator=' ', strip=True)
                    if text:
                        content["sections"].append({
                            "type": "heading",
                            "level": int(element.name[1]),
                            "text": text
                        })
                
                elif element.name in ['div', 'p', 'span']:
                    # Extract only direct text nodes, ignoring nested element content
                    direct_text = ' '.join(
                        text.strip() 
                        for text in element.find_all(string=True, recursive=False) 
                        if text.strip()
                    )
                    if direct_text:
                        content["sections"].append({
                            "type": "text",
                            "content": direct_text
                        })
                
                elif element.name in ['ul', 'ol']:
                    # Extract list items
                    items = [li.get_text(separator=' ', strip=True) for li in element.find_all('li')]
                    if items:
                        content["sections"].append({
                            "type": "list",
                            "style": element.name,
                            "items": items
                        })

            logger.info(f"Extracted {len(content['sections'])} content sections from {url}")
            return content

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                "url": url,
                "title": "",
                "sections": []
            }

    def crawl(self) -> List[Dict]:
        """Start crawling from all known routes."""
        logger.info(f"\nStarting crawl from {self.base_url}")
        
        try:
            # Crawl each defined route once
            for route in self.routes_to_crawl:
                url = urljoin(self.base_url, route)
                content = self.extract_content(url)
                if content["sections"]:
                    self.content_data.append(content)
                    logger.info(f"Successfully added content from: {url}")
            
            logger.info(f"\nCrawling completed. Found {len(self.content_data)} pages")
            return self.content_data
        finally:
            self.driver.quit()

    def format_for_vectorstore(self) -> List[Dict[str, str]]:
        """Format crawled content for vector store ingestion."""
        documents = []
        
        for page in self.content_data:
            # Build content in a structured way
            formatted_content = []
            
            # Add title as main heading
            if page["title"]:
                formatted_content.append(f"# {page['title']}\n")
            
            # Process sections in order
            current_section = []
            for section in page["sections"]:
                if section["type"] == "heading":
                    # If we have accumulated content, add it before the new heading
                    if current_section:
                        formatted_content.append("\n".join(current_section) + "\n\n")
                        current_section = []
                    
                    # Add heading
                    formatted_content.append(f"{'#' * section['level']} {section['text']}\n")
                
                elif section["type"] == "text":
                    current_section.append(section["content"])
                
                elif section["type"] == "list":
                    # Add any accumulated text before the list
                    if current_section:
                        formatted_content.append("\n".join(current_section) + "\n\n")
                        current_section = []
                    
                    # Format list items
                    if section["style"] == "ul":
                        formatted_content.append("\n".join(f"- {item}" for item in section["items"]) + "\n\n")
                    else:  # ol
                        formatted_content.append("\n".join(f"{i+1}. {item}" for i, item in enumerate(section["items"])) + "\n\n")
            
            # Add any remaining content
            if current_section:
                formatted_content.append("\n".join(current_section) + "\n\n")
            
            # Combine all content
            full_text = "".join(formatted_content).strip()
            
            # Add document with metadata
            documents.append({
                "content": full_text,
                "source": page["url"],
                "type": "webpage"
            })
        
        return documents 