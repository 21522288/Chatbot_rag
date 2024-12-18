"""Script to crawl and load the frontend website content into the chatbot's knowledge base."""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from pathlib import Path
from src.config.settings import CHROMA_DIR
from src.utils.web_crawler import WebCrawler
from src.data_processing.frontend_content import add_to_vectorstore
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Main function to crawl and load frontend content."""
    try:
        # Initialize web crawler
        logger.info("Initializing web crawler...")
        crawler = WebCrawler("http://localhost:3000")
        
        # Crawl the website
        logger.info("Starting website crawl...")
        crawler.crawl()
        
        # Format content for vector store
        logger.info("Formatting content for vector store...")
        documents = crawler.format_for_vectorstore()
        logger.info(f"Found {len(documents)} documents")
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        add_to_vectorstore(documents, CHROMA_DIR)
        logger.info("Successfully added frontend website content to vector store")
        
    except Exception as e:
        logger.error(f"Error loading frontend website content: {str(e)}")
        raise

if __name__ == "__main__":
    main() 