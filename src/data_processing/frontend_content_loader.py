"""Module for processing and loading frontend content into the chatbot's knowledge base."""
import os
from pathlib import Path
import traceback
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.models.embeddings import get_embedding_function
from src.utils.logger import get_logger

logger = get_logger(__name__)

def extract_text_from_html(html_content: str) -> str:
    """Extract readable text from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=' ', strip=True)

def process_file(file_path: Path) -> Dict[str, str]:
    """Process a single file and return its content with metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        logger.warning(f"Skipping binary file: {file_path}")
        return None
    
    # Process different file types
    if file_path.suffix == '.html':
        text = extract_text_from_html(content)
    elif file_path.suffix == '.js':
        # For minified JS files, we'll just extract any readable text
        # This includes strings, comments, and function names
        text = content
    elif file_path.suffix == '.json':
        # Parse JSON and convert to string representation
        try:
            json_content = json.loads(content)
            text = json.dumps(json_content, indent=2)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {file_path}, treating as text")
            text = content
    elif file_path.suffix == '.css':
        # Skip CSS files as they don't contain relevant content
        return None
    elif file_path.suffix in {'.map', '.ico', '.svg', '.png', '.jpg', '.jpeg', '.gif'}:
        # Skip source maps and binary files
        return None
    else:
        text = content

    # Skip empty content
    if not text.strip():
        return None

    return {
        "content": text,
        "source": str(file_path),
        "type": file_path.suffix[1:] if file_path.suffix else "txt"
    }

def load_frontend_content(frontend_dir: Path) -> List[Dict[str, str]]:
    """Load all relevant static content from the frontend directory."""
    documents = []
    
    # Define relevant file types for built frontend content
    relevant_extensions = {'.html', '.js', '.json', '.txt', '.md'}
    
    # Walk through the frontend directory
    for root, _, files in os.walk(frontend_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in relevant_extensions:
                try:
                    doc = process_file(file_path)
                    if doc:  # Only add if document was successfully processed
                        documents.append(doc)
                        logger.info(f"Processed {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
    
    return documents

def add_to_vectorstore(documents: List[Dict[str, str]], chroma_dir: Path) -> None:
    """Add documents to the vector store."""
    # Initialize text splitter with settings optimized for minified content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents
    texts = []
    metadatas = []
    for doc in documents:
        try:
            # Ensure content is a string
            content = str(doc["content"]) if doc["content"] is not None else ""
            if not content.strip():
                logger.warning(f"Skipping empty content from source: {doc['source']}")
                continue
                
            chunks = text_splitter.split_text(content)
            texts.extend(chunks)
            metadatas.extend([{
                "source": doc["source"],
                "type": doc["type"]
            } for _ in chunks])
        except Exception as e:
            logger.error(f"Error processing document from {doc['source']}: {str(e)}")
            # traceback
            logger.error(traceback.format_exc())
            continue
    
    if not texts:
        logger.warning("No valid texts to add to vector store")
        return
        
    # Initialize vector store
    embedding_function = get_embedding_function()
    vector_store = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embedding_function
    )
    
    # Add documents
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    vector_store.persist()
    logger.info(f"Added {len(texts)} chunks to vector store") 