"""Document loading and processing functionality."""
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

import os
import shutil
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

from src.config.settings import DATA_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.models.embeddings import get_embedding_function
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """Handles document loading, splitting, and storage in the vector database."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.embedding_function = get_embedding_function()
        self.cache_file = Path(CHROMA_DIR) / "document_cache.json"
        
    def _calculate_data_hash(self) -> str:
        """
        Calculate a hash of all documents in the data directory.
        This hash will change if any documents are added, removed, or modified.
        
        Returns:
            str: Hash of all documents
        """
        if not os.path.exists(DATA_DIR):
            return ""
            
        hash_md5 = hashlib.md5()
        
        for root, _, files in os.walk(DATA_DIR):
            for filename in sorted(files):  # Sort to ensure consistent order
                if filename.lower().endswith('.pdf'):
                    filepath = os.path.join(root, filename)
                    # Update hash with filename and last modified time
                    file_stat = os.stat(filepath)
                    file_info = f"{filepath}:{file_stat.st_mtime}:{file_stat.st_size}"
                    hash_md5.update(file_info.encode())
                    
        return hash_md5.hexdigest()
        
    def _load_cache(self) -> Optional[str]:
        """
        Load the cached document hash.
        
        Returns:
            Optional[str]: Cached hash if exists, None otherwise
        """
        if not self.cache_file.exists():
            return None
            
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                return cache_data.get('hash')
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
            
    def _save_cache(self, doc_hash: str) -> None:
        """
        Save the document hash to cache.
        
        Args:
            doc_hash: Hash to cache
        """
        try:
            # Ensure the directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump({'hash': doc_hash}, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            
    def _should_process_documents(self) -> bool:
        """
        Check if documents need to be processed by comparing current state with cache.
        
        Returns:
            bool: True if documents should be processed, False otherwise
        """
        current_hash = self._calculate_data_hash()
        cached_hash = self._load_cache()
        
        # Process if:
        # 1. No documents exist (empty hash)
        # 2. No cache exists
        # 3. Cache doesn't match current state
        should_process = (
            not current_hash or 
            cached_hash is None or 
            current_hash != cached_hash
        )
        
        if not should_process:
            logger.info("Documents unchanged since last processing, using cached version")
        
        return should_process
        
    def load_documents(self) -> List[Document]:
        """
        Load documents from the data directory.
        
        Returns:
            List[Document]: List of loaded documents
        """
        logger.info(f"Loading documents from {DATA_DIR}")
        try:
            loader = PyPDFDirectoryLoader(str(DATA_DIR))
            documents = loader.load()
            # Ensure metadata values are strings
            for doc in documents:
                doc.metadata = {k: str(v) for k, v in doc.metadata.items()}
            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
            
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        logger.info(f"Splitting documents with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}")
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(CHUNK_SIZE),
                chunk_overlap=int(CHUNK_OVERLAP),
                length_function=len,
                add_start_index=True
            )
            chunks = splitter.split_documents(documents)
            # Ensure metadata values are strings
            for chunk in chunks:
                chunk.metadata = {k: str(v) for k, v in chunk.metadata.items()}
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
        
    def process_and_store(self) -> None:
        """Load, split, and store documents in the vector database if needed."""
        if not self._should_process_documents():
            return
            
        logger.info("Changes detected in documents, processing...")
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self._store_in_chroma(chunks)
        
        # Save the new state to cache
        self._save_cache(self._calculate_data_hash())
        
    def _store_in_chroma(self, chunks: List[Document]) -> None:
        """
        Store document chunks in Chroma vector store.
        
        Args:
            chunks: List of document chunks to store
        """
        logger.info("Storing documents in Chroma")
        db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embedding_function
        )
        
        # Process chunks with IDs
        chunks_with_ids = self._calculate_chunk_ids(chunks)
        
        # Get existing documents
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"] if existing_items["ids"] else [])
        logger.info(f"Found {len(existing_ids)} existing documents in database")
        
        # Filter new chunks
        new_chunks = [
            chunk for chunk in chunks_with_ids 
            if chunk.metadata["id"] not in existing_ids
        ]
        
        if new_chunks:
            logger.info(f"Adding {len(new_chunks)} new documents to database")
            chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=chunk_ids)
        else:
            logger.info("No new documents to add")
            
    def _calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """
        Calculate unique IDs for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List[Document]: Chunks with calculated IDs in metadata
        """
        last_page_id = None
        current_chunk_index = 0
        
        for chunk in chunks:
            source = str(chunk.metadata.get("source", ""))
            page = str(chunk.metadata.get("page", "0"))
            current_page_id = f"{source}:{page}"
            
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
                
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            chunk.metadata["id"] = chunk_id
            last_page_id = current_page_id
            
        return chunks
        
    @staticmethod
    def clear_database() -> None:
        """Clear the existing vector database."""
        if os.path.exists(CHROMA_DIR):
            logger.warning("Clearing existing vector database")
            shutil.rmtree(CHROMA_DIR)
            logger.info("Vector database cleared successfully") 