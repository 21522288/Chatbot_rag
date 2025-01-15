"""Document loading and processing functionality."""
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

import os
import shutil
from typing import List, Optional
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=True
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
        
    def process_and_store(self) -> None:
        """Load, split, and store documents in the vector database."""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self._store_in_chroma(chunks)
        
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
        existing_ids = set(existing_items["ids"])
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
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
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