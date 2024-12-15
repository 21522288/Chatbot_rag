"""Core chatbot functionality for the dental clinic assistant."""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Tuple, Dict, Any, Generator, Optional, Union, AsyncGenerator
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import GenerationChunk
import asyncio
import re

from src.config.settings import (
    CHROMA_DIR,
    MODEL_NAME,
    HUGGINGFACE_API_TOKEN,
    DEFAULT_PROMPT_TEMPLATE
)
from src.models.embeddings import get_embedding_function
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DentalChatbot:
    """Dental clinic chatbot that provides information based on the clinic's documentation."""
    
    def __init__(self):
        """Initialize the chatbot with necessary components."""
        logger.info("Initializing DentalChatbot")
        self.embedding_function = get_embedding_function()
        self.vector_store = self._initialize_vector_store()
        self.prompt_template = ChatPromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)
        self.llm = self._initialize_llm()
        
    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize the vector store connection.
        
        Returns:
            Chroma: Initialized vector store
        """
        logger.info(f"Connecting to vector store at {CHROMA_DIR}")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embedding_function
        )
        
    def _initialize_llm(self) -> HuggingFaceHub:
        """
        Initialize the language model.
        
        Returns:
            HuggingFaceHub: Initialized language model
        """
        logger.info(f"Initializing language model: {MODEL_NAME}")
        return HuggingFaceHub(
            repo_id=MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            model_kwargs={
                'temperature': 0.5,
                'max_length': 2048,
                'max_new_tokens': 1024,
                'repetition_penalty': 1.1,
                'do_sample': True
            }
        )
        
    async def _stream_tokens(self, response: str) -> AsyncGenerator[str, None]:
        """
        Stream a response string token by token.
        
        Args:
            response: The response string to stream
            
        Returns:
            AsyncGenerator[str, None]: An async generator that yields tokens
        """
        tokens = re.findall(r'\w+|\W+', response)
        for token in tokens:
            yield token
            await asyncio.sleep(0.02)  # Small delay between tokens
        
    def _process_response(self, response: str) -> str:
        """
        Process the raw response to clean it and extract only the answer.
        Removes any token metadata and formatting artifacts.
        
        Args:
            response: Raw response from the language model
            
        Returns:
            str: Clean, processed response
        """
        # Remove any token metadata
        response = re.sub(r'data: {"type": "token", "content": "[^"]*"}', '', response)
        
        # Remove any "Human:" or "Context:" prefixes
        response = re.sub(r'^(Human:|Context:)\s*', '', response, flags=re.MULTILINE)
        
        # Extract content after "Assistant:" if present
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1]
            
        # Clean up extra whitespace while preserving newlines
        lines = response.split('\n')
        lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
        final_response = '\n'.join(filter(None, lines))
        
        return final_response.strip()
        
    def get_response(
        self, 
        query: str, 
        k: int = 5,
        streaming: bool = False
    ) -> Union[Tuple[str, List[Dict[str, Any]]], AsyncGenerator[str, None]]:
        """
        Get a response to a user query.
        
        Args:
            query: The user's question
            k: Number of relevant documents to retrieve
            streaming: Whether to stream the response token by token
            
        Returns:
            If streaming is False:
                Tuple[str, List[Dict]]: The response and sources
            If streaming is True:
                AsyncGenerator[str, None]: Token generator
        """
        logger.info(f"Processing query: {query}")
        
        # Search for relevant documents
        results = self.vector_store.similarity_search_with_score(query, k=k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Let the LLM handle the response appropriately based on the query type
        prompt = self.prompt_template.format(context=context_text, question=query)
        
        try:
            raw_response = self.llm.invoke(prompt)
            processed_response = self._process_response(raw_response)
            
            sources = [
                {
                    "id": doc.metadata.get("id"),
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "relevance_score": float(score)
                }
                for doc, score in results
            ]
            
            logger.info("Successfully generated response")
            logger.info(f"Final response to be returned: {processed_response}")
            
            if streaming:
                return self._stream_tokens(processed_response)
            return processed_response, sources
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    def get_sources(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the source documents for a query without generating a response.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List[Dict]: Metadata about the relevant sources
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "id": doc.metadata.get("id"),
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "relevance_score": float(score),
                "content": doc.page_content
            }
            for doc, score in results
        ] 