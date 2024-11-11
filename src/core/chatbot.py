"""Core chatbot functionality for the dental clinic assistant."""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Tuple, Dict, Any, Generator, Optional
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import GenerationChunk
import time

from src.config.settings import (
    CHROMA_DIR,
    MODEL_NAME,
    HUGGINGFACE_API_TOKEN,
    DEFAULT_PROMPT_TEMPLATE
)
from src.models.embeddings import get_embedding_function
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StreamingCallback(BaseCallbackHandler):
    """Callback handler for streaming tokens."""
    
    def __init__(self):
        """Initialize the streaming callback handler."""
        self.text = ""
        self.started_responding = False
        
    def on_llm_start(self, *args, **kwargs) -> None:
        """Called when LLM starts processing."""
        self.started_responding = False
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new tokens as they are generated."""
        # Only start capturing and printing after we get the first real response token
        if not token.strip() or token.startswith(('Human:', 'Assistant:', 'Context:', 'Question:')):
            return
            
        if not self.started_responding and token.strip():
            self.started_responding = True
            
        if self.started_responding:
            self.text += token
            print(token, end="", flush=True)

class DentalChatbot:
    """Dental clinic chatbot that provides information based on the clinic's documentation."""
    
    def __init__(self):
        """Initialize the chatbot with necessary components."""
        logger.info("Initializing DentalChatbot")
        self.embedding_function = get_embedding_function()
        self.vector_store = self._initialize_vector_store()
        self.prompt_template = ChatPromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)
        
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
        
    def _initialize_llm(self, streaming: bool = False) -> HuggingFaceHub:
        """
        Initialize the language model.
        
        Args:
            streaming: Whether to enable streaming mode
            
        Returns:
            HuggingFaceHub: Initialized language model
        """
        logger.info(f"Initializing language model: {MODEL_NAME}")
        
        callbacks = []
        if streaming:
            callbacks.append(StreamingCallback())
            
        return HuggingFaceHub(
            repo_id=MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            model_kwargs={
                'temperature': 0.5,
                'max_length': 2048,
                'max_new_tokens': 1024,
                'repetition_penalty': 1.1,
                'do_sample': True
            },
            callbacks=callbacks,
            streaming=streaming
        )
        
    def get_response(
        self, 
        query: str, 
        k: int = 5,
        streaming: bool = False
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get a response to a user query.
        
        Args:
            query: The user's question
            k: Number of relevant documents to retrieve
            streaming: Whether to stream the response token by token
            
        Returns:
            Tuple containing:
                - str: The chatbot's response
                - List[Dict]: Metadata about the sources used
        """
        logger.info(f"Processing query: {query}")
        
        # Search for relevant documents
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Prepare context from relevant documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Generate response using the language model
        prompt = self.prompt_template.format(context=context_text, question=query)
        
        try:
            # Initialize LLM with streaming if requested
            llm = self._initialize_llm(streaming=streaming)
            
            # Get response
            if streaming:
                callback_handler = StreamingCallback()
                llm.callbacks = [callback_handler]
                _ = llm.invoke(prompt)  # Actual streaming happens through callback
                response = callback_handler.text
            else:
                response = llm.invoke(prompt)
            
            # Extract sources information
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
            return response, sources
            
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