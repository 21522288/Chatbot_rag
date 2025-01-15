"""Embedding model configuration and functions."""
from langchain.embeddings import HuggingFaceEmbeddings
from src.config.settings import EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_embedding_function():
    """
    Get the embedding function using the specified model.
    
    Returns:
        HuggingFaceEmbeddings: The embedding model instance
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise 