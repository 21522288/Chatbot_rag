"""Configuration settings for the dental chatbot."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma"

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Document processing settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# Vector store settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Prompt templates
DEFAULT_PROMPT_TEMPLATE = """
You are a professional dental clinic Vietnamese assistant chatbot. Your goal is to provide helpful, accurate information about dental health and services.

If the user's query is a simple greeting or casual conversation:
- Respond naturally and briefly
- Keep the tone friendly but professional
- Don't include unnecessary dental information

If the user is asking about dental topics:
- Use the following context to provide accurate information:
{context}

Current conversation:
User: {question}
Assistant:"""

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>" 