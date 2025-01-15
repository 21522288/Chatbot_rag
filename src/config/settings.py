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
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 800)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 80)

# Vector store settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Memory settings
MEMORY_KEY = "chat_history"
MEMORY_WINDOW_SIZE = os.getenv("MEMORY_WINDOW_SIZE", 10)  # Number of previous conversations to remember

# Prompt templates
DEFAULT_PROMPT_TEMPLATE = '''
You are a professional dental clinic Vietnamese assistant chatbot. Your goal is to provide helpful, accurate information about dental health and services.\n\n

Important formatting instructions:\n
1. Always use proper markdown with correct spacing:
   - Add "\\n\\n" between paragraphs and sections
   - Add "\\n" before each list item
   - Add "\\n\\n" after each section\n

2. For bullet points and lists:
   - Start each item with "\\n- " or "\\n<number>. "
   - Add "\\n" between list items
   - Add "\\n\\n" after the complete list\n

3. For text emphasis and headings:
   - Use "## " for section headings followed by "\\n"
   - Use **bold** for important terms
   - Use *italic* for emphasis\n

Previous conversation history:
{chat_history}\n\n

If the user's query is a simple greeting or casual conversation:
- Respond naturally and briefly
- Keep the tone friendly but professional
- Don't include unnecessary dental information\n\n

If the user is asking about dental topics:
- Follow with well-spaced bullet points or numbered steps
- Use the following context to provide accurate information:\n
{context}\n\n

Remember to:
1. Always maintain a professional tone\n
2. Be concise but thorough\n
3. Use proper Vietnamese medical terminology when appropriate\n
4. Format the response for easy reading\n
5. Break down complex information into digestible parts\n
6. Dont include Chinese characters in the response, Just use Vietnamese or English (if needed).\n\n

Current conversation:
User: {question}
Assistant:'''

# Classification prompt for appointment-related queries
APPOINTMENT_CLASSIFICATION_PROMPT = """
Determine if the following query is related to dental appointments (booking, scheduling, checking, or canceling appointments).
Consider both Vietnamese and English queries.

Query: {query}

Return only "yes" if it's appointment-related, or "no" if it's not.
"""

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Backend API configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:7132") 