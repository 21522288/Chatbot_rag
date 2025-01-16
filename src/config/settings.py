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
DEFAULT_K_RETRIEVED_DOCS = int(os.getenv("DEFAULT_K_RETRIEVED_DOCS", 10))

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Vector store settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Memory settings
MEMORY_KEY = "chat_history"
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", 10))  # Number of previous conversations to remember

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
- Follow with well-spaced bullet points or numbered steps when providing information
- Use the following context to provide accurate information:\n
{context}\n\n

Remember to:
1. Always maintain a professional tone\n
2. Be concise but thorough\n
3. Use proper Vietnamese medical terminology when appropriate\n
4. Format the response for easy reading\n
5. Break down complex information into digestible parts\n
6. Dont include Chinese characters in the response, Just use Vietnamese or English (if needed).\n
7. Dont sign your name at the end of the response.\n
8. Only provide information that is available in the given context\n
9. If the question is not related to the dental clinic, just say that you don't have specific information about their query.\n
10. Never make up or provide uncertain information.\n\n

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

# Query condensing prompt template
QUERY_CONDENSING_PROMPT = """Given the following conversation history and a new question, rephrase the question to be standalone, incorporating any relevant context from the conversation history. 

Must follow the following rules:
1. Be concise and maximum in 2 sentences.
2. Dont include any information that is not relevant to the question.
3. Dont include any information that is not available in the conversation history.
4. If the new question is not related to the conversation history, just return the new question.

Chat History:
{history}

New Question: {question}

Standalone question:"""

# Appointment formatting prompt
APPOINTMENT_FORMATTING_PROMPT = """Current datetime is {time} on {weekday}, {date}
Based on the current datetime, follow the following rules:
- If customer want to book an appointment in a specific date, we should check if the date is in the future and suggest available future datetimes.
- We should say that we dont allow scheduling appointments in the past.
- Dont suggest any datetime slots in the past.
- The dental clinic only has these available time slots in a week: 08:00 - 20:00 from Monday to Saturday, and 08:00 - 17:00 on Sunday.
- A datetime slot is unavailable if it already has an appointment with status other than 'Chờ xác nhận' (Waiting for confirmation), 'Đã hủy' (Cancelled), or 'Chưa xử lí' (Unprocessed).
- Other time slots on the same date that have not been booked are still available for scheduling. Example: If customer ask to book an appointment on 05/01/2025 and there are 2 time slots 15:00 and 16:00 on 2025-02-05 that have been booked, we should suggest other time slots on 2025-02-05.
- You can only book appointments on available datetimes that have not been booked by other customers.
- The list of current appointments will only be provided if specifically requested.
- Ensure customer provide the name, phone number, date, time and branch to complete the appointment booking."""

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Backend API configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:7132") 