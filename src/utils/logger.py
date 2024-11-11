"""Logging configuration for the dental chatbot."""
from loguru import logger
import sys
from src.config.settings import LOG_LEVEL, LOG_FORMAT

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True
)

def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name) 