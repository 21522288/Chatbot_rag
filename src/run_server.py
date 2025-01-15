"""Script to run the FastAPI server."""
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import uvicorn
from src.utils.logger import get_logger

logger = get_logger(__name__)

def log_environment_variables():
    """Log all necessary environment variables for debugging."""
    env_vars = {
        'MODEL_NAME': os.getenv('MODEL_NAME', 'Not set'),
        'HUGGINGFACEHUB_API_TOKEN': 'Present' if os.getenv('HUGGINGFACEHUB_API_TOKEN') else 'Not set',
        'TRANSFORMERS_CACHE': os.getenv('TRANSFORMERS_CACHE', 'Not set'),
        'HF_HOME': os.getenv('HF_HOME', 'Not set'),
        'SENTENCE_TRANSFORMERS_HOME': os.getenv('SENTENCE_TRANSFORMERS_HOME', 'Not set'),
        'REQUESTS_TIMEOUT': os.getenv('REQUESTS_TIMEOUT', 'Not set'),
    }
    
    logger.info("Environment Variables:")
    for key, value in env_vars.items():
        logger.info(f"{key}: {value}")

def main():
    """Run the FastAPI server."""
    logger.info("Starting FastAPI server")
    
    # Log environment variables
    log_environment_variables()
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

if __name__ == "__main__":
    main() 