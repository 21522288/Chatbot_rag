"""Script to run the FastAPI server."""
import os
import sys
import ssl
import certifi
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import uvicorn
from src.utils.logger import get_logger

logger = get_logger(__name__)

def setup_ssl_certificates():
    """Setup SSL certificates for Windows."""
    try:
        certifi_path = certifi.where()
        logger.info(f"Setting up SSL certificates from: {certifi_path}")
        
        # Set the SSL certificate environment variables
        os.environ['SSL_CERT_FILE'] = certifi_path
        os.environ['REQUESTS_CA_BUNDLE'] = certifi_path
        
        # Verify the SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.load_verify_locations(certifi_path)
        logger.info("SSL certificates configured successfully")
    except Exception as e:
        logger.error(f"Error setting up SSL certificates: {e}")

def log_environment_variables():
    """Log all necessary environment variables for debugging."""
    env_vars = {
        'MODEL_NAME': os.getenv('MODEL_NAME', 'Not set'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'Not set'),
        'HUGGINGFACEHUB_API_TOKEN': 'Present' if os.getenv('HUGGINGFACEHUB_API_TOKEN') else 'Not set',
        'DEFAULT_K_RETRIEVED_DOCS': os.getenv('DEFAULT_K_RETRIEVED_DOCS', 'Not set'),
        'CHUNK_SIZE': os.getenv('CHUNK_SIZE', 'Not set'),
        'CHUNK_OVERLAP': os.getenv('CHUNK_OVERLAP', 'Not set'),
        'MEMORY_WINDOW_SIZE': os.getenv('MEMORY_WINDOW_SIZE', 'Not set'),
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
    
    # Setup SSL certificates
    setup_ssl_certificates()
    
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