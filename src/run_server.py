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

def main():
    """Run the FastAPI server."""
    logger.info("Starting FastAPI server")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

if __name__ == "__main__":
    main() 