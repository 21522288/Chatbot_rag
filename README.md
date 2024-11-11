# Dental Clinic Chatbot

A powerful chatbot assistant for dental clinics that uses RAG (Retrieval-Augmented Generation) to provide accurate information based on clinic documentation.

## Features

- Document processing and storage using ChromaDB vector database
- Semantic search for relevant information
- Natural language responses using HuggingFace models
- Interactive CLI interface
- Detailed source tracking for responses
- Comprehensive logging

## Project Structure

```
src/
├── config/
│   └── settings.py         # Configuration settings
├── core/
│   └── chatbot.py         # Core chatbot functionality
├── data_processing/
│   └── document_loader.py  # Document processing
├── models/
│   └── embeddings.py      # Embedding model configuration
├── utils/
│   └── logger.py          # Logging utilities
├── cli.py                 # Command-line interface
└── data/                  # Directory for PDF documents
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dental-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
MODEL_NAME=your_model_name
LOG_LEVEL=INFO
```

## Usage

### Loading Documents

1. Place your PDF documents in the `data/` directory
2. Process the documents:
```bash
python -m src.cli --reload-data
```

### Running the Chatbot

#### Interactive Mode
```bash
python -m src.cli
```

#### Single Query Mode
```bash
python -m src.cli --query "What are your opening hours?"
```

### Additional Commands

- Clear the database:
```bash
python -m src.cli --clear-db
```

- Reload data and start interactive mode:
```bash
python -m src.cli --reload-data
```

## Development

The codebase follows these principles:

1. **Modularity**: Each component has a single responsibility
2. **Type Safety**: Type hints are used throughout the codebase
3. **Error Handling**: Comprehensive error handling and logging
4. **Documentation**: Detailed docstrings and comments
5. **Configuration**: Externalized configuration in settings.py

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.