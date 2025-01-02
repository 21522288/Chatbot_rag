# Dental Clinic Chatbot

A chatbot that provides information about dental clinic services and procedures using RAG (Retrieval-Augmented Generation).

## Setup

1. Clone the repository
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and fill in your API keys and configuration

## Running the API Server

Start the FastAPI server:
```bash
python src/run_server.py
```

From now, you can access:
- Demo Chat UI: `http://localhost:8000/`
- API: `http://localhost:8000/api`
- Interactive API documentation: `http://localhost:8000/docs`

## API Endpoints

### Chat Endpoints

1. **Streaming Chat** - `/chat/stream` (POST)
   - Streams the chatbot's response using Server-Sent Events (SSE)
   - Request body:
     ```json
     {
       "query": "your question here",
       "k": 5  // optional, number of relevant documents to retrieve
     }
     ```
   - Returns a stream of SSE messages with the following types:
     - `token`: Individual response tokens as they're generated
     - `complete`: The complete response text
     - `sources`: Reference sources used
     - `error`: Any error that occurred

2. **Regular Chat** - `/chat` (POST)
   - Returns the complete response in a single request
   - Same request format as streaming chat
   - Returns:
     ```json
     {
       "response": "chatbot's response",
       "sources": [
         {
           "id": "source_id",
           "source": "document name",
           "page": 1,
           "relevance_score": 0.95
         }
       ]
     }
     ```

3. **Get Sources** - `/sources` (POST)
   - Retrieves relevant sources without generating a response
   - Same request format as chat endpoints
   - Returns an array of source documents with relevance scores

## Development

- The API server automatically reloads when code changes are detected
- API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc)
- Health check endpoint at `/health`