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

The API will be available at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

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

### Example Usage with JavaScript

```javascript
// Streaming chat example
const eventSource = new EventSource('/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: "What are your operating hours?"
  })
});

let responseText = '';

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'token':
      // Append token to the response text
      responseText += data.content;
      // Update UI with the token
      console.log('Streaming:', responseText);
      break;
      
    case 'complete':
      // Final complete response
      console.log('Complete response:', data.content);
      break;
      
    case 'sources':
      // Reference sources
      console.log('Sources:', data.content);
      eventSource.close();  // Close the connection after receiving sources
      break;
      
    case 'error':
      console.error('Error:', data.content);
      eventSource.close();
      break;
  }
};

eventSource.onerror = (error) => {
  console.error('EventSource error:', error);
  eventSource.close();
};

// Regular chat example
async function sendChat(query) {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ query })
  });
  const data = await response.json();
  console.log('Response:', data.response);
  console.log('Sources:', data.sources);
}
```

## Development

- The API server automatically reloads when code changes are detected
- API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc)
- Health check endpoint at `/health`

## Error Handling

The API uses standard HTTP status codes:
- 200: Successful response
- 400: Bad request (invalid input)
- 500: Server error

Errors include a detail message explaining what went wrong.