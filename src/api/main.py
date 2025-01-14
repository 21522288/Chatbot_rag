"""FastAPI application for the dental clinic chatbot."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

from src.core.chatbot import DentalChatbot
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Dental Clinic Chatbot API",
    description="API for interacting with the dental clinic chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = DentalChatbot()

class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    query: str
    k: Optional[int] = 5

class SourceResponse(BaseModel):
    """Response model for sources."""
    id: Optional[str]
    source: Optional[str]
    page: Optional[int]
    relevance_score: float
    content: Optional[str]

async def generate_streaming_response(query: str, k: int = 5):
    """Generate streaming response for chat endpoint."""
    try:
        # Get the token generator
        token_generator = await chatbot.get_response(query, k=k, streaming=True)
        response_text = ""
        
        # Stream tokens as they're generated
        async for token in token_generator:
            response_text += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # After streaming is complete, get sources
        response, sources = await chatbot.get_response(query, k=k, streaming=False)
        
        # Send the complete response and sources
        yield f"data: {json.dumps({'type': 'complete', 'content': response_text})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    finally:
        # Ensure we close the chatbot session
        await chatbot.close()

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    Streaming chat endpoint.
    Returns a server-sent events stream with the chatbot's response and sources.
    """
    try:
        # Parse the JSON body manually
        body = await request.json()
        chat_request = ChatRequest(**body)
        
        return StreamingResponse(
            generate_streaming_response(chat_request.query, chat_request.k),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
            }
        )
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Regular chat endpoint.
    Returns the chatbot's response and sources in a single response.
    """
    try:
        response, sources = await chatbot.get_response(request.query, k=request.k)
        return {
            "response": response,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await chatbot.close()

@app.post("/api/memory/clear")
async def clear_memory():
    """
    Clear the chatbot's conversation memory.
    This endpoint should be called when the frontend is refreshed.
    """
    try:
        chatbot.clear_memory()
        return {"status": "success", "message": "Memory cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sources")
async def get_sources(request: ChatRequest) -> List[SourceResponse]:
    """
    Get relevant sources for a query without generating a response.
    """
    try:
        sources = chatbot.get_sources(request.query, k=request.k)
        return [SourceResponse(**source) for source in sources]
    except Exception as e:
        logger.error(f"Error in sources endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Mount static files after all API routes are defined
app.mount("/", StaticFiles(directory="static", html=True), name="static") 