"""Core chatbot functionality for the dental clinic assistant."""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List, Tuple, Dict, Any, Generator, Optional, Union, AsyncGenerator
from langchain.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import GenerationChunk
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
import asyncio
import re
import aiohttp
from datetime import datetime

from src.config.settings import (
    CHROMA_DIR,
    MODEL_NAME,
    HUGGINGFACE_API_TOKEN,
    DEFAULT_PROMPT_TEMPLATE,
    BACKEND_API_URL,
    APPOINTMENT_CLASSIFICATION_PROMPT,
    MEMORY_KEY,
    MEMORY_WINDOW_SIZE
)
from src.models.embeddings import get_embedding_function
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DentalChatbot:
    """Dental clinic chatbot that provides information based on the clinic's documentation."""
    
    def __init__(self):
        """Initialize the chatbot with necessary components."""
        logger.info("Initializing DentalChatbot")
        self.embedding_function = get_embedding_function()
        self.vector_store = self._initialize_vector_store()
        self.prompt_template = ChatPromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)
        self.classification_prompt = ChatPromptTemplate.from_template(APPOINTMENT_CLASSIFICATION_PROMPT)
        self.llm = self._initialize_llm()
        self.session = None
        self.memory = self._initialize_memory()
        
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize the vector store connection.
        
        Returns:
            Chroma: Initialized vector store
        """
        logger.info(f"Connecting to vector store at {CHROMA_DIR}")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embedding_function
        )
        
    def _initialize_llm(self) -> ChatHuggingFace:
        """
        Initialize the language model.
        
        Returns:
            Union[ChatHuggingFace, HuggingFaceEndpoint]: Initialized language model
        """
        logger.info(f"Initializing language model: {MODEL_NAME}")
        
        # Create the base HuggingFaceEndpoint for ChatHuggingFace
        base_llm = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            task="text-generation",
            temperature=0.5,
            max_new_tokens=1024,
            repetition_penalty=1.1,
            do_sample=True,
            streaming=True,
            verbose=True
        )
        
        return ChatHuggingFace(
            llm=base_llm,
            streaming=True,
            verbose=True
        )
        
    def _initialize_memory(self) -> ConversationBufferWindowMemory:
        """Initialize conversation memory."""
        return ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            memory_key=MEMORY_KEY,
            return_messages=True
        )

    def _process_response(self, response: str) -> str:
        """
        Process the raw response to clean it and extract only the answer.
        Removes any token metadata and formatting artifacts.
        
        Args:
            response: Raw response from the language model
            
        Returns:
            str: Clean, processed response
        """
        # Remove any token metadata
        response = re.sub(r'data: {"type": "token", "content": "[^"]*"}', '', response)
        
        # Remove any "Human:" or "Context:" prefixes
        response = re.sub(r'^(Human:|Context:)\s*', '', response, flags=re.MULTILINE)
        
        # Extract content after "Assistant:" if present
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1]
            
        # Remove special tokens
        special_tokens = ['<|im_end|>', '<|endoftext|>', '<|end|>', '<|im_start|>']
        for token in special_tokens:
            response = response.replace(token, '')
            
        # Clean up extra whitespace while preserving newlines
        lines = response.split('\n')
        lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
        final_response = '\n'.join(filter(None, lines))
        
        return final_response.strip()
        
    async def _is_appointment_related(self, query: str) -> bool:
        """
        Determine if a query is related to appointments.
        
        Args:
            query: The user's question
            
        Returns:
            bool: True if the query is appointment-related, False otherwise
        """
        prompt = self.classification_prompt.format(query=query)
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        
        # Handle different response types
        response_text = (
            response.content if hasattr(response, 'content')
            else str(response) if response is not None
            else ""
        )
        
        return "yes" in response_text.strip().lower()[-5:]
                
    async def _get_appointments(self) -> Dict[str, Any]:
        """
        Fetch all appointments from the backend API.
        
        Returns:
            Dict[str, Any]: All appointments
        """
        await self._ensure_session()
        try:
            async with self.session.get(f"{BACKEND_API_URL}/api/Appointment") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching appointments: {str(e)}")
            return {"error": "Could not fetch appointment data"}

    def _format_appointments(self, appointments: List[Dict[str, Any]]) -> str:
        """
        Format appointments into a readable string.
        
        Args:
            appointments: List of appointment dictionaries
            
        Returns:
            str: Formatted appointments string
        """
        if not appointments or isinstance(appointments, dict) and "error" in appointments:
            return "Không có thông tin lịch hẹn."
            
        formatted = []
        for appt in appointments:
            formatted.append(
                f"- Lịch hẹn: {appt.get('maLichHen', 'N/A')}\n"
                f"  + Ngày: {appt.get('ngay', 'N/A')}\n"
                f"  + Giờ: {appt.get('gio', 'N/A')}\n"
                f"  + Trạng thái: {appt.get('trangThai', 'N/A')}\n"
            )
        
        return "Thông tin về các lịch hẹn hiện tại:\n" + "\n".join(formatted)

    async def get_response(
        self, 
        query: str, 
        k: int = 5,
        streaming: bool = False
    ) -> Union[AsyncGenerator[str, None], Tuple[str, List[Dict[str, Any]]]]:
        """
        Get a response to a user query, handling both general and appointment-related queries.
        
        Args:
            query: The user's question
            k: Number of relevant documents to retrieve
            streaming: Whether to stream the response token by token
            
        Returns:
            If streaming is False:
                Tuple[str, List[Dict]]: The response and sources
            If streaming is True:
                AsyncGenerator[str, None]: Token generator
        """
        logger.info(f"Processing query: {query}")

        try:
            # Get chat history
            chat_history = self.memory.load_memory_variables({}).get(MEMORY_KEY, [])
            
            # Check if query is appointment-related
            is_appointment = await self._is_appointment_related(query)
            
            logger.info(f"Appointment related: {is_appointment}")
            
            if is_appointment:
                # Get all appointments and format them
                appointments = await self._get_appointments()
                formatted_appointments = self._format_appointments(appointments)
                logger.info(f"Formatted appointments: {formatted_appointments}")
                
                # Format the prompt using the template
                prompt_value = self.prompt_template.format_messages(
                    context=formatted_appointments,
                    question=query,
                    chat_history=chat_history
                )
                
                sources = [{"source": "Appointments API", "relevance_score": 1.0}]
            else:
                # Handle general query using existing logic
                results = self.vector_store.similarity_search_with_score(query, k=k)
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                
                # Format the prompt using the template
                prompt_value = self.prompt_template.format_messages(
                    context=context_text,
                    question=query,
                    chat_history=chat_history
                )
                
                sources = [
                    {
                        "id": doc.metadata.get("id"),
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "relevance_score": float(score)
                    }
                    for doc, score in results
                ]

            if streaming:
                async def response_generator():
                    buffer = ""
                    try:
                        async for chunk in self.llm.astream(prompt_value):
                            if hasattr(chunk, 'content'):
                                text = chunk.content
                                text = text.replace("\\n", "\n")
                                if "<|im_end|>" in text:
                                    text = text.replace("<|im_end|>", "")
                                buffer += text
                                logger.debug(f"Streaming content: {text}")
                                # Simple consistent delay
                                if text.strip():  # Only delay for non-empty tokens
                                    await asyncio.sleep(0.02)
                                yield text
                            elif isinstance(chunk, str):
                                buffer += chunk
                                logger.debug(f"Streaming string: {chunk}")
                                if chunk.strip():  # Only delay for non-empty tokens
                                    await asyncio.sleep(0.02)
                                yield chunk
                        
                        # Save the complete response to memory after streaming
                        if buffer:
                            processed_buffer = self._process_response(buffer)
                            self.memory.save_context(
                                {"input": query},
                                {"output": processed_buffer}
                            )
                    except Exception as e:
                        logger.error(f"Error in streaming: {str(e)}")
                        yield f"Error during streaming: {str(e)}"
                    
                return response_generator()
            else:
                # Handle non-streaming response
                response = await asyncio.to_thread(self.llm.invoke, prompt_value)
                response_text = response.content if hasattr(response, 'content') else str(response)
                processed_response = self._process_response(response_text)
                
                # Save the conversation to memory
                self.memory.save_context(
                    {"input": query},
                    {"output": processed_response}
                )
                
                return processed_response, sources

        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            error_msg = "I apologize, but I encountered an error while processing your request."
            if streaming:
                async def error_generator():
                    yield error_msg
                return error_generator()
            return error_msg, []
            
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def get_sources(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the source documents for a query without generating a response.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List[Dict]: Metadata about the relevant sources
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "id": doc.metadata.get("id"),
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "relevance_score": float(score),
                "content": doc.page_content
            }
            for doc, score in results
        ]

    def clear_memory(self):
        """Clear the conversation memory."""
        logger.info("Clearing chatbot memory")
        self.memory = self._initialize_memory() 