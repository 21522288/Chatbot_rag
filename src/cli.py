"""Command-line interface for the dental clinic chatbot."""
import argparse
import sys
from typing import Optional

from src.core.chatbot import DentalChatbot
from src.data_processing.document_loader import DocumentProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Dental Clinic Chatbot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--reload-data",
        action="store_true",
        help="Reload and reprocess all documents"
    )
    
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the existing vector database"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (optional)"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode for responses"
    )
    
    return parser

def process_documents() -> None:
    """Process and store documents in the vector database."""
    try:
        processor = DocumentProcessor()
        processor.process_and_store()
        logger.info("Document processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        sys.exit(1)

def interactive_mode(chatbot: DentalChatbot, streaming: bool = True) -> None:
    """
    Run the chatbot in interactive mode.
    
    Args:
        chatbot: The chatbot instance
        streaming: Whether to enable streaming mode for responses
    """
    print("\nDental Clinic Chatbot")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'sources' to see the sources for the last response")
    print("-" * 50)
    
    last_sources = None
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ("quit", "exit"):
                break
                
            if query.lower() == "sources" and last_sources:
                print("\nSources used for the last response:")
                for source in last_sources:
                    print(f"\nSource: {source['source']}")
                    print(f"Page: {source['page']}")
                    print(f"Relevance Score: {source['relevance_score']:.2f}")
                continue
                
            if not query:
                continue
                
            print("\nChatbot:", end=" ", flush=True)
            response, sources = chatbot.get_response(query, streaming=streaming)
            if not streaming:
                print(response)
            print("\n")  # Add newline after streaming response
            last_sources = sources
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print("\nSorry, I encountered an error. Please try again.")

def process_single_query(chatbot: DentalChatbot, query: str, streaming: bool = True) -> None:
    """
    Process a single query and exit.
    
    Args:
        chatbot: The chatbot instance
        query: The query to process
        streaming: Whether to enable streaming mode for responses
    """
    try:
        print("\nChatbot:", end=" ", flush=True)
        response, sources = chatbot.get_response(query, streaming=streaming)
        if not streaming:
            print(response)
        print("\n")  # Add newline after streaming response
        
        print("\nSources:")
        for source in sources:
            print(f"\nSource: {source['source']}")
            print(f"Page: {source['page']}")
            print(f"Relevance Score: {source['relevance_score']:.2f}")
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        sys.exit(1)

def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        if args.clear_db:
            DocumentProcessor.clear_database()
            
        if args.reload_data:
            process_documents()
            
        chatbot = DentalChatbot()
        streaming = not args.no_stream
        
        if args.query:
            process_single_query(chatbot, args.query, streaming=streaming)
        else:
            interactive_mode(chatbot, streaming=streaming)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 