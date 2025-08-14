from langchain.prompts import ChatPromptTemplate
from typing import List
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.requests import QueryRequest
from models.responses import QueryResponse
from services.ml_service import ml_service
from services.book_service import book_service

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a friendly helper who loves stories and helping kids with their questions.

Here's some information from the book "{book_title}":
{context}

Question: {question}

Instructions:
- Answer using the information from "{book_title}" as much as possible
- Include quotes or parts of the story that help explain your answer
- If the story doesn't have the full answer, tell what you can figure out and what's missing
- Be playful, fun, and kind—make your answer feel like a story adventure!

Answer:
"""

class QueryService:
    """Service for handling book queries and generating responses."""
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a book query and return response."""
        query_text = request.query_text.strip()
        if not query_text:
            raise ValueError("Query text cannot be empty")
        
        logger.info(f"Processing query: '{query_text}'")
        
        # Extract entities
        entities = ml_service.extract_entities(query_text)
        logger.info(f"Extracted entities: {entities}")
        
        # Choose book
        book_score = None
        if request.force_book:
            target_book = request.force_book
            logger.info(f"Forced to use book: '{target_book}'")
        else:
            target_book, book_score = await book_service.find_best_book(query_text, entities)
            if not target_book:
                raise ValueError("No suitable book found for the query")
            
            if book_score > request.book_threshold:
                raise ValueError(
                    f"No book meets the threshold. Best match: '{target_book}' with score {book_score:.3f}"
                )

        # Get relevant chunks
        relevant_chunks = await book_service.get_book_chunks(
            query_text,
            target_book,
            entities,
            k=request.max_chunks,
            max_chunk_threshold=request.chunk_threshold,
        )

        if not relevant_chunks:
            raise ValueError(f"No relevant content found in '{target_book}'")

        # Generate response
        context_text = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, 
            question=query_text, 
            book_title=target_book
        )
        
        answer = await ml_service.generate_response(prompt)
        
        logger.info(f"Successfully generated response from '{target_book}' using {len(relevant_chunks)} chunks")
        
        return QueryResponse(
            answer=answer,
            source_book=target_book,
            chunks_used=len(relevant_chunks),
            book_score=book_score,
            entities_found=entities
        )

# Create singleton instance
query_service = QueryService()
