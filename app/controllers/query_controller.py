from typing import List
from ..models.schemas import QueryRequest, QueryResponse, BookMatch
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService
from ..utils.templates import FALLBACK_RESPONSE_TEMPLATE, NO_BOOK_RESPONSE_TEMPLATE
import logging

logger = logging.getLogger(__name__)


class QueryController:
    def __init__(self, rag_service: RAGService, llm_service: LLMService):
        self.rag_service = rag_service
        self.llm_service = llm_service

    def generate_fallback_response(self, query: str, context: str = "no_book") -> str:
        """Generate a helpful fallback response when no relevant content is found"""
        if context == "no_book":
            return NO_BOOK_RESPONSE_TEMPLATE.format(question=query)
        else:
            return FALLBACK_RESPONSE_TEMPLATE.format(question=query)

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main method to process book queries"""
        try:
            logger.info(f"Processing query: '{request.query}'")

            # Determine target book
            if request.force_book:
                target_book = request.force_book
                book_score = 0.0
                other_books = []
                logger.info(f"Forced to use book: '{target_book}'")
            else:
                target_book, book_score, other_books = (
                    self.rag_service.get_best_book_for_query(
                        request.query, request.book_threshold
                    )
                )

                if not target_book:
                    # Generate AI-like response for no suitable book found
                    fallback_answer = self.generate_fallback_response(
                        request.query, "no_book"
                    )
                    return QueryResponse(
                        query=request.query,
                        selected_book="No suitable book found",
                        book_score=book_score,
                        other_books=other_books,
                        answer=fallback_answer,
                        chunks_used=0,
                        success=True,
                        error=None,
                    )

            # Query the selected book
            relevant_chunks = self.rag_service.query_single_book(
                request.query,
                target_book,
                k=request.max_chunks,
                max_chunk_threshold=request.chunk_threshold,
            )

            if not relevant_chunks:
                # Generate AI-like response for no relevant chunks found
                fallback_answer = self.generate_fallback_response(
                    request.query, "no_chunks"
                )
                return QueryResponse(
                    query=request.query,
                    selected_book=target_book,
                    book_score=book_score,
                    other_books=other_books,
                    answer=fallback_answer,
                    chunks_used=0,
                    success=True,
                    error=None,
                )

            # Generate answer using LLM
            context_text = "\n\n---\n\n".join(
                [chunk.page_content for chunk in relevant_chunks]
            )
            answer = self.llm_service.generate_answer(
                context_text, request.query, target_book
            )

            logger.info(f"✅ Successfully processed query from '{target_book}'")

            return QueryResponse(
                query=request.query,
                selected_book=target_book,
                book_score=book_score,
                other_books=other_books,
                answer=answer,
                chunks_used=len(relevant_chunks),
                success=True,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"I apologize, but I encountered an unexpected error while processing your question: '{request.query}'. Please try again, or contact support if the issue persists. Error details: {str(e)}"
            return QueryResponse(
                query=request.query,
                selected_book="Error occurred",
                book_score=float("inf"),
                other_books=[],
                answer=error_response,
                chunks_used=0,
                success=False,
                error=str(e),
            )
