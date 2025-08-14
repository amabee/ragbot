from fastapi import APIRouter, Query
from typing import Optional
from models.requests import QueryRequest
from models.responses import QueryResponse
from controllers.query_controller import query_controller

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse, summary="Query Books")
async def query_books(request: QueryRequest):
    """
    Query the book database with a question and get an AI-generated answer.

    The system will:
    1. Extract entities from your query
    2. Find the most relevant book (unless force_book is specified)
    3. Retrieve relevant chunks from that book
    4. Generate a kid-friendly answer using the book content
    """
    return await query_controller.query_books(request)


@router.get("/", response_model=QueryResponse, summary="Query Books (GET)")
async def query_books_get(
    query_text: str = Query(..., description="The question about the book"),
    book_threshold: float = Query(
        default=1.2, description="Threshold for book selection"
    ),
    chunk_threshold: float = Query(
        default=1.0, description="Threshold for chunk relevance"
    ),
    max_chunks: int = Query(
        default=8, description="Maximum number of chunks to retrieve"
    ),
    force_book: Optional[str] = Query(
        default=None, description="Force query on specific book"
    ),
):
    """GET version of the query endpoint for simple testing."""
    request = QueryRequest(
        query_text=query_text,
        book_threshold=book_threshold,
        chunk_threshold=chunk_threshold,
        max_chunks=max_chunks,
        force_book=force_book,
    )
    return await query_controller.query_books(request)
