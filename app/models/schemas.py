from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    book_threshold: Optional[float] = 1.2
    chunk_threshold: Optional[float] = 1.0
    max_chunks: Optional[int] = 8
    force_book: Optional[str] = None

class BookMatch(BaseModel):
    name: str
    score: float

class QueryResponse(BaseModel):
    query: str
    selected_book: str
    book_score: float
    other_books: List[BookMatch]
    answer: str
    chunks_used: int
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class BooksListResponse(BaseModel):
    books: List[str]
    count: int
    error: Optional[str] = None
