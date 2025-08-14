from pydantic import BaseModel
from typing import List, Optional


class QueryResponse(BaseModel):
    answer: str
    source_book: str
    chunks_used: int
    book_score: Optional[float] = None
    entities_found: List[str] = []


class BookInfo(BaseModel):
    book_name: str
    score: Optional[float] = None


class AvailableBooksResponse(BaseModel):
    books: List[str]
    total_count: int


class HealthResponse(BaseModel):
    message: str
    status: str
    version: str = "1.0.0"
