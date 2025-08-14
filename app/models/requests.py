from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query_text: str = Field(..., description="The question about the book")
    book_threshold: float = Field(
        default=1.2, description="Threshold for book selection"
    )
    chunk_threshold: float = Field(
        default=1.0, description="Threshold for chunk relevance"
    )
    max_chunks: int = Field(
        default=8, description="Maximum number of chunks to retrieve"
    )
    force_book: Optional[str] = Field(
        default=None, description="Force query on specific book"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "Who is the main character?",
                "book_threshold": 1.2,
                "chunk_threshold": 1.0,
                "max_chunks": 8,
                "force_book": None,
            }
        }
