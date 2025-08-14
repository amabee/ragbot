from fastapi import APIRouter, HTTPException
from ..models.schemas import BooksListResponse
from ..config import settings
import logging

router = APIRouter(prefix="/api/v1", tags=["Books"])
logger = logging.getLogger(__name__)


@router.get("/books", response_model=BooksListResponse)
async def list_books():
    """Get list of available books"""
    try:
        if not settings.PER_BOOK_DIR.exists():
            return BooksListResponse(
                books=[], count=0, error="Books directory not found"
            )

        books = [d.name for d in settings.PER_BOOK_DIR.iterdir() if d.is_dir()]
        return BooksListResponse(books=books, count=len(books))

    except Exception as e:
        logger.error(f"Error listing books: {e}")
        raise HTTPException(status_code=500, detail=str(e))
