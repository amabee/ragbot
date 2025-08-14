from fastapi import HTTPException
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.responses import AvailableBooksResponse
from services.book_service import book_service

logger = logging.getLogger(__name__)


class BookController:
    """Controller for book-related operations."""

    async def get_available_books(self) -> AvailableBooksResponse:
        """Get all available books."""
        try:
            books = await book_service.get_available_books()
            return AvailableBooksResponse(books=books, total_count=len(books))
        except Exception as e:
            logger.error(f"Error in get_available_books: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error retrieving available books: {str(e)}"
            )


book_controller = BookController()
