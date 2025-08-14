from fastapi import APIRouter
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.responses import AvailableBooksResponse
from controllers.book_controller import book_controller

router = APIRouter(prefix="/books", tags=["books"])


@router.get("/", response_model=AvailableBooksResponse, summary="Get Available Books")
async def get_available_books():
    """Get list of all available books in the database."""
    return await book_controller.get_available_books()
