from fastapi import HTTPException
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.requests import QueryRequest
from models.responses import QueryResponse
from services.query_service import query_service

logger = logging.getLogger(__name__)


class QueryController:
    """Controller for query operations."""

    async def query_books(self, request: QueryRequest) -> QueryResponse:
        """Process a book query."""
        try:
            return await query_service.process_query(request)
        except ValueError as e:
            logger.warning(f"Client error in query: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Server error in query: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error processing query: {str(e)}"
            )


query_controller = QueryController()
