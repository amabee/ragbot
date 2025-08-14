from fastapi import APIRouter
from ..models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", message="RAG API is running and ready to serve queries"
    )
