from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from config import settings
from models.responses import HealthResponse
from routes import books, query
from services.ml_service import ml_service
from services.book_service import book_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("🚀 Starting Children's Book Query API...")

    try:
        await ml_service.initialize_models()
        await book_service.initialize_database()
        logger.info("✅ All services initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down Children's Book Query API...")


# Initialize FastAPI app
app = FastAPI(
    title="Children's Book Query API",
    description="A RAG-based API for querying children's books with entity-aware search",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(books.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


@app.get("/", response_model=HealthResponse, summary="Health Check")
async def root():
    """Simple health check endpoint."""
    return HealthResponse(
        message="Children's Book Query API is running!", status="healthy"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level,
    )
