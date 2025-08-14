from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import settings
from .routes import health, books, query
from .services.rag_service import RAGService
from .services.llm_service import LLMService
from .controllers.query_controller import QueryController
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables (will be initialized on startup)
rag_service = None
llm_service = None
query_controller = None

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(books.router)
app.include_router(query.router)


@app.on_event("startup")
async def startup_event():
    """Initialize models and databases on startup"""
    global rag_service, llm_service, query_controller

    try:
        logger.info("Initializing models and databases...")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize databases
        books_index_db = Chroma(
            persist_directory=settings.BOOKS_INDEX_DIR, embedding_function=embeddings
        )

        # Initialize services
        rag_service = RAGService(embeddings, books_index_db)
        llm_service = LLMService()

        # Initialize controllers
        query_controller = QueryController(rag_service, llm_service)

        logger.info("✅ All models and databases initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
