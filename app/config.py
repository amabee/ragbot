import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Settings
    API_TITLE = "RAG Book Query API"
    API_DESCRIPTION = "API for querying book content using RAG"
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Model Settings
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    
    # Database Settings
    BOOKS_INDEX_DIR = "books_index"
    PER_BOOK_DIR = Path("chroma_per_book")
    
    # Query Settings
    DEFAULT_BOOK_THRESHOLD = 1.2
    DEFAULT_CHUNK_THRESHOLD = 1.0
    DEFAULT_MAX_CHUNKS = 8

settings = Settings()
