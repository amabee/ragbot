from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Keys (mapped from uppercase env vars)
    groq_api_key: Optional[str] = None

    # Database settings (mapped from uppercase env vars)
    chroma_db_dir: str = "./chroma_db"
    chroma_api_key: Optional[str] = None
    chroma_tenant: Optional[str] = None
    chroma_db_name: str = "ragbot"

    # Model settings
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    spacy_model: str = "en_core_web_sm"

    # Paths
    books_index_path: str = "books_index"
    per_book_path: Path = Path("chroma_per_book")

    # Default thresholds
    default_book_threshold: float = 1.2
    default_chunk_threshold: float = 1.0
    default_max_chunks: int = 8

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"
        # This tells Pydantic to automatically map UPPERCASE env vars to lowercase field names
        case_sensitive = False


settings = Settings()
