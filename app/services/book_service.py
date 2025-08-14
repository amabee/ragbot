from langchain_chroma import Chroma
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import settings
from services.ml_service import ml_service

logger = logging.getLogger(__name__)

class BookService:
    """Service for book database operations."""
    
    def __init__(self):
        self.books_index_db: Optional[Chroma] = None
    
    async def initialize_database(self):
        """Initialize the books index database."""
        try:
            if not Path(settings.books_index_path).exists():
                raise FileNotFoundError(f"Books index not found at {settings.books_index_path}")
            
            self.books_index_db = Chroma(
                persist_directory=settings.books_index_path, 
                embedding_function=ml_service.embeddings
            )
            logger.info("✅ Books index database loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading books database: {e}")
            raise
    
    async def get_available_books(self) -> List[str]:
        """Get list of all available books."""
        if not self.books_index_db:
            raise RuntimeError("Books database not initialized")
        
        try:
            all_docs = self.books_index_db.similarity_search("", k=1000)
            book_names = list(set(
                doc.metadata.get("book_name", "") 
                for doc in all_docs 
                if doc.metadata.get("book_name")
            ))
            return sorted([name for name in book_names if name])
        except Exception as e:
            logger.error(f"Error retrieving books: {e}")
            raise
    
    async def find_best_book(self, query: str, entities: List[str]) -> Tuple[Optional[str], float]:
        """Find the best matching book for a query."""
        if not self.books_index_db:
            raise RuntimeError("Books database not initialized")
        
        results = self.books_index_db.similarity_search_with_score(query, k=10)
        
        if not results:
            logger.warning("No books matched the query")
            return None, float("inf")

        boosted_results = []
        entities_lower = [e.lower() for e in entities]
        
        for doc, score in results:
            boost = 0
            book_name = doc.metadata.get("book_name", "").lower()

            # Book name match boost
            if any(e in book_name for e in entities_lower):
                boost -= 0.3

            # Character metadata match boost
            per_book_dir = settings.per_book_path / book_name
            if per_book_dir.exists():
                try:
                    book_db = Chroma(
                        persist_directory=str(per_book_dir), 
                        embedding_function=ml_service.embeddings
                    )
                    all_meta = book_db._collection.get(include=["metadatas"])["metadatas"]
                    characters = {
                        str(c).lower()
                        for m in all_meta
                        if "characters" in m and isinstance(m["characters"], list)
                        for c in m["characters"]
                    }
                    if any(e in characters for e in entities_lower):
                        boost -= 0.5
                except Exception as e:
                    logger.warning(f"Error accessing book database for {book_name}: {e}")

            boosted_results.append((doc, score + boost))

        # Pick best
        boosted_results.sort(key=lambda x: x[1])
        best_doc, best_score = boosted_results[0]
        best_book = best_doc.metadata["book_name"]

        logger.info(f"Best matching book: '{best_book}' (score: {best_score:.3f})")
        return best_book, best_score
    
    async def get_book_chunks(self, query: str, book_name: str, entities: List[str], 
                            k: int = 10, max_chunk_threshold: float = 1.0) -> List:
        """Get relevant chunks from a specific book."""
        chroma_dir = settings.per_book_path / book_name
        if not chroma_dir.exists():
            logger.error(f"Book directory not found: {chroma_dir}")
            return []

        try:
            db = Chroma(persist_directory=str(chroma_dir), embedding_function=ml_service.embeddings)
            entities_lower = [e.lower() for e in entities]
            chunks_with_scores = db.similarity_search_with_score(query, k=k * 2)

            relevant_chunks = []
            logger.info(f"Evaluating chunks for '{book_name}':")
            
            for i, (chunk, score) in enumerate(chunks_with_scores):
                boost = 0
                if any(
                    e in [str(c).lower() for c in chunk.metadata.get("characters", [])]
                    for e in entities_lower
                ):
                    boost -= 0.3
                
                final_score = score + boost
                status = "INCLUDED" if final_score < max_chunk_threshold else "EXCLUDED"
                logger.info(f"   Chunk {i+1}: {final_score:.3f} {status}")
                
                if final_score < max_chunk_threshold:
                    relevant_chunks.append(chunk)

            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error querying book '{book_name}': {e}")
            return []

# Create singleton instance
book_service = BookService()
