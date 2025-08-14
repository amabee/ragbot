from pathlib import Path
from typing import List, Tuple, Optional
from langchain_chroma import Chroma
from ..models.schemas import BookMatch
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, embeddings, books_index_db):
        self.embeddings = embeddings
        self.books_index_db = books_index_db
    
    def get_best_book_for_query(
        self, 
        query: str, 
        max_score_threshold: float = None
    ) -> Tuple[Optional[str], float, List[BookMatch]]:
        """Find the single best book that matches the query"""
        if max_score_threshold is None:
            max_score_threshold = settings.DEFAULT_BOOK_THRESHOLD
            
        try:
            book_results_with_scores = self.books_index_db.similarity_search_with_score(query, k=10)
            
            if not book_results_with_scores:
                return None, float('inf'), []
            
            # Get the best match
            best_doc, best_score = book_results_with_scores[0]
            best_book = best_doc.metadata["book_name"]
            
            # Get other matches for reference
            other_matches = [
                BookMatch(name=doc.metadata["book_name"], score=score)
                for doc, score in book_results_with_scores[1:4]
            ]
            
            # Check threshold
            if best_score > max_score_threshold:
                return None, best_score, other_matches
                
            return best_book, best_score, other_matches
        
        except Exception as e:
            logger.error(f"Error in get_best_book_for_query: {e}")
            return None, float('inf'), []
    
    def query_single_book(
        self, 
        query: str, 
        book_name: str, 
        k: int = None, 
        max_chunk_threshold: float = None
    ) -> List:
        """Query a specific book's chunks"""
        if k is None:
            k = settings.DEFAULT_MAX_CHUNKS
        if max_chunk_threshold is None:
            max_chunk_threshold = settings.DEFAULT_CHUNK_THRESHOLD
            
        try:
            chroma_dir = settings.PER_BOOK_DIR / book_name
            if not chroma_dir.exists():
                logger.error(f"Book directory not found: {chroma_dir}")
                return []
            
            db = Chroma(persist_directory=str(chroma_dir), embedding_function=self.embeddings)
            chunks_with_scores = db.similarity_search_with_score(query, k=k)
            
            # Filter by threshold
            relevant_chunks = [
                chunk for chunk, score in chunks_with_scores 
                if score < max_chunk_threshold
            ]
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for '{book_name}'")
            return relevant_chunks
        
        except Exception as e:
            logger.error(f"Error querying book {book_name}: {e}")
            return []
