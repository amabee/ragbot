from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from pathlib import Path
import spacy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Paths
BOOKS_INDEX_PATH = "books_index"
PER_BOOK_PATH = Path("chroma_per_book")

# Prompt
PROMPT_TEMPLATE = """
You are a friendly helper who loves stories and helping kids with their questions.

Here's some information from the book "{book_title}":
{context}

Question: {question}

Instructions:
- Answer using the information from "{book_title}" as much as possible
- Include quotes or parts of the story that help explain your answer
- If the story doesn't have the full answer, tell what you can figure out and what's missing
- Be playful, fun, and kind—make your answer feel like a story adventure!

Answer:
"""

# Initialize FastAPI app
app = FastAPI(
    title="Book Query API",
    description="API for querying children's books using semantic search and LLM responses",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
nlp = None
embeddings = None
books_index_db = None
model = None

# Pydantic models
class QueryRequest(BaseModel):
    query_text: str = Field(..., description="The question or query about the book")
    book_threshold: Optional[float] = Field(1.2, description="Threshold for book selection")
    chunk_threshold: Optional[float] = Field(1.0, description="Threshold for chunk relevance")
    max_chunks: Optional[int] = Field(8, description="Maximum number of chunks to use")
    force_book: Optional[str] = Field(None, description="Force query to use a specific book")

class ChunkInfo(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    source_book: str
    chunks_used: int
    chunk_details: List[ChunkInfo]
    entities_found: List[str]
    book_selection_score: Optional[float]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    books_index_exists: bool

class BookListResponse(BaseModel):
    books: List[str]
    total_count: int

# Utility functions
def extract_entities(text: str) -> List[str]:
    """Extract PERSON/ORG/GPE entities from query."""
    if not nlp:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]

def get_best_book_for_query(query: str, books_index_db, embeddings):
    """Entity-aware book selection using both book names & characters from per-book DBs."""
    entities = [e.lower() for e in extract_entities(query)]
    results = books_index_db.similarity_search_with_score(query, k=10)

    if not results:
        logger.warning("No books matched the query")
        return None, float("inf")

    boosted_results = []
    for doc, score in results:
        boost = 0
        book_name = doc.metadata.get("book_name", "").lower()

        # Book name match boost
        if any(e in book_name for e in entities):
            boost -= 0.3

        # Character metadata match boost
        per_book_dir = PER_BOOK_PATH / book_name
        if per_book_dir.exists():
            try:
                book_db = Chroma(
                    persist_directory=str(per_book_dir), embedding_function=embeddings
                )
                all_meta = book_db._collection.get(include=["metadatas"])["metadatas"]
                characters = {
                    str(c).lower()
                    for m in all_meta
                    if "characters" in m and isinstance(m["characters"], list)
                    for c in m["characters"]
                }
                if any(e in characters for e in entities):
                    boost -= 0.5
            except Exception as e:
                logger.warning(f"Error accessing book DB for {book_name}: {e}")

        boosted_results.append((doc, score + boost))

    # Pick best
    boosted_results.sort(key=lambda x: x[1])
    best_doc, best_score = boosted_results[0]
    best_book = best_doc.metadata["book_name"]

    logger.info(f"Best matching book: '{best_book}' (score: {best_score:.3f})")
    return best_book, best_score

def query_single_book(query: str, book_name: str, embeddings, k: int = 10, max_chunk_threshold: float = 1.0):
    """Entity-aware chunk selection from a single book."""
    chroma_dir = PER_BOOK_PATH / book_name
    if not chroma_dir.exists():
        logger.error(f"Book directory not found: {chroma_dir}")
        return []

    try:
        db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
        entities = [e.lower() for e in extract_entities(query)]
        chunks_with_scores = db.similarity_search_with_score(query, k=k * 2)

        relevant_chunks = []
        chunk_details = []
        
        logger.info(f"Chunk relevance scores for '{book_name}':")
        for i, (chunk, score) in enumerate(chunks_with_scores):
            boost = 0
            if any(
                e in [str(c).lower() for c in chunk.metadata.get("characters", [])]
                for e in entities
            ):
                boost -= 0.3
            final_score = score + boost
            
            chunk_info = ChunkInfo(
                content=chunk.page_content,
                score=final_score,
                metadata=chunk.metadata
            )
            chunk_details.append(chunk_info)
            
            status = "INCLUDED" if final_score < max_chunk_threshold else "EXCLUDED"
            logger.info(f"   Chunk {i+1}: {final_score:.3f} {status}")
            
            if final_score < max_chunk_threshold:
                relevant_chunks.append(chunk)

        return relevant_chunks, chunk_details
    except Exception as e:
        logger.error(f"Error querying book '{book_name}': {e}")
        return [], []

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and databases on startup."""
    global nlp, embeddings, books_index_db, model
    
    try:
        logger.info("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        
        logger.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        logger.info("Loading chat model...")
        model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
        
        logger.info("Loading books index...")
        if Path(BOOKS_INDEX_PATH).exists():
            books_index_db = Chroma(
                persist_directory=BOOKS_INDEX_PATH, embedding_function=embeddings
            )
            logger.info("All models and databases loaded successfully!")
        else:
            logger.warning(f"Books index not found at {BOOKS_INDEX_PATH}")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down...")

# API Routes
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=all([nlp, embeddings, model]),
        books_index_exists=books_index_db is not None
    )

@app.get("/api/v1/books", response_model=BookListResponse)
async def list_books():
    """Get list of available books."""
    try:
        if not PER_BOOK_PATH.exists():
            raise HTTPException(status_code=404, detail="Books directory not found")
        
        books = [d.name for d in PER_BOOK_PATH.iterdir() if d.is_dir()]
        return BookListResponse(books=books, total_count=len(books))
    except Exception as e:
        logger.error(f"Error listing books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_books(request: QueryRequest):
    """Query books with a question."""
    try:
        if not all([nlp, embeddings, books_index_db, model]):
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        query_text = request.query_text
        logger.info(f"Processing query: '{query_text}'")
        
        # Extract entities
        entities_found = extract_entities(query_text)
        
        # Choose book
        book_selection_score = None
        if request.force_book:
            target_book = request.force_book
            logger.info(f"Forced to use book: '{target_book}'")
        else:
            target_book, book_selection_score = get_best_book_for_query(
                query_text, books_index_db, embeddings
            )
            if not target_book:
                raise HTTPException(status_code=404, detail="No suitable book found for the query")

        # Get chunks
        relevant_chunks, chunk_details = query_single_book(
            query_text,
            target_book,
            embeddings,
            k=request.max_chunks,
            max_chunk_threshold=request.chunk_threshold,
        )

        if not relevant_chunks:
            raise HTTPException(
                status_code=404, 
                detail=f"No relevant chunks found in '{target_book}'"
            )

        # Build final answer
        context_text = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, question=query_text, book_title=target_book
        )
        response = model.invoke(prompt)

        logger.info(f"Generated response from '{target_book}' using {len(relevant_chunks)} chunks")

        return QueryResponse(
            answer=response.content,
            source_book=target_book,
            chunks_used=len(relevant_chunks),
            chunk_details=[cd for cd in chunk_details if cd.score < request.chunk_threshold],
            entities_found=entities_found,
            book_selection_score=book_selection_score
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Book Query API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "books": "/api/v1/books", 
            "query": "/api/v1/query (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
