import spacy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from typing import List, Optional
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import settings

logger = logging.getLogger(__name__)

class MLService:
    """Service for managing ML models and NLP operations."""
    
    def __init__(self):
        self.nlp: Optional[spacy.Language] = None
        self.model: Optional[ChatGroq] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
    
    async def initialize_models(self):
        """Initialize all ML models."""
        try:
            logger.info("Loading ML models...")
            
            # Load spaCy model
            self.nlp = spacy.load(settings.spacy_model)
            logger.info("✅ spaCy model loaded")
            
            # Initialize ChatGroq model
            self.model = ChatGroq(model=settings.groq_model, temperature=0)
            logger.info("✅ ChatGroq model initialized")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("✅ Embeddings model loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract PERSON/ORG/GPE entities from text."""
        if not self.nlp:
            raise RuntimeError("spaCy model not initialized")
        
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response using the language model."""
        if not self.model:
            raise RuntimeError("Language model not initialized")
        
        response = self.model.invoke(prompt)
        return response.content

# Create singleton instance
ml_service = MLService()
