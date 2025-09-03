from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            await self.initialize()
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            await self.initialize()
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (same as embed_text but semantically different)"""
        return await self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if not self.model:
            return settings.EMBEDDING_DIMENSION
        return self.model.get_sentence_embedding_dimension()

# Global embedding service instance
embedding_service = EmbeddingService()