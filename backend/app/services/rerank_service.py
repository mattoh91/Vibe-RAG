from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class RerankService:
    def __init__(self):
        self.model = None
        self.model_name = settings.RERANK_MODEL
        
    async def initialize(self):
        """Initialize the cross-encoder reranking model"""
        try:
            logger.info(f"Loading reranking model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Successfully loaded reranking model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranking model: {e}")
            raise
    
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder model"""
        if not self.model:
            await self.initialize()
        
        if not documents:
            return documents
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for doc in documents:
                pairs.append([query, doc['text']])
            
            # Get reranking scores
            scores = self.model.predict(pairs)
            
            # Add rerank scores to documents and sort
            reranked_docs = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(scores[i])
                reranked_docs.append(doc_copy)
            
            # Sort by rerank score (descending)
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            # Return original documents if reranking fails
            return documents
    
    def is_initialized(self) -> bool:
        """Check if the reranking model is initialized"""
        return self.model is not None

# Global reranking service instance
rerank_service = RerankService()