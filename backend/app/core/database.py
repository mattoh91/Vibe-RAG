from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List, Dict, Any, Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        self.client = None
        self.collection_name = "documents"
        
    async def connect(self):
        """Connect to Qdrant database"""
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Collections: {len(collections.collections)}")
            
            # Create collection if it doesn't exist
            await self.ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def ensure_collection(self):
        """Ensure the documents collection exists"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add document chunks to the vector database"""
        try:
            points = []
            for doc in documents:
                point = PointStruct(
                    id=doc["id"],
                    vector=doc["embedding"],
                    payload={
                        "document_id": doc["document_id"],
                        "chunk_index": doc["chunk_index"],
                        "text": doc["text"],
                        "page_number": doc.get("page_number"),
                        "metadata": doc.get("metadata", {})
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} document chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar document chunks"""
        try:
            search_filter = None
            if document_ids:
                search_filter = {
                    "must": [
                        {
                            "key": "document_id",
                            "match": {"any": document_ids}
                        }
                    ]
                }
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "document_id": hit.payload["document_id"],
                    "chunk_index": hit.payload["chunk_index"],
                    "text": hit.payload["text"],
                    "page_number": hit.payload.get("page_number"),
                    "metadata": hit.payload.get("metadata", {})
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "document_id",
                                "match": {"value": document_id}
                            }
                        ]
                    }
                }
            )
            
            logger.info(f"Deleted document {document_id} from vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

# Global vector database instance
vector_db = VectorDatabase()