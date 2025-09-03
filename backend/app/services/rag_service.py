from typing import List, Dict, Any, Optional
import logging
import time
import uuid
from datetime import datetime

from app.services.pdf_service import PDFService
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.rerank_service import rerank_service
from app.core.database import vector_db
from app.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.pdf_service = PDFService()
        self.documents: Dict[str, Document] = {}  # In-memory document storage
        self.document_chunks: Dict[str, List[DocumentChunk]] = {}  # In-memory chunk storage
        
    async def initialize(self):
        """Initialize all services"""
        try:
            await embedding_service.initialize()
            await rerank_service.initialize()
            # Note: vector_db.connect() would be called here in production
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    async def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process a PDF document for RAG"""
        try:
            start_time = time.time()
            
            # Save uploaded file
            file_path = await self.pdf_service.save_uploaded_file(file_content, filename)
            
            # Process PDF and extract chunks
            document = await self.pdf_service.process_pdf(file_path, filename)
            if not document:
                return {
                    "success": False,
                    "message": "Failed to process PDF document"
                }
            
            # Extract text chunks
            chunks = await self.pdf_service._extract_text_chunks(file_path, document.id)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await embedding_service.embed_texts(chunk_texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Store in memory (in production, this would go to vector database)
            self.documents[document.id] = document
            self.document_chunks[document.id] = chunks
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed document {filename}: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return {
                "success": True,
                "document_id": document.id,
                "filename": filename,
                "chunk_count": len(chunks),
                "processing_time": processing_time,
                "message": "Document processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            return {
                "success": False,
                "message": f"Failed to process document: {str(e)}"
            }
    
    async def query_documents(
        self, 
        query: str, 
        max_results: int = 5,
        use_reranking: bool = True,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query documents using RAG"""
        try:
            start_time = time.time()
            
            if not llm_service.is_configured():
                return {
                    "success": False,
                    "message": "LLM service not configured. Please configure an LLM provider first."
                }
            
            # Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            
            # Retrieve similar chunks
            similar_chunks = await self._retrieve_similar_chunks(
                query_embedding, 
                max_results * 2,  # Get more for reranking
                document_ids
            )
            
            if not similar_chunks:
                return {
                    "success": True,
                    "query": query,
                    "response": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Rerank if enabled
            if use_reranking and rerank_service.is_initialized():
                similar_chunks = await rerank_service.rerank_documents(
                    query, 
                    similar_chunks, 
                    top_k=max_results
                )
            else:
                similar_chunks = similar_chunks[:max_results]
            
            # Generate response using LLM
            response = await llm_service.generate_rag_response(
                query=query,
                context_chunks=similar_chunks
            )
            
            # Prepare source information
            sources = []
            for chunk in similar_chunks:
                sources.append({
                    "document_id": chunk["document_id"],
                    "document_name": chunk.get("document_name", "Unknown"),
                    "page_number": chunk.get("page_number"),
                    "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "similarity_score": chunk.get("score", 0.0),
                    "rerank_score": chunk.get("rerank_score")
                })
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "query": query,
                "response": response,
                "sources": sources,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            return {
                "success": False,
                "message": f"Failed to process query: {str(e)}"
            }
    
    async def _retrieve_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar chunks using cosine similarity (in-memory implementation)"""
        try:
            import numpy as np
            
            all_chunks = []
            
            # Filter by document IDs if specified
            target_doc_ids = document_ids if document_ids else list(self.document_chunks.keys())
            
            for doc_id in target_doc_ids:
                if doc_id not in self.document_chunks:
                    continue
                    
                document = self.documents.get(doc_id)
                if not document:
                    continue
                
                chunks = self.document_chunks[doc_id]
                
                for chunk in chunks:
                    if chunk.embedding:
                        # Calculate cosine similarity
                        chunk_embedding = np.array(chunk.embedding)
                        query_emb = np.array(query_embedding)
                        
                        similarity = np.dot(chunk_embedding, query_emb) / (
                            np.linalg.norm(chunk_embedding) * np.linalg.norm(query_emb)
                        )
                        
                        all_chunks.append({
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "document_name": document.original_filename,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                            "page_number": chunk.page_number,
                            "metadata": chunk.metadata,
                            "score": float(similarity)
                        })
            
            # Sort by similarity score (descending)
            all_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            return all_chunks[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar chunks: {e}")
            return []
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of processed documents"""
        documents = []
        for doc in self.documents.values():
            documents.append({
                "document_id": doc.id,
                "filename": doc.original_filename,
                "upload_time": doc.upload_time.isoformat(),
                "status": doc.status.value,
                "page_count": doc.page_count,
                "chunk_count": doc.chunk_count
            })
        return documents
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks"""
        try:
            if document_id not in self.documents:
                return False
            
            document = self.documents[document_id]
            
            # Delete file from disk
            await self.pdf_service.delete_file(document.file_path)
            
            # Remove from memory
            del self.documents[document_id]
            if document_id in self.document_chunks:
                del self.document_chunks[document_id]
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

# Global RAG service instance
rag_service = RAGService()