from fastapi import APIRouter, HTTPException
import logging

from app.services.rag_service import rag_service
from app.schemas.requests import QueryRequest
from app.schemas.responses import QueryResponse, SourceInfo

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """Query documents using RAG"""
    try:
        # Process the query
        result = await rag_service.query_documents(
            query=query_request.query,
            max_results=query_request.max_results,
            use_reranking=query_request.use_reranking,
            document_ids=query_request.document_ids
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        # Convert sources to SourceInfo objects
        sources = []
        for source in result["sources"]:
            sources.append(SourceInfo(
                document_id=source["document_id"],
                document_name=source["document_name"],
                page_number=source.get("page_number"),
                chunk_text=source["text_preview"],
                relevance_score=source.get("similarity_score", 0.0)
            ))
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            sources=sources,
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@router.get("/history")
async def get_query_history():
    """Get query history (placeholder for future implementation)"""
    return {
        "message": "Query history feature not yet implemented",
        "queries": []
    }