from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import logging

from app.services.rag_service import rag_service
from app.schemas.responses import DocumentListResponse, DocumentInfo
from app.schemas.requests import DocumentUploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process document
        result = await rag_service.process_document(file_content, file.filename)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            status="completed",
            message=result["message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """Get list of uploaded documents"""
    try:
        documents = rag_service.get_documents()
        
        document_infos = []
        for doc in documents:
            document_infos.append(DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                upload_time=doc["upload_time"],
                status=doc["status"],
                page_count=doc.get("page_count"),
                chunk_count=doc.get("chunk_count")
            ))
        
        return DocumentListResponse(
            documents=document_infos,
            total_count=len(document_infos)
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = await rag_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status"""
    try:
        documents = rag_service.get_documents()
        
        for doc in documents:
            if doc["document_id"] == document_id:
                return {
                    "document_id": document_id,
                    "status": doc["status"],
                    "page_count": doc.get("page_count"),
                    "chunk_count": doc.get("chunk_count")
                }
        
        raise HTTPException(status_code=404, detail="Document not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")