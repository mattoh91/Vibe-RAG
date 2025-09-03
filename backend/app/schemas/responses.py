from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    upload_time: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int

class LLMTestResponse(BaseModel):
    success: bool
    provider: str
    model: Optional[str] = None
    message: str

class SourceInfo(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_text: str
    relevance_score: float

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[SourceInfo]
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime