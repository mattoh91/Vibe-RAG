from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None

class Document(BaseModel):
    document_id: str
    filename: str
    file_path: str
    status: DocumentStatus
    created_at: datetime
    processed_at: Optional[datetime] = None
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None
    chunks: List[DocumentChunk] = []
    metadata: Optional[Dict[str, Any]] = None