from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class LLMProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENROUTER = "openrouter"

class LLMConfigRequest(BaseModel):
    provider: LLMProvider
    api_key: str = Field(..., min_length=1)
    endpoint: Optional[str] = None  # For Azure OpenAI
    model_name: Optional[str] = None
    api_version: Optional[str] = None  # For Azure OpenAI
    deployment_name: Optional[str] = None  # For Azure OpenAI

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = Field(default=True)
    document_ids: Optional[List[str]] = None  # Filter by specific documents

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float