from pydantic import BaseModel
from typing import Optional
from app.schemas.requests import LLMProvider

class LLMConfig(BaseModel):
    provider: LLMProvider
    api_key: str
    endpoint: Optional[str] = None
    model_name: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None