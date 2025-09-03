from fastapi import APIRouter, HTTPException
import logging

from app.services.llm_service import llm_service
from app.schemas.requests import LLMConfigRequest, LLMProvider
from app.schemas.responses import LLMTestResponse
from app.models.llm_config import LLMConfig

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/configure")
async def configure_llm(config_request: LLMConfigRequest):
    """Configure LLM provider"""
    try:
        # Create LLM config
        config = LLMConfig(
            provider=config_request.provider,
            api_key=config_request.api_key,
            endpoint=config_request.endpoint,
            model_name=config_request.model_name,
            api_version=config_request.api_version,
            deployment_name=config_request.deployment_name
        )
        
        # Configure the LLM service
        llm_service.configure_llm(config)
        
        return {
            "message": f"Successfully configured {config_request.provider.value} provider",
            "provider": config_request.provider.value,
            "model": config_request.model_name
        }
        
    except Exception as e:
        logger.error(f"Failed to configure LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure LLM: {str(e)}")

@router.get("/providers")
async def get_providers():
    """Get list of available LLM providers"""
    return {
        "providers": [
            {
                "id": "azure_openai",
                "name": "Azure OpenAI",
                "requires_endpoint": True,
                "default_models": ["gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-4-turbo"]
            },
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "requires_endpoint": False,
                "default_models": ["anthropic/claude-sonnet-4", "openai/gpt-5", "x-ai/grok-4", "deepseek/deepseek-chat-v3.1", "google/gemini-2.5-flash"]
            },
            {
                "id": "anthropic",
                "name": "Anthropic Claude",
                "requires_endpoint": False,
                "default_models": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
            }
        ]
    }

@router.post("/test", response_model=LLMTestResponse)
async def test_llm_connection():
    """Test connection to configured LLM provider"""
    try:
        if not llm_service.is_configured():
            raise HTTPException(status_code=400, detail="No LLM provider configured")
        
        result = await llm_service.test_connection()
        
        return LLMTestResponse(
            success=result["success"],
            provider=result["provider"],
            model=result.get("model"),
            message=result["message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test LLM connection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test LLM connection: {str(e)}")

@router.get("/status")
async def get_llm_status():
    """Get current LLM configuration status"""
    try:
        if not llm_service.is_configured():
            return {
                "configured": False,
                "provider": None,
                "message": "No LLM provider configured"
            }
        
        return {
            "configured": True,
            "provider": llm_service.get_current_provider(),
            "message": "LLM provider configured and ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to get LLM status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM status: {str(e)}")