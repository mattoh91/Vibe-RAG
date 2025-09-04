from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import asyncio
from abc import ABC, abstractmethod

# LLM Client imports
import openai
import httpx

from app.models.llm_config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the LLM provider"""
        pass

class AzureOpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version or "2024-10-01-preview"
        )
        # For Azure OpenAI, use deployment_name as the model
        self.model = config.deployment_name or config.model_name or "gpt-35-turbo"
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI generation failed: {e}")
            raise
    
    async def test_connection(self) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return {
                "success": True,
                "model": self.model,
                "message": "Connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "model": self.model,
                "message": f"Connection failed: {str(e)}"
            }

class OpenRouterClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = config.model_name or "openai/gpt-3.5-turbo"
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    async def test_connection(self) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return {
                "success": True,
                "model": self.model,
                "message": "Connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "model": self.model,
                "message": f"Connection failed: {str(e)}"
            }



class LLMService:
    def __init__(self):
        self.current_client: Optional[BaseLLMClient] = None
        self.current_config: Optional[LLMConfig] = None
    
    def configure_llm(self, config: LLMConfig) -> None:
        """Configure the LLM service with a specific provider"""
        try:
            if config.provider == LLMProvider.AZURE_OPENAI:
                self.current_client = AzureOpenAIClient(config)
            elif config.provider == LLMProvider.OPENROUTER:
                self.current_client = OpenRouterClient(config)
            else:
                raise ValueError(f"Unsupported LLM provider: {config.provider}")
            
            self.current_config = config
            logger.info(f"Configured LLM service with provider: {config.provider}")
            
        except Exception as e:
            logger.error(f"Failed to configure LLM service: {e}")
            raise
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using the configured LLM"""
        if not self.current_client:
            raise ValueError("LLM service not configured. Please configure a provider first.")
        
        return await self.current_client.generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the current LLM provider"""
        if not self.current_client:
            return {
                "success": False,
                "message": "No LLM provider configured"
            }
        
        result = await self.current_client.test_connection()
        result["provider"] = self.current_config.provider.value
        return result
    
    def is_configured(self) -> bool:
        """Check if the LLM service is configured"""
        return self.current_client is not None
    
    def get_current_provider(self) -> Optional[str]:
        """Get the current LLM provider"""
        return self.current_config.provider.value if self.current_config else None
    
    async def generate_rag_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> str:
        """Generate a RAG response using retrieved context"""
        if not self.current_client:
            raise ValueError("LLM service not configured")
        
        # Build context from retrieved chunks
        context_text = "\n\n".join([
            f"Source: {chunk.get('document_name', 'Unknown')} (Page {chunk.get('page_number', 'N/A')})\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Create RAG prompt
        system_message = """You are a helpful assistant that answers questions based on the provided context. 
        Use only the information from the context to answer questions. If the context doesn't contain 
        enough information to answer the question, say so clearly. Always cite the sources when possible."""
        
        user_message = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return await self.generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3  # Lower temperature for more factual responses
        )

# Global LLM service instance
llm_service = LLMService()