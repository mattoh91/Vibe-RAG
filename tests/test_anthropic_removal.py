"""
Tests to verify that Anthropic API has been completely removed from the codebase.
"""
import pytest
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def test_anthropic_not_in_requirements():
    """Test that anthropic is not in requirements.txt"""
    requirements_path = backend_path / "requirements.txt"
    with open(requirements_path, 'r') as f:
        requirements_content = f.read().lower()
    
    assert "anthropic" not in requirements_content, "Anthropic should not be in requirements.txt"

def test_anthropic_not_in_config():
    """Test that ANTHROPIC_API_KEY is not in config"""
    config_path = backend_path / "app" / "core" / "config.py"
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    assert "ANTHROPIC_API_KEY" not in config_content, "ANTHROPIC_API_KEY should not be in config"

def test_anthropic_provider_not_in_enum():
    """Test that ANTHROPIC is not in LLMProvider enum"""
    from app.schemas.requests import LLMProvider
    
    provider_values = [provider.value for provider in LLMProvider]
    assert "anthropic" not in provider_values, "Anthropic should not be in LLMProvider enum"

def test_anthropic_not_in_llm_service():
    """Test that AnthropicClient is not in LLM service"""
    llm_service_path = backend_path / "app" / "services" / "llm_service.py"
    with open(llm_service_path, 'r') as f:
        service_content = f.read()
    
    assert "AnthropicClient" not in service_content, "AnthropicClient should not be in LLM service"
    assert "import anthropic" not in service_content, "anthropic import should not be in LLM service"

def test_anthropic_not_in_providers_endpoint():
    """Test that anthropic is not in the providers endpoint"""
    endpoint_path = backend_path / "app" / "api" / "endpoints" / "llm_config.py"
    with open(endpoint_path, 'r') as f:
        endpoint_content = f.read()
    
    # Check that anthropic is not mentioned in the providers list
    assert '"anthropic"' not in endpoint_content, "Anthropic should not be in providers endpoint"
    assert "Anthropic Claude" not in endpoint_content, "Anthropic Claude should not be in providers endpoint"

def test_llm_service_only_supports_valid_providers():
    """Test that LLM service only supports Azure OpenAI and OpenRouter"""
    from app.services.llm_service import LLMService
    from app.models.llm_config import LLMConfig
    from app.schemas.requests import LLMProvider
    
    service = LLMService()
    
    # Test that valid providers work (we'll test with mock data)
    valid_providers = [LLMProvider.AZURE_OPENAI, LLMProvider.OPENROUTER]
    
    for provider in valid_providers:
        config = LLMConfig(
            provider=provider,
            api_key="test_key",
            model_name="test_model"
        )
        # This should not raise an exception for valid providers
        try:
            service.configure_llm(config)
        except ValueError as e:
            if "Unsupported LLM provider" in str(e):
                pytest.fail(f"Valid provider {provider} should be supported")
        except Exception:
            # Other exceptions are fine (like API connection errors)
            pass

def test_no_anthropic_imports_in_codebase():
    """Test that there are no anthropic imports anywhere in the backend codebase"""
    backend_app_path = backend_path / "app"
    
    for py_file in backend_app_path.rglob("*.py"):
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Check for various forms of anthropic imports
        assert "import anthropic" not in content, f"Found 'import anthropic' in {py_file}"
        assert "from anthropic" not in content, f"Found 'from anthropic' in {py_file}"

if __name__ == "__main__":
    pytest.main([__file__])