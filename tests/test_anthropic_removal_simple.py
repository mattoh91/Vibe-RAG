"""
Simple tests to verify that Anthropic API has been completely removed from the codebase.
These tests don't require importing the modules.
"""
import pytest
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

def test_anthropic_not_in_requirements():
    """Test that anthropic is not in requirements.txt"""
    requirements_path = project_root / "backend" / "requirements.txt"
    with open(requirements_path, 'r') as f:
        requirements_content = f.read().lower()
    
    assert "anthropic" not in requirements_content, "Anthropic should not be in requirements.txt"

def test_anthropic_not_in_config():
    """Test that ANTHROPIC_API_KEY is not in config"""
    config_path = project_root / "backend" / "app" / "core" / "config.py"
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    assert "ANTHROPIC_API_KEY" not in config_content, "ANTHROPIC_API_KEY should not be in config"

def test_anthropic_not_in_llm_service():
    """Test that AnthropicClient is not in LLM service"""
    llm_service_path = project_root / "backend" / "app" / "services" / "llm_service.py"
    with open(llm_service_path, 'r') as f:
        service_content = f.read()
    
    assert "AnthropicClient" not in service_content, "AnthropicClient should not be in LLM service"
    assert "import anthropic" not in service_content, "anthropic import should not be in LLM service"

def test_anthropic_not_in_providers_endpoint():
    """Test that anthropic is not in the providers endpoint"""
    endpoint_path = project_root / "backend" / "app" / "api" / "endpoints" / "llm_config.py"
    with open(endpoint_path, 'r') as f:
        endpoint_content = f.read()
    
    # Check that anthropic is not mentioned in the providers list
    assert '"anthropic"' not in endpoint_content, "Anthropic should not be in providers endpoint"
    assert "Anthropic Claude" not in endpoint_content, "Anthropic Claude should not be in providers endpoint"

def test_anthropic_not_in_requests_schema():
    """Test that ANTHROPIC is not in LLMProvider enum"""
    requests_path = project_root / "backend" / "app" / "schemas" / "requests.py"
    with open(requests_path, 'r') as f:
        requests_content = f.read()
    
    # Check that ANTHROPIC is not in the enum
    assert 'ANTHROPIC = "anthropic"' not in requests_content, "ANTHROPIC should not be in LLMProvider enum"

def test_no_anthropic_imports_in_codebase():
    """Test that there are no anthropic imports anywhere in the backend codebase"""
    backend_app_path = project_root / "backend" / "app"
    
    for py_file in backend_app_path.rglob("*.py"):
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Check for various forms of anthropic imports
        assert "import anthropic" not in content, f"Found 'import anthropic' in {py_file}"
        assert "from anthropic" not in content, f"Found 'from anthropic' in {py_file}"

def test_no_claude_references_in_codebase():
    """Test that there are no Claude model references in the codebase"""
    backend_app_path = project_root / "backend" / "app"
    
    for py_file in backend_app_path.rglob("*.py"):
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Check for Claude model references (but allow in comments)
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            if 'claude-' in line.lower() and not line.strip().startswith('#'):
                pytest.fail(f"Found Claude model reference in {py_file}:{line_num}: {line.strip()}")

if __name__ == "__main__":
    pytest.main([__file__])