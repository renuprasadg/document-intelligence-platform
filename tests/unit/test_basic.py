"""
Basic tests to verify setup is working
"""
import pytest
from fastapi.testclient import TestClient
from knowledge_engine.main import app


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct response"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to GuardianRAG API"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "GuardianRAG"


def test_config_loads():
    """Test configuration loads from environment"""
    from knowledge_engine.core.config import settings
    
    assert settings.APP_NAME == "GuardianRAG"
    assert settings.OPENAI_API_KEY is not None
    assert settings.CHUNK_SIZE == 512
