"""
Unit tests for health endpoint
"""
import pytest
from fastapi.testclient import TestClient
from knowledge_engine.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health endpoint returns correct response"""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "GuardianRAG"
    assert "version" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_health_endpoint_uptime_increases():
    """Test uptime increases between calls"""
    import time
    
    response1 = client.get("/health")
    uptime1 = response1.json()["uptime_seconds"]
    
    time.sleep(0.1)
    
    response2 = client.get("/health")
    uptime2 = response2.json()["uptime_seconds"]
    
    assert uptime2 > uptime1
