"""
Unit tests for configuration
"""
import pytest
from knowledge_engine.core.config import get_settings, reset_settings_cache


def test_get_settings():
    """Test settings factory returns Settings instance"""
    settings = get_settings()
    
    assert settings.APP_NAME == "GuardianRAG"
    assert hasattr(settings, 'OPENAI_API_KEY')


def test_get_settings_cached():
    """Test settings are cached"""
    settings1 = get_settings()
    settings2 = get_settings()
    
    # Should return same instance
    assert settings1 is settings2


def test_reset_settings_cache():
    """Test settings cache can be reset"""
    settings1 = get_settings()
    
    # Reset cache
    reset_settings_cache()
    
    settings2 = get_settings()
    
    # Should be different instances after reset
    assert settings1 is not settings2
    
    # But with same values
    assert settings1.APP_NAME == settings2.APP_NAME
