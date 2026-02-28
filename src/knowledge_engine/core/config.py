"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # App Configuration
    APP_NAME: str = "GuardianRAG"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., description="OpenAI API Key")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4"
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 500
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = "./data/vectorstore"
    
    # Retrieval Configuration
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
