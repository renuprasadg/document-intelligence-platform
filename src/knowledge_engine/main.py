"""
GuardianRAG - Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge_engine.core.config import get_settings
from knowledge_engine.core.logging_config import setup_logging
from knowledge_engine.api.routes import health

# Get settings and set up logging
settings = get_settings()
setup_logging(settings.LOG_LEVEL)

app = FastAPI(
    title=settings.APP_NAME,
    description="Production RAG system for insurance policy Q&A",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # TODO: Initialize vector DB, load models, etc.
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # TODO: Close connections, cleanup resources
    pass


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to GuardianRAG API",
        "status": "running",
        "version": "0.1.0"
    }
