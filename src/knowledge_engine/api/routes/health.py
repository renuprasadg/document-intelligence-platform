"""
Health check endpoint
"""
from fastapi import APIRouter
from datetime import datetime
import time

from knowledge_engine.core.config import get_settings

router = APIRouter(tags=["health"])

# Track when server started
START_TIME = time.time()


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns server status and uptime
    """
    settings = get_settings()  # Factory pattern - no import-time crash
    uptime_seconds = time.time() - START_TIME
    
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }
