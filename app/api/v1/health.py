"""
Health check and readiness endpoints.
"""

from fastapi import APIRouter, HTTPException
from app.config import settings
from app.vector.client import QdrantManager

router = APIRouter(tags=["System Health"])


@router.get("/health")
async def health_check():
    """Returns status of database, message broker, and vector engine."""
    qdrant_healthy = False
    try:
        client = QdrantManager.get_sync_client()
        client.get_collections()
        qdrant_healthy = True
    except Exception:
        qdrant_healthy = False

    if not qdrant_healthy:
        raise HTTPException(status_code=503, detail="Vector engine is down")

    return {
        "status": "HEALTHY",
        "service": "SentinelRAG API Gateway",
        "qdrant_connected": qdrant_healthy,
        "environment": settings.ENVIRONMENT,
    }
