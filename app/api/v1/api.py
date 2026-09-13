"""
Aggregate router for API v1.
"""

from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.documents import router as documents_router
from app.api.v1.events import router as events_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(events_router)
api_v1_router.include_router(documents_router)
