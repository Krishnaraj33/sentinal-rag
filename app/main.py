"""
SentinelRAG: FastAPI Main Application Gateway.
Standard Compliance: SRS-SENTINEL-2026-V1.2.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_v1_router
from app.config import settings
from app.db.base import Base
from app.db.session import async_engine
from app.llm.embeddings import get_embedding_service
from app.vector.client import QdrantManager

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle hooks."""
    logger.info("Starting SentinelRAG Gateway (Environment: %s)...", settings.ENVIRONMENT)

    # Initialize relational database tables
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL relational schema initialized.")
    except Exception as exc:
        logger.warning("Could not automatically create tables on startup (%s). Ensure init.sql ran.", exc)

    # Initialize Qdrant collection and payload indexes
    try:
        embedding_svc = get_embedding_service()
        QdrantManager.init_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vector_size=embedding_svc.dimension,
        )
        logger.info("Qdrant collection and HNSW payload indexes ready.")
        # Pre-warm embedding model to prevent query cold-start delays
        embedding_svc.embed_text("SentinelRAG system initialization warmup")
        logger.info("Dense embedding model pre-warmed successfully.")
    except Exception as exc:
        logger.warning("Qdrant/Embedding initialization warning: %s", exc)

    yield

    logger.info("Shutting down SentinelRAG Gateway...")
    await async_engine.dispose()


app = FastAPI(
    title="SentinelRAG: Enterprise Knowledge Engine",
    description="ACL-Aware RAG with Pre-Retrieval Document-Level Security & Real-Time Vector Invalidation",
    version="1.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router)


@app.get("/")
def root_index():
    return {
        "engine": "SentinelRAG",
        "specification": "SRS-SENTINEL-2026-V1.2",
        "status": "OPERATIONAL",
        "documentation": "/docs",
    }
