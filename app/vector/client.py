"""
Qdrant vector engine client wrapper, collection setup, and lifecycle operations.
Supports both remote Qdrant instances and local in-memory instances for testing.
"""

import logging
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, VectorParams

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant client instances, collection initialization, and payload indexing."""

    _sync_client: Optional[QdrantClient] = None
    _async_client: Optional[AsyncQdrantClient] = None
    _is_in_memory: bool = False

    @classmethod
    def get_sync_client(cls, use_memory: bool = False) -> QdrantClient:
        """Returns a singleton or configured synchronous Qdrant client."""
        if cls._sync_client is None or (use_memory and not cls._is_in_memory):
            if use_memory or settings.QDRANT_HOST.lower() == ":memory:":
                logger.info("Initializing in-memory synchronous Qdrant client.")
                cls._sync_client = QdrantClient(":memory:")
                cls._is_in_memory = True
            else:
                cls._sync_client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=10.0,
                )
        return cls._sync_client

    @classmethod
    def get_async_client(cls, use_memory: bool = False) -> AsyncQdrantClient:
        """Returns an asynchronous Qdrant client."""
        if cls._async_client is None or (use_memory and not cls._is_in_memory):
            if use_memory or settings.QDRANT_HOST.lower() == ":memory:":
                logger.info("Initializing in-memory asynchronous Qdrant client.")
                cls._async_client = AsyncQdrantClient(":memory:")
                cls._is_in_memory = True
            else:
                cls._async_client = AsyncQdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=10.0,
                )
        return cls._async_client

    @classmethod
    def set_sync_client(cls, client: QdrantClient) -> None:
        """Overrides sync client for testing or custom lifecycle management."""
        cls._sync_client = client

    @classmethod
    def set_async_client(cls, client: AsyncQdrantClient) -> None:
        """Overrides async client for testing."""
        cls._async_client = client

    @classmethod
    def init_collection(
        cls,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> None:
        """
        Ensures the collection exists with Cosine distance and payload indexes.
        Creates payload indexes for fast HNSW traversal:
          - 'classification' (Keyword)
          - 'allowed_roles' (Keyword)
          - 'document_id' (Keyword)
        """
        c = client or cls.get_sync_client()
        coll_name = collection_name or settings.QDRANT_COLLECTION
        dim = vector_size or settings.EMBEDDING_DIM

        collections = c.get_collections().collections
        exists = any(col.name == coll_name for col in collections)

        if not exists:
            logger.info("Creating Qdrant collection '%s' (dim=%d, distance=Cosine)...", coll_name, dim)
            c.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        # Ensure payload indices exist for pre-filtering
        indexed_fields = ["classification", "allowed_roles", "document_id"]
        for field in indexed_fields:
            try:
                c.create_payload_index(
                    collection_name=coll_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                # Index may already exist
                logger.debug("Payload index for '%s' already exists or skipped: %s", field, exc)



def update_document_payload_in_place(
    client: QdrantClient,
    doc_id: str,
    payload_updates: Dict[str, Any],
    collection_name: Optional[str] = None,
) -> None:
    """
    Updates vector metadata in Qdrant in-place without recalculating dense embeddings.
    Standard Compliance: FR-1.4 & TEST-PERM-04.
    """
    coll_name = collection_name or settings.QDRANT_COLLECTION
    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=str(doc_id)))]
    )
    client.set_payload(
        collection_name=coll_name,
        payload=payload_updates,
        points=doc_filter,
    )
