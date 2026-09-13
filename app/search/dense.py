"""
Dense vector retrieval with native Qdrant pre-retrieval DLS filters.
Standard Compliance: SRS FR-1.2, FR-1.3 & Module 4.
"""

import logging
from typing import Any, Dict, List
from qdrant_client import QdrantClient

from app.config import settings
from app.core.security import UserClaims
from app.llm.embeddings import get_embedding_service
from app.vector.client import QdrantManager
from app.vector.filter_builder import FilterBuilder

logger = logging.getLogger(__name__)


def dense_search(
    query: str,
    claims: UserClaims,
    limit: int = 20,
    client: QdrantClient = None,
) -> List[Dict[str, Any]]:
    """
    Executes dense cosine similarity search in Qdrant with native pre-retrieval DLS filter.
    Guarantees no post-retrieval discards (FR-1.3).
    """
    q_client = client or QdrantManager.get_sync_client()
    embedding_svc = get_embedding_service()

    # Generate query dense vector
    query_vector = embedding_svc.embed_text(query)

    # Compile native Qdrant filter directly enforcing DLS
    dls_filter = FilterBuilder.build_qdrant_filter(claims)

    try:
        if hasattr(q_client, "query_points"):
            resp = q_client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=dls_filter,
                limit=limit,
                with_payload=True,
            )
            results = resp.points
        else:
            results = q_client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                query_filter=dls_filter,
                limit=limit,
                with_payload=True,
            )
    except Exception as exc:
        logger.error("Dense search failed in Qdrant: %s", exc)
        return []

    candidates = []
    for hit in results:
        payload = hit.payload or {}
        candidates.append({
            "id": str(hit.id),
            "score": float(hit.score),
            "document_id": str(payload.get("document_id", "")),
            "external_id": str(payload.get("external_id", payload.get("document_id", ""))),
            "title": str(payload.get("title", "")),
            "chunk_index": payload.get("chunk_index", 0),
            "classification": payload.get("classification", "public"),
            "allowed_roles": payload.get("allowed_roles", ["*"]),
            "content_hash": payload.get("content_hash", ""),
            "text": payload.get("text", ""),
        })

    return candidates
