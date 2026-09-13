"""
Hybrid search orchestration combining pre-filtered dense search, BM25, RRF, and Cross-Encoder.
Standard Compliance: SRS Module 4 & NFR-1.1, NFR-1.2, NFR-2.3.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from app.core.security import UserClaims
from app.search.bm25 import bm25_rank
from app.search.dense import dense_search
from app.search.reranker import get_reranker
from app.search.rrf import reciprocal_rank_fusion
from app.vector.client import QdrantManager


class RetrievalMetrics(BaseModel):
    """Execution timing and scan metrics."""
    retrieval_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float
    authorized_chunks_scanned: int


class HybridSearchResult(BaseModel):
    """Result of hybrid search with metrics."""
    chunks: List[Dict[str, Any]]
    metrics: RetrievalMetrics


class HybridEngine:
    """Coordinates hybrid retrieval pipeline with strict pre-retrieval security boundaries."""

    def __init__(self):
        self.reranker = get_reranker()

    def search(
        self,
        query: str,
        claims: UserClaims,
        top_k: int = 5,
        candidate_pool: int = 20,
    ) -> HybridSearchResult:
        """
        Executes hybrid retrieval:
          1. Dense pre-filtered search (Qdrant HNSW bitmask traversal)
          2. Early exit with empty result if 0 chunks authorized (Zero-Token Denial)
          3. BM25 sparse scoring across authorized candidates
          4. Reciprocal Rank Fusion (RRF, k=60)
          5. Local Cross-Encoder Re-ranking -> top_k
        """
        start_total = time.perf_counter()

        # Step 1: Pre-filtered Dense Vector Search
        start_retrieval = time.perf_counter()
        dense_candidates = dense_search(
            query=query,
            claims=claims,
            limit=candidate_pool,
        )
        end_retrieval = time.perf_counter()
        retrieval_latency_ms = (end_retrieval - start_retrieval) * 1000.0

        # Step 2: Zero-Token Denial Check (FR-4.3 & NFR-2.3)
        if not dense_candidates:
            total_latency_ms = (time.perf_counter() - start_total) * 1000.0
            return HybridSearchResult(
                chunks=[],
                metrics=RetrievalMetrics(
                    retrieval_latency_ms=round(retrieval_latency_ms, 2),
                    rerank_latency_ms=0.0,
                    total_latency_ms=round(total_latency_ms, 2),
                    authorized_chunks_scanned=0,
                ),
            )

        # Step 3: BM25 Sparse Scoring over authorized pool
        sparse_candidates = bm25_rank(query, dense_candidates)

        # Step 4: Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            top_n=candidate_pool,
        )

        # Step 5: Cross-Encoder Re-ranking
        start_rerank = time.perf_counter()
        final_chunks = self.reranker.rerank(
            query=query,
            candidates=fused_candidates,
            top_n=top_k,
        )
        end_rerank = time.perf_counter()
        rerank_latency_ms = (end_rerank - start_rerank) * 1000.0

        total_latency_ms = (time.perf_counter() - start_total) * 1000.0

        return HybridSearchResult(
            chunks=final_chunks,
            metrics=RetrievalMetrics(
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                rerank_latency_ms=round(rerank_latency_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                authorized_chunks_scanned=len(dense_candidates),
            ),
        )


_hybrid_engine_instance: Optional[HybridEngine] = None


def get_hybrid_engine() -> HybridEngine:
    """Returns singleton HybridEngine."""
    global _hybrid_engine_instance
    if _hybrid_engine_instance is None:
        _hybrid_engine_instance = HybridEngine()
    return _hybrid_engine_instance
