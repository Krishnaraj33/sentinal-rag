"""
Reciprocal Rank Fusion (RRF) for merging dense and sparse ranking lists.
Standard Compliance: SRS FR-4.1 (k=60).
"""

from typing import Any, Dict, List
from app.constants import RRF_K, RRF_TOP_CANDIDATES


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = RRF_K,
    top_n: int = RRF_TOP_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Fuses dense and sparse rankings using Reciprocal Rank Fusion:
    RRF(d) = sum(1 / (k + rank(r, d))) for each ranked list.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Dict[str, Any]] = {}

    # Accumulate dense ranks
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = str(doc["id"])
        chunk_store[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Accumulate sparse (BM25) ranks
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = str(doc["id"])
        if doc_id not in chunk_store:
            chunk_store[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort items by accumulated RRF score descending
    sorted_doc_ids = sorted(
        rrf_scores.keys(),
        key=lambda did: rrf_scores[did],
        reverse=True,
    )

    fused_results = []
    for doc_id in sorted_doc_ids[:top_n]:
        item = dict(chunk_store[doc_id])
        item["rrf_score"] = float(rrf_scores[doc_id])
        fused_results.append(item)

    return fused_results
