"""
Cross-Encoder re-ranking for contextual relevance refinement.
Standard Compliance: SRS FR-4.2 & NFR-1.2.
"""

import logging
from typing import Any, Dict, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Local Cross-Encoder re-ranking engine using ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._compressor = None
        self.enabled = settings.RERANKER_ENABLED

    def _get_compressor(self):
        if self._compressor is None and self.enabled:
            try:
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder
                logger.info("Loading HuggingFaceCrossEncoder: %s (cache: %s)", self.model_name, settings.HF_HOME)
                self._compressor = HuggingFaceCrossEncoder(model_name=self.model_name, model_kwargs={"cache_dir": settings.HF_HOME})
            except Exception as exc:
                logger.warning("Could not load HuggingFaceCrossEncoder (%s). Fallback scoring active.", exc)
                self._compressor = None
        return self._compressor

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks candidates using Cross-Encoder model and returns top N chunks.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_n]

        model = self._get_compressor()

        if model is not None:
            try:
                text_pairs = [(query, c.get("text", "")) for c in candidates]
                scores = model.score(text_pairs)
                reranked = []
                for candidate, score in zip(candidates, scores):
                    item = dict(candidate)
                    item["rerank_score"] = float(score)
                    reranked.append(item)
                reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
                return reranked[:top_n]
            except Exception as exc:
                logger.warning("Cross-encoder prediction error (%s). Using fallback scores.", exc)
        
        # Fallback heuristic: combination of RRF score and lexical match
        scores = []
        query_words = set(query.lower().split())
        for c in candidates:
            text_words = set(c.get("text", "").lower().split())
            overlap = len(query_words.intersection(text_words)) / max(len(query_words), 1)
            base = c.get("rrf_score", 0.0)
            scores.append(base + overlap)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]


_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Returns singleton CrossEncoderReranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
