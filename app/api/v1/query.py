"""
Identity-filtered hybrid retrieval and grounded synthesis endpoint.
Standard Compliance: SRS Section 8.1 (FR-1.1, FR-1.2, FR-4.3, FR-5.1, FR-5.2).
"""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from app.api.deps import get_current_user_claims
from app.core.security import UserClaims
from app.db.models import RetrievalAuditLog
from app.db.session import get_async_db
from app.llm.synthesizer import Citation, Synthesizer
from app.search.hybrid_engine import HybridEngine, get_hybrid_engine
from app.vector.filter_builder import FilterBuilder

router = APIRouter(prefix="", tags=["Retrieval & Query Engine"])


class QueryRequest(BaseModel):
    query: str = Field(..., example="What are our staging cluster credentials?")
    top_k: int = Field(5, ge=1, le=50, description="Top N contextual chunks to select")


class LatencyBreakdown(BaseModel):
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    authorized_chunks_scanned: int


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    compiled_filter: Dict[str, Any]
    metrics: LatencyBreakdown
    context_clearance_ceiling: Optional[str] = "public"


class ErrorResponse(BaseModel):
    error: str
    message: str


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Insufficient clearance or zero authorized documents."},
        401: {"model": ErrorResponse, "description": "Unauthorized bearer token."},
    },
)
async def execute_query(
    req: QueryRequest,
    claims: UserClaims = Depends(get_current_user_claims),
    db: AsyncSession = Depends(get_async_db),
    engine: HybridEngine = Depends(get_hybrid_engine),
):
    """
    Executes identity-filtered retrieval and generation:
      1. Translates user claims into native Qdrant bitmask pre-filter.
      2. Traverses vector space without post-filtering.
      3. Re-ranks top candidate chunks using local cross-encoder.
      4. Synthesizes grounded answer and returns tamper-evident citations.
      5. Commits immutable entry to retrieval audit logs.
    """
    start_total = time.perf_counter()

    # Step 1 & 2: Pre-filtered Hybrid Retrieval & Re-ranking
    search_result = engine.search(
        query=req.query,
        claims=claims,
        top_k=req.top_k,
    )

    compiled_filter_dict = FilterBuilder.to_dict(claims)

    # Step 3: Zero-Token Denial Check (FR-4.3 & Section 8.1)
    if not search_result.chunks:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "INSUFFICIENT_SECURITY_CLEARANCE",
                "message": "Query terminated: No authorized documents available for user clearance.",
            },
        )

    # Fast-fail Relevance Threshold (Skip LLM if top chunk is completely irrelevant)
    top_score = search_result.chunks[0].get("rerank_score", 0.0)
    # RRF fallback scores are typically < 0.1 for irrelevant, CrossEncoder is < -2.0
    if top_score < -2.0 or (0.0 < top_score < 0.05):
        total_latency_ms = (time.perf_counter() - start_total) * 1000.0
        metrics = LatencyBreakdown(
            retrieval_latency_ms=search_result.metrics.retrieval_latency_ms,
            rerank_latency_ms=search_result.metrics.rerank_latency_ms,
            generation_latency_ms=0.0,
            total_latency_ms=round(total_latency_ms, 2),
            authorized_chunks_scanned=search_result.metrics.authorized_chunks_scanned,
        )
        return QueryResponse(
            answer="I do not possess authorized context to answer this query.",
            citations=[],
            compiled_filter=compiled_filter_dict,
            metrics=metrics,
            context_clearance_ceiling=search_result.chunks[0].get("classification", "public"),
        )

    # Step 4: Grounded Answer Synthesis
    start_gen = time.perf_counter()
    synthesizer = Synthesizer()
    synthesis_result = await synthesizer.synthesize(
        query=req.query,
        context_chunks=search_result.chunks,
    )
    gen_latency_ms = (time.perf_counter() - start_gen) * 1000.0

    total_latency_ms = (time.perf_counter() - start_total) * 1000.0

    # Step 5: Asynchronous Audit Logging
    retrieved_chunk_ids = [str(c["id"]) for c in search_result.chunks]
    try:
        audit_log = RetrievalAuditLog(
            id=uuid.uuid4(),
            user_id=claims.sub,
            user_roles=claims.roles,
            user_clearance=claims.clearance,
            query_text=req.query,
            retrieved_chunk_ids=retrieved_chunk_ids,
            execution_time_ms=round(total_latency_ms, 2),
        )
        db.add(audit_log)
    except Exception as exc:
        # Audit logging failure should not break user retrieval path
        logger.warning("Audit logging failed: %s", exc)

    metrics = LatencyBreakdown(
        retrieval_latency_ms=search_result.metrics.retrieval_latency_ms,
        rerank_latency_ms=search_result.metrics.rerank_latency_ms,
        generation_latency_ms=round(gen_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        authorized_chunks_scanned=search_result.metrics.authorized_chunks_scanned,
    )

    return QueryResponse(
        answer=synthesis_result.answer,
        citations=synthesis_result.citations,
        compiled_filter=compiled_filter_dict,
        metrics=metrics,
        context_clearance_ceiling=synthesis_result.context_clearance_ceiling,
    )
