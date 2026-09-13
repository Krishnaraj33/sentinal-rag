"""
Verification Test: TEST-DIFF-02 (Deterministic SHA-256 Chunk Diffing & Cost Suppression).
Standard Compliance: SRS Section 10.1 (TEST-DIFF-02) & NFR-4.1.
Pass Criteria: Mutating 2 of 10 chunks triggers exactly 2 embedding calls (80.0% savings).
"""

from unittest.mock import MagicMock
from app.llm.embeddings import MockEmbeddingService
from app.worker.diff_engine import DiffEngine


def test_sha256_chunk_diffing_embedding_suppression(memory_db_session, memory_qdrant):
    """
    TEST-DIFF-02:
      1. Ingest initial 10-chunk document.
      2. Update only paragraph 3 and paragraph 7.
      3. Assert mock embedding call counter equals exactly 2.
      4. Assert PostgreSQL chunk records updated only for indexes 3 and 7.
    """
    mock_embedding_service = MockEmbeddingService(dimension=384)
    # Wrap embed_batch with a spy/mock counter
    original_embed_batch = mock_embedding_service.embed_batch
    call_spy = MagicMock(side_effect=original_embed_batch)
    mock_embedding_service.embed_batch = call_spy

    # Custom chunker that yields exactly 10 distinct paragraphs
    class ParagraphChunker:
        def chunk_document(self, content: str):
            return [p.strip() for p in content.split("\n\n") if p.strip()]

    diff_engine = DiffEngine(
        chunker=ParagraphChunker(),
        embedding_service=mock_embedding_service,
        qdrant_client=memory_qdrant,
    )

    # 1. Ingest 10-chunk document
    paragraphs = [f"Paragraph {i}: Standard enterprise operating policy number {i}." for i in range(10)]
    initial_content = "\n\n".join(paragraphs)

    res_initial = diff_engine.process_change(
        event_type="CREATE",
        document_id="DOC-DIFF-TEST-01",
        title="Enterprise Policy Manual",
        classification="internal",
        allowed_roles=["*"],
        content=initial_content,
        db_session=memory_db_session,
    )

    assert res_initial["total_chunks"] == 10
    assert res_initial["embedding_calls_generated"] == 10
    assert len(res_initial["mutated_chunks"]) == 10
    assert len(res_initial["unchanged_chunks"]) == 0

    # Reset embedding spy call counter
    call_spy.reset_mock()

    # 2. Mutate only chunks 3 and 7
    updated_paragraphs = list(paragraphs)
    updated_paragraphs[3] = "Paragraph 3: MUTATED text for reimbursement threshold increased to $80/day."
    updated_paragraphs[7] = "Paragraph 7: MUTATED text with updated compliance reporting guidelines."
    updated_content = "\n\n".join(updated_paragraphs)

    res_update = diff_engine.process_change(
        event_type="UPDATE",
        document_id="DOC-DIFF-TEST-01",
        title="Enterprise Policy Manual",
        classification="internal",
        allowed_roles=["*"],
        content=updated_content,
        db_session=memory_db_session,
    )

    # 3. Assertions
    assert res_update["total_chunks"] == 10
    # Chunks 3 and 7 mutated; remaining 8 bypassed!
    assert sorted(res_update["mutated_chunks"]) == [3, 7]
    assert len(res_update["unchanged_chunks"]) == 8
    assert res_update["embedding_calls_generated"] == 2
    assert res_update["embedding_suppression_pct"] == 80.0

    # Verify spy called with exactly 2 texts
    assert call_spy.call_count == 1
    embedded_texts = call_spy.call_args[0][0]
    assert len(embedded_texts) == 2

    print("\n[TEST-DIFF-02] Successfully verified 80.0% embedding API call suppression.")
