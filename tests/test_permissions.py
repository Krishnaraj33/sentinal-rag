"""
Verification Test: TEST-PERM-04 (In-Place Vector Reclassification).
Standard Compliance: SRS Section 10.1 (TEST-PERM-04) & FR-1.4.
Pass Criteria: Qdrant payload reflects classification: executive with 0 embedding calls.
"""

from unittest.mock import MagicMock
from app.config import settings
from app.llm.embeddings import MockEmbeddingService
from app.vector.hashing import generate_point_id
from app.worker.chunker import StructuralChunker
from app.worker.diff_engine import DiffEngine


def test_inplace_permissions_change_zero_embeddings(memory_db_session, memory_qdrant):
    """
    TEST-PERM-04:
      1. Ingest internal document.
      2. Issue PERMISSIONS_CHANGE event reclassifying to 'executive'.
      3. Verify Qdrant payload reflects classification: executive.
      4. Verify zero embedding calls were generated during reclassification.
    """
    mock_emb = MockEmbeddingService(dimension=384)
    original_embed_batch = mock_emb.embed_batch
    spy_embed_batch = MagicMock(side_effect=original_embed_batch)
    mock_emb.embed_batch = spy_embed_batch

    diff_engine = DiffEngine(
        chunker=StructuralChunker(),
        embedding_service=mock_emb,
        qdrant_client=memory_qdrant,
    )

    doc_id = "DOC-PERM-TEST-01"
    content = "# Internal Strategy\n\nQuarterly strategic objectives."

    # 1. Ingest as internal
    res_create = diff_engine.process_change(
        event_type="CREATE",
        document_id=doc_id,
        title="Internal Strategy",
        classification="internal",
        allowed_roles=["engineering"],
        content=content,
        db_session=memory_db_session,
    )
    assert spy_embed_batch.call_count == 1
    spy_embed_batch.reset_mock()

    internal_id = res_create["internal_id"]
    point_id_0 = generate_point_id(internal_id, 0)

    # 2. Issue PERMISSIONS_CHANGE
    res_perm = diff_engine.process_change(
        event_type="PERMISSIONS_CHANGE",
        document_id=doc_id,
        title="",
        classification="executive",
        allowed_roles=["c-suite"],
        content="",
        db_session=memory_db_session,
    )

    # 3. Assertions
    assert res_perm["status"] == "RECLASSIFIED"
    assert res_perm["embedding_calls_generated"] == 0
    # Crucial: verify mock embedding function was NEVER called
    assert spy_embed_batch.call_count == 0

    # 4. Verify Qdrant payload directly reflects 'executive'
    pts = memory_qdrant.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[point_id_0],
        with_payload=True,
    )
    assert len(pts) == 1
    point_payload = pts[0].payload
    assert point_payload["classification"] == "executive"
    assert point_payload["allowed_roles"] == ["c-suite"]

    print("\n[TEST-PERM-04] Successfully verified in-place reclassification with 0 embedding calls.")
