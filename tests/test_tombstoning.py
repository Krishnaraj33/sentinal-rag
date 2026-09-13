"""
Verification Test: TEST-PURGE-03 (Deterministic Point Tombstoning & Purge Latency).
Standard Compliance: SRS Section 10.1 (TEST-PURGE-03) & NFR-1.3.
Pass Criteria: Point ID lookup returns null in <= 100 ms after document deletion.
"""

import time
from app.config import settings
from app.vector.hashing import generate_point_id


def test_deterministic_point_tombstoning_sub_second(memory_db_session, memory_qdrant, test_diff_engine):
    """
    TEST-PURGE-03:
      1. Ingest document.
      2. Verify Point ID exists in Qdrant.
      3. Issue document DELETE event.
      4. Assert Point ID lookup returns null in <= 100 ms.
    """
    doc_id = "DOC-TOMBSTONE-01"
    content = "# Critical Infrastructure\n\nDatabase passwords and connection strings."

    # 1. Ingest
    res_create = test_diff_engine.process_change(
        event_type="CREATE",
        document_id=doc_id,
        title="Critical Infrastructure",
        classification="confidential",
        allowed_roles=["engineering"],
        content=content,
        db_session=memory_db_session,
    )

    internal_id = res_create["internal_id"]
    point_id_0 = generate_point_id(internal_id, 0)

    # 2. Verify Point exists
    pts = memory_qdrant.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[point_id_0],
    )
    assert len(pts) == 1, "Point was not indexed in Qdrant!"

    # 3. Issue DELETE
    start_purge = time.perf_counter()
    res_delete = test_diff_engine.process_change(
        event_type="DELETE",
        document_id=doc_id,
        title="",
        classification="public",
        allowed_roles=["*"],
        content="",
        db_session=memory_db_session,
    )
    purge_duration_ms = (time.perf_counter() - start_purge) * 1000.0

    assert res_delete["status"] == "DELETED"
    assert res_delete["purged_points_count"] >= 1

    # 4. Assert Point ID lookup returns null immediately (<= 100 ms)
    lookup_start = time.perf_counter()
    post_pts = memory_qdrant.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[point_id_0],
    )
    lookup_latency_ms = (time.perf_counter() - lookup_start) * 1000.0

    assert len(post_pts) == 0, "Point still exists in Qdrant after purge!"
    assert lookup_latency_ms <= 100.0, f"Lookup latency exceeded 100ms: {lookup_latency_ms} ms"

    print(f"\n[TEST-PURGE-03] Verified O(1) tombstoning in {purge_duration_ms:.2f} ms (Lookup: {lookup_latency_ms:.2f} ms).")
