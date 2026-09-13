"""
15-Case Integration Benchmark Test Suite.
Standard Compliance: SRS Section 10.2 (E2E-01 through E2E-15).
"""

import pytest
from app.config import settings
from app.core.security import UserClaims
from app.llm.synthesizer import Synthesizer
from app.search.hybrid_engine import HybridEngine
from app.worker.diff_engine import DiffEngine


@pytest.fixture
def seeded_benchmark_env(memory_db_session, memory_qdrant, test_diff_engine):
    """Sets up the canonical benchmark document corpus."""
    from scripts.seed_data import BENCHMARK_DOCUMENTS
    for doc in BENCHMARK_DOCUMENTS:
        test_diff_engine.process_change(
            event_type="CREATE",
            document_id=str(doc["external_id"]),
            title=str(doc["title"]),
            classification=str(doc["classification"]),
            allowed_roles=list(doc["allowed_roles"]),
            content=str(doc["content"]),
            db_session=memory_db_session,
        )
    return {
        "db": memory_db_session,
        "qdrant": memory_qdrant,
        "diff_engine": test_diff_engine,
    }


@pytest.mark.asyncio
async def test_15_case_integration_benchmark(seeded_benchmark_env):
    """Executes all 15 integration benchmark test cases from SRS Section 10.2."""
    db = seeded_benchmark_env["db"]
    qdrant = seeded_benchmark_env["qdrant"]
    diff_engine = seeded_benchmark_env["diff_engine"]
    engine = HybridEngine()
    synthesizer = Synthesizer()

    # Define test claims
    contractor = UserClaims(sub="u-contractor", roles=["contractor"], clearance="public")
    engineer = UserClaims(sub="u-engineer", roles=["engineering"], clearance="internal")
    executive = UserClaims(sub="u-ceo", roles=["c-suite", "engineering", "finance", "hr"], clearance="executive")
    employee = UserClaims(sub="u-emp", roles=["general"], clearance="internal")
    hr_user = UserClaims(sub="u-hr", roles=["hr"], clearance="confidential")
    sales_user = UserClaims(sub="u-sales", roles=["sales"], clearance="internal")
    legal_user = UserClaims(sub="u-legal", roles=["legal"], clearance="confidential")
    guest_user = UserClaims(sub="u-guest", roles=["guest"], clearance="public")

    # --- CASE E2E-01: Contractor queries staging credentials ---
    res_01 = engine.search("What are the staging database credentials?", contractor)
    assert len(res_01.chunks) == 0, "E2E-01 FAILED: Contractor must be denied access to internal infra!"

    # --- CASE E2E-02: Engineer queries container deployment / staging ---
    res_02 = engine.search("What are our container deployment standards?", engineer)
    assert len(res_02.chunks) > 0, "E2E-02 FAILED: Engineer must receive staging guidelines."
    assert any(c["external_id"] == "DOC-CORP-ENG-04" for c in res_02.chunks)

    # --- CASE E2E-03: Engineer queries executive bonus (No Leakage) ---
    res_03 = engine.search("What is the Q3 executive bonus structure?", engineer)
    # Must NOT retrieve executive document
    for c in res_03.chunks:
        assert c["classification"] != "executive", "E2E-03 FAILED: Context leakage of executive doc to engineer!"
        assert c["external_id"] != "DOC-EXEC-COMP-01", "E2E-03 FAILED: Executive compensation doc leaked!"

    # --- CASE E2E-04: Executive queries executive bonus ---
    res_04 = engine.search("What is the Q3 executive bonus structure?", executive)
    assert len(res_04.chunks) > 0, "E2E-04 FAILED: Executive must access executive compensation doc."
    assert any(c["external_id"] == "DOC-EXEC-COMP-01" for c in res_04.chunks)
    syn_04 = await synthesizer.synthesize("What is the Q3 executive bonus structure?", res_04.chunks)
    assert syn_04.context_clearance_ceiling == "executive"

    # --- CASE E2E-05: Pre-edit Travel Policy query ($50/day) ---
    res_05 = engine.search("What is our travel reimbursement limit?", employee)
    assert len(res_05.chunks) > 0
    assert any("50" in c["text"] for c in res_05.chunks)

    # --- CASE E2E-06: Action: Ingest Travel Policy v2 ($75/day) ---
    v2_content = (
        "# Corporate Travel Reimbursement Policy\n\n"
        "All employees traveling for business can claim reimbursement.\n"
        "The daily travel meal reimbursement limit is $75/day. "
        "All expense receipts must be retained and submitted via the Concur portal within 14 business days of return."
    )
    res_06 = diff_engine.process_change(
        event_type="UPDATE",
        document_id="DOC-HR-TRAVEL-01",
        title="Corporate Travel Reimbursement Policy",
        classification="internal",
        allowed_roles=["*"],
        content=v2_content,
        db_session=db,
    )
    assert res_06["status"] == "SUCCESS"

    # --- CASE E2E-07: Post-edit Travel Policy query ($75/day, proving stale vector purged) ---
    res_07 = engine.search("What is our travel reimbursement limit?", employee)
    assert len(res_07.chunks) > 0
    # Must retrieve $75 and NOT contain stale $50
    assert any("75" in c["text"] for c in res_07.chunks)
    assert not any("50" in c["text"] for c in res_07.chunks), "E2E-07 FAILED: Stale $50 vector was retrieved!"

    # --- CASE E2E-08: HR queries offboarding procedures ---
    res_08 = engine.search("What are the offboarding procedures?", hr_user)
    assert len(res_08.chunks) > 0
    assert any(c["external_id"] == "DOC-HR-OFFBOARD-01" for c in res_08.chunks)

    # --- CASE E2E-09: Sales queries offboarding procedures (Role Mismatch) ---
    res_09 = engine.search("What are the offboarding procedures?", sales_user)
    # Even though sales has internal clearance, HR offboarding requires 'hr' role and confidential clearance
    assert not any(c["external_id"] == "DOC-HR-OFFBOARD-01" for c in res_09.chunks), "E2E-09 FAILED: Role block failed!"

    # --- CASE E2E-10: Action: Issue DELETE for Staging Runbook ---
    res_10 = diff_engine.process_change(
        event_type="DELETE",
        document_id="DOC-CORP-ENG-04",
        title="",
        classification="internal",
        allowed_roles=["engineering"],
        content="",
        db_session=db,
    )
    assert res_10["status"] == "DELETED"

    # --- CASE E2E-11: Engineer queries staging clusters (Purge Verified) ---
    res_11 = engine.search("How do I access staging clusters?", engineer)
    assert not any(c["external_id"] == "DOC-CORP-ENG-04" for c in res_11.chunks), "E2E-11 FAILED: Deleted doc retrieved!"

    # --- CASE E2E-12: Multi-source synthesis (API Standards + 2026 Roadmap) ---
    res_12 = engine.search("Summarize API standards and 2026 roadmap.", engineer)
    doc_ids_in_context = {c["external_id"] for c in res_12.chunks}
    assert "DOC-ENG-API-01" in doc_ids_in_context, "E2E-12 FAILED: Missing API Guide chunk"
    assert "DOC-ENG-ROADMAP-01" in doc_ids_in_context, "E2E-12 FAILED: Missing 2026 Roadmap chunk"

    # --- CASE E2E-13: Guest queries WiFi password (internal doc) ---
    res_13 = engine.search("What is the enterprise WiFi password?", guest_user)
    assert len(res_13.chunks) == 0, "E2E-13 FAILED: Guest user must be denied internal office WiFi doc!"

    # --- CASE E2E-14: Legal queries standard liability clause ---
    res_14 = engine.search("What is our standard liability clause?", legal_user)
    assert len(res_14.chunks) > 0
    assert res_14.chunks[0]["external_id"] == "DOC-LEGAL-CONTRACT-01", "E2E-14 FAILED: Cross-encoder did not rank liability top-1!"

    # --- CASE E2E-15: Out of domain query -> Clean non-hallucination ---
    res_15 = engine.search("Explain our quantum key deployment.", engineer)
    syn_15 = await synthesizer.synthesize("Explain our quantum key deployment.", res_15.chunks)
    # Check that synthesis returns fallback / does not hallucinate
    assert ("do not possess authorized" in syn_15.answer.lower()
            or len(syn_15.citations) == 0), "E2E-15 FAILED: Did not gracefully reject out-of-domain query!"

    print("\n[SUCCESS] All 15 E2E Integration Benchmark test cases PASSED!")
