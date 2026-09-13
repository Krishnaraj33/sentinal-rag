"""
Verification Test: TEST-DLS-01 (Access Isolation & Zero Context Leakage).
Standard Compliance: SRS Section 10.1 & NFR-2.1.
Pass Criteria: 0 instances returned where classification > clearance or role sets do not intersect.
"""

import random
import uuid
import pytest
from qdrant_client.models import PointStruct

from app.config import settings
from app.constants import CLEARANCE_ORDER, PRESET_PERSONAS
from app.core.dls import can_access_document
from app.core.security import UserClaims
from app.search.dense import dense_search
from app.vector.client import QdrantManager


def test_dls_access_isolation_zero_leakage(memory_qdrant):
    """
    TEST-DLS-01: Ingests 1,000 synthetic chunks across diverse clearance levels and roles.
    Queries using 4 distinct identity tokens.
    Verifies 0.00% context leakage rate.
    """
    coll_name = settings.QDRANT_COLLECTION
    classifications = ["public", "internal", "confidential", "executive"]
    roles_pool = [["contractor"], ["engineering"], ["finance"], ["hr"], ["legal"], ["*"]]

    # Ingest 1,000 synthetic chunks
    points = []
    for i in range(1000):
        cls_level = random.choice(classifications)
        assigned_roles = random.choice(roles_pool)
        doc_id = f"DOC-SYNTH-{i % 50:03d}"
        payload = {
            "document_id": doc_id,
            "chunk_index": i % 5,
            "classification": cls_level,
            "allowed_roles": assigned_roles,
            "content_hash": f"hash_{i:06d}",
            "text": f"Synthetic enterprise record {i} regarding {cls_level} procedures for {assigned_roles}.",
        }
        # Random normalized vector
        vec = [random.uniform(-1.0, 1.0) for _ in range(384)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vec = [x / norm for x in vec]

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    memory_qdrant.upsert(collection_name=coll_name, points=points)

    total_queries_tested = 0
    leakage_violations = 0

    # Test across all preset personas
    for persona_key, p_data in PRESET_PERSONAS.items():
        claims = UserClaims(
            sub=str(p_data["sub"]),
            roles=list(p_data["roles"]),
            clearance=str(p_data["clearance"]),
        )

        for query_sample in ["corporate credentials", "reimbursement limit", "executive bonus", "internal procedures"]:
            total_queries_tested += 1
            results = dense_search(
                query=query_sample,
                claims=claims,
                limit=50,
                client=memory_qdrant,
            )

            for hit in results:
                hit_classification = hit.get("classification", "public")
                hit_roles = hit.get("allowed_roles", ["*"])

                is_allowed = can_access_document(
                    user_clearance=claims.clearance,
                    user_roles=claims.roles,
                    doc_classification=hit_classification,
                    doc_roles=hit_roles,
                )

                if not is_allowed:
                    leakage_violations += 1

    leakage_rate = (leakage_violations / max(total_queries_tested, 1)) * 100.0
    print(f"\n[TEST-DLS-01] Evaluated {total_queries_tested} queries. Violations: {leakage_violations}. Leakage: {leakage_rate:.2f}%")

    assert leakage_violations == 0, f"Detected {leakage_violations} context leakage violations!"
    assert leakage_rate == 0.00
