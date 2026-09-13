"""
Document-Level Security (DLS) dual-axis access evaluation.
Implements Ordinal ABAC Clearance Hierarchy and Set-Intersection RBAC.
Standard Compliance: SRS Section 4.
"""

from typing import List, Sequence
from app.constants import CLEARANCE_ORDER, get_accessible_classifications
from app.core.security import UserClaims


def can_access_document(
    user_clearance: str,
    user_roles: Sequence[str],
    doc_classification: str,
    doc_roles: Sequence[str],
) -> bool:
    """
    Evaluates whether a user can access a document chunk under the dual-axis security model.

    Condition:
        (doc_classification <= user_clearance) AND
        (doc_roles ∩ user_roles ≠ ∅ OR "*" in doc_roles)
    """
    clean_user_clearance = user_clearance.lower().strip()
    clean_doc_classification = doc_classification.lower().strip()

    # 1. Evaluate Ordinal ABAC Hierarchy
    user_rank = CLEARANCE_ORDER.get(clean_user_clearance, 0)
    doc_rank = CLEARANCE_ORDER.get(clean_doc_classification, 999)
    if doc_rank > user_rank:
        return False

    # 2. Evaluate Set-Intersection RBAC
    clean_user_roles = {r.lower().strip() for r in user_roles if r.strip()}
    clean_doc_roles = [r.lower().strip() for r in doc_roles if r.strip()]

    if "*" in clean_doc_roles:
        return True

    # Check set intersection
    if bool(clean_user_roles.intersection(set(clean_doc_roles))):
        return True

    return False


def verify_chunk_access(claims: UserClaims, chunk_payload: dict) -> bool:
    """
    Cryptographic/security verification that an individual chunk payload complies with claims.
    Used in assertion testing and audit validation.
    """
    doc_classification = chunk_payload.get("classification", "public")
    doc_roles = chunk_payload.get("allowed_roles", ["*"])
    return can_access_document(
        user_clearance=claims.clearance,
        user_roles=claims.roles,
        doc_classification=doc_classification,
        doc_roles=doc_roles,
    )
