"""
Builds native Qdrant payload filters from UserClaims.
Translates dual-axis security claims into HNSW pre-retrieval bitmask filters.
Standard Compliance: SRS FR-1.2 & FR-6.2.
"""

from typing import Any, Dict, List
from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.constants import get_accessible_classifications
from app.core.security import UserClaims


class FilterBuilder:
    """Constructs native Qdrant filters enforcing dual-axis DLS during graph traversal."""

    @staticmethod
    def build_qdrant_filter(claims: UserClaims) -> Filter:
        """
        Translates user claims into a native Qdrant Filter object.
        Applies:
          1. Ordinal ABAC: classification in [accessible_classifications]
          2. Set-Intersection RBAC: allowed_roles intersects [*] + [user_roles]
        """
        accessible_classifications = get_accessible_classifications(claims.clearance)
        roles_to_match = list(set(["*"] + [r.lower().strip() for r in claims.roles if r.strip()]))

        classification_condition = FieldCondition(
            key="classification",
            match=MatchAny(any=accessible_classifications),
        )

        roles_condition = FieldCondition(
            key="allowed_roles",
            match=MatchAny(any=roles_to_match),
        )

        return Filter(
            must=[
                classification_condition,
                roles_condition,
            ]
        )

    @staticmethod
    def to_dict(claims: UserClaims) -> Dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representation of the compiled Qdrant pre-filter.
        Used by the UI inspector and API metrics payload.
        """
        accessible_classifications = get_accessible_classifications(claims.clearance)
        roles_to_match = sorted(list(set(["*"] + [r.lower().strip() for r in claims.roles if r.strip()])))

        return {
            "filter_type": "pre_retrieval_dls",
            "must": [
                {
                    "key": "classification",
                    "match": {
                        "any": accessible_classifications
                    },
                    "description": f"ABAC Ordinal Hierarchy (≤ {claims.clearance})"
                },
                {
                    "key": "allowed_roles",
                    "match": {
                        "any": roles_to_match
                    },
                    "description": "RBAC Set-Intersection (Doc roles ∩ User roles or Wildcard)"
                }
            ]
        }
