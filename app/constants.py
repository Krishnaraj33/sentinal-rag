"""
Constants and static configurations for SentinelRAG.
Compliant with SRS-SENTINEL-2026-V1.2 specification.
"""

import uuid
from typing import Dict, List

# Ordinal ABAC Clearance Hierarchy: public < internal < confidential < executive
CLEARANCE_HIERARCHY: List[str] = [
    "public",
    "internal",
    "confidential",
    "executive",
]

CLEARANCE_ORDER: Dict[str, int] = {
    level: idx for idx, level in enumerate(CLEARANCE_HIERARCHY)
}

# Namespace DNS for deterministic UUIDv5 Point ID generation
POINT_ID_NAMESPACE: uuid.UUID = uuid.NAMESPACE_DNS

# Default Chunking Hyperparameters
DEFAULT_CHUNK_SIZE_TOKENS: int = 512
DEFAULT_CHUNK_OVERLAP_TOKENS: int = 64

# Default Reciprocal Rank Fusion (RRF) constant
RRF_K: int = 60

# Default Re-ranking top candidates pool and final selection
RRF_TOP_CANDIDATES: int = 20
DEFAULT_TOP_K_RESULTS: int = 5

# Pre-set Employee Personas as defined in SRS FR-6.1
PRESET_PERSONAS: Dict[str, Dict[str, object]] = {
    "contractor": {
        "name": "Guest / Contractor",
        "sub": "USR-CONTRACTOR-01",
        "clearance": "public",
        "roles": ["contractor"],
        "description": "External partner with baseline public clearance.",
    },
    "engineering": {
        "name": "Software Engineer",
        "sub": "USR-ENG-42",
        "clearance": "internal",
        "roles": ["engineering"],
        "description": "Internal engineering team member accessing technical docs.",
    },
    "finance": {
        "name": "Finance Manager",
        "sub": "USR-FIN-09",
        "clearance": "confidential",
        "roles": ["finance"],
        "description": "Finance personnel accessing confidential budget and payroll docs.",
    },
    "executive": {
        "name": "Chief Executive Officer",
        "sub": "USR-CEO-01",
        "clearance": "executive",
        "roles": ["c-suite", "engineering", "finance", "hr"],
        "description": "Executive officer with maximum clearance across all domains.",
    },
}


def get_accessible_classifications(user_clearance: str) -> List[str]:
    """
    Returns list of document classifications accessible by a user given their clearance level.
    Enforces Ordinal ABAC Hierarchy: doc_classification <= user_clearance.
    """
    clean_clearance = user_clearance.lower().strip()
    if clean_clearance not in CLEARANCE_ORDER:
        # Fallback to public if unrecognized
        return ["public"]
    user_order = CLEARANCE_ORDER[clean_clearance]
    return [
        level
        for level in CLEARANCE_HIERARCHY
        if CLEARANCE_ORDER[level] <= user_order
    ]
