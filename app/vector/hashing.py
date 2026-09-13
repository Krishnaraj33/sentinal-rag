"""
Cryptographic hashing and deterministic Point ID generation.
Standard Compliance: SRS Section 2 & Module 3.
"""

import hashlib
import re
import uuid
from typing import Tuple
from app.constants import POINT_ID_NAMESPACE


def normalize_text(text: str) -> str:
    """
    Normalizes chunk text for deterministic hashing:
    - Standardizes CRLF and LF newlines
    - Collapses multiple consecutive spaces/tabs into single space
    - Trims redundant blank lines while preserving markdown paragraphs
    - Trims leading and trailing whitespace
    """
    if not text:
        return ""
    # Standardize newlines
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace
    lines = []
    for line in normalized.split("\n"):
        # Strip trailing whitespace on each line and collapse spaces/tabs
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)
    # Re-join with single newlines and collapse 3+ consecutive newlines to 2
    joined = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", joined)
    return collapsed.strip()


def compute_chunk_hash(text: str) -> str:
    """
    Computes a cryptographic hexadecimal SHA-256 digest of normalized text.
    Provides tamper-proof verification and change-detection.
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_point_id(doc_id: str, chunk_index: int) -> str:
    """
    Generates a deterministic UUIDv5 Point ID derived from document ID and chunk index:
    Point ID = UUIDv5(Namespace_DNS, f"{doc_id}:{chunk_index}")
    Enables O(1) direct addressing and targeted invalidations.
    """
    seed_str = f"{str(doc_id).strip()}:{chunk_index}"
    return str(uuid.uuid5(POINT_ID_NAMESPACE, seed_str))


def verify_point_addressing(point_id_str: str, doc_id: str, chunk_index: int) -> bool:
    """Verifies that a given point ID matches the expected deterministic formula."""
    expected = generate_point_id(doc_id, chunk_index)
    return point_id_str.lower() == expected.lower()
