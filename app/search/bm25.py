"""
BM25 lexical scoring for sparse candidate ranking.
Standard Compliance: SRS FR-4.1.
"""

import re
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 lexical matching."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [token for token in clean.split() if token]


def bm25_rank(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranks candidate chunks using BM25Okapi lexical matching algorithm.
    Returns chunks sorted in descending order of BM25 score.
    """
    if not chunks:
        return []

    corpus_tokens = [tokenize(c.get("text", "")) for c in chunks]
    query_tokens = tokenize(query)

    if not query_tokens or not any(corpus_tokens):
        return chunks

    try:
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
    except Exception:
        # Fallback if corpus is empty or invalid
        scores = [0.0] * len(chunks)

    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        item = dict(chunk)
        item["bm25_score"] = float(score)
        scored_chunks.append(item)

    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored_chunks
