"""
Structural document chunker preserving tables and section boundaries.
Standard Compliance: SRS FR-2.2.
"""

from typing import List
from langchain_text_splitters import MarkdownTextSplitter
from app.constants import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_TOKENS
from app.vector.hashing import normalize_text


class StructuralChunker:
    """
    Parses Markdown and structured text using LangChain MarkdownTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )

    def chunk_document(self, content: str) -> List[str]:
        """
        Splits text content into structured chunks using LangChain.
        """
        normalized = normalize_text(content)
        if not normalized:
            return []

        chunks = self.splitter.split_text(normalized)
        return [normalize_text(c) for c in chunks if normalize_text(c)]
