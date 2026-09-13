"""
Dense vector embedding services.
Supports Sentence-Transformers, OpenAI-compatible APIs, and Deterministic Mock embeddings.
"""

import abc
import hashlib
import logging
import math
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(abc.ABC):
    """Abstract interface for dense vector embeddings."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimensionality."""
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates embedding vector for a single string."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of strings."""
        pass


class MockEmbeddingService(BaseEmbeddingService):
    """
    Deterministic pseudo-random vector generator for zero-cost unit testing.
    Generates unit-normalized vectors derived deterministically from text hashes.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        # Generate pseudo-random vector deterministically using SHA-256 seed
        seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self._dimension):
            byte_val = seed_hash[i % len(seed_hash)]
            # Map byte (0-255) to float [-1.0, 1.0]
            val = ((byte_val ^ (i & 0xFF)) / 127.5) - 1.0
            vec.append(val)
        # Unit-normalize vector for Cosine similarity
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]


class SentenceTransformersEmbeddingService(BaseEmbeddingService):
    """Local embedding service powered by HuggingFace / Sentence-Transformers."""

    def __init__(self, model_name: Optional[str] = None):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info("Loading HuggingFaceEmbeddings: %s", self.model_name)
        self._model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=settings.HF_HOME
        )
        self._dimension = settings.EMBEDDING_DIM

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._model.embed_documents(texts)


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Remote embedding service powered by OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from langchain_openai import OpenAIEmbeddings
        self.api_key = api_key or settings.LLM_API_KEY
        self.model = model or "text-embedding-3-small"
        self._dimension = settings.EMBEDDING_DIM
        self._model = OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.model,
            base_url=settings.LLM_BASE_URL
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._model.embed_documents(texts)


_embedding_service_instance: Optional[BaseEmbeddingService] = None


def get_embedding_service() -> BaseEmbeddingService:
    """Returns singleton instance of configured embedding service."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        provider = settings.EMBEDDING_PROVIDER.lower().strip()
        if provider == "mock":
            _embedding_service_instance = MockEmbeddingService(dimension=settings.EMBEDDING_DIM)
        elif provider == "openai":
            _embedding_service_instance = OpenAIEmbeddingService()
        else:
            try:
                _embedding_service_instance = SentenceTransformersEmbeddingService()
            except Exception as exc:
                logger.warning(
                    "Failed to initialize SentenceTransformer (%s). Falling back to MockEmbeddingService.",
                    exc,
                )
                _embedding_service_instance = MockEmbeddingService(dimension=settings.EMBEDDING_DIM)
    return _embedding_service_instance


def set_embedding_service(service: BaseEmbeddingService) -> None:
    """Sets embedding service singleton (useful for testing)."""
    global _embedding_service_instance
    _embedding_service_instance = service
