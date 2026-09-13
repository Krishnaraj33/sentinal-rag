"""
Pytest configuration and zero-token deterministic test fixtures.
Supports in-memory SQLite and in-memory Qdrant for hermetic local execution.
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient

from app.config import settings
from app.db.base import Base
from app.llm.embeddings import MockEmbeddingService, set_embedding_service
from app.vector.client import QdrantManager
from app.worker.chunker import StructuralChunker
from app.worker.diff_engine import DiffEngine


@pytest.fixture(scope="session", autouse=True)
def setup_mock_environment():
    """Forces deterministic mock embedding and Qdrant in-memory mode for tests."""
    mock_emb = MockEmbeddingService(dimension=384)
    set_embedding_service(mock_emb)
    settings.QDRANT_COLLECTION = "test_sentinel_kb"
    yield


@pytest.fixture
def memory_db_session():
    """Provides isolated in-memory SQLite database session for tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def memory_qdrant():
    """Provides fresh in-memory Qdrant instance for tests."""
    client = QdrantClient(":memory:")
    QdrantManager.set_sync_client(client)
    QdrantManager.init_collection(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        vector_size=384,
    )
    yield client


@pytest.fixture
def test_diff_engine(memory_qdrant):
    """Provides DiffEngine wired to in-memory Qdrant and Mock embeddings."""
    mock_emb = MockEmbeddingService(dimension=384)
    return DiffEngine(
        chunker=StructuralChunker(chunk_size=512, overlap=64),
        embedding_service=mock_emb,
        qdrant_client=memory_qdrant,
    )
