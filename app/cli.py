"""
Administrative CLI tool for SentinelRAG.
Commands:
  - python -m app.cli reindex : Rebuilds Qdrant vector index from PostgreSQL authoritative record (NFR-3.1)
  - python -m app.cli inspect : Inspects database and vector engine statistics
  - python -m app.cli seed    : Populates baseline benchmark documents
"""

import argparse
import sys
import uuid
from typing import List
from sqlalchemy import func, select
from qdrant_client.models import PointStruct

from app.config import settings
from app.db.models import Document, DocumentChunk
from app.db.session import sync_session_factory
from app.llm.embeddings import get_embedding_service
from app.vector.client import QdrantManager
from app.vector.hashing import generate_point_id


def reindex_vector_store():
    """Rebuilds the entire Qdrant vector index from PostgreSQL system of record (NFR-3.1)."""
    print("=" * 70)
    print("SentinelRAG Vector Reindexing Utility (NFR-3.1)")
    print("=" * 70)

    session = sync_session_factory()
    try:
        embedding_svc = get_embedding_service()
        q_client = QdrantManager.get_sync_client()
        coll_name = settings.QDRANT_COLLECTION

        print(f"[*] Re-initializing Qdrant collection '{coll_name}' (dim={embedding_svc.dimension})...")
        # Recreate collection cleanly
        try:
            q_client.delete_collection(collection_name=coll_name)
        except Exception:
            pass

        QdrantManager.init_collection(
            client=q_client,
            collection_name=coll_name,
            vector_size=embedding_svc.dimension,
        )

        # Retrieve all active documents and their chunks from PostgreSQL
        docs = session.execute(
            select(Document).where(Document.is_deleted == False)
        ).scalars().all()

        total_chunks = 0
        points_to_upsert: List[PointStruct] = []

        for doc in docs:
            chunks = session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == doc.id)
                .order_by(DocumentChunk.chunk_index)
            ).scalars().all()

            if not chunks:
                continue

            print(f"  -> Processing document: {doc.external_id} ('{doc.title}') - {len(chunks)} chunks")
            texts = [c.content for c in chunks]
            vectors = embedding_svc.embed_batch(texts)

            for c, vec in zip(chunks, vectors):
                point_id = generate_point_id(str(doc.id), c.chunk_index)
                payload = {
                    "document_id": str(doc.id),
                    "external_id": doc.external_id,
                    "title": doc.title,
                    "chunk_index": c.chunk_index,
                    "classification": doc.classification,
                    "allowed_roles": doc.allowed_roles,
                    "content_hash": c.content_hash,
                    "text": c.content,
                }
                points_to_upsert.append(
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                )
                total_chunks += 1

        if points_to_upsert:
            print(f"[*] Upserting {len(points_to_upsert)} points into Qdrant in batch...")
            q_client.upsert(
                collection_name=coll_name,
                points=points_to_upsert,
            )

        print(f"[SUCCESS] Reindexed {len(docs)} documents and {total_chunks} chunk vectors.")
    finally:
        session.close()


def inspect_storage():
    """Inspects PostgreSQL and Qdrant storage counts."""
    session = sync_session_factory()
    try:
        q_client = QdrantManager.get_sync_client()
        coll = settings.QDRANT_COLLECTION

        doc_count = session.scalar(select(func.count(Document.id)).where(Document.is_deleted == False))
        chunk_count = session.scalar(select(func.count(DocumentChunk.id)))

        try:
            q_count = q_client.count(collection_name=coll, exact=True).count
        except Exception:
            q_count = "N/A (unreachable)"

        print("=" * 50)
        print("SentinelRAG Storage Inspection")
        print("=" * 50)
        print(f"Active Documents in PostgreSQL: {doc_count}")
        print(f"Registered Chunks in PostgreSQL: {chunk_count}")
        print(f"Indexed Vectors in Qdrant:     {q_count}")
        print("=" * 50)
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="SentinelRAG Administrative CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("reindex", help="Rebuild vector index from PostgreSQL")
    subparsers.add_parser("inspect", help="Inspect storage statistics")
    subparsers.add_parser("seed", help="Seed standard benchmark documents")

    args = parser.parse_args()

    if args.command == "reindex":
        reindex_vector_store()
    elif args.command == "inspect":
        inspect_storage()
    elif args.command == "seed":
        from scripts.seed_data import seed_benchmark_corpus
        seed_benchmark_corpus()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
