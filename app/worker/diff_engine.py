"""
Deterministic SHA-256 chunk diffing engine with targeted point invalidation.
Standard Compliance: SRS Section 2, Module 3 (FR-3.1, FR-3.2, FR-3.3) & NFR-4.1.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, PointIdsList

from app.config import settings
from app.db.models import Document, DocumentChunk
from app.llm.embeddings import BaseEmbeddingService, get_embedding_service
from app.vector.client import QdrantManager, update_document_payload_in_place
from app.vector.hashing import compute_chunk_hash, generate_point_id, normalize_text
from app.worker.chunker import StructuralChunker

logger = logging.getLogger(__name__)


class DiffEngine:
    """
    Executes content diffing, deterministic point addressing, and targeted vector invalidation.
    Suppresses redundant embedding calls by comparing cryptographic SHA-256 digests.
    """

    def __init__(
        self,
        chunker: Optional[StructuralChunker] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        qdrant_client: Optional[QdrantClient] = None,
    ):
        self.chunker = chunker or StructuralChunker()
        self.embedding_service = embedding_service or get_embedding_service()
        self.qdrant_client = qdrant_client or QdrantManager.get_sync_client()

    def process_change(
        self,
        event_type: str,
        document_id: str,
        title: str,
        classification: str,
        allowed_roles: List[str],
        content: str,
        db_session: Session,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for CDC document events: CREATE, UPDATE, DELETE, PERMISSIONS_CHANGE.
        Idempotent execution guaranteed.
        """
        event_upper = event_type.upper().strip()
        clean_roles = [r.lower().strip() for r in allowed_roles if r.strip()] or ["*"]
        clean_classification = classification.lower().strip()

        # Ensure Qdrant collection exists
        QdrantManager.init_collection(
            client=self.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION,
            vector_size=self.embedding_service.dimension,
        )

        if event_upper == "DELETE":
            return self._handle_delete(document_id, db_session)
        elif event_upper == "PERMISSIONS_CHANGE":
            return self._handle_permissions_change(
                document_id, clean_classification, clean_roles, db_session
            )
        else:
            return self._handle_upsert(
                event_upper,
                document_id,
                title,
                clean_classification,
                clean_roles,
                content,
                db_session,
            )

    def _handle_upsert(
        self,
        event_type: str,
        external_id: str,
        title: str,
        classification: str,
        allowed_roles: List[str],
        content: str,
        db: Session,
    ) -> Dict[str, Any]:
        """Handles document CREATE and UPDATE events with chunk diffing."""
        # 1. Retrieve or create parent document record
        doc = db.execute(
            select(Document).where(Document.external_id == external_id)
        ).scalar_one_or_none()

        if doc is None:
            doc = Document(
                id=uuid.uuid4(),
                external_id=external_id,
                title=title,
                classification=classification,
                allowed_roles=allowed_roles,
                version=1,
                is_deleted=False,
            )
            db.add(doc)
            db.flush()
        else:
            doc.title = title
            doc.classification = classification
            doc.allowed_roles = allowed_roles
            doc.is_deleted = False
            doc.version += 1
            db.flush()

        doc_internal_id = str(doc.id)

        # 2. Query stored chunk hashes from PostgreSQL
        stored_chunks_rows = db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars().all()

        stored_by_index: Dict[int, DocumentChunk] = {
            c.chunk_index: c for c in stored_chunks_rows
        }

        # 3. Structural Chunking
        new_chunk_texts = self.chunker.chunk_document(content)
        total_incoming_chunks = len(new_chunk_texts)

        unchanged_indices: List[int] = []
        mutated_indices: List[int] = []
        orphaned_indices: List[int] = []
        points_to_upsert: List[PointStruct] = []
        chunks_to_embed: List[Tuple[int, str, str, str]] = []  # (index, text, hash, point_id)

        # 4. Content Hashing & Diff Evaluation
        for idx, raw_text in enumerate(new_chunk_texts):
            norm_text = normalize_text(raw_text)
            new_hash = compute_chunk_hash(norm_text)
            point_id = generate_point_id(doc_internal_id, idx)

            stored = stored_by_index.get(idx)
            if stored is not None and stored.content_hash == new_hash:
                # UNCHANGED: Hash matches stored record -> Skip embedding
                unchanged_indices.append(idx)
            else:
                # MUTATED: Content hash differs or index is newly created
                mutated_indices.append(idx)
                chunks_to_embed.append((idx, norm_text, new_hash, point_id))

        # 5. Detect ORPHANED chunks (indices no longer present in shortened document)
        for stored_idx in stored_by_index:
            if stored_idx >= total_incoming_chunks:
                orphaned_indices.append(stored_idx)

        # 6. Generate embeddings strictly for MUTATED chunks
        embedding_calls_count = len(chunks_to_embed)
        if chunks_to_embed:
            texts_to_embed = [item[1] for item in chunks_to_embed]
            vectors = self.embedding_service.embed_batch(texts_to_embed)

            for (idx, norm_text, new_hash, point_id), vec in zip(chunks_to_embed, vectors):
                point_uuid = uuid.UUID(point_id)
                stored_chunk = stored_by_index.get(idx)

                if stored_chunk is not None:
                    # Update existing chunk record in PostgreSQL
                    stored_chunk.content_hash = new_hash
                    stored_chunk.content = norm_text
                else:
                    # Insert new chunk record in PostgreSQL
                    new_chunk_record = DocumentChunk(
                        id=point_uuid,
                        document_id=doc.id,
                        chunk_index=idx,
                        content_hash=new_hash,
                        content=norm_text,
                    )
                    db.add(new_chunk_record)

                # Prepare Qdrant PointStruct at deterministic Point ID
                payload = {
                    "document_id": doc_internal_id,
                    "external_id": external_id,
                    "title": title,
                    "chunk_index": idx,
                    "classification": classification,
                    "allowed_roles": allowed_roles,
                    "content_hash": new_hash,
                    "text": norm_text,
                }
                points_to_upsert.append(
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                )

        # 7. Write mutated vectors to Qdrant at deterministic point IDs
        if points_to_upsert:
            self.qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points_to_upsert,
            )

        # 7.5 Update payloads for UNCHANGED chunks
        if unchanged_indices:
            unchanged_point_ids = [
                generate_point_id(doc_internal_id, idx) for idx in unchanged_indices
            ]
            self.qdrant_client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload={
                    "title": title,
                    "classification": classification,
                    "allowed_roles": allowed_roles,
                },
                points=unchanged_point_ids,
            )

        # 8. Delete ORPHANED points from Qdrant and PostgreSQL
        if orphaned_indices:
            orphaned_point_ids = [
                generate_point_id(doc_internal_id, idx) for idx in orphaned_indices
            ]
            # Targeted Point ID deletion in Qdrant
            self.qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=PointIdsList(points=orphaned_point_ids),
            )
            # Delete from PostgreSQL
            db.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == doc.id,
                    DocumentChunk.chunk_index.in_(orphaned_indices),
                )
            )

        savings_ratio = (
            (len(unchanged_indices) / total_incoming_chunks * 100.0)
            if total_incoming_chunks > 0
            else 0.0
        )

        return {
            "status": "SUCCESS",
            "event_type": event_type,
            "document_id": external_id,
            "internal_id": doc_internal_id,
            "total_chunks": total_incoming_chunks,
            "unchanged_chunks": unchanged_indices,
            "mutated_chunks": mutated_indices,
            "orphaned_chunks": orphaned_indices,
            "embedding_calls_generated": embedding_calls_count,
            "embedding_suppression_pct": round(savings_ratio, 1),
        }

    def _handle_delete(self, external_id: str, db: Session) -> Dict[str, Any]:
        """Handles document DELETE: atomic purge within 1 second."""
        doc = db.execute(
            select(Document).where(Document.external_id == external_id)
        ).scalar_one_or_none()

        if doc is None:
            return {"status": "NOT_FOUND", "document_id": external_id}

        doc_internal_id = str(doc.id)

        # Flag as deleted in PostgreSQL
        doc.is_deleted = True

        # Query all existing chunk IDs
        chunks = db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).scalars().all()
        point_ids = [str(c.id) for c in chunks]

        # Atomic batch deletion by deterministic IDs in Qdrant (FR-3.3)
        if point_ids:
            self.qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=PointIdsList(points=point_ids),
            )

        # Clean up relational chunk records
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))

        return {
            "status": "DELETED",
            "document_id": external_id,
            "purged_points_count": len(point_ids),
        }

    def _handle_permissions_change(
        self,
        external_id: str,
        classification: str,
        allowed_roles: List[str],
        db: Session,
    ) -> Dict[str, Any]:
        """In-place permissions update without re-embedding vectors (FR-1.4 & TEST-PERM-04)."""
        doc = db.execute(
            select(Document).where(Document.external_id == external_id)
        ).scalar_one_or_none()

        if doc is None:
            return {"status": "NOT_FOUND", "document_id": external_id}

        doc_internal_id = str(doc.id)

        # Update PostgreSQL metadata
        doc.classification = classification
        doc.allowed_roles = allowed_roles

        # Update Qdrant payloads in-place
        update_document_payload_in_place(
            client=self.qdrant_client,
            doc_id=doc_internal_id,
            payload_updates={
                "classification": classification,
                "allowed_roles": allowed_roles,
            },
        )

        return {
            "status": "RECLASSIFIED",
            "document_id": external_id,
            "new_classification": classification,
            "new_allowed_roles": allowed_roles,
            "embedding_calls_generated": 0,  # Zero-token in-place update!
        }
