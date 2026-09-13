"""
Celery worker background tasks for Change Data Capture (CDC) events.
Standard Compliance: SRS FR-2.1, FR-2.3 & Module 3.
"""

import logging
from typing import Any, Dict
from app.db.session import sync_session_factory
from app.worker.celery_app import celery_app
from app.worker.diff_engine import DiffEngine

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.process_document_change", bind=True, max_retries=3, default_retry_delay=5)
def process_document_change_task(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous Celery task processing incoming document change events.
    Executes idempotently and performs targeted vector diffing.
    """
    logger.info("Starting document change processing: %s (event=%s)",
                event_data.get("document_id"), event_data.get("event_type"))
    try:
        session = sync_session_factory()
        try:
            diff_engine = DiffEngine()
            result = diff_engine.process_change(
                event_type=event_data.get("event_type", "UPDATE"),
                document_id=event_data.get("document_id", ""),
                title=event_data.get("title", ""),
                classification=event_data.get("classification", "public"),
                allowed_roles=event_data.get("allowed_roles", ["*"]),
                content=event_data.get("content", ""),
                db_session=session,
            )
            session.commit()
            return result
        finally:
            session.close()
    except Exception as exc:
        logger.error("Error processing document change task: %s", exc, exc_info=True)
        # Auto-retry with backoff
        raise self.retry(exc=exc)


def execute_document_change_inline(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous / eager execution helper for testing and standalone mode.
    """
    session = sync_session_factory()
    try:
        diff_engine = DiffEngine()
        result = diff_engine.process_change(
            event_type=event_data.get("event_type", "UPDATE"),
            document_id=event_data.get("document_id", ""),
            title=event_data.get("title", ""),
            classification=event_data.get("classification", "public"),
            allowed_roles=event_data.get("allowed_roles", ["*"]),
            content=event_data.get("content", ""),
            db_session=session,
        )
        session.commit()
        return result
    finally:
        session.close()
