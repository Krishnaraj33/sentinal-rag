"""
Change Data Capture (CDC) document event webhook endpoint.
Standard Compliance: SRS Section 8.2 (FR-2.1, FR-2.3).
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import require_webhook_secret
from app.constants import CLEARANCE_HIERARCHY
from app.worker.tasks import execute_document_change_inline, process_document_change_task

router = APIRouter(prefix="/events", tags=["Change Data Capture (CDC)"])


class DocumentEventRequest(BaseModel):
    event_type: str = Field(..., example="UPDATE", description="CREATE | UPDATE | DELETE | PERMISSIONS_CHANGE")
    document_id: str = Field(..., example="DOC-CORP-ENG-04", description="External document reference identifier")
    title: str = Field("", example="Staging Environment Guidelines")
    classification: str = Field("internal", example="internal")
    allowed_roles: List[str] = Field(default_factory=lambda: ["engineering"])
    content: str = Field("", description="Raw document text or Markdown")
    timestamp: Optional[int] = Field(None, example=1773307448)


class DocumentEventResponse(BaseModel):
    status: str
    task_id: str
    execution_mode: str = "asynchronous"
    details: Optional[Dict[str, Any]] = None


@router.post(
    "/document-change",
    response_model=DocumentEventResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def handle_document_change(
    event: DocumentEventRequest,
    authenticated: bool = Depends(require_webhook_secret),
    sync_mode: bool = Query(False, description="Run synchronously for testing/eager execution"),
):
    """
    Asynchronous ingestion webhook for document mutations.
    Returns HTTP 202 within 50 ms and offloads diffing & invalidation to workers.
    """
    clean_event = event.event_type.upper().strip()
    if clean_event not in ["CREATE", "UPDATE", "DELETE", "PERMISSIONS_CHANGE"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event_type '{event.event_type}'. Must be CREATE, UPDATE, DELETE, or PERMISSIONS_CHANGE.",
        )

    clean_classification = event.classification.lower().strip()
    if clean_classification not in CLEARANCE_HIERARCHY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid classification '{event.classification}'.",
        )

    payload = event.model_dump()

    if sync_mode:
        # Synchronous execution mode (useful for integration tests & immediate validation)
        res = execute_document_change_inline(payload)
        return DocumentEventResponse(
            status="COMPLETED",
            task_id=f"sync-{uuid.uuid4()}",
            execution_mode="synchronous",
            details=res,
        )

    # Asynchronous queuing to Celery broker
    task_id = f"task-{uuid.uuid4()}"
    try:
        celery_task = process_document_change_task.delay(payload)
        task_id = str(celery_task.id)
    except Exception:
        # Fallback to inline execution if Redis/Celery is unavailable locally
        res = execute_document_change_inline(payload)
        return DocumentEventResponse(
            status="COMPLETED",
            task_id=task_id,
            execution_mode="local_fallback",
            details=res,
        )

    return DocumentEventResponse(
        status="QUEUED",
        task_id=f"celery-task-{task_id}",
        execution_mode="asynchronous",
    )
