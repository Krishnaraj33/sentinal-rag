"""
Celery asynchronous task broker configuration.
Standard Compliance: SRS Section 3 & Section 9.
"""

from celery import Celery
from app.config import settings

celery_app = Celery(
    "sentinel_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)
