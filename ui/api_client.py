"""
HTTP client for Streamlit UI communicating with FastAPI Gateway.
"""

import os
import requests
from typing import Any, Dict, List, Optional

DEFAULT_API_URL = os.environ.get("API_GATEWAY_URL", "http://0.0.0.0:8000")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "enterprise_webhook_secret_2026")


class SentinelApiClient:
    """Client for SentinelRAG FastAPI backend."""

    def __init__(self, base_url: str = DEFAULT_API_URL):
        self.base_url = base_url.rstrip("/")

    def get_personas(self, max_retries: int = 4, retry_delay: float = 1.0) -> Dict[str, Any]:
        """Fetches pre-configured personas and tokens with retry resilience."""
        import time
        url = f"{self.base_url}/api/v1/auth/personas"
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=5.0)
                resp.raise_for_status()
                data = resp.json()
                if data:
                    return data
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        if last_exc:
            raise last_exc
        return {}

    def query(self, query_text: str, token: str, top_k: int = 5) -> Dict[str, Any]:
        """Executes identity-filtered retrieval and generation."""
        url = f"{self.base_url}/api/v1/query"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {"query": query_text, "top_k": top_k}
        resp = requests.post(url, json=payload, headers=headers, timeout=120.0)
        if resp.status_code in (401, 403):
            return {"error_status": resp.status_code, **resp.json()}
        resp.raise_for_status()
        return resp.json()

    def list_documents(self, token: str) -> List[Dict[str, Any]]:
        """Lists registered documents and content."""
        url = f"{self.base_url}/api/v1/documents"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url, headers=headers, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def update_document(
        self,
        event_type: str,
        document_id: str,
        title: str,
        classification: str,
        allowed_roles: List[str],
        content: str,
        sync_mode: bool = True,
    ) -> Dict[str, Any]:
        """Submits document change webhook event."""
        url = f"{self.base_url}/api/v1/events/document-change?sync_mode={'true' if sync_mode else 'false'}"
        headers = {
            "X-Webhook-Secret": WEBHOOK_SECRET,
            "Content-Type": "application/json",
        }
        payload = {
            "event_type": event_type,
            "document_id": document_id,
            "title": title,
            "classification": classification,
            "allowed_roles": allowed_roles,
            "content": content,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    def get_security_stats(self, token: str) -> Dict[str, Any]:
        """Retrieves boundary isolation statistics."""
        url = f"{self.base_url}/api/v1/documents/inspect/stats"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url, headers=headers, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
