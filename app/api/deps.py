"""
FastAPI dependencies for JWT authentication, claims parsing, and webhook secret verification.
Standard Compliance: SRS FR-1.1 & FR-2.1.
"""

from typing import Optional
from fastapi import Header, HTTPException, status
from app.core.security import UserClaims, decode_access_token, verify_webhook_secret


def get_current_user_claims(
    authorization: Optional[str] = Header(None, description="Bearer <JWT_TOKEN>"),
) -> UserClaims:
    """
    Dependency extracting and validating UserClaims from Authorization Bearer token.
    Raises HTTP 401 if missing, invalid, or expired.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <TOKEN>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    try:
        claims = decode_access_token(token)
        return claims
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_webhook_secret(
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
) -> bool:
    """
    Dependency validating the webhook shared secret header.
    Raises HTTP 401 if secret does not match.
    """
    if not verify_webhook_secret(x_webhook_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Webhook-Secret header.",
        )
    return True
