"""
Security utilities: JWT token generation, claims extraction, and secret verification.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.constants import CLEARANCE_HIERARCHY


class UserClaims(BaseModel):
    """Normalized claims extracted from validated JWT token."""
    sub: str = Field(..., description="Subject unique user identifier")
    roles: List[str] = Field(default_factory=list, description="Assigned organizational roles")
    clearance: str = Field("public", description="Ordinal ABAC security clearance level")

    @field_validator("clearance")
    @classmethod
    def validate_clearance(cls, v: str) -> str:
        clean = v.lower().strip()
        if clean not in CLEARANCE_HIERARCHY:
            raise ValueError(f"Invalid clearance '{v}'. Must be one of: {CLEARANCE_HIERARCHY}")
        return clean

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, v: List[str]) -> List[str]:
        return [r.lower().strip() for r in v if r.strip()]


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Encodes JWT bearer token containing user claims."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> UserClaims:
    """Decodes and validates JWT bearer token, returning validated UserClaims."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        sub: Optional[str] = payload.get("sub")
        roles: List[str] = payload.get("roles", [])
        clearance: str = payload.get("clearance", "public")
        if sub is None:
            raise ValueError("JWT token missing 'sub' claim")
        return UserClaims(sub=sub, roles=roles, clearance=clearance)
    except (JWTError, ValueError) as exc:
        raise ValueError(f"Could not validate credentials: {str(exc)}") from exc


import hmac

def verify_webhook_secret(secret_header: Optional[str]) -> bool:
    """Validates incoming X-Webhook-Secret against system configuration."""
    if not secret_header:
        return False
    return hmac.compare_digest(secret_header.strip().encode('utf-8'), settings.WEBHOOK_SECRET.strip().encode('utf-8'))
