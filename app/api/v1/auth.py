"""
Authentication and Persona Token management endpoints.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.constants import CLEARANCE_HIERARCHY, PRESET_PERSONAS
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication & Personas"])


class TokenRequest(BaseModel):
    """Payload for generating custom or persona JWT bearer tokens."""
    sub: str = Field(..., example="USR-ENG-42")
    roles: List[str] = Field(default_factory=lambda: ["engineering"])
    clearance: str = Field("internal", example="internal")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    sub: str
    roles: List[str]
    clearance: str


@router.post("/token", response_model=TokenResponse)
async def generate_token(req: TokenRequest):
    """Generates a valid JWT token signed with system secret for specified claims."""
    clean_clearance = req.clearance.lower().strip()
    if clean_clearance not in CLEARANCE_HIERARCHY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid clearance '{req.clearance}'. Must be one of: {CLEARANCE_HIERARCHY}",
        )

    claims = {
        "sub": req.sub,
        "roles": req.roles,
        "clearance": clean_clearance,
    }
    token = create_access_token(claims)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        sub=req.sub,
        roles=req.roles,
        clearance=clean_clearance,
    )


@router.get("/personas")
async def list_preset_personas():
    """
    Returns pre-configured personas (Contractor, Engineer, Finance, CEO)
    along with pre-computed JWT tokens for instantaneous testing and UI switching.
    """
    results: Dict[str, Any] = {}
    for key, p in PRESET_PERSONAS.items():
        claims = {
            "sub": str(p["sub"]),
            "roles": list(p["roles"]),
            "clearance": str(p["clearance"]),
        }
        token = create_access_token(claims)
        results[key] = {
            **p,
            "token": token,
        }
    return results
