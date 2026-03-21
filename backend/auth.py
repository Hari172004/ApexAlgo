"""
auth.py — Firebase Authentication + JWT License Verification
"""

import os
import logging
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from fastapi import HTTPException, Header
from typing import Optional
from backend.database import is_license_valid

logger = logging.getLogger("apexalgo.auth")

_firebase_initialized = False


def init_firebase():
    global _firebase_initialized
    if _firebase_initialized:
        return
    # [DEV MOCK] Bypass actual Firebase init
    logger.info("[Auth] Firebase mocked for local development.")
    _firebase_initialized = True


def verify_firebase_token(id_token: str) -> dict:
    """Verify a Firebase ID token and return the decoded payload."""
    # [DEV MOCK] Always accept the mock token
    return {"uid": "admin_101", "email": "admin@apexalgo.com", "plan": "ELITE"}


def get_current_user(authorization: str = Header(None)) -> dict:
    """FastAPI dependency — extract and verify Firebase token from auth header."""
    # [DEV MOCK] Always return the mock user
    return {"uid": "admin_101", "email": "admin@apexalgo.com", "plan": "ELITE"}


def require_valid_license(user: Optional[dict] = None, user_id: Optional[str] = None) -> str:
    """
    Check that the user has an active paid license.
    Returns the plan name if valid.
    Raises 403 if not.
    """
    # [DEV MOCK] Always return Elite plan
    return "ELITE"


def require_pro_or_elite(user: dict) -> str:
    """Require Pro or Elite plan (for funded mode and BTC access)."""
    plan = require_valid_license(user=user)
    if plan not in ("PRO", "ELITE"):
        raise HTTPException(
            status_code=403,
            detail="Funded account mode requires Pro or Elite plan."
        )
    return plan


def require_elite(user: dict) -> str:
    """Require Elite plan."""
    plan = require_valid_license(user=user)
    if plan != "ELITE":
        raise HTTPException(
            status_code=403,
            detail="This feature requires the Elite plan."
        )
    return plan
