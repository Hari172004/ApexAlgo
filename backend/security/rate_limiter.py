"""
rate_limiter.py — API Rate Limiting and IP Blocking
Uses `slowapi` to protect FastAPI endpoints against DDoS and abuse.
"""

import logging
from fastapi import Request, HTTPException, status
from typing import Callable
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("apexalgo.ratelimit")

# In-memory dictionary tracking repeated abusers (in production, use Redis)
ABUSE_RECORDS = {}

# 1. Initialize the global Limiter
# By default, we use IP address for anonymous requests
limiter = Limiter(key_func=get_remote_address)

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler bound to FastAPI app to return 429 status code
    and log the abuse.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.warning(f"[RateLimit] ⚠️ 429 Too Many Requests -> IP: {client_ip} | Path: {request.url.path}")
    
    # Track repeated limit hits
    ABUSE_RECORDS[client_ip] = ABUSE_RECORDS.get(client_ip, 0) + 1
    if ABUSE_RECORDS[client_ip] > 5:
        logger.error(f"[RateLimit] 🚨 IP {client_ip} is repeatedly abusing the API. Blocking required.")
        # Trigger: Threat detector alert
        
    return _rate_limit_exceeded_handler(request, exc)

# ── Dynamic Key Functions ───────────────────────────────────────────────────

def get_user_id(request: Request) -> str:
    """
    Extracts user_id from the injected request state.
    Requires an auth middleware to populate `request.state.user_id`.
    Falls back to IP address if the user is not authenticated.
    """
    if hasattr(request, "state") and hasattr(request.state, "user_id"):
        return request.state.user_id
    return get_remote_address(request)

def limit_by_user() -> Callable:
    """
    User-level rate limit: 100 requests per minute max.
    Usage in FastAPI route: @limiter.limit("100/minute", key_func=get_user_id)
    """
    return get_user_id

def limit_by_ip() -> Callable:
    """
    Strict IP-level limit: 200 requests per minute max.
    Usage in FastAPI route: @limiter.limit("200/minute")
    """
    return get_remote_address

def get_global_limiter() -> Limiter:
    """Returns the configured SlowAPI limiter."""
    return limiter

def setup_app_rate_limiting(app):
    """
    Binds the limiter and the exception handler to a FastAPI app instance.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    logger.info("[RateLimit] Global Rate Limiter attached to FastAPI.")
