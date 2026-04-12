"""
API Key Authentication Middleware for TalentLens AI.
Validates X-API-Key header. Static keys loaded from environment.
Demo keys are pre-populated for local development.
"""

import os
import time
import logging
from collections import deque
from fastapi import Header, HTTPException
from typing import Optional

logger = logging.getLogger("talentlens.auth")

# ── Key Registry ──────────────────────────────────────────────────────────────
# Format: key → {name, tier, rate_limit_per_min}
DEMO_KEYS = {
    "demo-key-talentlens":  {"name": "Demo Key",       "tier": "free",  "rpm": 20},
    "dev-key-internal":     {"name": "Dev Internal",   "tier": "pro",   "rpm": 60},
    "investor-preview-key": {"name": "Investor Demo",  "tier": "pro",   "rpm": 60},
}

def _load_keys() -> dict:
    """Load API keys from env var + pre-loaded demo keys."""
    keys = dict(DEMO_KEYS)
    env_keys = os.getenv("API_KEYS", "")
    for raw_key in env_keys.split(","):
        k = raw_key.strip()
        if k:
            keys[k] = {"name": "Env Key", "tier": "pro", "rpm": 60}
    return keys

API_KEY_REGISTRY: dict = _load_keys()

# Sliding-window usage tracking: key → deque of timestamps
_usage: dict[str, deque] = {k: deque() for k in API_KEY_REGISTRY}


def _check_rate_limit(key: str, rpm: int) -> bool:
    """Sliding window rate limiter. Returns False if limit exceeded."""
    now = time.time()
    window = _usage.setdefault(key, deque())
    # Evict old entries outside the 60-second window
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= rpm:
        return False
    window.append(now)
    return True


async def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """
    FastAPI dependency — validates X-API-Key header.
    Pass as a dependency on any route that requires auth.
    Returns the key metadata dict on success.
    """
    # If no keys are configured beyond demos, be lenient in dev
    if x_api_key is None:
        # Allow requests through without a key if we're in dev mode
        env = os.getenv("ENV", "development").lower()
        if env == "development":
            return {"name": "Anonymous Dev", "tier": "free", "rpm": 10}
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_api_key",
                "message": "X-API-Key header is required.",
                "hint": "Use 'demo-key-talentlens' for free access.",
            },
        )

    meta = API_KEY_REGISTRY.get(x_api_key)
    if not meta:
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": "The provided API key is not valid.",
                "hint": "Use 'demo-key-talentlens' for free demo access.",
            },
        )

    # Rate limiting
    if not _check_rate_limit(x_api_key, meta["rpm"]):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit of {meta['rpm']} requests/minute exceeded.",
                "retry_after": 60,
            },
            headers={"Retry-After": "60"},
        )

    logger.debug(f"Auth OK: {meta['name']} (tier={meta['tier']})")
    return meta


def get_api_keys_summary() -> list:
    """Return sanitized list of registered keys for API docs."""
    return [
        {
            "key": k[:8] + "...",
            "name": v["name"],
            "tier": v["tier"],
            "rpm": v["rpm"],
        }
        for k, v in API_KEY_REGISTRY.items()
    ]


def get_demo_keys() -> list:
    """Return the full demo keys for display in UI."""
    return [
        {"key": k, "name": v["name"], "tier": v["tier"]}
        for k, v in DEMO_KEYS.items()
    ]
