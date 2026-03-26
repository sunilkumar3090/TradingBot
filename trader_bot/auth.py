"""
auth.py — OAuth flow for Zerodha Kite Connect.

DISCLAIMER: This software is for educational purposes only.
All trading involves significant financial risk. The developer
assumes no liability for losses incurred through use of this system.

Daily token refresh is required per SEBI/exchange regulations.
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from kiteconnect import KiteConnect

from config.settings import get_settings

logger = logging.getLogger(__name__)

TOKEN_CACHE_FILE = Path(".kite_token_cache.json")
settings = get_settings()


def get_login_url() -> str:
    """Generate the Kite Connect OAuth login URL."""
    kite = KiteConnect(api_key=settings.kite_api_key)
    url = kite.login_url()
    logger.info("Visit the following URL to log in: %s", url)
    return url


def generate_session(request_token: str) -> dict:
    """
    Exchange the request_token (from OAuth redirect) for an access_token.
    Caches the token with today's date to support daily refresh detection.
    """
    kite = KiteConnect(api_key=settings.kite_api_key)
    try:
        session = kite.generate_session(
            request_token=request_token,
            api_secret=settings.kite_api_secret,
        )
        access_token = session["access_token"]
        _cache_token(access_token)
        logger.info("Session generated successfully. Token cached.")
        return session
    except Exception as exc:
        logger.error("Failed to generate session: %s", exc)
        raise


def get_kite_client() -> KiteConnect:
    """
    Return an authenticated KiteConnect instance.
    Loads access_token from cache if today's token exists,
    otherwise raises RuntimeError prompting re-login.
    """
    access_token = _load_cached_token()
    if not access_token:
        raise RuntimeError(
            "No valid access token found. "
            "Call get_login_url() and complete OAuth flow first."
        )
    kite = KiteConnect(api_key=settings.kite_api_key)
    kite.set_access_token(access_token)
    return kite


def _cache_token(access_token: str) -> None:
    """Persist today's access token to a local JSON file."""
    payload = {
        "access_token": access_token,
        "date": str(date.today()),
    }
    TOKEN_CACHE_FILE.write_text(json.dumps(payload))
    # Restrict file permissions: owner read/write only
    TOKEN_CACHE_FILE.chmod(0o600)


def _load_cached_token() -> Optional[str]:
    """
    Load the access token from cache if it was generated today.
    Returns None if the token is missing or stale (from a previous day).
    """
    if not TOKEN_CACHE_FILE.exists():
        return None
    try:
        payload = json.loads(TOKEN_CACHE_FILE.read_text())
        cached_date = payload.get("date")
        if cached_date == str(date.today()):
            return payload.get("access_token")
        logger.info("Cached token is from %s — re-login required.", cached_date)
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Could not parse token cache: %s", exc)
        return None


def invalidate_token() -> None:
    """Remove cached token (e.g., on logout or error)."""
    if TOKEN_CACHE_FILE.exists():
        TOKEN_CACHE_FILE.unlink()
        logger.info("Token cache invalidated.")
