"""
Session management — token generation, creation, validation, and expiry.

Sessions are stored in SQLite and validated on every request via
the FastAPI dependency in `dependencies.py`.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone

from backend.database.repositories import sessions as session_repo

logger = logging.getLogger(__name__)

# Session token length (bytes before URL-safe base64 encoding).
TOKEN_BYTES = 48

# Default session lifetime.
SESSION_LIFETIME_HOURS = 24


def create_session(user_id: int, lifetime_hours: int | None = None) -> dict:
    """
    Create a new authenticated session for a user.
    Returns dict: { session_id, token, expires_at }.
    """
    hours = lifetime_hours if lifetime_hours is not None else SESSION_LIFETIME_HOURS
    session = session_repo.create_session(user_id, hours)
    logger.info("Session created for user_id=%d", user_id)
    return session


def validate_token(token: str) -> dict | None:
    """
    Validate a session token.
    Returns the session dict if valid, None if invalid or expired.
    """
    return session_repo.validate_session(token)


def invalidate_token(token: str) -> None:
    """Invalidate (logout) a session by token."""
    session_repo.expire_session_by_token(token)


def invalidate_all_user_sessions(user_id: int) -> int:
    """Invalidate all sessions for a user (e.g. password change). Returns count."""
    return session_repo.expire_all_user_sessions(user_id)


def cleanup() -> int:
    """Remove old expired sessions. Returns count deleted."""
    return session_repo.cleanup_expired_sessions()
