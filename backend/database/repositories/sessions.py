"""
Session repository — create, validate, expire, and cleanup auth sessions.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)

# Default session lifetime: 24 hours.
DEFAULT_SESSION_LIFETIME_HOURS = 24


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def create_session(user_id: int, lifetime_hours: int = DEFAULT_SESSION_LIFETIME_HOURS) -> dict:
    """
    Create a new session for the given user.
    Returns dict with 'token', 'expires_at', and 'session_id'.
    """
    token = secrets.token_urlsafe(48)
    now = _utc_now()
    expires_at = now + timedelta(hours=lifetime_hours)

    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, _iso(expires_at)),
        )
        session_id = cursor.lastrowid

    logger.info("Created session %d for user_id=%d, expires=%s", session_id, user_id, _iso(expires_at))
    return {
        "session_id": session_id,
        "token": token,
        "expires_at": _iso(expires_at),
    }


def validate_session(token: str) -> dict | None:
    """
    Validate a session token. Returns the session row (as dict) if valid
    and not expired, or None if invalid/expired.

    Automatically deactivates expired sessions.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE token = ? AND is_active = 1",
        (token,),
    ).fetchone()

    if row is None:
        return None

    session = dict(row)
    expires_at = datetime.strptime(session["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    if _utc_now() > expires_at:
        # Session has expired — deactivate it.
        expire_session(session["id"])
        return None

    return session


def expire_session(session_id: int) -> None:
    """Mark a session as inactive."""
    with transaction() as conn:
        conn.execute(
            "UPDATE sessions SET is_active = 0 WHERE id = ?",
            (session_id,),
        )
    logger.info("Expired session %d", session_id)


def expire_session_by_token(token: str) -> None:
    """Expire a session by its token (used for logout)."""
    with transaction() as conn:
        conn.execute(
            "UPDATE sessions SET is_active = 0 WHERE token = ?",
            (token,),
        )
    logger.debug("Expired session by token")


def expire_all_user_sessions(user_id: int) -> int:
    """Expire all active sessions for a user. Returns count expired."""
    with transaction() as conn:
        cursor = conn.execute(
            "UPDATE sessions SET is_active = 0 WHERE user_id = ? AND is_active = 1",
            (user_id,),
        )
        count = cursor.rowcount
    logger.info("Expired %d sessions for user_id=%d", count, user_id)
    return count


def cleanup_expired_sessions() -> int:
    """
    Remove sessions that have been expired for more than 7 days.
    Returns the number of sessions deleted.
    """
    cutoff = _utc_now() - timedelta(days=7)
    with transaction() as conn:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE is_active = 0 AND expires_at < ?",
            (_iso(cutoff),),
        )
        count = cursor.rowcount
    if count > 0:
        logger.info("Cleaned up %d expired sessions", count)
    return count
