"""
Chat repository — CRUD for chat sessions and messages.

Provides persistent chat history that survives server restarts.
"""

import json
import logging
from datetime import datetime, timezone

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Chat Sessions ────────────────────────────────────────────────────────

def create_chat_session(user_id: int, title: str = "New Chat") -> dict:
    """Create a new chat session. Returns the session dict."""
    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
            (user_id, title),
        )
        session_id = cursor.lastrowid

    return get_chat_session(session_id)


def get_chat_session(session_id: int) -> dict | None:
    """Return a single chat session by id, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM chat_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def list_chat_sessions(user_id: int) -> list[dict]:
    """List all chat sessions for a user, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def rename_chat_session(session_id: int | dict, title: str) -> bool:
    """Rename a chat session. Returns True if found."""
    if isinstance(session_id, dict):
        session_id = session_id["id"]
    with transaction() as conn:
        cursor = conn.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )
        return cursor.rowcount > 0


def delete_chat_session(session_id: int | dict) -> bool:
    """Delete a chat session and its messages (cascaded). Returns True if found."""
    if isinstance(session_id, dict):
        session_id = session_id["id"]
    with transaction() as conn:
        cursor = conn.execute(
            "DELETE FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted chat session %d", session_id)
    return deleted


# Aliases for backwards compatibility with tests and callers
def create_session(user_id: int, title: str = "New Chat") -> int:
    """Create a chat session and return its id."""
    session = create_chat_session(user_id, title)
    return session["id"]


def get_session(session_id: int | dict) -> dict | None:
    if isinstance(session_id, dict):
        session_id = session_id["id"]
    return get_chat_session(session_id)


def get_messages(session_id: int | dict) -> list[dict]:
    if isinstance(session_id, dict):
        session_id = session_id["id"]
    return list_messages(session_id)


rename_session = rename_chat_session
delete_session = delete_chat_session


# ── Chat Messages ────────────────────────────────────────────────────────

def add_message(
    session_id: int | dict,
    role: str,
    content: str,
    *,
    model_id: str | None = None,
    tool_calls: list | None = None,
    sources: list | None = None,
    execution_ms: int | None = None,
) -> dict:
    """
    Append a message to a chat session. Returns the message dict.

    tool_calls and sources are stored as JSON strings.
    """
    if isinstance(session_id, dict):
        session_id = session_id["id"]
    tool_calls_json = json.dumps(tool_calls) if tool_calls else None
    sources_json = json.dumps(sources) if sources else None

    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_messages
                (session_id, role, content, model_id, tool_calls, sources, execution_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, model_id, tool_calls_json, sources_json, execution_ms),
        )
        message_id = cursor.lastrowid

        # Touch the chat session's updated_at.
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (_utc_now_iso(), session_id),
        )

    return get_message(message_id)


def _hydrate_message(msg: dict) -> dict:
    """Deserializes JSON fields and extracts deliverable and coding_result metadata."""
    if not msg:
        return msg

    msg["tool_calls"] = json.loads(msg["tool_calls"]) if msg.get("tool_calls") and isinstance(msg["tool_calls"], str) else msg.get("tool_calls")
    msg["sources"] = json.loads(msg["sources"]) if msg.get("sources") and isinstance(msg["sources"], str) else msg.get("sources")

    deliverable = None
    coding_res = None
    if msg.get("tool_calls") and isinstance(msg["tool_calls"], list):
        for tc in msg["tool_calls"]:
            if isinstance(tc, dict):
                if tc.get("tool", "").startswith("docgen_") and tc.get("status") == "success":
                    deliverable = tc.get("result")
                    break
                elif "type" in tc and "filename" in tc and tc.get("status") == "success":
                    deliverable = tc
                    break

        sb_attempts = [
            tc for tc in msg["tool_calls"]
            if isinstance(tc, dict) and (tc.get("tool") == "sandbox_execute" or "attempt" in tc)
        ]
        if sb_attempts:
            ver = next(
                (a for a in sb_attempts if a.get("status") == "verified" or a.get("exit_code") == 0),
                None,
            )
            tgt = ver if ver else sb_attempts[-1]
            c_status = (
                "VERIFIED" if (tgt.get("exit_code") == 0 or tgt.get("status") == "verified")
                else ("BLOCKED" if tgt.get("status") in ("error", "blocked") else "FAILED")
            )
            coding_res = {
                "task_type": "CODING",
                "model": msg.get("model_id") or "qwen2.5-coder:3b",
                "status": c_status,
                "code": tgt.get("code", ""),
                "stdout": tgt.get("stdout", ""),
                "stderr": tgt.get("stderr", ""),
                "exit_code": tgt.get("exit_code"),
                "correction_attempts": len(sb_attempts),
                "execution_details": tgt.get("sandbox_info"),
                "sandbox_info": tgt.get("sandbox_info"),
            }

    msg["deliverable"] = deliverable
    msg["coding_result"] = coding_res
    if coding_res and not msg.get("task_type"):
        msg["task_type"] = "CODING"
    return msg


def get_message(message_id: int) -> dict | None:
    """Return a single message by id, deserializing JSON fields."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM chat_messages WHERE id = ?",
        (message_id,),
    ).fetchone()

    if row is None:
        return None

    return _hydrate_message(dict(row))


def list_messages(session_id: int) -> list[dict]:
    """Return all messages in a session, in chronological order."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()

    return [_hydrate_message(dict(r)) for r in rows]


def get_session_with_messages(session_id: int) -> dict | None:
    """Return a chat session with all its messages embedded."""
    session = get_chat_session(session_id)
    if session is None:
        return None
    session["messages"] = list_messages(session_id)
    return session
