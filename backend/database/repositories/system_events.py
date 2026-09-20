"""
System events repository — logs system-level events like startup, shutdown,
model loading, and errors.
"""

import json
import logging

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)


def log_event(
    event_type: str,
    message: str,
    *,
    severity: str = "info",
    details: dict | None = None,
) -> int:
    """
    Log a system event. Returns the event id.

    Args:
        event_type: Category (e.g. 'startup', 'shutdown', 'model_load', 'error').
        message: Human-readable description.
        severity: One of 'info', 'warning', 'error', 'critical'.
        details: Optional JSON-serializable context.
    """
    details_json = json.dumps(details) if details else None

    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO system_events (event_type, severity, message, details)
            VALUES (?, ?, ?, ?)
            """,
            (event_type, severity, message, details_json),
        )
        return cursor.lastrowid


def query_events(
    *,
    event_type: str | None = None,
    severity: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Query system events with optional filters. Returns newest first."""
    conditions = []
    params = []

    if event_type is not None:
        conditions.append("event_type = ?")
        params.append(event_type)
    if severity is not None:
        conditions.append("severity = ?")
        params.append(severity)
    if start_date is not None:
        conditions.append("created_at >= ?")
        params.append(start_date)
    if end_date is not None:
        conditions.append("created_at <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    conn = get_connection()
    rows = conn.execute(
        f"SELECT * FROM system_events {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    results = []
    for row in rows:
        event = dict(row)
        event["details"] = json.loads(event["details"]) if event["details"] else None
        results.append(event)
    return results
