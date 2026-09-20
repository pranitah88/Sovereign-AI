"""
Audit log repository — append-only writes and filtered queries.

Every security-relevant action in the system is recorded here.
"""

import json
import logging

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)


def write_log(
    action: str,
    outcome: str,
    *,
    user_id: int | None = None,
    username: str | None = None,
    target: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> int:
    """
    Append an audit log entry. Returns the log entry id.

    Args:
        action: What happened (e.g. 'login', 'chat_message', 'file_upload').
        outcome: 'success', 'failure', or 'denied'.
        user_id: The acting user's id (None for system events).
        username: Denormalized username for log durability.
        target: What was acted on (e.g. 'chat_session:42').
        details: Arbitrary JSON-serializable context.
        ip_address: Client IP address.
    """
    details_json = json.dumps(details) if details else None

    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_logs
                (user_id, username, action, target, outcome, details, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, action, target, outcome, details_json, ip_address),
        )
        return cursor.lastrowid


def query_logs(
    *,
    user_id: int | None = None,
    action: str | None = None,
    outcome: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """
    Query audit logs with optional filters.

    Dates should be ISO-8601 format (e.g. '2026-01-01T00:00:00Z').
    Returns newest-first.
    """
    conditions = []
    params = []

    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action is not None:
        conditions.append("action = ?")
        params.append(action)
    if outcome is not None:
        conditions.append("outcome = ?")
        params.append(outcome)
    if start_date is not None:
        conditions.append("created_at >= ?")
        params.append(start_date)
    if end_date is not None:
        conditions.append("created_at <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT * FROM audit_logs
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    conn = get_connection()
    rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        entry = dict(row)
        entry["details"] = json.loads(entry["details"]) if entry["details"] else None
        results.append(entry)
    return results


def count_logs(
    *,
    user_id: int | None = None,
    action: str | None = None,
    outcome: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """Count audit log entries matching the given filters."""
    conditions = []
    params = []

    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action is not None:
        conditions.append("action = ?")
        params.append(action)
    if outcome is not None:
        conditions.append("outcome = ?")
        params.append(outcome)
    if start_date is not None:
        conditions.append("created_at >= ?")
        params.append(start_date)
    if end_date is not None:
        conditions.append("created_at <= ?")
        params.append(end_date)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    conn = get_connection()
    row = conn.execute(f"SELECT COUNT(*) as cnt FROM audit_logs {where_clause}", params).fetchone()
    return row["cnt"]


# Aliases for backwards compatibility with tests and services
record_audit_log = write_log


def query_audit_logs(
    *,
    user_id: int | None = None,
    action: str | None = None,
    outcome: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Query audit logs and return (logs, total_count)."""
    logs = query_logs(
        user_id=user_id,
        action=action,
        outcome=outcome,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    total = count_logs(
        user_id=user_id,
        action=action,
        outcome=outcome,
        start_date=start_date,
        end_date=end_date,
    )
    return logs, total
