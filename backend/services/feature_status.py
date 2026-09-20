"""
Feature status service.

Maps every PRD feature to its current status, backed by the
feature_status SQLite table.
"""

import logging

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)


def get_all_features() -> dict:
    """
    Return all features grouped by category.

    Returns:
        {"category_name": [{"feature_key": ..., "display_name": ..., "status": ...}, ...]}
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT feature_key, display_name, category, status, notes, updated_at "
        "FROM feature_status ORDER BY category, feature_key"
    ).fetchall()

    grouped = {}
    for row in rows:
        r = dict(row)
        cat = r.pop("category")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(r)

    return grouped


def get_feature_status(feature_key: str) -> str | None:
    """Return the status of a single feature, or None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT status FROM feature_status WHERE feature_key = ?",
        (feature_key,),
    ).fetchone()
    return row["status"] if row else None


def update_feature_status(feature_key: str, status: str, notes: str | None = None) -> bool:
    """Update a feature's status. Returns True if found."""
    fields = ["status = ?"]
    values = [status]

    if notes is not None:
        fields.append("notes = ?")
        values.append(notes)

    values.append(feature_key)

    with transaction() as conn:
        cursor = conn.execute(
            f"UPDATE feature_status SET {', '.join(fields)} WHERE feature_key = ?",
            values,
        )
        return cursor.rowcount > 0


def is_feature_available(feature_key: str) -> bool:
    """Check if a feature is implemented and available."""
    status = get_feature_status(feature_key)
    return status == "implemented"
