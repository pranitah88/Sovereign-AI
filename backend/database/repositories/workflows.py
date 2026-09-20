"""
Workflow run repository — tracks n8n workflow executions.
"""

import json
import logging

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)


def record_workflow_run(
    workflow_name: str,
    trigger_source: str,
    *,
    triggered_by: int | None = None,
    input_data: dict | None = None,
    n8n_execution_id: str | None = None,
) -> int:
    """Record a triggered workflow run. Returns the run id."""
    input_json = json.dumps(input_data) if input_data else None

    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO workflow_runs
                (workflow_name, n8n_execution_id, trigger_source, input_data, triggered_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            (workflow_name, n8n_execution_id, trigger_source, input_json, triggered_by),
        )
        return cursor.lastrowid


def update_workflow_run(
    run_id: int,
    *,
    status: str | None = None,
    n8n_execution_id: str | None = None,
    output_data: dict | None = None,
    error_info: str | None = None,
    completed_at: str | None = None,
) -> bool:
    """Update a workflow run's status/output. Returns True if found."""
    fields = []
    values = []

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if n8n_execution_id is not None:
        fields.append("n8n_execution_id = ?")
        values.append(n8n_execution_id)
    if output_data is not None:
        fields.append("output_data = ?")
        values.append(json.dumps(output_data))
    if error_info is not None:
        fields.append("error_info = ?")
        values.append(error_info)
    if completed_at is not None:
        fields.append("completed_at = ?")
        values.append(completed_at)

    if not fields:
        return False

    values.append(run_id)
    with transaction() as conn:
        cursor = conn.execute(
            f"UPDATE workflow_runs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        return cursor.rowcount > 0


def get_workflow_run(run_id: int) -> dict | None:
    """Return a workflow run by id."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM workflow_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    run = dict(row)
    run["input_data"] = json.loads(run["input_data"]) if run["input_data"] else None
    run["output_data"] = json.loads(run["output_data"]) if run["output_data"] else None
    return run


def list_workflow_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    """List workflow runs, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM workflow_runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()

    results = []
    for row in rows:
        run = dict(row)
        run["input_data"] = json.loads(run["input_data"]) if run["input_data"] else None
        run["output_data"] = json.loads(run["output_data"]) if run["output_data"] else None
        results.append(run)
    return results
