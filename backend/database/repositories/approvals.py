"""
Action Approvals repository — tracks human-in-the-loop approvals for high-risk actions.

Enforces:
1. Classification of actions into LOW, MEDIUM, and HIGH risk.
2. Mandatory human approval for HIGH risk actions before execution.
3. AI is explicitly prohibited from self-approving any high-risk action.
"""

import json
import logging
from typing import Any

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)

ACTION_RISK_CLASSIFICATION = {
    # LOW RISK
    "read_document": "LOW",
    "search_kb": "LOW",
    "summarize": "LOW",
    "chat": "LOW",
    # MEDIUM RISK
    "generate_document": "MEDIUM",
    "modify_workspace": "MEDIUM",
    # HIGH RISK
    "delete_file": "HIGH",
    "modify_controlled_document": "HIGH",
    "modify_database": "HIGH",
    "execute_production_action": "HIGH",
    "system_config_change": "HIGH",
}


def classify_action_risk(action_type: str) -> str:
    """Return risk classification: LOW, MEDIUM, or HIGH."""
    return ACTION_RISK_CLASSIFICATION.get(action_type.lower(), "HIGH")


def requires_human_approval(action_type: str) -> bool:
    """True if action is HIGH risk and requires human approval."""
    return classify_action_risk(action_type) == "HIGH"


def ensure_approvals_schema() -> None:
    """Ensure action_approvals table exists."""
    with transaction() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_approvals (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                requesting_user     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                requesting_username TEXT NOT NULL,
                action_type         TEXT NOT NULL,
                risk_level          TEXT NOT NULL CHECK(risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
                affected_resource   TEXT NOT NULL,
                proposed_payload    TEXT,
                status              TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
                decision_reason     TEXT,
                approving_user      INTEGER REFERENCES users(id) ON DELETE SET NULL,
                approving_username  TEXT,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                decided_at          TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_approvals_status ON action_approvals(status)")


def propose_action(
    requesting_user_id: int,
    requesting_username: str,
    action_type: str,
    affected_resource: str,
    proposed_payload: dict | None = None,
) -> dict:
    """
    Propose an action. If HIGH risk, recorded as 'pending' for human review.
    If LOW/MEDIUM risk, can proceed directly.
    """
    ensure_approvals_schema()
    risk = classify_action_risk(action_type)
    payload_json = json.dumps(proposed_payload or {})

    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO action_approvals
                (requesting_user, requesting_username, action_type, risk_level, affected_resource, proposed_payload, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (requesting_user_id, requesting_username, action_type, risk, affected_resource, payload_json),
        )
        approval_id = cursor.lastrowid

    logger.info("Created approval proposal #%d for '%s' (risk=%s)", approval_id, action_type, risk)
    return {
        "approval_id": approval_id,
        "action_type": action_type,
        "risk_level": risk,
        "requires_approval": risk == "HIGH",
        "status": "pending",
        "affected_resource": affected_resource,
    }


def list_pending_approvals() -> list[dict]:
    """List all pending action approvals requiring human decision."""
    ensure_approvals_schema()
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, requesting_user, requesting_username, action_type, risk_level,
               affected_resource, proposed_payload, status, created_at
        FROM action_approvals
        WHERE status = 'pending'
        ORDER BY created_at ASC
        """
    ).fetchall()

    result = []
    for r in rows:
        item = dict(r)
        try:
            item["proposed_payload"] = json.loads(item.get("proposed_payload") or "{}")
        except Exception:
            pass
        result.append(item)
    return result


def get_approval(approval_id: int) -> dict | None:
    """Retrieve an approval record by id."""
    ensure_approvals_schema()
    conn = get_connection()
    row = conn.execute("SELECT * FROM action_approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    try:
        item["proposed_payload"] = json.loads(item.get("proposed_payload") or "{}")
    except Exception:
        pass
    return item


def decide_approval(
    approval_id: int,
    decision: str,
    approving_user_id: int,
    approving_username: str,
    reason: str = "",
) -> bool:
    """
    Approve or reject an action proposal.
    AI is strictly prohibited from approving its own action.
    """
    ensure_approvals_schema()
    if decision not in ("approved", "rejected"):
        raise ValueError(f"Invalid decision '{decision}'. Must be 'approved' or 'rejected'.")

    approval = get_approval(approval_id)
    if not approval:
        raise ValueError(f"Approval proposal #{approval_id} not found.")

    if approval["status"] != "pending":
        raise ValueError(f"Approval proposal #{approval_id} has already been decided ({approval['status']}).")

    # Anti-self-approval rule: AI is strictly forbidden from approving high-risk proposals
    if approving_username.lower() in ("ai", "system", "agent", "assistant"):
        raise ValueError("AI agents cannot approve high-risk action proposals. Human approval required.")

    if approval["requesting_user"] == approving_user_id:
        raise ValueError("Users cannot approve their own high-risk action proposals. Independent human review required.")

    with transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE action_approvals
            SET status = ?, approving_user = ?, approving_username = ?, decision_reason = ?,
                decided_at = (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            WHERE id = ?
            """,
            (decision, approving_user_id, approving_username, reason, approval_id),
        )
        return cursor.rowcount > 0
