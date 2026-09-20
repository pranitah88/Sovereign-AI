"""
Centralized audit logging service.

Provides a single function callable from any module to record
security-relevant actions into the audit_logs table.
"""

import logging

from backend.database.repositories import audit as audit_repo

logger = logging.getLogger(__name__)

# Standardized security audit event types
LOGIN_SUCCESS = "LOGIN_SUCCESS"
LOGIN_FAILURE = "LOGIN_FAILURE"
UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
DOCUMENT_REJECTED = "DOCUMENT_REJECTED"
DOCUMENT_DELETE = "DOCUMENT_DELETE"
RAG_ACCESS = "RAG_ACCESS"
RAG_ACCESS_DENIED = "RAG_ACCESS_DENIED"
TOOL_CALL = "TOOL_CALL"
TOOL_DENIED = "TOOL_DENIED"
CODE_EXECUTION = "CODE_EXECUTION"
CODE_EXECUTION_BLOCKED = "CODE_EXECUTION_BLOCKED"
APPROVAL_CREATED = "APPROVAL_CREATED"
APPROVAL_APPROVED = "APPROVAL_APPROVED"
APPROVAL_REJECTED = "APPROVAL_REJECTED"
PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
MODEL_SELECTED = "MODEL_SELECTED"
MODEL_REJECTED = "MODEL_REJECTED"
SECURITY_STATUS_CHECK = "SECURITY_STATUS_CHECK"
DOCUMENT_GENERATED = "DOCUMENT_GENERATED"
DOCUMENT_GENERATION_FAILED = "DOCUMENT_GENERATION_FAILED"



def audit_log(
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
    Record an audit log entry. Returns the log entry id.

    This is the single entry point for all audit logging in the system.
    Called from API endpoints, agent tools, file operations, and admin actions.

    Args:
        action: What happened (e.g. 'login', 'chat_message', 'file_upload',
                'sandbox_execute', 'user_create', 'model_toggle').
        outcome: 'success', 'failure', or 'denied'.
        user_id: The acting user's id (None for system-level events).
        username: Denormalized username (survives user deletion).
        target: What was acted on (e.g. 'chat_session:42', 'file:report.pdf').
        details: Arbitrary dict with extra context.
        ip_address: Client IP address.
    """
    log_id = audit_repo.write_log(
        action=action,
        outcome=outcome,
        user_id=user_id,
        username=username,
        target=target,
        details=details,
        ip_address=ip_address,
    )

    logger.info(
        "AUDIT | action=%s outcome=%s user=%s target=%s",
        action,
        outcome,
        username or user_id or "system",
        target or "-",
    )

    return log_id
