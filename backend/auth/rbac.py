"""
Role-Based Access Control (RBAC) enforcement.

Provides decorators and a permission matrix mapping roles to API endpoint
groups. Used alongside `dependencies.py` for route-level protection.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status

from backend.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

# ── Role aliases & normalization ─────────────────────────────────────────
ROLE_ALIASES = {
    "admin": "administrator",
    "administrator": "administrator",
    "engineer": "engineer",
    "reviewer": "reviewer",
    "viewer": "viewer",
    "auditor": "auditor",
    "document_manager": "document_manager",
}


def normalize_roles(roles: list[str] | set[str]) -> set[str]:
    """Normalize role names (case-insensitive and map aliases like admin -> administrator)."""
    normalized = set()
    for r in roles:
        r_clean = str(r).strip().lower()
        normalized.add(ROLE_ALIASES.get(r_clean, r_clean))
    return normalized


# ── Permission matrix ────────────────────────────────────────────────────
# Maps endpoint groups -> which roles are allowed.
# The `require_role` dependency in `dependencies.py` does the actual check;
# this matrix is the single source of truth for RBAC policy.

PERMISSION_MATRIX: dict[str, set[str]] = {
    # Chat — all authenticated users
    "chat": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},

    # Voice assistant — all authenticated users
    "voice": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},

    # RAG query — all authenticated users
    "rag_query": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},

    # Document management — upload, index, delete
    "document_manage": {"administrator", "engineer", "document_manager"},

    # Agent invocation — multi-step tasks
    "agent_invoke": {"administrator", "engineer", "reviewer"},

    # Code sandbox — execution (strictly engineer & admin)
    "sandbox_execute": {"administrator", "engineer"},

    # Document generation — docx/xlsx/pptx
    "docgen": {"administrator", "engineer"},

    # Model registry — view
    "model_view": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},

    # Model registry — modify (add, enable/disable)
    "model_manage": {"administrator"},

    # User management
    "user_manage": {"administrator"},

    # Audit logs — view
    "audit_view": {"administrator", "auditor"},

    # System status — view
    "status_view": {"administrator", "engineer", "reviewer", "auditor"},

    # Feature status — view (public to all authenticated)
    "feature_status": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},

    # Admin panel
    "admin": {"administrator"},

    # Human approval workflow for high-risk actions
    "approval_view": {"administrator", "reviewer"},
    "approval_manage": {"administrator", "reviewer"},
}


def get_allowed_roles(permission_key: str) -> set[str]:
    """
    Return the set of roles allowed for a given permission key.
    Raises KeyError if the permission key is not defined.
    """
    if permission_key not in PERMISSION_MATRIX:
        raise KeyError(f"Unknown permission key: '{permission_key}'")
    return PERMISSION_MATRIX[permission_key]


def require_permission(permission_key: str):
    """
    FastAPI dependency factory that checks the RBAC permission matrix.

    Usage:
        @router.post("/sandbox/execute",
                      dependencies=[Depends(require_permission("sandbox_execute"))])
        def execute_code(): ...
    """
    allowed_roles = get_allowed_roles(permission_key)

    async def _enforce(
        user: Annotated[dict, Depends(get_current_user)],
    ) -> dict:
        user_roles = normalize_roles(user.get("roles", []))

        if not user_roles & allowed_roles:
            logger.warning(
                "RBAC denied: user '%s' (roles=%s) -> permission '%s' (requires %s)",
                user["username"],
                user_roles,
                permission_key,
                allowed_roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: requires one of {sorted(allowed_roles)}",
            )

        return user

    return _enforce


def check_permission(user: dict, permission_key: str) -> bool:
    """
    Programmatic permission check (non-dependency).
    Returns True if the user has the required role, False otherwise.
    """
    allowed_roles = get_allowed_roles(permission_key)
    user_roles = normalize_roles(user.get("roles", []))
    return bool(user_roles & allowed_roles)
