"""
Tests for Role-Based Access Control (RBAC) enforcement and permission matrix.
"""

import pytest
from backend.auth.rbac import check_permission, get_allowed_roles, PERMISSION_MATRIX


def test_permission_matrix_structure():
    assert "chat" in PERMISSION_MATRIX
    assert "admin" in PERMISSION_MATRIX
    assert "user_manage" in PERMISSION_MATRIX
    assert "sandbox_execute" in PERMISSION_MATRIX


def test_admin_has_full_permissions():
    admin_user = {"username": "admin", "roles": ["administrator"]}
    for perm in PERMISSION_MATRIX:
        assert check_permission(admin_user, perm) is True, f"Admin should have {perm}"


def test_engineer_permissions():
    engineer_user = {"username": "engineer1", "roles": ["engineer"]}

    # Allowed
    assert check_permission(engineer_user, "chat") is True
    assert check_permission(engineer_user, "rag_query") is True
    assert check_permission(engineer_user, "sandbox_execute") is True
    assert check_permission(engineer_user, "docgen") is True

    # Denied
    assert check_permission(engineer_user, "user_manage") is False
    assert check_permission(engineer_user, "admin") is False
    assert check_permission(engineer_user, "audit_view") is False


def test_auditor_permissions():
    auditor_user = {"username": "auditor1", "roles": ["auditor"]}

    assert check_permission(auditor_user, "audit_view") is True
    assert check_permission(auditor_user, "chat") is True
    assert check_permission(auditor_user, "sandbox_execute") is False
    assert check_permission(auditor_user, "user_manage") is False


def test_viewer_restricted_permissions():
    viewer_user = {"username": "viewer1", "roles": ["viewer"]}

    assert check_permission(viewer_user, "chat") is True
    assert check_permission(viewer_user, "rag_query") is True
    assert check_permission(viewer_user, "sandbox_execute") is False
    assert check_permission(viewer_user, "document_manage") is False
    assert check_permission(viewer_user, "admin") is False


def test_invalid_permission_key():
    with pytest.raises(KeyError):
        get_allowed_roles("nonexistent_permission")
