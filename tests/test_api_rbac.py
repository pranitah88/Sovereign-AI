"""
Integration tests for RBAC enforcement across FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.database.seed import seed_database
from backend.database.repositories import users as users_repo


@pytest.fixture(scope="module")
def client():
    seed_database()
    with TestClient(app) as c:
        yield c


def _get_token(client, username, password="changeme123"):
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    return res.json()["token"]


def test_admin_can_access_admin_users(client):
    token = _get_token(client, "admin")
    res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_auditor_denied_admin_users(client):
    # Ensure auditor exists
    if not users_repo.get_user_by_username("auditor_test"):
        users_repo.create_user("auditor_test", "changeme123", "Auditor", ["auditor"])

    token = _get_token(client, "auditor_test")
    res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403
    assert "Permission denied" in res.json()["detail"]


def test_auditor_can_access_audit_logs(client):
    if not users_repo.get_user_by_username("auditor_test"):
        users_repo.create_user("auditor_test", "changeme123", "Auditor", ["auditor"])

    token = _get_token(client, "auditor_test")
    res = client.get("/api/audit/logs", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200


def test_viewer_denied_sandbox_execution(client):
    if not users_repo.get_user_by_username("viewer_test"):
        users_repo.create_user("viewer_test", "changeme123", "Viewer", ["viewer"])

    token = _get_token(client, "viewer_test")
    res = client.post(
        "/api/sandbox/execute",
        json={"code": "print(1)"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403
    assert "Permission denied" in res.json()["detail"]
