"""
Integration tests for authentication API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.database.seed import seed_database


@pytest.fixture(scope="module")
def client():
    seed_database()
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_login_success(client):
    res = client.post("/api/auth/login", json={"username": "admin", "password": "1234567890"})
    assert res.status_code == 200
    data = res.json()
    assert "token" in data
    assert data["user"]["username"] == "admin"
    assert "administrator" in data["user"]["roles"]


def test_login_wrong_password(client):
    res = client.post("/api/auth/login", json={"username": "admin", "password": "WrongPassword"})
    assert res.status_code == 401
    assert "Invalid credentials" in res.json()["detail"]


def test_get_current_user_authenticated(client):
    login_res = client.post("/api/auth/login", json={"username": "admin", "password": "1234567890"})
    token = login_res.json()["token"]

    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    user = res.json()
    assert user["username"] == "admin"


def test_get_current_user_unauthorized(client):
    client.cookies.clear()
    res = client.get("/api/auth/me")
    assert res.status_code == 401
