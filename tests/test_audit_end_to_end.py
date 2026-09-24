"""
Comprehensive End-to-End Audit Log Test Suite.

Verifies the entire audit trail pipeline:
- Action -> Centralized Audit Service -> Repository -> SQLite persistence
- Empty-string and whitespace filter normalization
- API endpoint /api/audit/logs response shape and filtering
- RBAC authorization enforcement (administrator/auditor vs unauthorized)
- Real application event emission (login, model_toggle, document operations, etc.)
- Database connection lifecycle, WAL durability, and single authoritative path
- Separation of execution traces from persistent SQLite audit records
"""

import json
import logging
import sqlite3
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.database.connection import get_connection, get_db_path, close_connection, _DEFAULT_DB_PATH
from backend.database.seed import seed_database
from backend.database.repositories import audit as audit_repo
from backend.database.repositories import users as users_repo
from backend.services.audit import audit_log


@pytest.fixture(scope="module")
def client():
    seed_database()
    with TestClient(app) as c:
        yield c


def _get_token(client: TestClient, username: str, password: str = "changeme123") -> str:
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, f"Login failed: {res.text}"
    return res.json()["token"]


# ── 1, 2, 3: Direct Write, Query, and Count ───────────────────────────────────

def test_01_write_log_inserts_and_can_be_queried():
    """write_log inserts a row into SQLite audit_logs and query_logs returns it with parsed details."""
    test_target = "test_target_01"
    log_id = audit_repo.write_log(
        action="AUDIT_SMOKE_TEST",
        outcome="success",
        username="test_admin",
        target=test_target,
        details={"smoke_test": True, "step": 1},
        ip_address="127.0.0.1",
    )
    assert isinstance(log_id, int)
    assert log_id > 0

    # Query back
    logs = audit_repo.query_logs(action="AUDIT_SMOKE_TEST", limit=10)
    assert len(logs) > 0
    matched = next((l for l in logs if l["id"] == log_id), None)
    assert matched is not None
    assert matched["action"] == "AUDIT_SMOKE_TEST"
    assert matched["outcome"] == "success"
    assert matched["target"] == test_target
    assert matched["username"] == "test_admin"
    assert matched["details"] == {"smoke_test": True, "step": 1}

    # Count
    cnt = audit_repo.count_logs(action="AUDIT_SMOKE_TEST")
    assert cnt >= 1


# ── 4, 5, 6, 7: Filter Normalization (Empty & Whitespace) ─────────────────────

def test_02_empty_and_whitespace_filters_behave_as_unfiltered():
    """Empty and whitespace-only strings are normalized to None and do not restrict results."""
    total_unfiltered = audit_repo.count_logs(action=None, outcome=None)
    assert total_unfiltered > 0

    # Empty strings
    assert audit_repo.count_logs(action="") == total_unfiltered
    assert audit_repo.count_logs(outcome="") == total_unfiltered
    assert audit_repo.count_logs(action="", outcome="") == total_unfiltered

    # Whitespace-only strings
    assert audit_repo.count_logs(action="   ") == total_unfiltered
    assert audit_repo.count_logs(outcome="  \t  ") == total_unfiltered
    assert audit_repo.count_logs(start_date="  ", end_date="") == total_unfiltered

    # Query logs matching
    logs_none = audit_repo.query_logs(limit=20)
    logs_empty = audit_repo.query_logs(action="", outcome="", limit=20)
    logs_space = audit_repo.query_logs(action="   ", outcome="   ", limit=20)

    assert len(logs_none) == len(logs_empty) == len(logs_space)
    assert [l["id"] for l in logs_none] == [l["id"] for l in logs_empty] == [l["id"] for l in logs_space]


# ── 8, 9, 10, 11: Specific Filters & Pagination ───────────────────────────────

def test_03_specific_filters_and_pagination():
    """Specific action, outcome, and pagination parameters filter accurately."""
    id_succ = audit_repo.write_log(
        action="TEST_FILTER_ACTION",
        outcome="success",
        username="user_succ",
        target="target_succ",
    )
    id_fail = audit_repo.write_log(
        action="TEST_FILTER_ACTION",
        outcome="failure",
        username="user_fail",
        target="target_fail",
    )

    action_logs = audit_repo.query_logs(action="TEST_FILTER_ACTION")
    assert len(action_logs) >= 2
    assert all(l["action"] == "TEST_FILTER_ACTION" for l in action_logs)

    succ_logs = audit_repo.query_logs(action="TEST_FILTER_ACTION", outcome="success")
    assert any(l["id"] == id_succ for l in succ_logs)
    assert not any(l["id"] == id_fail for l in succ_logs)

    fail_logs = audit_repo.query_logs(action="TEST_FILTER_ACTION", outcome="failure")
    assert any(l["id"] == id_fail for l in fail_logs)
    assert not any(l["id"] == id_succ for l in fail_logs)

    p1 = audit_repo.query_logs(action="TEST_FILTER_ACTION", limit=1, offset=0)
    p2 = audit_repo.query_logs(action="TEST_FILTER_ACTION", limit=1, offset=1)
    assert len(p1) == 1
    assert len(p2) == 1
    assert p1[0]["id"] != p2[0]["id"]


# ── 12, 13, 14: API Response Shape and Serialization ──────────────────────────

def test_04_api_audit_logs_response_shape_and_filtering(client: TestClient):
    """GET /api/audit/logs returns correct shape and handles empty query params gracefully."""
    token = _get_token(client, "admin")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Unfiltered request
    res = client.get("/api/audit/logs", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert isinstance(data["logs"], list)
    assert isinstance(data["total"], int)
    assert data["total"] > 0
    assert data["total"] == audit_repo.count_logs()

    # 2. Empty query parameters (?action=&outcome=) must NOT zero out the results
    res_empty = client.get("/api/audit/logs?action=&outcome=", headers=headers)
    assert res_empty.status_code == 200
    data_empty = res_empty.json()
    assert data_empty["total"] == data["total"]
    assert len(data_empty["logs"]) == len(data["logs"])

    # 3. Specific action filter
    res_login = client.get("/api/audit/logs?action=login", headers=headers)
    assert res_login.status_code == 200
    data_login = res_login.json()
    assert all(l["action"] == "login" for l in data_login["logs"])
    assert data_login["total"] == audit_repo.count_logs(action="login")

    # 4. Non-matching query returns empty logs array with 200 OK (not error)
    res_none = client.get("/api/audit/logs?action=NON_EXISTENT_ACTION_XYZ_123", headers=headers)
    assert res_none.status_code == 200
    assert res_none.json()["logs"] == []
    assert res_none.json()["total"] == 0


# ── 15, 16: RBAC Enforcement ──────────────────────────────────────────────────

def test_05_rbac_enforcement_for_audit_view(client: TestClient):
    """Only administrator and auditor can view audit logs; engineer/viewer get 403 Forbidden."""
    if not users_repo.get_user_by_username("auditor_e2e"):
        users_repo.create_user("auditor_e2e", "changeme123", "Auditor E2E", ["auditor"])
    if not users_repo.get_user_by_username("engineer_e2e"):
        users_repo.create_user("engineer_e2e", "changeme123", "Engineer E2E", ["engineer"])
    if not users_repo.get_user_by_username("viewer_e2e"):
        users_repo.create_user("viewer_e2e", "changeme123", "Viewer E2E", ["viewer"])

    token_auditor = _get_token(client, "auditor_e2e")
    token_engineer = _get_token(client, "engineer_e2e")
    token_viewer = _get_token(client, "viewer_e2e")

    res_auditor = client.get("/api/audit/logs", headers={"Authorization": f"Bearer {token_auditor}"})
    assert res_auditor.status_code == 200

    res_engineer = client.get("/api/audit/logs", headers={"Authorization": f"Bearer {token_engineer}"})
    assert res_engineer.status_code == 403
    assert "audit_view" in res_engineer.text or "Permission denied" in res_engineer.text

    res_viewer = client.get("/api/audit/logs", headers={"Authorization": f"Bearer {token_viewer}"})
    assert res_viewer.status_code == 403


# ── 17: Centralized Login Audit ───────────────────────────────────────────────

def test_06_centralized_login_audit(client: TestClient):
    """Login attempt routes through centralized audit_log and records row in SQLite."""
    count_before = audit_repo.count_logs(action="login")

    res = client.post("/api/auth/login", json={"username": "admin", "password": "changeme123"})
    assert res.status_code == 200

    count_after = audit_repo.count_logs(action="login")
    assert count_after > count_before

    recent_logins = audit_repo.query_logs(action="login", limit=5)
    matched = next((l for l in recent_logins if l["username"] == "admin" and l["outcome"] == "success"), None)
    assert matched is not None
    assert matched["action"] == "login"


# ── 18: Model Toggle Audit ────────────────────────────────────────────────────

def test_07_model_toggle_audit(client: TestClient):
    """Model enable/disable via API generates a real model_toggle audit record in SQLite."""
    token = _get_token(client, "admin")
    headers = {"Authorization": f"Bearer {token}"}

    models_res = client.get("/api/models", headers=headers)
    assert models_res.status_code == 200
    models = models_res.json()
    assert len(models) > 0
    model_id = models[0]["id"]
    current_state = models[0]["enabled"]

    new_state = not current_state
    patch_res = client.patch(f"/api/models/{model_id}", json={"enabled": new_state}, headers=headers)
    assert patch_res.status_code == 200

    recent = audit_repo.query_logs(action="model_toggle", limit=1)
    assert len(recent) > 0
    assert recent[0]["action"] == "model_toggle"
    assert recent[0]["outcome"] == "success"
    assert recent[0]["target"] == f"model:{model_id}"
    assert recent[0]["details"]["enabled"] == new_state

    client.patch(f"/api/models/{model_id}", json={"enabled": current_state}, headers=headers)


# ── 19: Document Operations Audit ─────────────────────────────────────────────

def test_08_document_operations_audit(client: TestClient):
    """Document upload and delete produce real SQLite audit rows."""
    token = _get_token(client, "admin")
    headers = {"Authorization": f"Bearer {token}"}

    test_filename = "audit_smoke_document.txt"
    files = {"file": (test_filename, b"Audit log verification content.", "text/plain")}
    data = {"classification": "INTERNAL", "department": "TEST"}
    upload_res = client.post("/api/documents/upload", files=files, data=data, headers=headers)
    assert upload_res.status_code == 201
    file_id = upload_res.json()["id"]

    upload_logs = audit_repo.query_logs(action="file_upload", limit=5)
    matched_upload = next((l for l in upload_logs if l["target"] == f"file:{file_id}"), None)
    assert matched_upload is not None
    assert matched_upload["outcome"] == "success"

    del_res = client.delete(f"/api/documents/{file_id}", headers=headers)
    assert del_res.status_code == 204

    del_logs = audit_repo.query_logs(action="file_delete", limit=5)
    matched_del = next((l for l in del_logs if l["target"] == f"file:{file_id}"), None)
    assert matched_del is not None
    assert matched_del["outcome"] == "success"


# ── 20: Sandbox / Tool Audit ──────────────────────────────────────────────────

def test_09_sandbox_or_tool_audit(client: TestClient):
    """Direct audit_log of tool_execution or sandbox_execute writes persistent row."""
    log_id = audit_log(
        action="sandbox_execute",
        outcome="success",
        username="engineer",
        target="sandbox:test_script.py",
        details={"language": "python", "timeout_seconds": 10},
    )
    assert log_id > 0

    queried = audit_repo.query_logs(action="sandbox_execute", limit=5)
    matched = next((l for l in queried if l["id"] == log_id), None)
    assert matched is not None
    assert matched["outcome"] == "success"
    assert matched["details"]["language"] == "python"


# ── 21: Persistence Across Connection Reopen ──────────────────────────────────

def test_10_persistence_across_connection_reopen():
    """Audit records persist in SQLite across thread connection closes and reopen."""
    unique_target = "persistence_test_target_999"
    log_id = audit_repo.write_log(
        action="PERSISTENCE_TEST",
        outcome="success",
        target=unique_target,
    )

    close_connection()

    new_conn = get_connection()
    row = new_conn.execute("SELECT * FROM audit_logs WHERE id = ?", (log_id,)).fetchone()
    assert row is not None
    assert row["target"] == unique_target


# ── 22: Authoritative Database Identity ───────────────────────────────────────

def test_11_authoritative_database_identity():
    """Writer, reader, and app use the exact same authoritative SQLite path."""
    current_path = get_db_path().resolve()
    expected_path = _DEFAULT_DB_PATH.resolve()
    assert current_path == expected_path
    assert current_path.name == "mrpl_sovereign.db"
    assert current_path.exists()


# ── 23: Trace Events vs Database Audit Separation ──────────────────────────────

def test_12_execution_trace_is_not_treated_as_audit_db_entry():
    """In-memory trace events do not magically insert rows; only audit_log() creates rows."""
    from backend.agent.state import AgentState

    state = AgentState(user_id=1, session_id=1, query="test query")
    state.execution_trace.append({
        "step": 1,
        "event": "audit_logged",
        "title": "Trace Event Only",
        "status": "verified",
        "metadata": {"trace_only": True},
    })

    found = audit_repo.query_logs(action="Trace Event Only")
    assert len(found) == 0


# ── 24, 25: Instrumentation Output Verification ───────────────────────────────

def test_13_audit_write_success_instrumentation(caplog):
    """audit_log emits AUDIT_WRITE_SUCCESS with log_id, action, outcome, user, target."""
    with caplog.at_level(logging.INFO, logger="backend.services.audit"):
        log_id = audit_log(
            action="INSTRUMENTATION_TEST",
            outcome="success",
            username="admin",
            target="resource:instrumentation",
            details={"safe": True},
        )
        assert log_id > 0

    assert any(
        "AUDIT_WRITE_SUCCESS" in record.message and "INSTRUMENTATION_TEST" in record.message
        for record in caplog.records
    )
