"""
Unit and Integration tests for Knowledge Map service, RBAC access controls,
and the Knowledge Base API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.auth.session import create_session
from backend.database.repositories import users as users_repo
from backend.services.knowledge_map import (
    CATEGORY_METADATA,
    get_knowledge_map_data,
    invalidate_knowledge_map_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure cache is clean before tests."""
    invalidate_knowledge_map_cache()
    yield
    invalidate_knowledge_map_cache()


def test_knowledge_map_data_admin_clearance():
    """Admin with CONFIDENTIAL clearance should see all categories and documents."""
    km = get_knowledge_map_data(user_roles=["administrator"], user_clearance="CONFIDENTIAL")

    root = km.get("root", {})
    categories = km.get("categories", [])

    assert root["id"] == "root_kb"
    assert root["total_documents"] >= 185
    assert root["total_chunks"] >= 21000
    assert root["user_clearance"] == "CONFIDENTIAL"
    assert "Indexed" in root["indexed_status"]

    # Verify categories exist
    category_codes = [c["category_code"] for c in categories]
    assert "01_Refinery_Manufacturing" in category_codes
    assert "03_Safety_HSE" in category_codes
    assert "09_Confidential" in category_codes

    # Confidential category should be visible to admin
    conf_cat = next(c for c in categories if c["category_code"] == "09_Confidential")
    assert conf_cat["document_count"] >= 6
    assert any("P&ID" in d["filename"] or "p&id" in d["filename"].lower() for d in conf_cat["documents"])


def test_knowledge_map_data_public_rbac():
    """Viewer with PUBLIC clearance must NOT see confidential category or files."""
    km = get_knowledge_map_data(user_roles=["viewer"], user_clearance="PUBLIC")

    root = km.get("root", {})
    categories = km.get("categories", [])

    assert root["user_clearance"] == "PUBLIC"
    assert root["max_clearance_level"] == 1

    category_codes = [c["category_code"] for c in categories]
    # Confidential category MUST be completely absent for unauthorized users
    assert "09_Confidential" not in category_codes

    # Ensure no confidential files appear in any returned category
    for cat in categories:
        for doc in cat["documents"]:
            assert doc["classification"] != "CONFIDENTIAL"
            assert "P&ID" not in doc["filename"]
            assert "Vendor negotiations" not in doc["filename"]


def test_knowledge_map_unit_detection():
    """Refinery process units (CDU, HCU, PFCCU, P&ID) should be accurately tagged."""
    km = get_knowledge_map_data(user_roles=["administrator"], user_clearance="CONFIDENTIAL")

    all_docs = []
    for cat in km.get("categories", []):
        all_docs.extend(cat["documents"])

    # Check for unit detection
    units_found = set()
    for doc in all_docs:
        for u in doc.get("units_detected", []):
            units_found.add(u)

    assert "PFCCU" in units_found or "CDU" in units_found or "HCU" in units_found


def test_knowledge_map_endpoint_authenticated():
    """Test GET /api/documents/knowledge-map with a valid admin token."""
    client = TestClient(app)

    # Get or ensure admin user
    user = users_repo.get_user_by_username("admin")
    if not user:
        user_id = users_repo.create_user("admin", "testpass123", ["administrator"], clearance="CONFIDENTIAL")
    else:
        user_id = user["id"]

    session = create_session(user_id=user_id)
    token = session["token"]

    response = client.get(
        "/api/documents/knowledge-map",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "root" in data
    assert "categories" in data
    assert data["root"]["total_documents"] >= 185


def test_corpus_endpoint_authenticated():
    """Test GET /api/documents/corpus endpoint returns flat list of authoritative documents."""
    client = TestClient(app)

    user = users_repo.get_user_by_username("admin")
    session = create_session(user_id=user["id"])
    token = session["token"]

    response = client.get(
        "/api/documents/corpus",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "documents" in data
    assert data["total"] >= 185
