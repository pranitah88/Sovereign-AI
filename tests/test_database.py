"""
Tests for SQLite repositories: users, chat sessions, audit logs, and files.
"""

import pytest
from backend.database.seed import seed_database
from backend.database.connection import transaction
from backend.database.repositories import (
    users as users_repo,
    chat as chat_repo,
    audit as audit_repo,
    files as files_repo,
    models as models_repo,
)


@pytest.fixture(autouse=True)
def setup_db():
    seed_database()


def test_user_repository_crud():
    user = users_repo.get_user_by_username("admin")
    assert user is not None
    assert user["username"] == "admin"

    # Clean up test user if previously created
    test_username = "test_engineer_99"
    existing = users_repo.get_user_by_username(test_username)
    if existing:
        with transaction() as conn:
            conn.execute("DELETE FROM user_roles WHERE user_id = ?", (existing["id"],))
            conn.execute("DELETE FROM users WHERE id = ?", (existing["id"],))

    user_id = users_repo.create_user(
        username=test_username,
        password="TestPassword123!",
        display_name="Test Engineer",
        roles=["engineer"],
    )
    assert user_id > 0

    fetched = users_repo.get_user_by_id(user_id)
    assert fetched is not None
    assert fetched["username"] == test_username
    assert "engineer" in fetched["roles"]

    # Clean up test user
    with transaction() as conn:
        conn.execute("DELETE FROM user_roles WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def test_chat_repository_crud():
    admin = users_repo.get_user_by_username("admin")
    session_id = chat_repo.create_session(admin["id"], "Test Session")
    assert session_id is not None

    # Add message
    msg_id = chat_repo.add_message(
        session_id=session_id,
        role="user",
        content="What is the crude distillation capacity of MRPL?",
    )
    assert msg_id is not None

    # Get session with messages
    session_data = chat_repo.get_session(session_id)
    assert session_data is not None
    assert session_data["title"] == "Test Session"

    messages = chat_repo.get_messages(session_id)
    assert len(messages) >= 1
    assert messages[-1]["content"] == "What is the crude distillation capacity of MRPL?"

    # Rename session
    renamed = chat_repo.rename_session(session_id, "Updated Title")
    assert renamed is True

    # Delete session
    deleted = chat_repo.delete_session(session_id)
    assert deleted is True


def test_audit_repository():
    admin = users_repo.get_user_by_username("admin")
    audit_repo.record_audit_log(
        action="test_action",
        outcome="success",
        user_id=admin["id"],
        username=admin["username"],
        target="system",
        details={"test_key": "test_val"},
    )

    logs, total = audit_repo.query_audit_logs(action="test_action")
    assert total >= 1
    assert logs[0]["action"] == "test_action"
    assert logs[0]["username"] == "admin"


def test_model_registry_repository():
    models = models_repo.list_models()
    assert len(models) >= 1
    gemma = next((m for m in models if "gemma" in m["name"].lower()), None)
    assert gemma is not None
    assert gemma["is_active"] == 1
