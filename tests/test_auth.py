"""
Tests for authentication: Argon2 password hashing and session management.
"""

import pytest
from backend.auth.password import hash_password, verify_password
from backend.auth.session import create_session, validate_token, invalidate_token
from backend.database.seed import seed_database
from backend.database.repositories import users as users_repo


@pytest.fixture(autouse=True)
def setup_db():
    seed_database()


def test_password_hashing():
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert hashed.startswith("$argon2id$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_session_lifecycle():
    user = users_repo.get_user_by_username("admin")
    assert user is not None

    # Create session
    session_data = create_session(user["id"])
    token = session_data["token"]
    assert token is not None
    assert len(token) >= 32

    # Validate session
    validated_session = validate_token(token)
    assert validated_session is not None
    assert validated_session["user_id"] == user["id"]

    # Invalidate session
    invalidate_token(token)

    # Validate again — should fail
    expired = validate_token(token)
    assert expired is None
