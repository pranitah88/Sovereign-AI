"""
Authentication API routes — login, logout, current user info.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, field_validator

from backend.auth.dependencies import get_current_session, get_current_user
from backend.auth.password import verify_password
from backend.auth.session import create_session, invalidate_token
from backend.database.repositories import audit as audit_repo
from backend.database.repositories import users as users_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Request/Response models ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

    @field_validator("username", "password")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class LoginResponse(BaseModel):
    token: str
    expires_at: str
    user: dict


class UserInfoResponse(BaseModel):
    id: int
    username: str
    display_name: str
    roles: list[str]


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, request: Request, response: Response):
    """Authenticate a user and return a session token."""
    client_ip = request.client.host if request.client else None

    user = users_repo.get_user_by_username(body.username)

    if user is None or not verify_password(user["password_hash"], body.password):
        audit_repo.write_log(
            action="login",
            outcome="failure",
            username=body.username,
            ip_address=client_ip,
            details={"reason": "invalid_credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not user["is_active"]:
        audit_repo.write_log(
            action="login",
            outcome="denied",
            user_id=user["id"],
            username=user["username"],
            ip_address=client_ip,
            details={"reason": "account_deactivated"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    session = create_session(user["id"])

    # Set cookie for browser clients.
    response.set_cookie(
        key="session_token",
        value=session["token"],
        httponly=True,
        samesite="strict",
        secure=False,  # localhost only — no HTTPS in air-gap
        max_age=86400,
    )

    roles = users_repo.get_user_roles(user["id"])

    audit_repo.write_log(
        action="login",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        ip_address=client_ip,
    )

    return LoginResponse(
        token=session["token"],
        expires_at=session["expires_at"],
        user={
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "roles": roles,
        },
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    session: dict = Depends(get_current_session),
    response: Response = None,
):
    """Invalidate the current session."""
    invalidate_token(session["token"])

    if response:
        response.delete_cookie("session_token")

    audit_repo.write_log(
        action="logout",
        outcome="success",
        user_id=session["user_id"],
    )


@router.get("/me", response_model=UserInfoResponse)
async def me(user: dict = Depends(get_current_user)):
    """Return the authenticated user's info."""
    return UserInfoResponse(
        id=user["id"],
        username=user["username"],
        display_name=user["display_name"],
        roles=user["roles"],
    )
