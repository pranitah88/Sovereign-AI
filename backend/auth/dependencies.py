"""
FastAPI dependency functions for authentication and authorization.

Used as `Depends(...)` in route handlers to extract and validate
the current user from the request.
"""

import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from backend.auth.session import validate_token
from backend.database.repositories import users as users_repo

logger = logging.getLogger(__name__)


def _extract_token(
    authorization: Annotated[str | None, Header()] = None,
    request: Request = None,
) -> str:
    """
    Extract the session token from the Authorization header (Bearer scheme)
    or from a cookie named 'session_token'.
    """
    # Try Authorization header first.
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]

    # Fall back to query param for direct download/view actions
    if request and request.query_params.get("token"):
        return request.query_params["token"]

    # Fall back to cookie.
    if request and request.cookies.get("session_token"):
        return request.cookies["session_token"]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_session(
    authorization: Annotated[str | None, Header()] = None,
    request: Request = None,
) -> dict:
    """
    FastAPI dependency: validate the session token and return the session dict.
    Raises 401 if the token is missing, invalid, or expired.
    """
    token = _extract_token(authorization, request)
    session = validate_token(token)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return session


async def get_current_user(
    session: Annotated[dict, Depends(get_current_session)],
) -> dict:
    """
    FastAPI dependency: return the authenticated user dict.
    Includes 'roles' as a list of role names.
    Raises 401 if the user is deactivated.
    """
    user = users_repo.get_user_by_id(session["user_id"])

    if user is None or not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated",
        )

    user["roles"] = users_repo.get_user_roles(user["id"])
    return user


def require_role(*required_roles: str):
    """
    Factory that returns a FastAPI dependency enforcing role membership.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role("administrator"))])
        def admin_endpoint(): ...
    """

    async def _check_role(
        user: Annotated[dict, Depends(get_current_user)],
    ) -> dict:
        user_roles = set(user.get("roles", []))
        required = set(required_roles)

        if not user_roles & required:
            logger.warning(
                "RBAC denied: user '%s' (roles=%s) attempted access requiring %s",
                user["username"],
                user_roles,
                required,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(required_roles)}",
            )

        return user

    return _check_role


def require_any_role(roles: list[str]):
    """Alias for require_role(*roles) for list input."""
    return require_role(*roles)
