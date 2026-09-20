"""
Admin API endpoints — user management.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.database.repositories import users as users_repo
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    username: str
    display_name: str
    password: str
    roles: list[str] = []

    @field_validator("username", "display_name", "password")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    roles: list[str] | None = None
    is_active: bool | None = None


@router.get("/users")
async def list_users(
    user: Annotated[dict, Depends(require_permission("user_manage"))],
):
    """List all users with their roles."""
    users = users_repo.list_users(include_inactive=True)
    for u in users:
        u["roles"] = users_repo.get_user_roles(u["id"])
    return users


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserRequest,
    user: Annotated[dict, Depends(require_permission("user_manage"))],
):
    """Create a new user (admin only)."""
    existing = users_repo.get_user_by_username(body.username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already exists")

    user_id = users_repo.create_user(
        username=body.username,
        display_name=body.display_name,
        password=body.password,
    )

    for role_name in body.roles:
        try:
            users_repo.assign_role(user_id, role_name)
        except ValueError as exc:
            logger.warning("Skipping invalid role '%s': %s", role_name, exc)

    audit_log(
        action="user_create",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"user:{user_id}",
        details={"new_username": body.username, "roles": body.roles},
    )

    new_user = users_repo.get_user_by_id(user_id)
    new_user["roles"] = users_repo.get_user_roles(user_id)
    return new_user


@router.patch("/users/{target_user_id}")
async def update_user(
    target_user_id: int,
    body: UpdateUserRequest,
    user: Annotated[dict, Depends(require_permission("user_manage"))],
):
    """Update a user's profile or roles (admin only)."""
    target = users_repo.get_user_by_id(target_user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    if body.display_name is not None or body.is_active is not None:
        users_repo.update_user(
            target_user_id,
            display_name=body.display_name,
            is_active=body.is_active,
        )

    if body.roles is not None:
        # Replace all roles.
        current_roles = users_repo.get_user_roles(target_user_id)
        for role in current_roles:
            users_repo.revoke_role(target_user_id, role)
        for role in body.roles:
            try:
                users_repo.assign_role(target_user_id, role)
            except ValueError:
                logger.warning("Skipping invalid role '%s'", role)

    audit_log(
        action="user_update",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"user:{target_user_id}",
        details={"changes": body.model_dump(exclude_none=True)},
    )

    updated = users_repo.get_user_by_id(target_user_id)
    updated["roles"] = users_repo.get_user_roles(target_user_id)
    return updated


@router.delete("/users/{target_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    target_user_id: int,
    user: Annotated[dict, Depends(require_permission("user_manage"))],
):
    """Deactivate a user (soft delete)."""
    if target_user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    if not users_repo.deactivate_user(target_user_id):
        raise HTTPException(status_code=404, detail="User not found")

    audit_log(
        action="user_deactivate",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"user:{target_user_id}",
    )
