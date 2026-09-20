"""
Code sandbox API endpoint.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    timeout_seconds: int = 30


@router.post("/execute")
async def execute_code(
    body: ExecuteRequest,
    user: Annotated[dict, Depends(require_permission("sandbox_execute"))],
):
    """Execute code in an isolated sandbox environment."""
    if not body.code.strip():
        raise HTTPException(status_code=400, detail="Empty code")

    try:
        from backend.services.sandbox import execute_code as sandbox_exec

        result = sandbox_exec(
            code=body.code,
            language=body.language,
            timeout_seconds=body.timeout_seconds,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Code sandbox is not yet available",
        )

    audit_log(
        action="sandbox_execute",
        outcome=result.get("status", "unknown"),
        user_id=user["id"],
        username=user["username"],
        details={
            "language": body.language,
            "code_length": len(body.code),
            "status": result.get("status"),
        },
    )

    return result
