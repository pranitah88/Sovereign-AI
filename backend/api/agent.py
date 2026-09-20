"""
Agent invocation API endpoints.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


class InvokeRequest(BaseModel):
    query: str
    session_id: int | None = None
    has_image: bool = False
    has_scanned_pdf: bool = False


@router.post("/invoke")
async def invoke_agent(
    body: InvokeRequest,
    user: Annotated[dict, Depends(require_permission("agent_invoke"))],
):
    """Invoke the LangGraph agent for a multi-step task."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        from backend.agent.graph import run_agent

        result = await run_agent(
            query=body.query.strip(),
            user_id=user["id"],
            session_id=body.session_id,
            has_image=body.has_image,
            has_scanned_pdf=body.has_scanned_pdf,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Agent orchestrator is not yet available",
        )
    except TimeoutError as te:
        logger.error("Agent timeout during invocation: %s", te)
        audit_log(
            action="agent_invoke",
            outcome="failure",
            user_id=user["id"],
            username=user["username"],
            details={"error": str(te), "query": body.query[:200]},
        )
        raise HTTPException(status_code=504, detail=str(te))
    except RuntimeError as re:
        logger.error("Agent runtime error during invocation: %s", re)
        audit_log(
            action="agent_invoke",
            outcome="failure",
            user_id=user["id"],
            username=user["username"],
            details={"error": str(re), "query": body.query[:200]},
        )
        raise HTTPException(status_code=500, detail=str(re))

    audit_log(
        action="agent_invoke",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        details={"query": body.query[:200]},
    )

    return result


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: int,
    user: Annotated[dict, Depends(require_permission("agent_invoke"))],
):
    """Get the status and steps of an agent task."""
    from backend.database.connection import get_connection

    conn = get_connection()
    task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    task_dict = dict(task)

    # Ownership check.
    if task_dict["user_id"] != user["id"] and "administrator" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Access denied")

    steps = conn.execute(
        "SELECT * FROM task_steps WHERE task_id = ? ORDER BY step_index",
        (task_id,),
    ).fetchall()
    task_dict["steps"] = [dict(s) for s in steps]

    return task_dict
