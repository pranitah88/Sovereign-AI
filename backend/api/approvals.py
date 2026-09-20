"""
Action Approvals API endpoints — human-in-the-loop review for high-risk operations.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.database.repositories import approvals as approvals_repo
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


class ProposeActionRequest(BaseModel):
    action_type: str
    affected_resource: str
    proposed_payload: Optional[dict] = None


class DecisionRequest(BaseModel):
    reason: str = ""


@router.get("/pending")
async def list_pending(
    user: Annotated[dict, Depends(require_permission("approval_view"))],
):
    """List pending high-risk actions awaiting human approval."""
    return approvals_repo.list_pending_approvals()


@router.post("/propose", status_code=status.HTTP_201_CREATED)
async def propose(
    body: ProposeActionRequest,
    user: Annotated[dict, Depends(get_current_user)],
):
    """Propose an action. If classified HIGH risk, awaits reviewer/admin decision."""
    res = approvals_repo.propose_action(
        requesting_user_id=user["id"],
        requesting_username=user["username"],
        action_type=body.action_type,
        affected_resource=body.affected_resource,
        proposed_payload=body.proposed_payload,
    )

    audit_log(
        action="action_proposed",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"approval:{res['approval_id']}",
        details={"action_type": body.action_type, "risk_level": res["risk_level"]},
    )
    return res


@router.post("/{approval_id}/approve")
async def approve_action(
    approval_id: int,
    body: DecisionRequest,
    user: Annotated[dict, Depends(require_permission("approval_manage"))],
):
    """Approve a high-risk action (requires administrator or reviewer role)."""
    try:
        success = approvals_repo.decide_approval(
            approval_id=approval_id,
            decision="approved",
            approving_user_id=user["id"],
            approving_username=user["username"],
            reason=body.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not success:
        raise HTTPException(status_code=404, detail="Approval proposal not found")

    audit_log(
        action="action_approval_decided",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"approval:{approval_id}",
        details={"decision": "approved", "reason": body.reason},
    )
    return {"id": approval_id, "status": "approved", "approver": user["username"]}


@router.post("/{approval_id}/reject")
async def reject_action(
    approval_id: int,
    body: DecisionRequest,
    user: Annotated[dict, Depends(require_permission("approval_manage"))],
):
    """Reject a high-risk action (requires administrator or reviewer role)."""
    try:
        success = approvals_repo.decide_approval(
            approval_id=approval_id,
            decision="rejected",
            approving_user_id=user["id"],
            approving_username=user["username"],
            reason=body.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not success:
        raise HTTPException(status_code=404, detail="Approval proposal not found")

    audit_log(
        action="action_approval_decided",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"approval:{approval_id}",
        details={"decision": "rejected", "reason": body.reason},
    )
    return {"id": approval_id, "status": "rejected", "approver": user["username"]}
