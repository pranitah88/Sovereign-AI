"""
Audit log viewer API endpoint.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.auth.rbac import require_permission
from backend.database.repositories import audit as audit_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/logs")
async def get_audit_logs(
    user: Annotated[dict, Depends(require_permission("audit_view"))],
    user_id: int | None = Query(None, description="Filter by acting user id"),
    action: str | None = Query(None, description="Filter by action type"),
    outcome: str | None = Query(None, description="Filter by outcome"),
    start_date: str | None = Query(None, description="ISO-8601 start date"),
    end_date: str | None = Query(None, description="ISO-8601 end date"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Paginated audit log query with filters. Admin/Auditor only."""
    logs = audit_repo.query_logs(
        user_id=user_id,
        action=action,
        outcome=outcome,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )

    total = audit_repo.count_logs(
        user_id=user_id,
        action=action,
        outcome=outcome,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "logs": logs,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
