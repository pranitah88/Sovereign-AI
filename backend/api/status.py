"""
System status and monitoring API endpoints.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.auth.rbac import require_permission
from backend.database.connection import get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("/system")
async def system_status(
    user: Annotated[dict, Depends(require_permission("status_view"))],
):
    """GPU, RAM, disk, and model status."""
    try:
        from backend.services.system_monitor import get_system_status
        return get_system_status()
    except ImportError:
        return {"status": "unavailable", "reason": "system_monitor not implemented"}


@router.get("/network")
async def network_status(
    user: Annotated[dict, Depends(require_permission("status_view"))],
):
    """Network interface status (air-gap proof)."""
    try:
        from backend.services.system_monitor import get_network_status
        return get_network_status()
    except ImportError:
        return {"status": "unavailable", "reason": "system_monitor not implemented"}


@router.get("/services")
async def service_health(
    user: Annotated[dict, Depends(require_permission("status_view"))],
):
    """ChromaDB, Ollama, n8n health checks."""
    try:
        from backend.services.system_monitor import get_service_health
        return get_service_health()
    except ImportError:
        return {"status": "unavailable", "reason": "system_monitor not implemented"}


@router.get("/security")
async def security_status(
    user: Annotated[dict, Depends(require_permission("feature_status"))],
):
    """Air-gap and system security diagnostics (nothing leaves the premises)."""
    try:
        from backend.services.system_monitor import get_security_status
        return get_security_status()
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


@router.get("/features")
async def feature_status(
    user: Annotated[dict, Depends(require_permission("feature_status"))],
):
    """Feature status map — implemented/in-progress/planned/unavailable."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT feature_key, display_name, category, status, notes, updated_at "
        "FROM feature_status ORDER BY category, feature_key"
    ).fetchall()

    features = {}
    for row in rows:
        r = dict(row)
        category = r.pop("category")
        if category not in features:
            features[category] = []
        features[category].append(r)

    return {"features": features}


@router.get("/diagnostics")
async def system_diagnostics(
    user: Annotated[dict, Depends(require_permission("status_view"))],
):
    """Run comprehensive system diagnostics (Doctor) across all subsystems."""
    from backend.services.diagnostics import run_system_diagnostics
    return run_system_diagnostics()


@router.get("/seal")
async def network_seal(
    user: Annotated[dict, Depends(require_permission("feature_status"))],
):
    """Retrieve verified Network Seal status and zero-egress telemetry."""
    from backend.services.network_seal import get_network_seal_status
    return get_network_seal_status()

