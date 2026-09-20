"""
Model registry API endpoints.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.database.repositories import models as models_repo
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


class ToggleModelRequest(BaseModel):
    enabled: bool


@router.get("")
async def list_models(
    user: Annotated[dict, Depends(require_permission("model_view"))],
):
    """List all models in the registry."""
    return models_repo.list_models(enabled_only=False)


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    user: Annotated[dict, Depends(require_permission("model_view"))],
):
    """Get a single model by id."""
    model = models_repo.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.patch("/{model_id}")
async def toggle_model(
    model_id: str,
    body: ToggleModelRequest,
    user: Annotated[dict, Depends(require_permission("model_manage"))],
):
    """Enable or disable a model (admin only)."""
    if not models_repo.set_model_enabled(model_id, body.enabled):
        raise HTTPException(status_code=404, detail="Model not found")

    audit_log(
        action="model_toggle",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"model:{model_id}",
        details={"enabled": body.enabled},
    )

    return {"id": model_id, "enabled": body.enabled}


@router.get("/{model_id}/status")
async def model_status(
    model_id: str,
    user: Annotated[dict, Depends(require_permission("model_view"))],
):
    """Check if a model is currently loaded in Ollama."""
    import requests

    model = models_repo.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        loaded_models = [m["name"] for m in resp.json().get("models", [])]
        is_loaded = model["ollama_model_name"] in loaded_models
    except requests.RequestException:
        is_loaded = None  # Ollama unreachable

    return {
        "id": model_id,
        "ollama_model_name": model["ollama_model_name"],
        "is_loaded": is_loaded,
        "enabled": model["enabled"],
    }
