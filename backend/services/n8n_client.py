"""
n8n workflow integration client.

Triggers n8n webhooks and tracks workflow execution status.
"""

import logging

import requests

from backend.database.repositories import workflows as workflow_repo

logger = logging.getLogger(__name__)

# Default n8n base URL (self-hosted, localhost).
N8N_BASE_URL = "http://localhost:5678"
N8N_WEBHOOK_BASE = f"{N8N_BASE_URL}/webhook"

# Default timeout for n8n webhook calls.
N8N_TIMEOUT = 30


def trigger_webhook(
    webhook_path: str,
    payload: dict,
    *,
    workflow_name: str | None = None,
    triggered_by: int | None = None,
) -> dict:
    """
    Trigger an n8n webhook and record the workflow run.

    Args:
        webhook_path: The webhook path (e.g. '/document-ingest').
        payload: JSON payload to send.
        workflow_name: Human-readable workflow name for tracking.
        triggered_by: User id who triggered this.

    Returns:
        {"run_id": int, "status": str, "n8n_response": dict | None}
    """
    wf_name = workflow_name or f"webhook:{webhook_path}"

    run_id = workflow_repo.record_workflow_run(
        workflow_name=wf_name,
        trigger_source="api",
        triggered_by=triggered_by,
        input_data=payload,
    )

    url = f"{N8N_WEBHOOK_BASE}{webhook_path}"

    try:
        resp = requests.post(url, json=payload, timeout=N8N_TIMEOUT)
        resp.raise_for_status()
        n8n_response = resp.json() if resp.content else {}

        execution_id = n8n_response.get("executionId")
        workflow_repo.update_workflow_run(
            run_id,
            status="completed",
            n8n_execution_id=execution_id,
            output_data=n8n_response,
        )

        logger.info("n8n webhook %s triggered successfully (run_id=%d)", webhook_path, run_id)
        return {"run_id": run_id, "status": "completed", "n8n_response": n8n_response}

    except requests.RequestException as exc:
        workflow_repo.update_workflow_run(
            run_id,
            status="failed",
            error_info=str(exc),
        )
        logger.error("n8n webhook %s failed: %s", webhook_path, exc)
        return {"run_id": run_id, "status": "failed", "n8n_response": None}


def get_workflow_status(run_id: int) -> dict | None:
    """Get a workflow run's status by id."""
    return workflow_repo.get_workflow_run(run_id)


def list_recent_runs(limit: int = 50) -> list[dict]:
    """List recent workflow runs."""
    return workflow_repo.list_workflow_runs(limit=limit)


def check_n8n_health() -> dict:
    """Check if n8n is reachable."""
    try:
        resp = requests.get(f"{N8N_BASE_URL}/healthz", timeout=5)
        return {"status": "healthy" if resp.status_code == 200 else "unhealthy"}
    except requests.RequestException:
        return {"status": "unavailable"}
