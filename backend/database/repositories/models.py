"""
Model registry repository — runtime mirror of model_registry.yaml in SQLite.

Synced from YAML on startup; provides fast runtime lookups without
re-reading YAML on every request.
"""

import json
import logging
from pathlib import Path

import yaml

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)

_REGISTRY_YAML = Path(__file__).resolve().parent.parent.parent.parent / "config" / "model_registry.yaml"


def sync_from_yaml(yaml_path: Path | None = None) -> int:
    """
    Read model_registry.yaml and upsert all models into the SQLite
    model_registry table. Returns the number of models synced.
    """
    path = yaml_path or _REGISTRY_YAML

    if not path.exists():
        raise FileNotFoundError(f"Model registry YAML not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    models = registry.get("models", [])
    if not models:
        logger.warning("No models found in %s", path)
        return 0

    with transaction() as conn:
        for model in models:
            conn.execute(
                """
                INSERT INTO model_registry
                    (id, display_name, provider, ollama_model_name, endpoint,
                     modality, max_context_window, runtime_context_window,
                     vram_gb_estimate, enabled, task_tags, priority, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name = excluded.display_name,
                    provider = excluded.provider,
                    ollama_model_name = excluded.ollama_model_name,
                    endpoint = excluded.endpoint,
                    modality = excluded.modality,
                    max_context_window = excluded.max_context_window,
                    runtime_context_window = excluded.runtime_context_window,
                    vram_gb_estimate = excluded.vram_gb_estimate,
                    enabled = excluded.enabled,
                    task_tags = excluded.task_tags,
                    priority = excluded.priority,
                    notes = excluded.notes,
                    synced_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                """,
                (
                    model["id"],
                    model["display_name"],
                    model["provider"],
                    model["ollama_model_name"],
                    model["endpoint"],
                    json.dumps(model.get("modality", ["text"])),
                    model.get("max_context_window", 8192),
                    model.get("runtime_context_window", 8192),
                    model.get("vram_gb_estimate"),
                    int(model.get("enabled", True)),
                    json.dumps(model.get("task_tags", [])),
                    model.get("priority", 1),
                    model.get("notes", "").strip() if model.get("notes") else None,
                ),
            )

    logger.info("Synced %d models from %s", len(models), path)
    return len(models)


def list_models(enabled_only: bool = True) -> list[dict]:
    """Return all models from the registry."""
    conn = get_connection()
    query = "SELECT * FROM model_registry"
    if enabled_only:
        query += " WHERE enabled = 1"
    query += " ORDER BY priority ASC, id ASC"

    rows = conn.execute(query).fetchall()
    results = []
    for row in rows:
        model = dict(row)
        model["modality"] = json.loads(model["modality"])
        model["task_tags"] = json.loads(model["task_tags"])
        model["enabled"] = bool(model["enabled"])
        model["name"] = model.get("display_name", "")
        model["is_active"] = 1 if model["enabled"] else 0
        results.append(model)
    return results


def get_model(model_id: str) -> dict | None:
    """Return a single model by id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM model_registry WHERE id = ?",
        (model_id,),
    ).fetchone()

    if row is None:
        return None

    model = dict(row)
    model["modality"] = json.loads(model["modality"])
    model["task_tags"] = json.loads(model["task_tags"])
    model["enabled"] = bool(model["enabled"])
    model["name"] = model.get("display_name", "")
    model["is_active"] = 1 if model["enabled"] else 0
    return model


def set_model_enabled(model_id: str, enabled: bool) -> bool:
    """Enable or disable a model. Returns True if the model existed."""
    with transaction() as conn:
        cursor = conn.execute(
            "UPDATE model_registry SET enabled = ? WHERE id = ?",
            (int(enabled), model_id),
        )
        return cursor.rowcount > 0
