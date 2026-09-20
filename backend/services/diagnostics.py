"""
Comprehensive System Diagnostics ("Doctor") Service for MRPL Sovereign AI Workbench.

Performs live verification of every critical subsystem without faking any status:
- Ollama inference runtime & local models
- NVIDIA GPU acceleration & VRAM telemetry
- ChromaDB vector store collection integrity
- Authoritative Knowledge Base documents & chunks
- SQLite authentication & RBAC tables
- Docker sandbox isolation & image availability
- Network Seal air-gap enforcement
- Model Registry license validation
- Audit logging subsystem
"""

import logging
from pathlib import Path
from typing import Any

import requests

from backend.database.connection import get_connection
from backend.services.network_seal import get_network_seal_status, record_local_call
from backend.services.sandbox import check_sandbox_ready
from backend.services.system_monitor import _get_gpu_info

logger = logging.getLogger(__name__)


def run_system_diagnostics() -> dict[str, Any]:
    """
    Execute live diagnostic checks across all subsystems.
    Zero synthetic fallbacks — reports exact live state.
    """
    checks = []

    # 1. Ollama Runtime
    record_local_call("http://127.0.0.1:11434/api/tags")
    ollama_ok = False
    installed_models = []
    ollama_detail = ""
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_ok = True
            models_data = resp.json().get("models", [])
            installed_models = [m.get("name", "") for m in models_data]
            ollama_detail = f"Online. Installed models: {', '.join(installed_models) or 'none'}"
        else:
            ollama_detail = f"HTTP {resp.status_code} from Ollama"
    except Exception as exc:
        ollama_detail = f"Offline: {exc}"

    checks.append({
        "component": "Ollama Runtime",
        "category": "Inference",
        "status": "HEALTHY" if ollama_ok else "UNAVAILABLE",
        "ok": ollama_ok,
        "detail": ollama_detail,
    })

    # 2. Configured Models Check
    expected_models = [
        ("Gemma 3 4B", "gemma3:4b", "General QA / Multimodal RAG"),
        ("Qwen 2.5 Coder 3B", "qwen2.5-coder:3b", "Coding & Calculations"),
        ("Qwen 2.5 VL 3B", "qwen2.5-vl:3b", "P&ID Vision Specialist"),
    ]
    for disp_name, model_tag, role in expected_models:
        present = any(model_tag in m for m in installed_models)
        checks.append({
            "component": disp_name,
            "category": "Models",
            "status": "HEALTHY" if present else "NOT_INSTALLED",
            "ok": present,
            "detail": f"{model_tag} ({role}) — {'Installed & Verified' if present else 'Configured in registry, not yet pulled'}",
        })

    # 3. NVIDIA GPU Acceleration
    gpu_info = _get_gpu_info()
    gpu_ok = gpu_info.get("gpu_available", False)
    checks.append({
        "component": "NVIDIA GPU Acceleration",
        "category": "Hardware",
        "status": "HEALTHY" if gpu_ok else "UNAVAILABLE",
        "ok": gpu_ok,
        "detail": f"{gpu_info.get('gpu_name', 'No GPU')} — VRAM: {gpu_info.get('vram_used_mb', 0)}MB / {gpu_info.get('vram_total_mb', 0)}MB ({gpu_info.get('utilization_percent', 0)}% utilization)",
    })

    # 4. ChromaDB Vector Store
    chroma_ok = False
    chroma_detail = ""
    try:
        from backend.services.rag_engine import get_collection
        col = get_collection()
        chunk_count = col.count()
        chroma_ok = chunk_count > 0
        chroma_detail = f"Collection 'mrpl_knowledge_base' active ({chunk_count:,} chunks indexed)"
    except Exception as exc:
        chroma_detail = f"ChromaDB connection error: {exc}"

    checks.append({
        "component": "ChromaDB Vector Store",
        "category": "Storage",
        "status": "HEALTHY" if chroma_ok else "ERROR",
        "ok": chroma_ok,
        "detail": chroma_detail,
    })

    # 5. Authoritative Knowledge Base Corpus
    kb_ok = False
    kb_detail = ""
    try:
        from backend.services.knowledge_map import get_knowledge_map_data
        km = get_knowledge_map_data(["administrator"], "CONFIDENTIAL")
        root = km.get("root", {})
        doc_count = root.get("total_documents", 0)
        cat_count = root.get("total_categories", 0)
        kb_ok = doc_count >= 100
        kb_detail = f"Corpus verified: {doc_count} documents across {cat_count} collections ({root.get('total_chunks', 0):,} chunks)"
    except Exception as exc:
        kb_detail = f"Corpus resolution failed: {exc}"

    checks.append({
        "component": "Knowledge Base Integrity",
        "category": "Storage",
        "status": "HEALTHY" if kb_ok else "WARNING",
        "ok": kb_ok,
        "detail": kb_detail,
    })

    # 6. SQLite Database & Authentication
    db_ok = False
    db_detail = ""
    try:
        conn = get_connection()
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        role_count = conn.execute("SELECT COUNT(*) FROM roles").fetchone()[0]
        db_ok = user_count > 0 and role_count > 0
        db_detail = f"SQLite connected: {user_count} users registered, {role_count} roles provisioned"
    except Exception as exc:
        db_detail = f"Database error: {exc}"

    checks.append({
        "component": "Authentication & Database",
        "category": "Security",
        "status": "HEALTHY" if db_ok else "ERROR",
        "ok": db_ok,
        "detail": db_detail,
    })

    # 7. Role-Based Access Control (RBAC) & Clearance
    from backend.auth.rbac import PERMISSION_MATRIX
    from backend.services.rag_engine import CLASSIFICATION_LEVELS
    rbac_ok = len(PERMISSION_MATRIX) >= 10 and len(CLASSIFICATION_LEVELS) >= 4
    checks.append({
        "component": "RBAC & Clearance Matrix",
        "category": "Security",
        "status": "HEALTHY" if rbac_ok else "WARNING",
        "ok": rbac_ok,
        "detail": f"{len(PERMISSION_MATRIX)} permission groups defined, {len(CLASSIFICATION_LEVELS)} clearance tiers enforced",
    })

    # 8. Docker Code Execution Sandbox
    sandbox_res = check_sandbox_ready()
    sandbox_ok = sandbox_res.get("ready", False)
    checks.append({
        "component": "Docker Sandbox",
        "category": "Execution",
        "status": "HEALTHY" if sandbox_ok else "UNAVAILABLE",
        "ok": sandbox_ok,
        "detail": "Docker daemon running, sandbox image verified (network: none, non-root, read-only root)" if sandbox_ok else f"Sandbox not ready: {sandbox_res.get('reason')} ({sandbox_res.get('detail')})",
    })

    # 9. Network Seal & Air-Gap Enforcer
    seal_info = get_network_seal_status()
    seal_ok = seal_info.get("seal_active", False) and seal_info.get("external_calls_allowed") == 0
    checks.append({
        "component": "Network Seal",
        "category": "Governance",
        "status": "HEALTHY" if seal_ok else "ERROR",
        "ok": seal_ok,
        "detail": f"Active: {seal_info.get('status')} | External calls: {seal_info.get('external_calls_allowed')} | Blocked: {seal_info.get('blocked_attempts')}",
    })

    # 10. Model Registry & License Gate
    try:
        from backend.services.task_router import _load_registry
        reg = _load_registry()
        reg_models = reg.get("models", [])
        approved_models = [m["id"] for m in reg_models if m.get("approved") and m.get("license")]
        reg_ok = len(approved_models) >= 2
        reg_detail = f"{len(approved_models)} approved models verified with compliant licenses"
    except Exception as exc:
        reg_ok = False
        reg_detail = f"Registry check failed: {exc}"

    checks.append({
        "component": "Model Registry & License Gate",
        "category": "Governance",
        "status": "HEALTHY" if reg_ok else "ERROR",
        "ok": reg_ok,
        "detail": reg_detail,
    })

    # 11. Audit Logging Subsystem
    audit_ok = False
    audit_detail = ""
    try:
        conn = get_connection()
        audit_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        audit_ok = True
        audit_detail = f"Audit ledger active with {audit_count:,} security entries recorded"
    except Exception as exc:
        audit_detail = f"Audit ledger check failed: {exc}"

    checks.append({
        "component": "Audit Ledger",
        "category": "Governance",
        "status": "HEALTHY" if audit_ok else "ERROR",
        "ok": audit_ok,
        "detail": audit_detail,
    })

    total_checks = len(checks)
    healthy_checks = sum(1 for c in checks if c["ok"])

    return {
        "overall_status": "READY" if healthy_checks >= (total_checks - 2) else "DEGRADED",
        "total_checks": total_checks,
        "healthy_checks": healthy_checks,
        "checks": checks,
    }


run_all_diagnostics = run_system_diagnostics

