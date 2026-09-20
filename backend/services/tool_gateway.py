"""
Tool Permission Gateway — enforces strict role-based access control,
parameter validation, and audit logging for all agent and LLM tool invocations.

The LLM is NEVER the security boundary; all tool executions must pass through
this gateway before invoking any underlying system or service capabilities.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Set

from backend.auth.rbac import normalize_roles
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

# ── Tool Definitions ─────────────────────────────────────────────────────────

TOOL_SPECIFICATIONS = {
    "READ_DOCUMENT": {
        "name": "READ_DOCUMENT",
        "description": "Read or inspect document content from the approved knowledge base",
        "operation_type": "READ",
        "allowed_roles": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},
        "required_params": [],
    },
    "SEARCH_KB": {
        "name": "SEARCH_KB",
        "description": "Search the local knowledge base using hybrid vector and lexical retrieval",
        "operation_type": "READ",
        "allowed_roles": {"administrator", "engineer", "reviewer", "document_manager", "auditor", "viewer"},
        "required_params": ["query"],
    },
    "GENERATE_DOCUMENT": {
        "name": "GENERATE_DOCUMENT",
        "description": "Generate engineering reports and summaries (.docx, .xlsx, .pptx)",
        "operation_type": "WRITE",
        "allowed_roles": {"administrator", "engineer"},
        "required_params": ["title"],
    },
    "EXECUTE_CODE": {
        "name": "EXECUTE_CODE",
        "description": "Execute calculation or analysis code inside the isolated Docker sandbox",
        "operation_type": "EXECUTE",
        "allowed_roles": {"administrator", "engineer"},
        "required_params": ["code"],
    },
    "VISION_VERIFY": {
        "name": "VISION_VERIFY",
        "description": "Inspect P&ID engineering drawings, extract equipment tags, and deterministically validate against MRPL registry",
        "operation_type": "READ",
        "allowed_roles": {"administrator", "engineer", "reviewer"},
        "required_params": ["image_path"],
    },
}

# Map aliases and underlying tool functions to high-level tool specifications
TOOL_ALIAS_MAP = {
    "search_kb": "SEARCH_KB",
    "rag_search": "SEARCH_KB",
    "read_document": "READ_DOCUMENT",
    "file_list": "READ_DOCUMENT",
    "file_read": "READ_DOCUMENT",
    "ocr_extract": "READ_DOCUMENT",
    "generate_document": "GENERATE_DOCUMENT",
    "docgen_pdf": "GENERATE_DOCUMENT",
    "docgen_docx": "GENERATE_DOCUMENT",
    "docgen_xlsx": "GENERATE_DOCUMENT",
    "docgen_pptx": "GENERATE_DOCUMENT",
    "execute_code": "EXECUTE_CODE",
    "sandbox_execute": "EXECUTE_CODE",
    "vision_verify": "VISION_VERIFY",
    "verify_pid_drawing": "VISION_VERIFY",
}


def get_tool_spec(name: str) -> dict:
    """Resolve tool name or alias to canonical tool specification."""
    canonical = TOOL_ALIAS_MAP.get(name.lower(), name.upper())
    if canonical not in TOOL_SPECIFICATIONS:
        raise KeyError(f"Unknown or unapproved tool: '{name}'")
    return TOOL_SPECIFICATIONS[canonical]


def validate_tool_parameters(canonical_name: str, kwargs: dict) -> tuple[bool, str | None]:
    """
    Validate tool parameters against security rules.
    Prevents path traversal, empty required fields, and malicious inputs.
    """
    spec = TOOL_SPECIFICATIONS[canonical_name]
    for param in spec["required_params"]:
        if param not in kwargs or kwargs[param] is None:
            return False, f"Missing required parameter '{param}'"
        if isinstance(kwargs[param], str) and not kwargs[param].strip():
            return False, f"Parameter '{param}' cannot be empty"

    # Specific path traversal validation for file/document operations
    for key in ("file_path", "filename", "directory"):
        if key in kwargs and kwargs[key]:
            val = str(kwargs[key])
            if ".." in val or "/../" in val or "\\..\\" in val:
                return False, f"Directory traversal detected in parameter '{key}'"

    # Code execution parameter validation
    if canonical_name == "EXECUTE_CODE":
        lang = str(kwargs.get("language", "python")).lower()
        if lang not in ("python", "py"):
            return False, f"Unsupported sandbox language '{lang}'. Only Python is approved."

    return True, None


def check_tool_permission(tool_name: str, user_roles: list[str] | set[str]) -> bool:
    """Check if the given user roles are authorized to execute the tool."""
    try:
        spec = get_tool_spec(tool_name)
    except KeyError:
        return False
    normalized = normalize_roles(user_roles)
    return bool(normalized & spec["allowed_roles"])


def execute_tool_secure(tool_name: str, user: dict | None = None, **kwargs) -> dict:
    """
    Gateway entry point for executing tools with role authorization,
    parameter validation, and audit logging.
    """
    try:
        spec = get_tool_spec(tool_name)
    except KeyError as exc:
        logger.warning("Unapproved tool attempted: '%s'", tool_name)
        return {"tool": tool_name, "status": "forbidden", "error": str(exc)}

    canonical = spec["name"]
    roles = user.get("roles", ["viewer"]) if user else ["viewer"]
    user_id = user.get("id") if user else None
    username = user.get("username", "anonymous") if user else "anonymous"

    # 1. Authorization check
    if not check_tool_permission(canonical, roles):
        logger.warning(
            "Tool execution denied: user '%s' (roles=%s) attempted unauthorized tool '%s' (%s)",
            username, roles, canonical, tool_name
        )
        audit_log(
            action="tool_execution",
            outcome="denied",
            user_id=user_id,
            username=username,
            target=f"tool:{canonical}",
            details={"tool_requested": tool_name, "reason": "Role unauthorized"},
        )
        return {
            "tool": tool_name,
            "status": "forbidden",
            "error": f"Permission denied: roles {roles} not permitted for tool '{canonical}'",
        }

    # 2. Parameter validation
    valid, err = validate_tool_parameters(canonical, kwargs)
    if not valid:
        logger.warning("Tool parameter validation failed for '%s': %s", canonical, err)
        audit_log(
            action="tool_execution",
            outcome="denied",
            user_id=user_id,
            username=username,
            target=f"tool:{canonical}",
            details={"tool_requested": tool_name, "error": err},
        )
        return {"tool": tool_name, "status": "invalid_params", "error": err}

    # 3. Execution via underlying tool function
    from backend.agent.tools import get_tool
    try:
        # Pass user_roles to search_kb / rag_search if supported
        if canonical == "SEARCH_KB":
            kwargs["user_roles"] = roles

        tool_fn = get_tool(tool_name if tool_name in TOOL_ALIAS_MAP else tool_name.lower())
        result = tool_fn(**kwargs)
        status_code = "success"
        error_msg = None
    except KeyError:
        # Fallback to canonical dispatcher
        try:
            if canonical == "SEARCH_KB":
                from backend.services.rag_engine import query_knowledge_base
                result = query_knowledge_base(kwargs["query"], top_k=kwargs.get("top_k", 5), user_roles=roles)
            elif canonical == "EXECUTE_CODE":
                from backend.services.sandbox import execute_code
                result = execute_code(kwargs["code"], language=kwargs.get("language", "python"), timeout_seconds=kwargs.get("timeout_seconds", 30))
            elif canonical == "GENERATE_DOCUMENT":
                if tool_name.lower() in ("docgen_pdf", "pdf"):
                    from backend.services.docgen import generate_pdf
                    result = generate_pdf(
                        title=kwargs.get("title", "DOCUMENT ANALYSIS REPORT"),
                        content=kwargs.get("content", ""),
                        document_name=kwargs.get("document_name"),
                        filename=kwargs.get("filename"),
                        metadata=kwargs.get("metadata"),
                    )
                elif tool_name.lower() in ("docgen_xlsx", "xlsx"):
                    from backend.services.docgen import generate_xlsx
                    result = generate_xlsx(
                        title=kwargs.get("title", "Report"),
                        data=kwargs.get("data", []),
                        headers=kwargs.get("headers"),
                        metadata=kwargs.get("metadata"),
                        filename=kwargs.get("filename"),
                    )
                elif tool_name.lower() in ("docgen_pptx", "pptx"):
                    from backend.services.docgen import generate_pptx
                    result = generate_pptx(
                        title=kwargs.get("title", "Presentation"),
                        slides=kwargs.get("slides", []),
                        metadata=kwargs.get("metadata"),
                        filename=kwargs.get("filename"),
                    )
                else:
                    from backend.services.docgen import generate_docx
                    result = generate_docx(
                        title=kwargs.get("title", "DOCUMENT ANALYSIS REPORT"),
                        content=kwargs.get("content", ""),
                        document_name=kwargs.get("document_name"),
                        filename=kwargs.get("filename"),
                        metadata=kwargs.get("metadata"),
                    )
            elif canonical == "VISION_VERIFY":
                from backend.services.vision_verification import verify_pid_drawing
                report = verify_pid_drawing(
                    kwargs["image_path"],
                    user=user,
                    extracted_text_hint=kwargs.get("extracted_text_hint"),
                )
                result = report.to_dict()
            else:
                result = {"error": f"No implementation for tool {canonical}"}
            status_code = "success"
            error_msg = None
        except Exception as exc:
            logger.error("Error executing tool '%s': %s", canonical, exc)
            status_code = "error"
            error_msg = str(exc)
            result = None
    except Exception as exc:
        logger.error("Error executing tool '%s': %s", canonical, exc)
        status_code = "error"
        error_msg = str(exc)
        result = None

    # 4. Audit logging
    audit_log(
        action="tool_execution",
        outcome=status_code,
        user_id=user_id,
        username=username,
        target=f"tool:{canonical}",
        details={
            "tool_requested": tool_name,
            "operation_type": spec["operation_type"],
            "status": status_code,
            "error": error_msg,
        },
    )

    if status_code == "error":
        return {"tool": tool_name, "status": "error", "error": error_msg}

    return {"tool": tool_name, "status": "success", "result": result}
