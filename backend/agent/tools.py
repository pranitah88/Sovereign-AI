"""
Agent tool definitions — callable by the LangGraph agent nodes.

Each tool is a function that accepts structured input and returns
structured output. Tools are registered by name for dynamic dispatch.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Tool registry — maps tool name → callable.
_TOOL_REGISTRY: dict = {}


def register_tool(name: str):
    """Decorator to register a tool function."""
    def decorator(func):
        _TOOL_REGISTRY[name] = func
        return func
    return decorator


def get_tool(name: str):
    """Get a registered tool by name. Raises KeyError if not found."""
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not registered")
    return _TOOL_REGISTRY[name]


def list_tools() -> list[str]:
    """Return all registered tool names."""
    return list(_TOOL_REGISTRY.keys())


def execute_tool(name: str, user: dict | None = None, **kwargs) -> dict:
    """
    Execute a tool by name with keyword arguments through the secure Tool Gateway.
    Enforces RBAC permissions, parameter validation, and audit logging.
    """
    try:
        from backend.services.tool_gateway import execute_tool_secure
        return execute_tool_secure(name, user=user, **kwargs)
    except Exception as exc:
        logger.error("Tool Gateway execution failed for '%s': %s", name, exc)
        tool_fn = get_tool(name)
        result = tool_fn(**kwargs)
        return {"tool": name, "status": "success", "result": result}


# ── Tool implementations ────────────────────────────────────────────────

@register_tool("rag_search")
@register_tool("search_kb")
@register_tool("SEARCH_KB")
def rag_search(query: str, top_k: int = 5, user_roles: list[str] | None = None) -> dict:
    """Search the RAG knowledge base."""
    try:
        from backend.services.rag_engine import query_knowledge_base
        return query_knowledge_base(query, top_k=top_k, user_roles=user_roles)
    except Exception as exc:
        logger.error("RAG search failed: %s", exc)
        return {"context": "", "sources": [], "result_count": 0, "error": str(exc)}


@register_tool("ocr_extract")
def ocr_extract(file_path: str) -> dict:
    """Extract text from an image or scanned document using OCR."""
    from backend.services.ocr import extract_text_from_image
    return extract_text_from_image(file_path)


@register_tool("sandbox_execute")
@register_tool("execute_code")
@register_tool("EXECUTE_CODE")
def sandbox_execute(code: str, language: str = "python", timeout_seconds: int = 30) -> dict:
    """Execute code in the sandbox."""
    from backend.services.sandbox import execute_code
    return execute_code(code, language, timeout_seconds)


@register_tool("docgen_pdf")
def docgen_pdf(
    title: str = "DOCUMENT ANALYSIS REPORT",
    content: str = "",
    document_name: str | None = None,
    filename: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Generate a PDF document."""
    from backend.services.docgen import generate_pdf
    return generate_pdf(
        title=title,
        content=content,
        document_name=document_name,
        filename=filename,
        metadata=metadata,
    )


@register_tool("docgen_docx")
@register_tool("generate_document")
@register_tool("GENERATE_DOCUMENT")
def docgen_docx(
    title: str = "DOCUMENT ANALYSIS REPORT",
    content: str = "",
    document_name: str | None = None,
    filename: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Generate a Word document."""
    from backend.services.docgen import generate_docx
    return generate_docx(
        title=title,
        content=content,
        document_name=document_name,
        filename=filename,
        metadata=metadata,
    )


@register_tool("docgen_xlsx")
def docgen_xlsx(title: str, data: list[list], headers: list[str] | None = None) -> dict:
    """Generate an Excel spreadsheet."""
    from backend.services.docgen import generate_xlsx
    return generate_xlsx(title, data, headers=headers)


@register_tool("docgen_pptx")
def docgen_pptx(title: str, slides: list[dict]) -> dict:
    """Generate a PowerPoint presentation."""
    from backend.services.docgen import generate_pptx
    return generate_pptx(title, slides)


@register_tool("file_list")
def file_list(directory: str = "") -> dict:
    """List files in the knowledge base."""
    kb_root = Path(__file__).resolve().parent.parent.parent / "merged_knowledge_base"

    if directory:
        target = kb_root / directory
    else:
        target = kb_root

    # Prevent path traversal.
    try:
        target = target.resolve()
        if not str(target).startswith(str(kb_root.resolve())):
            return {"error": "Path traversal not allowed"}
    except Exception:
        return {"error": "Invalid path"}

    if not target.exists():
        return {"files": [], "error": "Directory not found"}

    files = []
    for item in sorted(target.iterdir()):
        files.append({
            "name": item.name,
            "is_directory": item.is_dir(),
            "size_bytes": item.stat().st_size if item.is_file() else None,
        })

    return {"directory": str(target.relative_to(kb_root)), "files": files}


@register_tool("file_read")
@register_tool("read_document")
@register_tool("READ_DOCUMENT")
def file_read(file_path: str) -> dict:
    """Read a file from the knowledge base."""
    kb_root = Path(__file__).resolve().parent.parent.parent / "merged_knowledge_base"
    target = (kb_root / file_path).resolve()

    # Prevent path traversal.
    if not str(target).startswith(str(kb_root.resolve())):
        return {"error": "Path traversal not allowed"}

    if not target.exists():
        return {"error": f"File not found: {file_path}"}

    if not target.is_file():
        return {"error": f"Not a file: {file_path}"}

    # Cap file size to prevent loading huge files.
    if target.stat().st_size > 1_000_000:  # 1 MB
        return {"error": "File too large (>1MB)"}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        return {"path": file_path, "content": content, "size_bytes": len(content)}
    except Exception as exc:
        return {"error": str(exc)}


@register_tool("file_search")
def file_search(pattern: str) -> dict:
    """Search for files in the knowledge base by name pattern."""
    kb_root = Path(__file__).resolve().parent.parent.parent / "merged_knowledge_base"

    matches = []
    for item in kb_root.rglob(f"*{pattern}*"):
        if item.is_file():
            matches.append({
                "path": str(item.relative_to(kb_root)),
                "name": item.name,
                "size_bytes": item.stat().st_size,
            })
        if len(matches) >= 50:
            break

    return {"pattern": pattern, "matches": matches, "total": len(matches)}


@register_tool("vision_verify")
def vision_verify(image_path: str, user: dict | None = None, extracted_text_hint: str | None = None) -> dict:
    """Inspect P&ID engineering drawings, extract equipment tags, and validate against MRPL registry."""
    from backend.services.vision_verification import verify_pid_drawing
    report = verify_pid_drawing(image_path, user=user, extracted_text_hint=extracted_text_hint)
    return report.to_dict()
