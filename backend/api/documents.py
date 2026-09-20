"""
Document management API endpoints — upload, list, delete, reindex.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.database.repositories import files as files_repo
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Upload storage directory.
_UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"

# Allowed & prohibited extensions
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx", ".xlsx", ".pptx", ".json", ".md", ".log", ".png", ".jpg", ".jpeg"}
PROHIBITED_EXTENSIONS = {".exe", ".bat", ".cmd", ".ps1", ".sh", ".dll", ".so", ".bin", ".vbs", ".js", ".py", ".com", ".scr"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_CLASSIFICATIONS = {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "HIGHLY_CONFIDENTIAL"}


def _ensure_upload_dir() -> Path:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return _UPLOAD_DIR


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    classification: str = Form("INTERNAL"),
    department: str = Form("GENERAL"),
    version: str = Form("1.0"),
    user: dict = Depends(require_permission("document_manage")),
):
    """Upload a document file with security validation and dispatch automatic background indexing."""
    raw_name = file.filename or ""
    if not raw_name.strip():
        raise HTTPException(status_code=400, detail="No filename provided")

    # 1. Path traversal protection
    if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
        logger.warning("Upload rejected: path traversal detected in filename '%s' from user '%s'", raw_name, user["username"])
        audit_log(
            action="file_upload_security_violation",
            outcome="denied",
            user_id=user["id"],
            username=user["username"],
            target=f"filename:{raw_name}",
            details={"reason": "Path traversal attempt in filename", "raw_filename": raw_name},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: Path traversal sequences are strictly prohibited.",
        )

    # 2. Extension validation
    ext = Path(raw_name).suffix.lower()
    if ext in PROHIBITED_EXTENSIONS or ext not in ALLOWED_EXTENSIONS:
        logger.warning("Upload rejected: prohibited extension '%s' for file '%s' by user '%s'", ext, raw_name, user["username"])
        audit_log(
            action="file_upload_security_violation",
            outcome="denied",
            user_id=user["id"],
            username=user["username"],
            target=f"filename:{raw_name}",
            details={"reason": "Prohibited or unsupported file extension", "extension": ext},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload rejected: File extension '{ext}' is not permitted in the knowledge base.",
        )

    # 3. Read content and enforce size limits
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        audit_log(
            action="file_upload_security_violation",
            outcome="denied",
            user_id=user["id"],
            username=user["username"],
            target=f"filename:{raw_name}",
            details={"reason": "File size exceeded limit", "size_bytes": len(content)},
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed upload size (50 MB). Received: {len(content)} bytes.",
        )

    # 4. Classification validation
    clean_cls = classification.strip().upper()
    if clean_cls not in ALLOWED_CLASSIFICATIONS:
        clean_cls = "INTERNAL"

    upload_dir = _ensure_upload_dir()

    # Generate a unique stored filename to isolate original input from internal disk structure
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = upload_dir / stored_name

    try:
        with open(stored_path, "wb") as f:
            f.write(content)
    except IOError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save file") from exc

    file_id = files_repo.record_upload(
        user_id=user["id"],
        original_name=raw_name,
        stored_path=str(stored_path),
        file_size_bytes=len(content),
        mime_type=file.content_type,
        classification=clean_cls,
        department=department.strip().upper(),
        owner=user["username"],
        version=version.strip(),
    )

    audit_log(
        action="file_upload",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"file:{file_id}",
        details={"filename": raw_name, "size": len(content), "classification": clean_cls},
    )

    # Automatic background indexing with classification metadata
    background_tasks.add_task(
        _run_indexing,
        file_id=file_id,
        stored_path=str(stored_path),
        original_name=raw_name,
        category=None,
        classification=clean_cls,
        department=department.strip().upper(),
        owner=user["username"],
        version=version.strip(),
    )

    return {
        "id": file_id,
        "original_name": raw_name,
        "file_size_bytes": len(content),
        "classification": clean_cls,
        "department": department.strip().upper(),
        "index_status": "pending",
    }


@router.get("/knowledge-map")
async def get_knowledge_map(
    user: dict = Depends(get_current_user),
):
    """Retrieve the authoritative Knowledge Map structure from ChromaDB with RBAC clearance filtering."""
    from backend.services.knowledge_map import get_knowledge_map_data
    return get_knowledge_map_data(
        user_roles=user.get("roles", []),
        user_clearance=user.get("clearance"),
    )


@router.get("/corpus")
async def list_corpus_documents(
    user: dict = Depends(get_current_user),
):
    """Retrieve all indexed documents across the full knowledge base corpus."""
    from backend.services.knowledge_map import get_knowledge_map_data
    km_data = get_knowledge_map_data(
        user_roles=user.get("roles", []),
        user_clearance=user.get("clearance"),
    )
    all_docs = []
    for cat in km_data.get("categories", []):
        all_docs.extend(cat.get("documents", []))
    return {
        "root": km_data.get("root", {}),
        "total": len(all_docs),
        "documents": all_docs,
    }


@router.get("")
async def list_documents(
    source: str | None = None,
    user: dict = Depends(require_permission("document_manage")),
):
    """List documents. Defaults to uploaded files for backward compatibility, or full corpus if requested."""
    if source in ("all", "corpus", "kb"):
        from backend.services.knowledge_map import get_knowledge_map_data
        km = get_knowledge_map_data(user_roles=user.get("roles", []), user_clearance=user.get("clearance"))
        docs = []
        for cat in km.get("categories", []):
            docs.extend(cat.get("documents", []))
        return docs

    # Admins see all uploads; others see their own.
    if "administrator" in user.get("roles", []):
        return files_repo.list_uploaded_files()
    return files_repo.list_uploaded_files(user["id"])


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    file_id: int,
    user: dict = Depends(require_permission("document_manage")),
):
    """Delete an uploaded document."""
    file_record = files_repo.get_uploaded_file(file_id)
    if file_record is None:
        raise HTTPException(status_code=404, detail="File not found")

    # Ownership check (admin can delete any).
    if file_record["user_id"] != user["id"] and "administrator" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Access denied")

    # Remove physical file.
    stored = Path(file_record["stored_path"])
    if stored.exists():
        stored.unlink()

    files_repo.delete_uploaded_file(file_id)
    try:
        from backend.services.knowledge_map import invalidate_knowledge_map_cache
        invalidate_knowledge_map_cache()
    except Exception:
        pass

    audit_log(
        action="file_delete",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"file:{file_id}",
        details={"filename": file_record["original_name"]},
    )


from fastapi.responses import FileResponse


def _run_indexing(
    file_id: int,
    stored_path: str,
    original_name: str | None = None,
    category: str | None = None,
    classification: str = "INTERNAL",
    department: str = "GENERAL",
    owner: str = "SYSTEM",
    version: str = "1.0",
    doc_id: str | None = None,
):
    """Background task to index document into ChromaDB with data classification."""
    files_repo.update_index_status(file_id, "indexing")
    try:
        from backend.services.rag_engine import index_document
        res = index_document(
            stored_path,
            original_name=original_name,
            file_id=file_id,
            category=category,
            classification=classification,
            department=department,
            owner=owner,
            version=version,
            doc_id=doc_id,
        )
        if res.get("status") in ("completed", "indexed"):
            files_repo.update_index_status(
                file_id,
                "completed",
                ocr_applied=bool(res.get("ocr_pages", 0) > 0),
                error_message=None,
            )
            try:
                from backend.services.knowledge_map import invalidate_knowledge_map_cache
                invalidate_knowledge_map_cache()
            except Exception:
                pass
            logger.info("Successfully indexed file %d (%s)", file_id, res.get("file_name"))
        else:
            err_msg = res.get("error") or f"Indexing returned status: {res.get('status')}"
            files_repo.update_index_status(file_id, "failed", error_message=err_msg)
            logger.warning("Indexing file %d returned status: %s (%s)", file_id, res.get("status"), err_msg)
    except Exception as exc:
        logger.error("Background indexing failed for file %d: %s", file_id, exc)
        files_repo.update_index_status(file_id, "failed", error_message=str(exc))


@router.get("/{file_id}/download")
async def download_document(
    file_id: int,
    user: dict = Depends(require_permission("document_manage")),
):
    """Download an uploaded document."""
    file_record = files_repo.get_uploaded_file(file_id)
    if file_record is None:
        raise HTTPException(status_code=404, detail="File not found")

    stored = Path(file_record["stored_path"])
    if not stored.exists():
        raise HTTPException(status_code=404, detail="Physical file not found on disk")

    return FileResponse(
        path=str(stored),
        filename=file_record["original_name"],
        media_type=file_record.get("mime_type") or "application/octet-stream",
    )


@router.get("/generated/list")
async def list_generated_documents(
    user: dict = Depends(get_current_user),
):
    """List generated files (reports, spreadsheets, presentations)."""
    if "administrator" in user.get("roles", []):
        return files_repo.list_generated_files()
    return files_repo.list_generated_files(user["id"])


MIME_MAP = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "csv": "text/csv",
    "txt": "text/plain",
}


@router.get("/generated/{file_id}/download")
async def download_generated_document(
    file_id: int,
    view: bool = False,
    inline: bool = False,
    user: dict = Depends(get_current_user),
):
    """Download or view a generated file."""
    file_record = files_repo.get_generated_file(file_id)
    if file_record is None:
        raise HTTPException(status_code=404, detail="File not found")

    if file_record["user_id"] != user["id"] and "administrator" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Access denied")

    stored = Path(file_record["stored_path"])
    if not stored.exists():
        raise HTTPException(status_code=404, detail="Physical file not found on disk")

    ft = (file_record.get("file_type") or "").lower()
    if not ft and "." in file_record["original_name"]:
        ft = file_record["original_name"].rsplit(".", 1)[-1].lower()

    media_type = MIME_MAP.get(ft, "application/octet-stream")
    disposition = "inline" if (view or inline) else "attachment"
    headers = {
        "Content-Disposition": f'{disposition}; filename="{file_record["original_name"]}"',
    }

    return FileResponse(
        path=str(stored),
        filename=file_record["original_name"],
        media_type=media_type,
        headers=headers,
    )


# ── Outputs Router (/api/outputs/{output_id}) ────────────────────────────

outputs_router = APIRouter(prefix="/api/outputs", tags=["outputs"])


@outputs_router.get("/{output_id}")
@outputs_router.get("/{output_id}/download")
async def get_output_file(
    output_id: int,
    view: bool = False,
    inline: bool = False,
    user: dict = Depends(get_current_user),
):
    """Secure endpoint for downloading or viewing deliverables by output id."""
    return await download_generated_document(
        file_id=output_id,
        view=view,
        inline=inline,
        user=user,
    )



@router.post("/{file_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_document(
    file_id: int,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("document_manage")),
):
    """Trigger re-indexing of an uploaded document into ChromaDB."""
    file_record = files_repo.get_uploaded_file(file_id)
    if file_record is None:
        raise HTTPException(status_code=404, detail="File not found")

    # Mark as pending indexing.
    files_repo.update_index_status(file_id, "indexing")

    audit_log(
        action="file_reindex",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"file:{file_id}",
    )

    background_tasks.add_task(
        _run_indexing,
        file_id=file_id,
        stored_path=file_record["stored_path"],
        original_name=file_record.get("original_name"),
        category=file_record.get("category"),
    )

    return {"id": file_id, "index_status": "indexing", "message": "Re-indexing queued"}

