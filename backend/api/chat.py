"""
Chat API endpoints — session management and message handling.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.database.repositories import chat as chat_repo
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Request/Response models ──────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class RenameSessionRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str
    role: str = "user"


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    user: Annotated[dict, Depends(require_permission("chat"))],
):
    """Create a new chat session."""
    session = chat_repo.create_chat_session(user["id"], body.title)
    audit_log(
        action="chat_session_create",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"chat_session:{session['id']}",
    )
    return session


@router.get("/sessions")
async def list_sessions(
    user: Annotated[dict, Depends(require_permission("chat"))],
):
    """List the current user's chat sessions."""
    return chat_repo.list_chat_sessions(user["id"])


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: int,
    user: Annotated[dict, Depends(require_permission("chat"))],
):
    """Get a chat session with all its messages."""
    session = chat_repo.get_session_with_messages(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    from backend.auth.rbac import normalize_roles
    user_roles = normalize_roles(user.get("roles", []))

    # Ensure the session belongs to this user (or user is admin).
    if session["user_id"] != user["id"] and "administrator" not in user_roles:
        audit_log(
            action="chat_access_violation",
            outcome="denied",
            user_id=user["id"],
            username=user["username"],
            target=f"chat_session:{session_id}",
            details={"reason": "Unauthorized access to private chat session", "session_owner_id": session["user_id"]},
        )
        raise HTTPException(status_code=403, detail="Access denied: Cannot access another user's private session")

    return session


@router.post("/sessions/{session_id}/messages", status_code=status.HTTP_201_CREATED)
async def send_message(
    session_id: int,
    body: SendMessageRequest,
    user: Annotated[dict, Depends(require_permission("chat"))],
):
    """
    Send a message to a chat session.

    For MVP, this stores the user message. The agent invocation
    (which produces the assistant response) is handled by the
    agent module via the /api/agent/invoke endpoint or inline.
    """
    session = chat_repo.get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    from backend.auth.rbac import normalize_roles
    user_roles = normalize_roles(user.get("roles", []))
    if session["user_id"] != user["id"] and "administrator" not in user_roles:
        raise HTTPException(status_code=403, detail="Access denied: Cannot post to another user's private session")

    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    message = chat_repo.add_message(
        session_id=session_id,
        role=body.role,
        content=body.content.strip(),
    )

    audit_log(
        action="chat_message",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"chat_session:{session_id}",
    )

    return message


@router.patch("/sessions/{session_id}")
async def rename_session(
    session_id: int,
    body: RenameSessionRequest,
    user: Annotated[dict, Depends(require_permission("chat"))],
):
    """Rename a chat session."""
    session = chat_repo.get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    from backend.auth.rbac import normalize_roles
    user_roles = normalize_roles(user.get("roles", []))
    if session["user_id"] != user["id"] and "administrator" not in user_roles:
        raise HTTPException(status_code=403, detail="Access denied: Cannot rename another user's private session")

    chat_repo.rename_chat_session(session_id, body.title)
    return {"id": session_id, "title": body.title}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int,
    user: Annotated[dict, Depends(require_permission("chat"))],
):
    """Delete a chat session and all its messages."""
    session = chat_repo.get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    from backend.auth.rbac import normalize_roles
    user_roles = normalize_roles(user.get("roles", []))
    if session["user_id"] != user["id"] and "administrator" not in user_roles:
        raise HTTPException(status_code=403, detail="Access denied: Cannot delete another user's private session")

    chat_repo.delete_chat_session(session_id)
    audit_log(
        action="chat_session_delete",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        target=f"chat_session:{session_id}",
    )


# ── Local Vision Analysis Endpoint ───────────────────────────────────────

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/sessions/{session_id}/image-analysis")
async def analyze_chat_image(
    session_id: int,
    user: Annotated[dict, Depends(require_permission("chat"))],
    file: UploadFile = File(...),
    query: str = Form(None),
    prompt: str = Form(None),
):
    """
    Analyze an uploaded image using the approved local Ollama vision model.
    Validates MIME type, extension, size, and filename.
    Stores temporarily in local storage, processes on-premise without external network egress,
    extracts equipment tags, runs deterministic P&ID verification, records audit trace,
    and deterministically deletes the temporary file.

    FAIL-CLOSED: If vision model inference fails, returns a controlled failure response.
    Does NOT fabricate visual observations from failed inference.
    """
    import hashlib
    import re
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid
    import requests

    from backend.auth.rbac import normalize_roles
    from backend.database.repositories import approvals as approvals_repo
    from backend.database.repositories import files as files_repo
    from backend.services.docgen import generate_pdf
    from backend.services.network_seal import record_local_call
    from backend.services.task_router import select_vision_model
    from backend.services.vision_verification import (
        extract_visual_text_evidence_v2,
        extract_process_labels,
        extract_grounded_equipment_tags,
        extract_equipment_tags_from_text,
        validate_equipment_tags,
        build_grounded_vision_prompt,
        run_vision_analysis,
        is_vision_failure_response,
        VisionInferenceResult,
        VISION_STATUS_COMPLETED,
        VISION_STATUS_FAILED,
        VISION_STATUS_OCR_FALLBACK,
        VISION_GENERATION_OPTIONS,
    )

    effective_query = (prompt or query or "Analyze this image and explain the diagram, visible components, labels, and structure.").strip()

    # 1. Verify session exists and ownership
    session = chat_repo.get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    user_roles = normalize_roles(user.get("roles", []))
    if session["user_id"] != user["id"] and "administrator" not in user_roles:
        raise HTTPException(status_code=403, detail="Access denied: Cannot upload to another user's session")

    # 2. Validate filename and extension
    raw_filename = file.filename or "uploaded_image.png"
    sanitized_name = Path(raw_filename).name
    ext = Path(sanitized_name).suffix.lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed formats: .jpg, .jpeg, .png, .webp",
        )

    # Validate Content-Type
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_IMAGE_MIMES and "image/" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported MIME type '{content_type}'. Must be an image (.jpg, .jpeg, .png, .webp)",
        )

    # 3. Read and validate file size (Max 10MB, Min 1 byte)
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty (0 bytes)")
    if file_size > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds 10 MB limit (Received: {file_size / (1024 * 1024):.1f}MB)",
        )

    # 4. Immutable per-request identity & deterministic SHA-256 hash
    analysis_id = f"vis_{uuid.uuid4().hex[:10]}"
    image_hash = hashlib.sha256(file_bytes).hexdigest()
    trace_id = f"trace_vis_{uuid.uuid4().hex[:8]}"
    execution_trace = []

    def log_trace(event: str, title: str, trace_status: str, details: dict | None = None):
        trace_data = {
            "trace_id": trace_id,
            "analysis_id": analysis_id,
            "image_hash": image_hash,
            "filename": sanitized_name,
        }
        if details:
            trace_data.update(details)
        execution_trace.append({
            "event": event,
            "title": title,
            "status": trace_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": trace_data,
        })

    # 5. Safe temporary local storage
    temp_dir = Path("data/temp_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_filename = f"mrpl_img_{session_id}_{analysis_id}_{sanitized_name}"
    temp_path = temp_dir / temp_filename

    try:
        # Step: IMAGE_UPLOAD_STARTED
        log_trace(
            "IMAGE_UPLOAD_STARTED",
            f"Image upload initiated: {sanitized_name}",
            "running",
            {"size_bytes": file_size, "content_type": content_type},
        )

        # Step: IMAGE_RECEIVED
        log_trace(
            "IMAGE_RECEIVED",
            f"Image received: {sanitized_name}",
            "allowed",
            {"filename": sanitized_name, "size_bytes": file_size, "content_type": content_type},
        )

        # Step: IMAGE_HASH_COMPUTED
        log_trace(
            "IMAGE_HASH_COMPUTED",
            f"Computed immutable SHA-256 hash: {image_hash[:16]}...",
            "verified",
            {"image_hash": image_hash, "analysis_id": analysis_id},
        )

        # Step: FILE_VALIDATED
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        log_trace(
            "FILE_VALIDATED",
            "File extension, MIME type, and size validated",
            "verified",
            {"extension": ext, "content_type": content_type, "size_kb": round(file_size / 1024, 2)},
        )

        # Step: RBAC_CHECK
        user_clearance = user.get("clearance", "INTERNAL")
        log_trace(
            "RBAC_CHECK",
            f"User permission and clearance authorized ({user_clearance})",
            "allowed",
            {"username": user["username"], "roles": user_roles, "clearance": user_clearance},
        )

        # Step: VISION_MODEL_SELECTED
        vision_model_entry = select_vision_model()
        model_name = vision_model_entry.get("ollama_model_name", "gemma3:4b")
        model_id = vision_model_entry.get("id", "gemma3_4b")
        model_endpoint = vision_model_entry.get("endpoint", "http://127.0.0.1:11434/api/generate")

        log_trace(
            "VISION_MODEL_SELECTED",
            f"Selected approved local vision model: {model_name}",
            "verified",
            {
                "model": model_name,
                "model_id": model_id,
                "provider": "ollama (local on-premise)",
                "license": vision_model_entry.get("license", "Open"),
            },
        )

        # ── VISION EVIDENCE EXTRACTION ──────────────────────────────────
        # Delegated to vision_verification.py (single owner)
        log_trace(
            "VISION_ANALYSIS_STARTED",
            "Vision evidence extraction started",
            "running",
            {"model": model_name, "endpoint": model_endpoint, "analysis_id": analysis_id},
        )

        evidence_result = extract_visual_text_evidence_v2(
            temp_path,
            model_endpoint=model_endpoint,
            model_name=model_name,
            analysis_id=analysis_id,
            image_hash=image_hash,
        )

        # Classify evidence extraction outcome
        evidence_text = evidence_result.text if evidence_result.success else ""
        evidence_source = evidence_result.source
        evidence_ocr_text = evidence_result.ocr_text

        if not evidence_result.success:
            log_trace(
                "VISION_INFERENCE_FAILED",
                f"Vision evidence extraction failed: {evidence_result.error_code}",
                "failed",
                {
                    "error_code": evidence_result.error_code,
                    "error": evidence_result.error[:300],
                    "retried": evidence_result.retried,
                },
            )
            if evidence_result.retried:
                log_trace(
                    "VISION_RETRY_STARTED",
                    "Vision evidence extraction retry attempted",
                    "failed",
                    {"retried": True},
                )
        else:
            log_trace(
                "VISUAL_EVIDENCE_EXTRACTED",
                f"Visual evidence extracted (source: {evidence_source})",
                "verified",
                {
                    "source": evidence_source,
                    "evidence_characters": len(evidence_text),
                    "retried": evidence_result.retried,
                },
            )

        # Extract process labels and grounded tags ONLY from successfully extracted evidence
        process_labels = []
        initial_grounded_tags = []
        if evidence_result.success and evidence_text:
            process_labels = extract_process_labels(evidence_text)
            initial_grounded_tags = extract_grounded_equipment_tags(evidence_text)

        # ── VISION ANALYSIS CALL ────────────────────────────────────────
        # Delegated to vision_verification.py run_vision_analysis() (single owner)
        log_trace(
            "OLLAMA_INFERENCE",
            f"Grounded vision analysis dispatched to local {model_name}",
            "running",
            {"endpoint": model_endpoint, "query": effective_query[:120]},
        )

        analysis_result = run_vision_analysis(
            image_bytes=file_bytes,
            user_query=effective_query,
            model_endpoint=model_endpoint,
            model_name=model_name,
            visual_evidence=evidence_text,
            process_labels=process_labels,
            grounded_tags=initial_grounded_tags,
            analysis_id=analysis_id,
            image_hash=image_hash,
        )

        # ── FAIL-CLOSED GATING ──────────────────────────────────────────
        vision_analysis_text = analysis_result.text if analysis_result.success else ""
        vision_error_internal = analysis_result.error  # Preserved for trace, NOT for user

        if analysis_result.success:
            vision_status = VISION_STATUS_COMPLETED
            log_trace(
                "VISION_ANALYSIS_COMPLETED",
                "Local multimodal visual analysis completed successfully",
                "verified",
                {
                    "characters": len(vision_analysis_text),
                    "model": model_name,
                    "retried": analysis_result.retried,
                    "analysis_id": analysis_id,
                },
            )
        elif evidence_result.success and evidence_source == "ocr":
            # Vision model failed but OCR had content — OCR fallback
            vision_status = VISION_STATUS_OCR_FALLBACK
            log_trace(
                "VISION_INFERENCE_FAILED",
                f"Vision analysis inference failed: {analysis_result.error_code}",
                "failed",
                {
                    "error_code": analysis_result.error_code,
                    "error": analysis_result.error[:300],
                    "retried": analysis_result.retried,
                },
            )
            if analysis_result.retried:
                log_trace(
                    "VISION_RETRY_STARTED",
                    "Vision analysis retry attempted and failed",
                    "failed",
                    {"retried": True},
                )
            log_trace(
                "VISION_OCR_FALLBACK",
                "Falling back to OCR-derived evidence (vision model failed)",
                "partial",
                {
                    "ocr_characters": len(evidence_ocr_text),
                    "vision_error": analysis_result.error[:200],
                },
            )
        else:
            # Both vision and OCR failed — FAIL CLOSED
            vision_status = VISION_STATUS_FAILED
            log_trace(
                "VISION_INFERENCE_FAILED",
                f"Vision analysis inference failed: {analysis_result.error_code}",
                "failed",
                {
                    "error_code": analysis_result.error_code,
                    "error": analysis_result.error[:300],
                    "retried": analysis_result.retried,
                },
            )
            if analysis_result.retried:
                log_trace(
                    "VISION_RETRY_STARTED",
                    "Vision analysis retry attempted and failed",
                    "failed",
                    {"retried": True},
                )
            log_trace(
                "VISION_ANALYSIS_FAILED",
                "Vision analysis failed — no visual conclusions will be made",
                "failed",
                {
                    "vision_error": analysis_result.error[:300],
                    "ocr_available": bool(evidence_ocr_text),
                },
            )

        # ── DOMAIN RELEVANCE EVALUATION (Post-Analysis) ─────────────────
        # Evaluated AFTER visual analysis: Unrelated images are NOT treated as model failure
        if vision_status != VISION_STATUS_FAILED:
            if process_labels or initial_grounded_tags:
                log_trace(
                    "VISION_RELEVANCE_CHECKED",
                    "Domain relevance assessed: Industrial refinery schematic / P&ID",
                    "verified",
                    {"domain": "mrpl_refinery", "scope": "in_scope"},
                )
            else:
                log_trace(
                    "VISION_RELEVANCE_CHECKED",
                    "Domain relevance assessed: General technical architecture / Non-refinery diagram",
                    "allowed",
                    {"domain": "general_technical", "scope": "in_scope_general"},
                )

        # ── DETERMINISTIC VERIFICATION ──────────────────────────────────
        # ONLY run if vision inference or OCR actually succeeded
        requires_human_review = False
        approval_id = None
        output_grounded_tags = []
        validation_result = validate_equipment_tags([])

        if vision_status == VISION_STATUS_FAILED:
            # FAIL CLOSED: Do NOT run verification on failed vision output
            verification_status = "VISION_ANALYSIS_FAILED"
            process_labels = []

        elif vision_status == VISION_STATUS_OCR_FALLBACK:
            output_grounded_tags = extract_grounded_equipment_tags(evidence_ocr_text) if evidence_ocr_text else []
            process_labels = extract_process_labels(evidence_ocr_text) if evidence_ocr_text else []

            if not output_grounded_tags:
                verification_status = "VERIFIED_NO_TAGS"
                validation_result = validate_equipment_tags([])
            else:
                validation_result = validate_equipment_tags(output_grounded_tags)
                if validation_result.requires_human_review:
                    requires_human_review = True
                    verification_status = "REQUIRES_REVIEW"
                else:
                    verification_status = "VERIFIED_TAGS_FOUND"

            log_trace(
                "VISION_VERIFICATION_COMPLETED",
                f"OCR-based verification: {verification_status}",
                "verified" if not requires_human_review else "insufficient",
                {
                    "status": verification_status,
                    "source": "ocr_only",
                    "detected_tags_count": len(output_grounded_tags),
                    "process_labels": process_labels[:8],
                },
            )

        else:
            # VISION_ANALYSIS_COMPLETED: Full verification pipeline
            output_grounded_tags = extract_grounded_equipment_tags(
                evidence_text,
                proposed_text=vision_analysis_text,
            )

            if not output_grounded_tags:
                verification_status = "VERIFIED_NO_TAGS"
                validation_result = validate_equipment_tags([])

                log_trace(
                    "VISION_VERIFICATION_COMPLETED",
                    "Deterministic Tag Check: No verifiable equipment tags detected (Process/Architecture diagram)",
                    "verified",
                    {
                        "status": verification_status,
                        "detected_tags_count": 0,
                        "process_labels": process_labels,
                        "registry_verification_performed": False,
                    },
                )
            else:
                validation_result = validate_equipment_tags(output_grounded_tags)
                if validation_result.requires_human_review:
                    requires_human_review = True
                    verification_status = "REQUIRES_REVIEW"
                    reason = (
                        f"P&ID contains {len(validation_result.mismatched_tags)} unregistered equipment tag(s): "
                        f"{', '.join(validation_result.mismatched_tags)}."
                    )

                    try:
                        prop = approvals_repo.propose_action(
                            requesting_user_id=user["id"],
                            requesting_username=user["username"],
                            action_type="PID_DRAWING_VERIFICATION",
                            affected_resource=f"image:{sanitized_name}",
                            proposed_payload={
                                "drawing_name": sanitized_name,
                                "matched_tags": validation_result.matched_tags,
                                "unregistered_tags": validation_result.mismatched_tags,
                                "reason": reason,
                            },
                        )
                        approval_id = prop.get("approval_id")
                    except Exception as ex:
                        logger.warning("Could not persist approval proposal: %s", ex)

                    log_trace(
                        "VISION_REQUIRES_REVIEW",
                        f"Deterministic Tag Check: {verification_status} (Unknown tags flagged)",
                        "insufficient",
                        {
                            "status": verification_status,
                            "matched_tags": validation_result.matched_tags,
                            "unregistered_tags": validation_result.mismatched_tags,
                            "approval_id": approval_id,
                        },
                    )
                else:
                    verification_status = "VERIFIED_TAGS_FOUND"
                    log_trace(
                        "VISION_VERIFICATION_COMPLETED",
                        f"Deterministic Tag Check: {verification_status} (All tags matched MRPL register)",
                        "verified",
                        {
                            "status": verification_status,
                            "matched_tags": validation_result.matched_tags,
                            "unregistered_tags": [],
                        },
                    )

        # ── AUDIT LOG ───────────────────────────────────────────────────
        audit_log(
            action="chat_image_analysis",
            outcome="success" if vision_status != VISION_STATUS_FAILED else "failure",
            user_id=user["id"],
            username=user["username"],
            target=f"chat_session:{session_id}",
            details={
                "analysis_id": analysis_id,
                "image_hash": image_hash,
                "filename": sanitized_name,
                "file_size": file_size,
                "model": model_name,
                "tags_detected": len(output_grounded_tags),
                "verification_status": verification_status,
                "vision_status": vision_status,
                "vision_error": vision_error_internal[:200] if vision_error_internal else "",
            },
        )

        log_trace(
            "AUDIT",
            "Immutable audit ledger entry recorded",
            "allowed",
            {"action": "chat_image_analysis", "target": f"chat_session:{session_id}", "analysis_id": analysis_id},
        )

        # ── FORMAT FINAL ASSISTANT CONTENT ──────────────────────────────
        if vision_status == VISION_STATUS_FAILED:
            # FAIL CLOSED: Controlled failure response
            final_assistant_content = (
                "Visual analysis could not be completed because the local vision model "
                "failed during inference. No visual conclusions were made."
            )

        elif vision_status == VISION_STATUS_OCR_FALLBACK:
            ocr_label_summary = ", ".join(process_labels[:8]) if process_labels else "None detected"
            ocr_tags_summary = ", ".join(output_grounded_tags) if output_grounded_tags else "None detected"
            final_assistant_content = (
                "The vision model was unable to complete visual analysis. "
                "The following findings are derived from OCR text extraction only "
                "and do not represent full visual understanding of the diagram.\n\n"
                f"**OCR-Detected Process Labels**: {ocr_label_summary}\n"
                f"**OCR-Detected Equipment Tags**: {ocr_tags_summary}\n\n"
                "⚠️ Flow direction, equipment relationships, and complete visual structure "
                "could not be verified from OCR text alone."
            )
            if output_grounded_tags and validation_result.matched_tags:
                verification_addendum = (
                    f"\n\n---\n### 🔍 Deterministic Equipment Verification (OCR-derived)\n"
                    f"**Overall Status**: `{verification_status}`\n\n"
                )
                for tag in validation_result.matched_tags:
                    details = validation_result.details.get(tag, {})
                    unit_name = details.get("unit", "Refinery")
                    service = details.get("service", "Operational Asset")
                    desc = details.get("name", "Equipment")
                    verification_addendum += f"- `✓ {tag}`: **{desc}** ({unit_name} — {service})\n"
                final_assistant_content += verification_addendum

        else:
            # VISION_ANALYSIS_COMPLETED: Normal success path
            if output_grounded_tags:
                verification_addendum = (
                    f"\n\n---\n### 🔍 Deterministic Equipment Verification\n"
                    f"**Overall Status**: `{verification_status}`\n\n"
                )
                if validation_result.matched_tags:
                    verification_addendum += "**Registered Equipment Tags**:\n"
                    for tag in validation_result.matched_tags:
                        details = validation_result.details.get(tag, {})
                        unit_name = details.get("unit", "Refinery")
                        service = details.get("service", "Operational Asset")
                        desc = details.get("name", "Equipment")
                        verification_addendum += f"- `✓ {tag}`: **{desc}** ({unit_name} — {service})\n"

                if validation_result.mismatched_tags:
                    verification_addendum += (
                        f"\n**⚠️ Unregistered Tags Flagged for Review**:\n"
                    )
                    for tag in validation_result.mismatched_tags:
                        verification_addendum += f"- `✕ {tag}`: *Tag not registered in MRPL Active Equipment Register*\n"
                    if approval_id:
                        verification_addendum += f"\n> **Human Review Action**: Proposal `#{approval_id}` registered in Human Approval Queue.\n"
            else:
                process_summary = ", ".join(process_labels[:8]) if process_labels else "Standard components/units"
                verification_addendum = (
                    f"\n\n---\n### 🔍 Deterministic Verification Status: No Verifiable Equipment Tags\n"
                    f"- **Image Type**: Architectural schematic / process flow with functional labels.\n"
                    f"- **Detected Labels**: {process_summary}\n"
                    f"- **Equipment Tag Registry**: No registry verification performed (no refinery equipment tags present).\n"
                )

            final_assistant_content = f"{vision_analysis_text}{verification_addendum}"

        # ── REPORT & PDF DELIVERABLE GENERATION ─────────────────────────
        # Determine whether user query requests a PDF or report deliverable
        wants_pdf = bool(re.search(
            r"\b(pdf|report|document|generate\s+(?:a\s+)?report|create\s+(?:a\s+)?report|export)\b",
            effective_query,
            re.IGNORECASE,
        ))
        deliverable = None

        if wants_pdf:
            log_trace(
                "REPORT_GENERATION_STARTED",
                "Dynamic report generation initiated for visual analysis",
                "running",
                {"analysis_id": analysis_id, "image_hash": image_hash},
            )

            if vision_status == VISION_STATUS_FAILED:
                # FAIL CLOSED: Failed vision must NOT produce a normal factual report or PDF
                log_trace(
                    "VISION_ANALYSIS_FAILED",
                    "Cannot generate visual report — vision model failed and no verified visual evidence available",
                    "failed",
                    {"analysis_id": analysis_id, "image_hash": image_hash},
                )
                final_assistant_content = (
                    "Visual analysis could not be completed because the local vision model "
                    "failed during inference. A verified visual report or PDF cannot be generated "
                    "without verified visual evidence."
                )
            else:
                # Build structured report object tied to current image and analysis
                report_title = f"Visual Analysis Report: {sanitized_name}"
                report_filename = f"report_{analysis_id}.pdf"

                structured_report = {
                    "analysis_id": analysis_id,
                    "image_hash": image_hash,
                    "request": effective_query,
                    "vision_status": vision_status,
                    "source": evidence_source,
                    "title": report_title,
                    "document_name": sanitized_name,
                    "filename": sanitized_name,
                    "summary": (
                        vision_analysis_text[:400] + "..."
                        if len(vision_analysis_text) > 400
                        else vision_analysis_text
                    ),
                    "verification_status": verification_status,
                    "detected_elements": output_grounded_tags or process_labels,
                    "limitations": [
                        "Analysis performed on-premise without external network egress.",
                        "Equipment tags and process units verified against visual text evidence.",
                        "Report reflects visual observations from the uploaded file.",
                    ],
                }

                log_trace(
                    "REPORT_GENERATED",
                    "Structured report object synthesized from current visual analysis",
                    "verified",
                    {"analysis_id": analysis_id, "report_title": report_title},
                )

                log_trace(
                    "PDF_GENERATION_STARTED",
                    f"Generating dynamic PDF deliverable: {report_filename}",
                    "running",
                    {"filename": report_filename, "analysis_id": analysis_id},
                )

                file_res = generate_pdf(
                    title=f"VISUAL ANALYSIS REPORT: {sanitized_name}",
                    content=vision_analysis_text,
                    document_name=sanitized_name,
                    filename=report_filename,
                    metadata={
                        "analysis_id": analysis_id,
                        "image_hash": image_hash,
                        "source": evidence_source,
                        "request": effective_query,
                        "verification_status": verification_status,
                        "document_name": sanitized_name,
                    },
                    image_bytes=file_bytes,
                    report_data=structured_report,
                )

                pdf_out_path = Path(file_res["path"])
                file_id = files_repo.record_generated_file(
                    user_id=user["id"],
                    file_type="pdf",
                    original_name=report_filename,
                    stored_path=str(pdf_out_path),
                    file_size_bytes=file_res["file_size_bytes"],
                    task_id=None,
                )

                download_url = f"/api/documents/generated/{file_id}/download?analysis_id={analysis_id}"
                deliverable = {
                    "status": "success",
                    "type": "pdf",
                    "title": report_title,
                    "filename": report_filename,
                    "output_id": file_id,
                    "download_url": download_url,
                    "file_size_bytes": file_res["file_size_bytes"],
                    "path": str(pdf_out_path),
                    "analysis_id": analysis_id,
                    "image_hash": image_hash,
                }

                log_trace(
                    "PDF_GENERATED",
                    f"PDF deliverable compiled and verified on disk ({file_res['file_size_bytes']} bytes)",
                    "verified",
                    {
                        "output_id": file_id,
                        "filename": report_filename,
                        "size_bytes": file_res["file_size_bytes"],
                        "path": str(pdf_out_path),
                    },
                )

                log_trace(
                    "PDF_RETURNED",
                    f"PDF artifact ready for download: {download_url}",
                    "allowed",
                    {"download_url": download_url, "analysis_id": analysis_id},
                )

                # Append deliverable notice to assistant message
                final_assistant_content += (
                    f"\n\n---\n### 📄 Generated Visual Analysis Report (PDF)\n"
                    f"- **Filename:** `{report_filename}`\n"
                    f"- **Analysis ID:** `{analysis_id}`\n"
                    f"- **Image Hash (SHA-256):** `{image_hash[:16]}...{image_hash[-8:]}`\n"
                    f"- **Size:** {round(file_res['file_size_bytes'] / 1024, 1)} KB\n\n"
                    f"The report includes visual observations, process/architecture interpretation, verification details, and the embedded source diagram."
                )

        # Persist messages in chat session database
        user_msg_content = f"📷 [Attached Image: {sanitized_name}] {effective_query}"
        user_msg = chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=user_msg_content,
        )

        asst_msg = chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=final_assistant_content,
            model_id=model_id,
        )
        if asst_msg:
            asst_msg["execution_trace"] = execution_trace
            asst_msg["trace_id"] = trace_id
            asst_msg["analysis_id"] = analysis_id
            asst_msg["image_hash"] = image_hash
            if deliverable:
                asst_msg["deliverable"] = deliverable

        # Map verification status for backward-compatible API response
        if vision_status == VISION_STATUS_FAILED:
            verif_summary_status = "VISION_ANALYSIS_FAILED"
        elif verification_status in ("VERIFIED_TAGS_FOUND", "VERIFIED_APPROVED"):
            verif_summary_status = "VERIFIED"
        elif verification_status == "REQUIRES_REVIEW":
            verif_summary_status = "REQUIRES_REVIEW"
        elif verification_status == "VERIFIED_NO_TAGS":
            verif_summary_status = "NO_VERIFIABLE_TAGS"
        else:
            verif_summary_status = verification_status

        return {
            "status": "success" if vision_status != VISION_STATUS_FAILED else "vision_failed",
            "session_id": session_id,
            "filename": sanitized_name,
            "analysis_id": analysis_id,
            "image_hash": image_hash,
            "task_type": "VISION",
            "model": model_name,
            "model_id": model_id,
            "analysis": vision_analysis_text if vision_status != VISION_STATUS_FAILED else "",
            "message": final_assistant_content,
            "deliverable": deliverable,
            "verification": {
                "status": verif_summary_status,
                "is_approved": not requires_human_review and vision_status != VISION_STATUS_FAILED,
                "requires_human_review": requires_human_review,
                "equipment_tags": output_grounded_tags,
                "matched_tags": validation_result.matched_tags if output_grounded_tags else [],
                "unknown_tags": validation_result.mismatched_tags if output_grounded_tags else [],
                "process_labels": process_labels,
                "approval_id": approval_id,
            },
            "tags_detected": output_grounded_tags,
            "verification_status": verification_status,
            "vision_status": vision_status,
            "vision_error": "",  # Do NOT expose raw error to user
            "requires_human_review": requires_human_review,
            "approval_id": approval_id,
            "execution_trace": execution_trace,
            "trace_id": trace_id,
            "user_message": user_msg,
            "assistant_message": asst_msg,
        }

    finally:
        # Guarantee zero disk leakage: Always remove temporary image file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning("Could not delete temporary image %s: %s", temp_path, e)

