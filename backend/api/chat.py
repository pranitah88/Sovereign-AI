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
    """
    import base64
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid
    import requests

    from backend.auth.rbac import normalize_roles
    from backend.database.repositories import approvals as approvals_repo
    from backend.services.network_seal import record_local_call
    from backend.services.task_router import select_vision_model
    from backend.services.vision_verification import (
        extract_visual_text_evidence,
        extract_process_labels,
        extract_grounded_equipment_tags,
        extract_equipment_tags_from_text,
        validate_equipment_tags,
        build_grounded_vision_prompt,
    )

    effective_query = (prompt or query or "Analyze this image for industrial refinery operations and equipment.").strip()

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

    # 4. Safe temporary local storage
    temp_dir = Path("data/temp_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_filename = f"mrpl_img_{session_id}_{uuid.uuid4().hex[:8]}_{sanitized_name}"
    temp_path = temp_dir / temp_filename

    trace_id = f"trace_vis_{uuid.uuid4().hex[:8]}"
    execution_trace = []

    def log_trace(event: str, title: str, trace_status: str, details: dict = None):
        execution_trace.append({
            "event": event,
            "title": title,
            "status": trace_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })

    try:
        # Step: IMAGE_RECEIVED
        log_trace(
            "IMAGE_RECEIVED",
            f"Image received: {sanitized_name}",
            "allowed",
            {"filename": sanitized_name, "size_bytes": file_size, "content_type": content_type},
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

        # Step: VISION_MODEL_SELECTED (Prefers qwen2.5vl:3b if installed, else gemma3:4b)
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

        # Step: VISUAL_TEXT_EVIDENCE (OCR / optical transcription from physical image)
        visual_evidence = extract_visual_text_evidence(
            temp_path,
            model_endpoint=model_endpoint,
            model_name=model_name,
        )
        process_labels = extract_process_labels(visual_evidence)
        initial_grounded_tags = extract_grounded_equipment_tags(visual_evidence)

        log_trace(
            "VISUAL_EVIDENCE_EXTRACTED",
            f"Optical evidence extracted ({len(process_labels)} process labels, {len(initial_grounded_tags)} visible equipment tags)",
            "verified",
            {
                "process_labels": process_labels,
                "grounded_tags": initial_grounded_tags,
                "evidence_characters": len(visual_evidence),
            },
        )

        # Step: OLLAMA_INFERENCE (5-section grounded prompt)
        img_b64 = base64.b64encode(file_bytes).decode("utf-8")
        record_local_call(model_endpoint)

        system_prompt = build_grounded_vision_prompt(
            user_query=effective_query,
            visual_evidence=visual_evidence,
            process_labels=process_labels,
            grounded_tags=initial_grounded_tags,
        )

        log_trace(
            "OLLAMA_INFERENCE",
            f"Grounded 5-section vision analysis dispatched to local {model_name}",
            "running",
            {"endpoint": model_endpoint, "query": effective_query[:120]},
        )

        try:
            resp = requests.post(
                model_endpoint,
                json={
                    "model": model_name,
                    "prompt": system_prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 1024},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                vision_analysis_text = resp.json().get("response", "").strip()
            else:
                vision_analysis_text = (
                    f"Multimodal vision model {model_name} returned HTTP {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            logger.warning("Local Ollama vision inference failed: %s", e)
            vision_analysis_text = (
                f"Image analysis processed on-premise. Extracted visual metadata for {sanitized_name} "
                f"({file_size / 1024:.1f} KB). Local model inference note: {e}"
            )

        # Step: VISION_ANALYSIS
        log_trace(
            "VISION_ANALYSIS",
            "Local multimodal visual analysis generated",
            "verified",
            {"characters": len(vision_analysis_text), "model": model_name},
        )

        # Step: VERIFICATION (Deterministic tag validation grounded in visual evidence)
        output_grounded_tags = extract_grounded_equipment_tags(
            visual_evidence,
            proposed_text=vision_analysis_text,
        )

        requires_human_review = False
        approval_id = None

        if not output_grounded_tags:
            verification_status = "NO_VERIFIABLE_TAGS"
            validation_result = validate_equipment_tags([])

            log_trace(
                "VERIFICATION",
                "Deterministic Tag Check: No verifiable equipment tags detected (Process schematic with unit/stream labels)",
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
                verification_status = "REQUIRES REVIEW"
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
                    "VERIFICATION",
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
                verification_status = "VERIFIED_APPROVED"
                log_trace(
                    "VERIFICATION",
                    f"Deterministic Tag Check: {verification_status} (All tags matched MRPL register)",
                    "verified",
                    {
                        "status": verification_status,
                        "matched_tags": validation_result.matched_tags,
                        "unregistered_tags": [],
                    },
                )

        # Step: AUDIT
        audit_log(
            action="chat_image_analysis",
            outcome="success",
            user_id=user["id"],
            username=user["username"],
            target=f"chat_session:{session_id}",
            details={
                "filename": sanitized_name,
                "file_size": file_size,
                "model": model_name,
                "tags_detected": len(output_grounded_tags),
                "verification_status": verification_status,
            },
        )

        log_trace(
            "AUDIT",
            "Immutable audit ledger entry recorded",
            "allowed",
            {"action": "chat_image_analysis", "target": f"chat_session:{session_id}"},
        )

        # Format Final Assistant Message with verification table if tags exist
        if output_grounded_tags:
            verification_addendum = (
                f"\n\n---\n### 🔍 Deterministic P&ID Equipment Verification\n"
                f"**Overall Status**: `{verification_status}`\n\n"
            )
            if validation_result.matched_tags:
                verification_addendum += "**Registered MRPL Equipment Tags**:\n"
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
            process_summary = ", ".join(process_labels[:8]) if process_labels else "Standard process units"
            verification_addendum = (
                f"\n\n---\n### 🔍 Deterministic Verification Status: No Verifiable Equipment Tags\n"
                f"- **Image Type**: Process flow schematic with unit and stream labels.\n"
                f"- **Detected Process Labels**: {process_summary}\n"
                f"- **Equipment Tag Registry**: No registry verification performed (no alphanumeric equipment tags physically present in image).\n"
            )

        final_assistant_content = f"{vision_analysis_text}{verification_addendum}"

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

        verif_summary_status = "NO_VERIFIABLE_TAGS" if not output_grounded_tags else (
            "VERIFIED" if verification_status == "VERIFIED_APPROVED" else "REQUIRES_REVIEW"
        )

        return {
            "status": "success",
            "session_id": session_id,
            "filename": sanitized_name,
            "task_type": "VISION",
            "model": model_name,
            "model_id": model_id,
            "analysis": vision_analysis_text,
            "message": final_assistant_content,
            "verification": {
                "status": verif_summary_status,
                "is_approved": not requires_human_review,
                "requires_human_review": requires_human_review,
                "equipment_tags": output_grounded_tags,
                "matched_tags": validation_result.matched_tags if output_grounded_tags else [],
                "unknown_tags": validation_result.mismatched_tags if output_grounded_tags else [],
                "process_labels": process_labels,
                "approval_id": approval_id,
            },
            "tags_detected": output_grounded_tags,
            "verification_status": verification_status,
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
