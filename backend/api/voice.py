"""
MRPL Sovereign AI Workbench — Voice Assistant API Endpoints.
Sovereign on-premise speech recognition, multilingual detection, and speech synthesis.
"""

import io
import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.auth.rbac import require_permission
from backend.database.repositories import chat as chat_repo
from backend.services.voice.voice_service import get_voice_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".flac"}
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15 MB


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@router.get("/status")
async def get_voice_status(
    user: Annotated[dict, Depends(require_permission("voice"))],
):
    """Return local ASR and TTS engine health and readiness."""
    service = get_voice_service()
    return service.get_status()


@router.get("/greeting")
async def get_voice_greeting(
    user: Annotated[dict, Depends(require_permission("voice"))],
    language: str = "en",
):
    """
    Return short natural conversational greeting text and local synthesized audio.
    Used when the voice assistant session starts so the assistant greets the user hands-free.
    """
    service = get_voice_service()
    return service.get_greeting(language=language)


@router.post("/transcribe")
async def transcribe_speech(
    user: Annotated[dict, Depends(require_permission("voice"))],
    file: UploadFile = File(...),
    language: str = Form(None),
):
    """
    Transcribe uploaded audio file and detect language.
    Does not execute reasoning query.
    """
    ext = Path(file.filename or "audio.webm").suffix.lower()
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty (0 bytes)")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds 15 MB limit ({len(audio_bytes) / (1024*1024):.1f}MB)",
        )

    service = get_voice_service()
    result = service.transcribe_audio(audio_bytes, language_hint=language)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.get("error") or "Speech recognition engine error",
        )

    return result


@router.post("/process")
async def process_voice_query(
    user: Annotated[dict, Depends(require_permission("voice"))],
    file: UploadFile | None = File(None),
    session_id: int | None = Form(None),
    language: str = Form(None),
    confirmed_query: str | None = Form(None),
    clarification_context: str | None = Form(None),
):
    """
    Full sovereign voice pipeline:
    Audio -> Local ASR -> Language Detection -> Existing Governance Pipeline -> Validated Text -> Local TTS.
    """
    service = get_voice_service()

    # If user confirmed query text directly without re-uploading audio
    if confirmed_query and not file:
        audio_data = b""
    else:
        if not file:
            raise HTTPException(status_code=400, detail="Missing audio file or confirmed query text")

        ext = Path(file.filename or "audio.webm").suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
            )

        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file exceeds 15 MB limit ({len(audio_bytes) / (1024*1024):.1f}MB)",
            )
        audio_data = audio_bytes

    # Session ownership verification if session_id provided
    if session_id is not None:
        session = chat_repo.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        from backend.auth.rbac import normalize_roles
        user_roles = normalize_roles(user.get("roles", []))
        if session["user_id"] != user["id"] and "administrator" not in user_roles:
            raise HTTPException(status_code=403, detail="Access denied: Cannot post to another user's session")

    result = await service.process_voice_query(
        audio_data=audio_data,
        user=user,
        session_id=session_id,
        language_hint=language,
        confirmed_query=confirmed_query,
        clarification_context=clarification_context,
    )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error") or "Voice processing error",
        )

    return result


@router.post("/tts")
async def generate_tts(
    body: TTSRequest,
    user: Annotated[dict, Depends(require_permission("voice"))],
):
    """
    Synthesize validated text into speech WAV stream strictly on-premise.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    service = get_voice_service()
    tts_res = service.tts.synthesize(body.text, language=body.language)

    if tts_res.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=tts_res.get("error") or "TTS engine unavailable",
        )

    return Response(
        content=tts_res["audio_bytes"],
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=voice_response.wav",
            "X-Voice-Duration": str(tts_res.get("duration_seconds", 0.0)),
        },
    )


@router.post("/process_stream")
async def process_voice_stream(
    user: Annotated[dict, Depends(require_permission("voice"))],
    file: UploadFile | None = File(None),
    session_id: int | None = Form(None),
    language: str | None = Form(None),
    confirmed_query: str | None = Form(None),
    clarification_context: str | None = Form(None),
    turn_id: int | None = Form(None),
):
    """
    Real-time conversational streaming voice pipeline (SSE):
    Audio -> Streaming ASR -> Governance -> Fast Path vs RAG Routing
    -> Streaming Ollama tokens -> Sentence Buffer -> Streaming Audio Chunks (TTS).
    """
    service = get_voice_service()

    if confirmed_query and not file:
        audio_data = b""
    else:
        if not file:
            raise HTTPException(status_code=400, detail="Missing audio file or confirmed query text")

        ext = Path(file.filename or "audio.webm").suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
            )

        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file exceeds 15 MB limit ({len(audio_bytes) / (1024*1024):.1f}MB)",
            )
        audio_data = audio_bytes

    if session_id is not None:
        session = chat_repo.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        from backend.auth.rbac import normalize_roles
        user_roles = normalize_roles(user.get("roles", []))
        if session["user_id"] != user["id"] and "administrator" not in user_roles:
            raise HTTPException(status_code=403, detail="Access denied: Cannot post to another user's session")

    async def sse_event_generator():
        try:
            async for event in service.stream_voice_query(
                audio_data=audio_data,
                user=user,
                session_id=session_id,
                language_hint=language,
                confirmed_query=confirmed_query,
                clarification_context=clarification_context,
                turn_id=turn_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error("SSE stream error in process_voice_stream: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'error': str(exc), 'turn_id': turn_id})}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")


@router.post("/partial_transcribe")
async def partial_transcribe_speech(
    user: Annotated[dict, Depends(require_permission("voice"))],
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """
    Low-latency partial transcription of live audio slices for real-time UI interim text.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        return {"text": "", "status": "empty"}

    service = get_voice_service()
    if hasattr(service.asr, "transcribe_partial"):
        return service.asr.transcribe_partial(audio_bytes, language=language)
    return service.transcribe_audio(audio_bytes, language_hint=language)


@router.post("/interrupt")
async def interrupt_voice(
    user: Annotated[dict, Depends(require_permission("voice"))],
    session_id: int | None = Form(None),
):
    """
    Barge-in interruption signal from client. Immediately stops server-side audio generation.
    """
    service = get_voice_service()
    service.interrupt_session(session_id)
    try:
        from backend.services.audit import audit_log
        audit_log(
            action="voice_interrupted",
            outcome="success",
            user_id=user["id"],
            username=user.get("username", "anonymous"),
            target=f"session:{session_id or 'direct'}",
            details={"reason": "User barge-in detected"},
        )
    except Exception as exc:
        logger.warning("Could not log voice interruption: %s", exc)

    return {"status": "interrupted", "session_id": session_id}
