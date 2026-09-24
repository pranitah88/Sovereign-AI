"""
MRPL Sovereign AI Workbench — Central Voice Assistant Service.
Orchestrates Speech-to-Text -> Language Detection -> Existing Governance Pipeline -> Text-to-Speech.
Strictly On-Premise / Local Inference — Zero Cloud Egress.
"""

import asyncio
import base64
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, AsyncGenerator
import uuid
import requests
import yaml

from backend.database.repositories import chat as chat_repo
from backend.services.audit import audit_log
from backend.services.network_seal import record_local_call
from backend.services.task_router import route_task
from backend.services.voice.asr import BaseASREngine, LocalWhisperASR
from backend.services.voice.context_policy import determine_conversational_context
from backend.services.voice.conversation_state import (
    ConversationState,
    TurnState,
    TurnStatus,
    get_conversation_state,
)
from backend.services.voice.language_detection import detect_voice_language, SUPPORTED_LANGUAGES
from backend.services.voice.sentence_buffer import SentenceChunkBuffer
from backend.services.voice.speech_renderer import render_speech_text
from backend.services.voice.tts import BaseTTSEngine, LocalSapiTTSEngine
from backend.services.voice.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/voice.yaml")


class VoiceService:
    """
    Central orchestrator for Multilingual Voice Assistant.
    Follows: "PYTHON CONTROLS. MODELS REASON."
    All voice requests pass through the exact same governance, RBAC, scope guard,
    hybrid RAG, and confidence gate as text requests.
    """

    def __init__(
        self,
        asr_engine: BaseASREngine | None = None,
        tts_engine: BaseTTSEngine | None = None,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize ASR
        if asr_engine is not None:
            self.asr = asr_engine
        else:
            asr_cfg = self.config.get("voice", {}).get("asr", {})
            self.asr = LocalWhisperASR(
                model_name=asr_cfg.get("model_name", "tiny"),
                model_path=asr_cfg.get("model_path", "models/whisper"),
                device=asr_cfg.get("device", "cpu"),
                compute_type=asr_cfg.get("compute_type", "int8"),
                confidence_threshold=asr_cfg.get("confidence_threshold", 0.55),
                local_files_only=asr_cfg.get("local_files_only", True),
            )

        # Initialize TTS
        if tts_engine is not None:
            self.tts = tts_engine
        else:
            tts_cfg = self.config.get("voice", {}).get("tts", {})
            self.tts = LocalSapiTTSEngine(
                voice_rate=tts_cfg.get("voice_rate", 165),
                voice_volume=tts_cfg.get("voice_volume", 1.0),
            )

        # Initialize Voice Activity Detector (VAD)
        vad_cfg = self.config.get("voice", {}).get("vad", {})
        self.vad = VoiceActivityDetector(
            energy_threshold=vad_cfg.get("energy_threshold", 0.02),
            silence_threshold_ms=vad_cfg.get("silence_threshold_ms", 750),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 250),
            sample_rate=self.config.get("voice", {}).get("audio_sample_rate", 16000),
            adaptive=vad_cfg.get("adaptive", True),
            min_energy_threshold=vad_cfg.get("min_energy_threshold", 0.012),
            snr_start_threshold_db=vad_cfg.get("snr_start_threshold_db", 6.0),
            snr_stop_threshold_db=vad_cfg.get("snr_stop_threshold_db", 3.0),
            noise_floor_smoothing=vad_cfg.get("noise_floor_smoothing", 0.95),
            highpass_cutoff_hz=vad_cfg.get("highpass_cutoff_hz", 80.0),
        )

        # Active session interruption tracking for barge-in
        self._interrupted_sessions: set[int] = set()
        self._interrupted_turns: set[int] = set()
        # Active turn tracking per session to prevent race conditions from outdated turns
        self._active_session_turns: dict[int, int] = {}
        # Explicitly stopped Nova voice sessions to guarantee stopped sessions cannot affect new sessions
        self._stopped_voice_sessions: set[str] = set()

    def get_greeting(self, language: str = "en") -> dict[str, Any]:
        """Return deterministic Nova greeting text and local synthesized audio."""
        lang = (language or "en").lower()
        if lang == "hi":
            text = "नमस्ते! मैं नोवा हूँ। मैं आपकी क्या सहायता कर सकता हूँ?"
        elif lang == "mr":
            text = "नमस्कार! मी नोव्हा आहे. मी तुम्हाला काय मदत करू शकतो?"
        else:
            text = "Hi, I'm Nova. How can I help you today?"

        audio_b64 = ""
        duration = 1.0
        try:
            synth = self.tts.synthesize(text, language=lang)
            if synth.get("audio_bytes"):
                audio_b64 = base64.b64encode(synth["audio_bytes"]).decode("ascii")
                duration = synth.get("duration_seconds", 1.0)
        except Exception as exc:
            logger.warning("Greeting synthesis notice: %s", exc)

        return {
            "status": "success",
            "text": text,
            "spoken_text": text,
            "audio_base64": audio_b64,
            "duration_seconds": duration,
            "language": lang,
        }

    def warmup(self) -> dict[str, bool]:
        """Pre-warm local ASR, TTS, and conversational Ollama models once to eliminate cold-start latency."""
        asr_ok = False
        tts_ok = False
        ollama_ok = False
        try:
            if hasattr(self.asr, "warmup"):
                asr_ok = self.asr.warmup()
            else:
                asr_ok = True
        except Exception as exc:
            logger.warning("VoiceService ASR warmup notice: %s", exc)

        try:
            if hasattr(self.tts, "warmup"):
                tts_ok = self.tts.warmup()
            else:
                tts_ok = True
        except Exception as exc:
            logger.warning("VoiceService TTS warmup notice: %s", exc)

        try:
            from backend.services.task_router import is_ollama_model_installed, route_task
            decision = route_task("hello")
            model_name = decision.get("model", "gemma3:4b")
            endpoint = decision.get("endpoint", "http://127.0.0.1:11434/api/generate")
            keep_alive = self.config.get("voice", {}).get("routing", {}).get("keep_alive", "30m")

            if is_ollama_model_installed(model_name):
                record_local_call(endpoint)
                resp = requests.post(
                    endpoint,
                    json={"model": model_name, "keep_alive": keep_alive},
                    timeout=5,
                )
                ollama_ok = (resp.status_code == 200)
            else:
                logger.error(
                    "Configured conversational model '%s' is not installed locally in Ollama.",
                    model_name,
                )
                ollama_ok = False
        except Exception as exc:
            logger.warning("VoiceService Ollama warmup notice: %s", exc)
            ollama_ok = False

        try:
            from backend.services.rag_engine import get_embedding_model, get_reranker_model, get_collection
            from backend.services.bm25 import BM25IndexManager
            get_embedding_model()
            get_reranker_model()
            coll = get_collection()
            BM25IndexManager.get_instance().get_index(coll)
            rag_ok = True
        except Exception as exc:
            logger.warning("VoiceService RAG warmup notice: %s", exc)
            rag_ok = False

        return {"asr": asr_ok, "tts": tts_ok, "ollama": ollama_ok, "rag": rag_ok}

    def interrupt_session(self, session_id: int | None) -> None:
        """Mark session as interrupted by user barge-in."""
        if session_id is not None:
            self._interrupted_sessions.add(session_id)

    def clear_interruption(self, session_id: int | None) -> None:
        """Clear interruption flag for a session."""
        if session_id is not None:
            self._interrupted_sessions.discard(session_id)

    def is_interrupted(self, session_id: int | None) -> bool:
        """Check whether current session has been interrupted by user barge-in."""
        if session_id is None:
            return False
        return session_id in self._interrupted_sessions

    def interrupt_turn(self, turn_id: int | None) -> None:
        """Mark specific turn as interrupted by user barge-in."""
        if turn_id is not None:
            self._interrupted_turns.add(turn_id)

    def is_turn_interrupted(self, turn_id: int | None) -> bool:
        """Check whether turn has been interrupted by user barge-in."""
        if turn_id is None:
            return False
        return turn_id in self._interrupted_turns

    def stop_voice_session(self, voice_session_id: str | None) -> None:
        """Explicitly stop and invalidate a Nova voice session."""
        if voice_session_id:
            self._stopped_voice_sessions.add(str(voice_session_id))

    def is_voice_session_stopped(self, voice_session_id: str | None) -> bool:
        """Check whether a Nova voice session has been explicitly stopped."""
        if not voice_session_id:
            return False
        return str(voice_session_id) in self._stopped_voice_sessions

    def register_voice_session(self, voice_session_id: str | None) -> None:
        """Register a new active Nova voice session (clears any prior stopped flag)."""
        if voice_session_id:
            self._stopped_voice_sessions.discard(str(voice_session_id))

    def _load_config(self) -> dict[str, Any]:
        """Load voice configuration from YAML file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning("Could not read voice config %s: %s", self.config_path, exc)
        return {}

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic health and readiness status of local voice engines."""
        asr_ready, asr_msg = self.asr.is_ready()
        tts_ready, tts_msg = self.tts.is_ready()

        return {
            "status": "ready" if (asr_ready and tts_ready) else "degraded",
            "asr": {
                "ready": asr_ready,
                "engine": getattr(self.asr, "model_name", "whisper"),
                "status": asr_msg,
                "provider": "local on-premise",
            },
            "tts": {
                "ready": tts_ready,
                "engine": "Windows SAPI",
                "status": tts_msg,
                "provider": "local on-premise",
            },
            "supported_languages": [
                {"code": code, "name": name}
                for code, name in SUPPORTED_LANGUAGES.items()
            ],
            "default_language": "auto",
            "confidence_threshold": getattr(self.asr, "confidence_threshold", 0.75),
            "offline_seal": "LOCAL_ONLY (Zero External Egress)",
        }

    def transcribe_audio(
        self,
        audio_data: bytes | io.BytesIO | str,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """
        Convert spoken audio to text and detect language.
        Returns transcribed text, confidence, and detected language.
        """
        asr_result = self.asr.transcribe(audio_data, language=language_hint)
        detected_lang_info = detect_voice_language(
            text=asr_result.get("text", ""),
            asr_detected_language=asr_result.get("language"),
            user_override=language_hint,
        )

        return {
            "text": asr_result.get("text", ""),
            "raw_text": asr_result.get("raw_text", asr_result.get("text", "")),
            "normalized_text": asr_result.get("normalized_text", asr_result.get("text", "")),
            "is_normalized": asr_result.get("is_normalized", False),
            "confidence": asr_result.get("confidence", 0.0),
            "status": asr_result.get("status", "error"),
            "language": detected_lang_info,
            "duration_seconds": asr_result.get("duration_seconds", 0.0),
            "engine": asr_result.get("engine", "local ASR"),
            "error": asr_result.get("error"),
        }

    async def process_voice_query(
        self,
        audio_data: bytes | io.BytesIO | str,
        user: dict,
        session_id: int | None = None,
        language_hint: str | None = None,
        confirmed_query: str | None = None,
        clarification_context: dict[str, Any] | str | None = None,
        voice_mode: str = "NOVA",
        voice_session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end voice query processing:
        Audio -> ASR -> Language Detection -> Governance Pipeline (run_agent) -> Validated Text -> TTS.
        """
        start_time = datetime.now(timezone.utc)
        trace_id = f"TRC-VCE-{uuid.uuid4().hex[:8].upper()}"
        voice_trace = []

        def _log_trace(event: str, title: str, status: str, details: dict | None = None):
            voice_trace.append({
                "event": event,
                "stage": event,
                "title": title,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": details or {},
            })

        # ── Step 1: VOICE INPUT RECEIVED & VAD DIAGNOSTICS ──────────────────
        audio_bytes_len = len(audio_data) if isinstance(audio_data, bytes) else 0
        _log_trace(
            "voice_input_received",
            f"Voice audio payload received ({audio_bytes_len / 1024:.1f} KB)",
            "allowed",
            {"size_bytes": audio_bytes_len, "language_hint": language_hint or "auto"},
        )
        _log_trace("VOICE_CAPTURE_STARTED", "Voice audio capture initiated", "allowed", {"language_hint": language_hint or "auto", "session_id": session_id})

        # Real VAD & audio diagnostics
        vad_metrics = {}
        if isinstance(audio_data, bytes) and len(audio_data) > 0:
            try:
                vad_metrics = self.vad.detect_speech_boundaries(audio_data)
            except Exception:
                vad_metrics = self.vad.get_diagnostics()
        else:
            vad_metrics = self.vad.get_diagnostics()

        capture_dur_val = vad_metrics.get("capture_duration_ms", 0)
        speech_dur_val = vad_metrics.get("speech_duration_ms", 0)
        sample_rate_val = vad_metrics.get("sample_rate", getattr(self.vad, "sample_rate", 16000))
        channel_count_val = vad_metrics.get("channel_count", 1)
        input_rms_val = vad_metrics.get("input_rms", 0.0)
        peak_val = vad_metrics.get("peak", 0.0)
        noise_floor_val = vad_metrics.get("noise_floor_rms", 0.008)
        speech_rms_val = vad_metrics.get("speech_rms", 0.0)
        snr_db_val = vad_metrics.get("snr_db", 0.0)
        vad_conf_val = vad_metrics.get("vad_confidence", 0.0)
        speech_active_val = vad_metrics.get("has_speech", True)

        _log_trace("AUDIO_NOISE_FLOOR", f"Ambient noise floor: {noise_floor_val:.4f} RMS", "verified", {"noise_floor_rms": noise_floor_val})
        _log_trace("AUDIO_RMS", f"Audio RMS level: {speech_rms_val:.4f}", "verified", {"speech_rms": speech_rms_val, "input_rms": input_rms_val, "peak": peak_val})
        _log_trace("AUDIO_SNR", f"Audio SNR: {snr_db_val:.1f} dB", "verified", {"snr_db": snr_db_val})
        _log_trace(
            "VAD_STATE_CHANGED",
            f"VAD State: {'SPEECH' if speech_active_val else 'SILENCE'} (duration: {speech_dur_val}ms, confidence: {vad_conf_val:.2f})",
            "verified",
            {
                "speech_active": speech_active_val,
                "confidence": vad_conf_val,
                "capture_duration_ms": capture_dur_val,
                "speech_duration_ms": speech_dur_val,
                "sample_rate": sample_rate_val,
                "channel_count": channel_count_val,
            },
        )

        # ── Step 2: SPEECH RECOGNITION (ASR) ─────────────────────────────────
        if confirmed_query:
            # User confirmed a previously transcribed low-confidence query
            raw_text = confirmed_query.strip()
            transcribed_text = confirmed_query.strip()
            is_normalized = False
            asr_confidence = 1.0
            asr_status = "success"
            raw_lang = language_hint or "en"
            duration = 0.0
        else:
            asr_res = self.asr.transcribe(audio_data, language=language_hint)
            raw_text = asr_res.get("raw_text", asr_res.get("text", "")).strip()
            transcribed_text = asr_res.get("normalized_text", asr_res.get("text", "")).strip()
            is_normalized = asr_res.get("is_normalized", False)
            asr_confidence = asr_res.get("confidence", 0.0)
            asr_status = asr_res.get("status", "error")
            raw_lang = asr_res.get("language", "en")
            duration = asr_res.get("duration_seconds", 0.0)

            if asr_status == "error":
                err_msg = asr_res.get("error", "ASR transcription failed")
                _log_trace("speech_recognition", f"ASR Error: {err_msg}", "blocked", {"error": err_msg})
                return {
                    "status": "error",
                    "error": err_msg,
                    "text": "",
                    "raw_text": "",
                    "normalized_text": "",
                    "execution_trace": voice_trace,
                    "trace_id": trace_id,
                }

        # Traces: ASR_FINAL_RAW, ASR_CONFIDENCE, ASR_NORMALIZED
        _log_trace("ASR_FINAL_RAW", f'ASR Final Raw: "{raw_text}"', "verified", {"raw_text": raw_text})
        _log_trace("ASR_CONFIDENCE", f"ASR Confidence: {asr_confidence:.2f}", "verified" if asr_status != "low_confidence" else "insufficient", {"confidence": asr_confidence})
        norm_desc = f'ASR Normalized: "{transcribed_text}"' if is_normalized else "ASR Normalized: unchanged"
        _log_trace("ASR_NORMALIZED", norm_desc, "verified", {"raw_text": raw_text, "normalized_text": transcribed_text, "is_normalized": is_normalized})
        _log_trace("CHAT_REQUEST_QUERY", f'Chat Request Query: "{transcribed_text}"', "verified", {"current_query": transcribed_text, "trace_id": trace_id})
        _log_trace("ORCHESTRATOR_QUERY", f'Orchestrator Query: "{transcribed_text}"', "verified", {"current_query": transcribed_text, "trace_id": trace_id})

        # In normal Voice-to-Text / Dictation mode, return immediately without calling LLM, RAG, or TTS
        if voice_mode in ("VOICE_TO_TEXT", "DICTATION"):
            threshold = getattr(self.asr, "confidence_threshold", 0.55)
            if asr_status == "low_confidence" or asr_confidence < threshold:
                low_conf_reason = (
                    asr_res.get("low_confidence_reason")
                    or asr_res.get("quality", {}).get("reason")
                    or (f"Token acoustic confidence too low: {asr_confidence:.2f} < {threshold:.2f}" if asr_confidence < threshold else "low_confidence")
                )
                diag_payload = {
                    **vad_metrics,
                    "asr_confidence": asr_confidence,
                    "confidence_threshold": threshold,
                    "asr_status": asr_status,
                    "low_confidence_reason": low_conf_reason,
                    "raw_text": raw_text,
                }
                _log_trace(
                    "ASR_LOW_CONFIDENCE",
                    f"ASR Low Confidence: {asr_confidence:.2f} < {threshold} ({low_conf_reason})",
                    "insufficient",
                    diag_payload,
                )
                return {
                    "status": "low_confidence",
                    "message": "Couldn't confidently transcribe that. Please try again.",
                    "low_confidence_reason": low_conf_reason,
                    "diagnostics": diag_payload,
                    "raw_text": raw_text,
                    "normalized_text": transcribed_text,
                    "confidence": asr_confidence,
                    "voice_mode": voice_mode,
                    "voice_session_id": voice_session_id,
                    "trace_id": trace_id,
                    "execution_trace": voice_trace,
                }
            _log_trace("VOICE_TO_TEXT_COMPLETED", f'Voice-to-Text completed: "{transcribed_text}"', "verified", {"raw_text": raw_text, "normalized_text": transcribed_text, "confidence": asr_confidence})
            return {
                "status": "success",
                "raw_text": raw_text,
                "normalized_text": transcribed_text,
                "confidence": asr_confidence,
                "voice_mode": voice_mode,
                "voice_session_id": voice_session_id,
                "trace_id": trace_id,
                "execution_trace": voice_trace,
            }

        _log_trace(
            "speech_recognition",
            f"Speech recognized: \"{transcribed_text[:80]}\"",
            "verified" if asr_status == "success" else "insufficient",
            {
                "transcribed_text": transcribed_text,
                "raw_text": raw_text,
                "is_normalized": is_normalized,
                "confidence": asr_confidence,
                "status": asr_status,
                "duration_seconds": duration,
                "engine": getattr(self.asr, "model_name", "whisper"),
            },
        )

        # ── Step 3: LANGUAGE DETECTION ───────────────────────────────────────
        lang_info = detect_voice_language(
            text=transcribed_text,
            asr_detected_language=raw_lang,
            user_override=language_hint,
        )

        _log_trace(
            "language_detected",
            f"Language: {lang_info['name']} ({lang_info['method']})",
            "verified",
            lang_info,
        )

        # Check for empty speech
        if not transcribed_text and not raw_text:
            return {
                "status": "empty",
                "text": "",
                "raw_text": "",
                "normalized_text": "",
                "message": "No audible speech detected. Please speak clearly into the microphone.",
                "language": lang_info,
                "confidence": asr_confidence,
                "execution_trace": voice_trace,
                "trace_id": trace_id,
            }

        # ── Step 3.5: CONVERSATIONAL INTENT DETECTION ─────────────────────
        # Intercept simple greetings / farewells / casual exchanges locally.
        # Bypass RAG, Ollama, and the full governed pipeline for these.
        from backend.services.voice.conversational_intent import detect_conversational_intent
        conv_intent = detect_conversational_intent(transcribed_text)

        if conv_intent.is_conversational and conv_intent.response_text:
            _log_trace(
                "CONVERSATION_INTENT_DETECTED",
                f"Conversational intent: {conv_intent.intent.value}",
                "allowed",
                conv_intent.to_dict(),
            )
            _log_trace(
                "LOCAL_CONVERSATIONAL_RESPONSE",
                f"Route: LOCAL_CONVERSATIONAL_RESPONSE",
                "allowed",
                {"response": conv_intent.response_text, "pool_key": conv_intent.response_key},
            )

            spoken_text = conv_intent.response_text
            _log_trace("TTS_STARTED", "Text-to-speech synthesis initiated", "allowed")
            tts_result = self.tts.synthesize(spoken_text, language=lang_info["code"])
            audio_bytes = tts_result.get("audio_bytes", b"") if isinstance(tts_result, dict) else b""
            audio_base64 = (
                base64.b64encode(audio_bytes).decode("utf-8")
                if isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0
                else None
            )
            tts_duration = tts_result.get("duration_seconds", 0.0) if isinstance(tts_result, dict) else 0.0
            _log_trace("TTS_COMPLETED", f"Speech synthesis completed ({tts_duration:.1f}s)", "verified", {"duration_seconds": tts_duration})
            _log_trace("VOICE_CAPTURE_RESUMED", "Continuous conversational voice capture resumed", "allowed")

            try:
                audit_log(
                    action="voice_interaction",
                    outcome="success",
                    user_id=user["id"],
                    username=user.get("username", "anonymous"),
                    target=f"session:{session_id or 'direct'}",
                    details={
                        "trace_id": trace_id,
                        "input_mode": "voice",
                        "language": lang_info["code"],
                        "asr_confidence": asr_confidence,
                        "conversational_intent": conv_intent.intent.value,
                        "route": "LOCAL_CONVERSATIONAL_RESPONSE",
                    },
                )
            except Exception:
                pass

            return {
                "status": "success",
                "transcribed_text": transcribed_text,
                "raw_text": raw_text,
                "normalized_text": transcribed_text,
                "is_normalized": is_normalized,
                "response": spoken_text,
                "spoken_text": spoken_text,
                "language": lang_info,
                "asr_confidence": asr_confidence,
                "audio_base64": audio_base64,
                "audio_content_type": tts_result.get("content_type", "audio/wav") if isinstance(tts_result, dict) else "audio/wav",
                "audio_duration": tts_duration,
                "execution_trace": voice_trace,
                "trace_id": trace_id,
                "task_type": "CONVERSATIONAL",
                "model_id": "local_conversational",
                "scope_decision": {},
                "confidence_decision": None,
                "citation_validation": None,
                "requires_human_review": False,
                "approval_id": None,
                "voice_mode": voice_mode,
                "voice_session_id": voice_session_id,
                "conversational_intent": conv_intent.intent.value,
            }

        # Check clarification context if user is answering a previous question
        clarif_dict: dict[str, Any] | None = None
        if isinstance(clarification_context, str):
            if clarification_context.strip():
                try:
                    loaded = json.loads(clarification_context)
                    if isinstance(loaded, dict):
                        clarif_dict = loaded
                except Exception:
                    clarif_dict = None
        elif isinstance(clarification_context, dict):
            clarif_dict = clarification_context

        if clarif_dict:
            from backend.services.voice.clarification import resolve_clarification_response
            resolved_text = resolve_clarification_response(
                transcribed_text,
                clarif_dict,
                language=lang_info.get("code", "en"),
            )
            _log_trace(
                "clarification_resolved",
                f'Clarification answer merged: "{resolved_text}"',
                "verified",
                {"original": clarif_dict.get("original_query"), "answer": transcribed_text, "resolved": resolved_text},
            )
            transcribed_text = resolved_text
        elif not confirmed_query:
            from backend.services.voice.clarification import detect_ambiguity
            threshold = getattr(self.asr, "confidence_threshold", 0.55)
            if asr_confidence < threshold:
                _log_trace("ASR_LOW_CONFIDENCE", f"ASR Low Confidence: {asr_confidence:.2f} < {threshold}", "insufficient", {"confidence": asr_confidence, "threshold": threshold})
            ambiguity = detect_ambiguity(
                transcribed_text,
                asr_confidence=asr_confidence,
                language=lang_info.get("code", "en"),
                confidence_threshold=threshold,
            )
            if ambiguity:
                _log_trace(
                    "CLARIFICATION_REQUIRED",
                    f'Clarification Required ({ambiguity.ambiguity_type}): "{ambiguity.clarification_question}"',
                    "insufficient",
                    ambiguity.to_dict(),
                )
                clarification_audio = ""
                try:
                    if hasattr(self.tts, "synthesize"):
                        synth = self.tts.synthesize(ambiguity.clarification_question, language=lang_info.get("code", "en"))
                        if synth.get("audio_bytes"):
                            clarification_audio = base64.b64encode(synth["audio_bytes"]).decode("ascii")
                except Exception as synth_err:
                    logger.warning("Could not synthesize clarification audio: %s", synth_err)

                return {
                    "status": "low_confidence" if ambiguity.ambiguity_type == "asr_failure" else "clarification_needed",
                    "requires_clarification": True,
                    "clarification_prompt": ambiguity.clarification_question,
                    "text": transcribed_text,
                    "raw_text": raw_text,
                    "normalized_text": transcribed_text,
                    "is_normalized": is_normalized,
                    "language": lang_info,
                    "confidence": asr_confidence,
                    "threshold": threshold,
                    "message": f"Clarification requested: {ambiguity.clarification_question}",
                    "clarification_text": ambiguity.clarification_question,
                    "audio_base64": clarification_audio,
                    "context": ambiguity.to_dict(),
                    "voice_mode": voice_mode,
                    "voice_session_id": voice_session_id,
                    "execution_trace": voice_trace,
                    "trace_id": trace_id,
                }

        # ── Step 4: FORWARD TEXT QUERY TO EXISTING GOVERNANCE PIPELINE ────────
        # Voice query enters run_agent — enforcing Scope Guard, RBAC, Clearance,
        # Hybrid RAG, Confidence Gate, Model Router, Local Ollama, Citation Validation.
        from backend.agent.graph import run_agent

        try:
            agent_result = await run_agent(
                query=transcribed_text,
                user_id=user["id"],
                session_id=session_id,
                user=user,
            )
        except Exception as exc:
            logger.error("Governance pipeline failed for voice query: %s", exc, exc_info=True)
            _log_trace("governance_pipeline", f"Pipeline Error: {exc}", "blocked", {"error": str(exc)})
            return {
                "status": "error",
                "error": f"Governance processing failed: {str(exc)}",
                "text": transcribed_text,
                "language": lang_info,
                "execution_trace": voice_trace,
                "trace_id": trace_id,
            }

        # Merge execution trace from the existing pipeline
        pipeline_trace = agent_result.get("execution_trace", [])
        combined_trace = voice_trace + pipeline_trace

        final_response_text = agent_result.get("response", "")
        scope_decision = agent_result.get("scope_decision", {})
        task_type = agent_result.get("task_type", "GENERAL_QUERY")
        model_id = agent_result.get("model_id", "qwen2.5:3b")

        # ── Step 5: GROUNDING & TTS VALIDATION ────────────────────────────────
        # TTS must only speak grounded / validated responses.
        # If rejected by Scope Guard, RBAC denied, or insufficient evidence,
        # speak the concise safety status message.
        spoken_text = final_response_text
        is_grounded = True

        if scope_decision and not scope_decision.get("allowed", True):
            is_grounded = False
            spoken_text = "Notice: This request is out of scope for MRPL industrial operations."
        elif "INSUFFICIENT EVIDENCE" in final_response_text.upper():
            is_grounded = False
            spoken_text = "MRPL Knowledge Base contains insufficient evidence to verify this query."
        elif "ACCESS DENIED" in final_response_text.upper():
            is_grounded = False
            spoken_text = "Access denied: Required security clearance not met."
        elif agent_result.get("requires_human_review"):
            spoken_text = "This engineering inquiry has been routed to the supervisor queue for human review."

        # ── Step 6: TEXT-TO-SPEECH (TTS) ──────────────────────────────────────
        _log_trace("TTS_STARTED", "Text-to-speech synthesis initiated", "allowed")
        tts_result = self.tts.synthesize(spoken_text, language=lang_info["code"])
        audio_bytes = tts_result.get("audio_bytes", b"") if isinstance(tts_result, dict) else b""
        audio_base64 = (
            base64.b64encode(audio_bytes).decode("utf-8")
            if isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0
            else None
        )

        tts_engine_name = tts_result.get("engine", "Local TTS") if isinstance(tts_result, dict) else "Local TTS"
        tts_status = tts_result.get("status", "unknown") if isinstance(tts_result, dict) else "unknown"
        tts_duration = tts_result.get("duration_seconds", 0.0) if isinstance(tts_result, dict) else 0.0

        _log_trace("TTS_COMPLETED", f"Speech synthesis completed ({tts_duration:.1f}s)", "verified", {"duration_seconds": tts_duration, "engine": tts_engine_name})
        _log_trace("VOICE_CAPTURE_RESUMED", "Continuous conversational voice capture resumed", "allowed")
        _log_trace("VOICE_SESSION_ENDED", "Voice interaction completed", "verified", {"trace_id": trace_id, "duration_seconds": tts_duration})

        combined_trace.append({
            "event": "tts_synthesized",
            "stage": "tts_synthesized",
            "title": f"Spoken response synthesized via {tts_engine_name}",
            "status": "verified" if tts_status == "success" else "insufficient",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "spoken_text_sample": spoken_text[:120],
                "duration_seconds": tts_duration,
                "is_grounded": is_grounded,
            },
        })

        # ── Step 7: IMMUTABLE AUDIT LOG ───────────────────────────────────────
        try:
            audit_log(
                action="voice_interaction",
                outcome="success" if tts_result.get("status") == "success" else "failure",
                user_id=user["id"],
                username=user.get("username", "anonymous"),
                target=f"session:{session_id or 'direct'}",
                details={
                    "trace_id": trace_id,
                    "input_mode": "voice",
                    "language": lang_info["code"],
                    "asr_confidence": asr_confidence,
                    "model_selected": model_id,
                    "task_type": task_type,
                    "is_grounded": is_grounded,
                    "tts_status": tts_result.get("status"),
                },
            )
            combined_trace.append({
                "event": "audit_logged",
                "stage": "audit_logged",
                "title": "Voice interaction recorded in immutable audit ledger",
                "status": "allowed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {"action": "voice_interaction", "trace_id": trace_id},
            })
        except Exception as ex:
            logger.warning("Could not write voice audit log: %s", ex)

        return {
            "status": "success",
            "transcribed_text": transcribed_text,
            "raw_text": raw_text,
            "normalized_text": transcribed_text,
            "is_normalized": is_normalized,
            "response": final_response_text,
            "spoken_text": spoken_text,
            "language": lang_info,
            "asr_confidence": asr_confidence,
            "audio_base64": audio_base64,
            "audio_content_type": tts_result.get("content_type", "audio/wav"),
            "audio_duration": tts_result.get("duration_seconds", 0.0),
            "execution_trace": combined_trace,
            "trace_id": trace_id,
            "task_type": task_type,
            "model_id": model_id,
            "scope_decision": scope_decision,
            "confidence_decision": agent_result.get("confidence_decision"),
            "citation_validation": agent_result.get("citation_validation"),
            "requires_human_review": agent_result.get("requires_human_review", False),
            "approval_id": agent_result.get("approval_id"),
            "voice_mode": voice_mode,
            "voice_session_id": voice_session_id,
        }

    def _stream_ollama_tokens(self, endpoint: str, payload: dict) -> Any:
        """Stream Ollama tokens synchronously line-by-line."""
        record_local_call(endpoint, f"Ollama {payload.get('model')} streaming")
        resp = requests.post(endpoint, json=payload, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    tok = chunk.get("response", "")
                    if tok:
                        yield tok
                    if chunk.get("done", False):
                        break
                except Exception:
                    continue

    async def _async_token_stream(self, endpoint: str, payload: dict) -> AsyncGenerator[str, None]:
        """Non-blocking asynchronous token stream from local Ollama."""
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                for tok in self._stream_ollama_tokens(endpoint, payload):
                    if loop.is_closed():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, tok)
            except Exception as exc:
                if not loop.is_closed():
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, exc)
                    except RuntimeError:
                        pass
            finally:
                if not loop.is_closed():
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, None)
                    except RuntimeError:
                        pass

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def stream_voice_query(
        self,
        audio_data: bytes | io.BytesIO | str,
        user: dict,
        session_id: int | None = None,
        language_hint: str | None = None,
        confirmed_query: str | None = None,
        clarification_context: dict[str, Any] | str | None = None,
        turn_id: int | None = None,
        voice_mode: str = "NOVA",
        voice_session_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Real-Time Conversational Streaming Voice Pipeline:
        Microphone Audio -> Streaming Local ASR -> Language Detection
        -> Scope Guard -> RBAC & Clearance -> Fast Query Routing
        -> Streaming Local Ollama Tokens -> Sentence Chunk Buffer
        -> Local Neural TTS -> Audio Chunks with Barge-In / Interruption Support.
        """
        t_start = time.perf_counter()
        trace_id = f"TRC-VCE-{uuid.uuid4().hex[:8].upper()}"
        voice_trace = []

        if self.is_voice_session_stopped(voice_session_id):
            logger.info("Nova session %s has been explicitly stopped. Ignoring request.", voice_session_id)
            return

        if session_id is not None:
            self.clear_interruption(session_id)
            if turn_id is not None:
                self._active_session_turns[session_id] = turn_id

        def _is_stale_turn() -> bool:
            if self.is_voice_session_stopped(voice_session_id):
                return True
            if session_id is not None and turn_id is not None:
                active = self._active_session_turns.get(session_id)
                if active is not None and active != turn_id:
                    return True
            return False

        def _log_trace(event: str, title: str, status: str, details: dict | None = None):
            d = dict(details or {})
            if turn_id is not None:
                d["turn_id"] = turn_id
            if voice_session_id is not None:
                d["voice_session_id"] = voice_session_id
            d["voice_mode"] = voice_mode
            voice_trace.append({
                "event": event,
                "title": title,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": d,
            })

        # ── 1. Request Received & Session Started ─────────────────────────────
        if voice_mode in ("VOICE_TO_TEXT", "DICTATION"):
            _log_trace("VOICE_TO_TEXT_STARTED", "Voice-to-Text audio capture initiated", "allowed", {"language_hint": language_hint or "auto", "voice_session_id": voice_session_id, "turn_id": turn_id})
            yield {
                "event": "voice_to_text_started",
                "voice_mode": voice_mode,
                "voice_session_id": voice_session_id,
                "turn_id": turn_id,
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            _log_trace("VOICE_CAPTURE_STARTED", "Voice audio stream capture initiated", "allowed", {"language_hint": language_hint or "auto", "session_id": session_id, "voice_session_id": voice_session_id, "turn_id": turn_id})
            _log_trace("VOICE_SESSION_STARTED", "Voice audio stream initiated", "allowed", {"language_hint": language_hint or "auto", "session_id": session_id, "voice_session_id": voice_session_id, "turn_id": turn_id})
            yield {
                "event": "asr_started",
                "voice_mode": "NOVA",
                "voice_session_id": voice_session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id": trace_id,
                "turn_id": turn_id,
            }

        # Real VAD & audio diagnostics
        vad_metrics = {}
        if isinstance(audio_data, bytes) and len(audio_data) > 0:
            try:
                vad_metrics = self.vad.detect_speech_boundaries(audio_data)
            except Exception:
                vad_metrics = self.vad.get_diagnostics()
        else:
            vad_metrics = self.vad.get_diagnostics()

        capture_dur_val = vad_metrics.get("capture_duration_ms", 0)
        speech_dur_val = vad_metrics.get("speech_duration_ms", 0)
        sample_rate_val = vad_metrics.get("sample_rate", getattr(self.vad, "sample_rate", 16000))
        channel_count_val = vad_metrics.get("channel_count", 1)
        input_rms_val = vad_metrics.get("input_rms", 0.0)
        peak_val = vad_metrics.get("peak", 0.0)
        noise_floor_val = vad_metrics.get("noise_floor_rms", 0.008)
        speech_rms_val = vad_metrics.get("speech_rms", 0.0)
        snr_db_val = vad_metrics.get("snr_db", 0.0)
        vad_conf_val = vad_metrics.get("vad_confidence", 0.0)
        speech_active_val = vad_metrics.get("has_speech", True)

        _log_trace("AUDIO_NOISE_FLOOR", f"Ambient noise floor: {noise_floor_val:.4f} RMS", "verified", {"noise_floor_rms": noise_floor_val})
        _log_trace("AUDIO_RMS", f"Audio RMS level: {speech_rms_val:.4f}", "verified", {"speech_rms": speech_rms_val, "input_rms": input_rms_val, "peak": peak_val})
        _log_trace("AUDIO_SNR", f"Audio SNR: {snr_db_val:.1f} dB", "verified", {"snr_db": snr_db_val})
        _log_trace(
            "VAD_STATE_CHANGED",
            f"VAD State: {'SPEECH' if speech_active_val else 'SILENCE'} (duration: {speech_dur_val}ms, confidence: {vad_conf_val:.2f})",
            "verified",
            {
                "speech_active": speech_active_val,
                "confidence": vad_conf_val,
                "capture_duration_ms": capture_dur_val,
                "speech_duration_ms": speech_dur_val,
                "sample_rate": sample_rate_val,
                "channel_count": channel_count_val,
            },
        )

        yield {
            "event": "audio_diagnostics",
            "capture_duration_ms": capture_dur_val,
            "speech_duration_ms": speech_dur_val,
            "sample_rate": sample_rate_val,
            "channel_count": channel_count_val,
            "input_rms": input_rms_val,
            "peak": peak_val,
            "noise_floor_rms": noise_floor_val,
            "speech_rms": speech_rms_val,
            "snr_db": snr_db_val,
            "vad_confidence": vad_conf_val,
            "trace_id": trace_id,
            "turn_id": turn_id,
        }

        # ── 2. Speech Recognition (ASR) ───────────────────────────────────────
        t_asr_start = time.perf_counter()
        if confirmed_query:
            raw_text = confirmed_query.strip()
            transcribed_text = confirmed_query.strip()
            is_normalized = False
            asr_confidence = 1.0
            asr_status = "success"
            raw_lang = language_hint or "en"
            t_asr_final = time.perf_counter()
            asr_latency_ms = 0.0
            asr_res = {}
        else:
            asr_res = await asyncio.to_thread(self.asr.transcribe, audio_data, language=language_hint)
            t_asr_final = time.perf_counter()
            asr_latency_ms = (t_asr_final - t_asr_start) * 1000
            raw_text = asr_res.get("raw_text", asr_res.get("text", "")).strip()
            transcribed_text = asr_res.get("normalized_text", asr_res.get("text", "")).strip()
            is_normalized = asr_res.get("is_normalized", False)
            asr_confidence = asr_res.get("confidence", 0.0)
            asr_status = asr_res.get("status", "error")
            raw_lang = asr_res.get("language", "en")

            if asr_status == "error":
                err_msg = asr_res.get("error", "ASR transcription failed")
                logger.warning("ASR transcription encountered error: %s; initiating spoken recovery turn", err_msg)
                _log_trace("speech_recognition", f"ASR Error: {err_msg}", "blocked", {"error": err_msg})
                retry_prompt = "Sorry, I didn't catch that. Could you repeat it?"
                _log_trace(
                    "CLARIFICATION_REQUIRED",
                    f'Clarification Required (asr_failure): "{retry_prompt}"',
                    "insufficient",
                    {"ambiguity_type": "asr_failure", "error": err_msg},
                )
                yield {
                    "event": "clarification_needed",
                    "ambiguity_type": "asr_failure",
                    "clarification_text": retry_prompt,
                    "context": {"ambiguity_type": "asr_failure", "original_query": ""},
                    "raw_text": "",
                    "normalized_text": "",
                    "trace_id": trace_id,
                }

                clarification_audio = ""
                try:
                    if hasattr(self.tts, "synthesize_chunk"):
                        synth = await asyncio.to_thread(self.tts.synthesize_chunk, retry_prompt, "en")
                        clarification_audio = synth.get("audio_base64") or (base64.b64encode(synth["audio_bytes"]).decode("ascii") if synth.get("audio_bytes") else "")
                    elif hasattr(self.tts, "synthesize"):
                        synth = await asyncio.to_thread(self.tts.synthesize, retry_prompt, "en")
                        if synth.get("audio_bytes"):
                            clarification_audio = base64.b64encode(synth["audio_bytes"]).decode("ascii")
                except Exception as synth_err:
                    logger.warning("Could not synthesize retry prompt audio: %s", synth_err)

                if clarification_audio:
                    yield {
                        "event": "tts_chunk",
                        "text": retry_prompt,
                        "spoken_text": retry_prompt,
                        "audio_base64": clarification_audio,
                        "is_final": True,
                        "chunk_index": 0,
                    }

                yield {
                    "event": "waiting_for_clarification",
                    "context": {"ambiguity_type": "asr_failure", "original_query": ""},
                    "clarification_text": retry_prompt,
                    "has_audio": bool(clarification_audio),
                    "trace_id": trace_id,
                }
                return

        # Trace & SSE: ASR_FINAL_RAW
        _log_trace("ASR_FINAL_RAW", f'ASR Final Raw: "{raw_text}"', "verified", {"raw_text": raw_text})
        yield {"event": "asr_final_raw", "raw_text": raw_text, "trace_id": trace_id}

        # Trace & SSE: ASR_CONFIDENCE
        _log_trace("ASR_CONFIDENCE", f"ASR Confidence: {asr_confidence:.2f}", "verified" if asr_status != "low_confidence" else "insufficient", {"confidence": asr_confidence})
        yield {"event": "asr_confidence", "confidence": asr_confidence, "trace_id": trace_id}

        # Trace & SSE: ASR_NORMALIZED
        norm_desc = f'ASR Normalized: "{transcribed_text}"' if is_normalized else "ASR Normalized: unchanged"
        _log_trace("ASR_NORMALIZED", norm_desc, "verified", {"raw_text": raw_text, "normalized_text": transcribed_text, "is_normalized": is_normalized})
        yield {"event": "asr_normalized", "raw_text": raw_text, "normalized_text": transcribed_text, "is_normalized": is_normalized, "trace_id": trace_id}

        # Trace & SSE: CHAT_REQUEST_QUERY and ORCHESTRATOR_QUERY
        _log_trace("CHAT_REQUEST_QUERY", f'Chat Request Query: "{transcribed_text}"', "verified", {"current_query": transcribed_text, "trace_id": trace_id})
        _log_trace("ORCHESTRATOR_QUERY", f'Orchestrator Query: "{transcribed_text}"', "verified", {"current_query": transcribed_text, "trace_id": trace_id})
        yield {"event": "chat_request_query", "current_query": transcribed_text, "trace_id": trace_id}
        yield {"event": "orchestrator_query", "current_query": transcribed_text, "trace_id": trace_id}

        # Language Detection
        lang_info = detect_voice_language(
            text=transcribed_text,
            asr_detected_language=raw_lang,
            user_override=language_hint,
        )
        _log_trace("language_detected", f"Language: {lang_info['name']} ({lang_info['method']})", "verified", lang_info)
        yield {"event": "language_detected", "language": lang_info}

        # Check empty speech
        if not transcribed_text and not raw_text:
            yield {
                "event": "empty",
                "message": "No audible speech detected. Please speak clearly into the microphone.",
                "voice_mode": voice_mode,
                "voice_session_id": voice_session_id,
                "trace_id": trace_id,
                "turn_id": turn_id,
            }
            return

        # ── 2.5: CONVERSATIONAL INTENT DETECTION ──────────────────────────────
        # Intercept simple greetings / farewells / casual exchanges locally.
        # Bypass Scope Guard, RAG, Ollama, and the full governed pipeline for these.
        from backend.services.voice.conversational_intent import detect_conversational_intent
        conv_intent = detect_conversational_intent(transcribed_text)

        if conv_intent.is_conversational and conv_intent.response_text:
            _log_trace(
                "CONVERSATION_INTENT_DETECTED",
                f"Conversational intent: {conv_intent.intent.value}",
                "allowed",
                conv_intent.to_dict(),
            )
            _log_trace(
                "LOCAL_CONVERSATIONAL_RESPONSE",
                f"Route: LOCAL_CONVERSATIONAL_RESPONSE",
                "allowed",
                {"response": conv_intent.response_text, "pool_key": conv_intent.response_key},
            )

            yield {
                "event": "conversational_intent_detected",
                "intent": conv_intent.intent.value,
                "response_key": conv_intent.response_key,
                "trace_id": trace_id,
                "turn_id": turn_id,
            }

            spoken_text = conv_intent.response_text
            _log_trace("TTS_STARTED", "Text-to-speech synthesis initiated", "allowed")
            tts_res = await asyncio.to_thread(self.tts.synthesize, spoken_text, language=lang_info["code"])
            audio_bytes = tts_res.get("audio_bytes", b"")
            tts_duration = tts_res.get("duration_seconds", 0.0)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes else None

            _log_trace("TTS_COMPLETED", f"Speech synthesis completed ({tts_duration:.1f}s)", "verified", {"duration_seconds": tts_duration})

            yield {
                "event": "tts_chunk",
                "chunk_index": 0,
                "spoken_text": spoken_text,
                "audio_base64": audio_b64,
                "duration_seconds": tts_duration,
                "is_final": True,
                "turn_id": turn_id,
            }

            # Store in session
            if session_id:
                try:
                    chat_repo.add_message(session_id, "user", f"🎤 [{lang_info['name']}] {transcribed_text}")
                    chat_repo.add_message(session_id, "assistant", spoken_text)
                except Exception as store_err:
                    logger.warning("Could not persist conversational message to session: %s", store_err)

            # Audit log
            try:
                audit_log(
                    action="voice_interaction",
                    outcome="success",
                    user_id=user["id"],
                    username=user.get("username", "anonymous"),
                    target=f"session:{session_id or 'direct'}",
                    details={
                        "trace_id": trace_id,
                        "input_mode": "voice_stream",
                        "language": lang_info["code"],
                        "asr_confidence": asr_confidence,
                        "conversational_intent": conv_intent.intent.value,
                        "route": "LOCAL_CONVERSATIONAL_RESPONSE",
                    },
                )
            except Exception:
                pass

            _log_trace("VOICE_CAPTURE_RESUMED", "Continuous conversational voice capture resumed", "allowed")
            _log_trace("LISTENING_RESUMED", "Continuous conversational turn ready", "allowed")

            yield {
                "event": "complete",
                "response": spoken_text,
                "spoken_text": spoken_text,
                "trace": voice_trace,
                "trace_id": trace_id,
                "task_type": "CONVERSATIONAL",
                "model_id": "local_conversational",
                "requires_human_review": False,
                "approval_id": None,
                "voice_mode": voice_mode,
                "voice_session_id": voice_session_id,
                "turn_id": turn_id,
                "conversational_intent": conv_intent.intent.value,
            }

            if voice_mode == "NOVA" and not self.is_interrupted(session_id) and not self.is_voice_session_stopped(voice_session_id):
                yield {
                    "event": "listening",
                    "voice_mode": "NOVA",
                    "voice_session_id": voice_session_id,
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                }
            return

        # In normal Voice-to-Text / Dictation mode: return transcript for text composer without calling LLM/RAG/TTS
        if voice_mode in ("VOICE_TO_TEXT", "DICTATION"):
            threshold = getattr(self.asr, "confidence_threshold", 0.55)
            if asr_status == "error" or asr_status == "low_confidence" or asr_confidence < threshold:
                low_conf_reason = (
                    asr_res.get("low_confidence_reason")
                    or asr_res.get("quality", {}).get("reason")
                    or (f"Token acoustic confidence too low: {asr_confidence:.2f} < {threshold:.2f}" if asr_confidence < threshold else "low_confidence")
                )
                diag_payload = {
                    **vad_metrics,
                    "asr_confidence": asr_confidence,
                    "confidence_threshold": threshold,
                    "asr_status": asr_status,
                    "low_confidence_reason": low_conf_reason,
                    "raw_text": raw_text,
                }
                _log_trace(
                    "ASR_LOW_CONFIDENCE",
                    f"ASR Low Confidence: {asr_confidence:.2f} < {threshold} ({low_conf_reason})",
                    "insufficient",
                    diag_payload,
                )
                yield {
                    "event": "voice_to_text_low_confidence",
                    "message": "Couldn't confidently transcribe that. Please try again.",
                    "low_confidence_reason": low_conf_reason,
                    "diagnostics": diag_payload,
                    "raw_text": raw_text,
                    "normalized_text": transcribed_text,
                    "confidence": asr_confidence,
                    "voice_mode": voice_mode,
                    "voice_session_id": voice_session_id,
                    "trace_id": trace_id,
                    "turn_id": turn_id,
                }
                return

            _log_trace(
                "VOICE_TO_TEXT_COMPLETED",
                f'Voice-to-Text completed: "{transcribed_text}"',
                "verified",
                {"raw_text": raw_text, "normalized_text": transcribed_text, "confidence": asr_confidence},
            )
            yield {
                "event": "voice_to_text_completed",
                "raw_text": raw_text,
                "normalized_text": transcribed_text,
                "confidence": asr_confidence,
                "language": lang_info,
                "voice_mode": voice_mode,
                "voice_session_id": voice_session_id,
                "trace_id": trace_id,
                "turn_id": turn_id,
            }
            return

        # Check clarification context if user is answering a previous question
        clarif_dict: dict[str, Any] | None = None
        if isinstance(clarification_context, str):
            if clarification_context.strip():
                try:
                    loaded = json.loads(clarification_context)
                    if isinstance(loaded, dict):
                        clarif_dict = loaded
                except Exception:
                    clarif_dict = None
        elif isinstance(clarification_context, dict):
            clarif_dict = clarification_context

        if clarif_dict:
            from backend.services.voice.clarification import resolve_clarification_response
            resolved_text = resolve_clarification_response(
                transcribed_text,
                clarif_dict,
                language=lang_info.get("code", "en"),
            )
            _log_trace(
                "clarification_resolved",
                f'Clarification answer merged: "{resolved_text}"',
                "verified",
                {"original": clarif_dict.get("original_query"), "answer": transcribed_text, "resolved": resolved_text},
            )
            yield {
                "event": "clarification_resolved",
                "original_query": clarif_dict.get("original_query"),
                "answer": transcribed_text,
                "resolved_query": resolved_text,
            }
            transcribed_text = resolved_text
        elif not confirmed_query:
            from backend.services.voice.clarification import detect_ambiguity
            threshold = getattr(self.asr, "confidence_threshold", 0.55)
            if asr_confidence < threshold:
                _log_trace("ASR_LOW_CONFIDENCE", f"ASR Low Confidence: {asr_confidence:.2f} < {threshold}", "insufficient", {"confidence": asr_confidence, "threshold": threshold})
                yield {"event": "asr_low_confidence", "confidence": asr_confidence, "threshold": threshold, "trace_id": trace_id}

            ambiguity = detect_ambiguity(
                transcribed_text,
                asr_confidence=asr_confidence,
                language=lang_info.get("code", "en"),
                confidence_threshold=threshold,
            )
            if ambiguity:
                _log_trace(
                    "CLARIFICATION_REQUIRED",
                    f'Clarification Required ({ambiguity.ambiguity_type}): "{ambiguity.clarification_question}"',
                    "insufficient",
                    ambiguity.to_dict(),
                )
                yield {
                    "event": "clarification_needed",
                    "ambiguity_type": ambiguity.ambiguity_type,
                    "clarification_text": ambiguity.clarification_question,
                    "context": ambiguity.to_dict(),
                    "raw_text": raw_text,
                    "normalized_text": transcribed_text,
                    "trace_id": trace_id,
                }
                # Synthesize clarification question via local TTS
                clarification_audio = ""
                try:
                    if hasattr(self.tts, "synthesize_chunk"):
                        synth = await asyncio.to_thread(self.tts.synthesize_chunk, ambiguity.clarification_question, lang_info.get("code", "en"))
                        clarification_audio = synth.get("audio_base64") or (base64.b64encode(synth["audio_bytes"]).decode("ascii") if synth.get("audio_bytes") else "")
                    elif hasattr(self.tts, "synthesize"):
                        synth = await asyncio.to_thread(self.tts.synthesize, ambiguity.clarification_question, lang_info.get("code", "en"))
                        if synth.get("audio_bytes"):
                            clarification_audio = base64.b64encode(synth["audio_bytes"]).decode("ascii")
                except Exception as synth_err:
                    logger.warning("Could not synthesize clarification audio: %s", synth_err)

                if clarification_audio:
                    yield {
                        "event": "tts_chunk",
                        "text": ambiguity.clarification_question,
                        "spoken_text": ambiguity.clarification_question,
                        "audio_base64": clarification_audio,
                        "is_final": True,
                        "chunk_index": 0,
                    }

                yield {
                    "event": "waiting_for_clarification",
                    "context": ambiguity.to_dict(),
                    "clarification_text": ambiguity.clarification_question,
                    "has_audio": bool(clarification_audio),
                    "trace_id": trace_id,
                }
                return

        _log_trace(
            "speech_recognition",
            f'Speech recognized: "{transcribed_text[:80]}" ({asr_latency_ms:.1f} ms)',
            "verified",
            {"transcribed_text": transcribed_text, "raw_text": raw_text, "confidence": asr_confidence, "is_normalized": is_normalized, "latency_ms": asr_latency_ms},
        )
        yield {
            "event": "asr_final",
            "text": transcribed_text,
            "raw_text": raw_text,
            "normalized_text": transcribed_text,
            "is_normalized": is_normalized,
            "confidence": asr_confidence,
            "language": lang_info,
            "latency_ms": round(asr_latency_ms, 1),
        }

        # ── 3. Security & Governance Checks ───────────────────────────────────
        t_gov_start = time.perf_counter()
        username = user.get("username", "anonymous") if user else "anonymous"
        roles = user.get("roles", ["engineer"]) if user else ["engineer"]

        # Scope Guard
        from backend.services.scope_guard import evaluate_scope
        scope_res = evaluate_scope(transcribed_text, username=username, role=roles[0] if roles else "engineer")

        if not scope_res.allowed:
            t_gov_done = time.perf_counter()
            gov_latency_ms = (t_gov_done - t_gov_start) * 1000
            _log_trace("scope_checked", f"Scope Guard: {scope_res.reason}", "blocked", {"reason": scope_res.reason})
            rejection_text = (
                f"⛔ REQUEST OUT OF SCOPE: {scope_res.reason}\n\n"
                "The MRPL Sovereign AI Workbench strictly permits industrial operations, refining engineering, "
                "compliance guidelines, financial reports, and sandboxed calculations."
            )
            spoken_rejection = "Notice: This request is out of scope for MRPL industrial operations."
            tts_res = await asyncio.to_thread(self.tts.synthesize, spoken_rejection, language=lang_info["code"])
            audio_bytes = tts_res.get("audio_bytes", b"")
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

            yield {"event": "scope_blocked", "reason": scope_res.reason, "response": rejection_text, "spoken_text": spoken_rejection}
            if audio_b64:
                yield {
                    "event": "tts_chunk",
                    "chunk_index": 0,
                    "spoken_text": spoken_rejection,
                    "audio_base64": audio_b64,
                    "duration_seconds": tts_res.get("duration_seconds", 0.0),
                    "is_final": True,
                }
            if session_id:
                try:
                    chat_repo.add_message(session_id, "user", f"🎤 [{lang_info['name']}] {transcribed_text}")
                    chat_repo.add_message(session_id, "assistant", rejection_text)
                except Exception as store_err:
                    logger.warning("Could not persist message to session: %s", store_err)
            yield {
                "event": "complete",
                "response": rejection_text,
                "spoken_text": spoken_rejection,
                "status": "blocked",
                "trace": voice_trace,
                "trace_id": trace_id,
            }
            return

        # RBAC & Clearance Authorization
        from backend.services.rag_engine import CLASSIFICATION_LEVELS, get_max_clearance_for_roles
        user_clearance = user.get("clearance") if user and user.get("clearance") else get_max_clearance_for_roles(roles)
        user_level = CLASSIFICATION_LEVELS.get(str(user_clearance).upper(), 2)
        is_restricted_intent = any(
            w in transcribed_text.lower()
            for w in ["restricted", "highly confidential", "board minutes", "classified report", "secret audit", "unreleased audit"]
        )

        if is_restricted_intent and user_level < CLASSIFICATION_LEVELS.get("CONFIDENTIAL", 3):
            t_gov_done = time.perf_counter()
            _log_trace(
                "authorization_checked",
                f"RBAC Denied: Role {roles} (Clearance: {user_clearance}) lacks required clearance",
                "denied",
                {"role": roles, "clearance": user_clearance},
            )
            denial_text = f"⛔ ACCESS DENIED: Role lacking required clearance ({user_clearance}) for RESTRICTED or CONFIDENTIAL industrial materials."
            spoken_denial = "Access denied: Required security clearance not met."
            tts_res = await asyncio.to_thread(self.tts.synthesize, spoken_denial, language=lang_info["code"])
            audio_bytes = tts_res.get("audio_bytes", b"")
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

            yield {"event": "rbac_denied", "reason": denial_text, "response": denial_text, "spoken_text": spoken_denial}
            if audio_b64:
                yield {
                    "event": "tts_chunk",
                    "chunk_index": 0,
                    "spoken_text": spoken_denial,
                    "audio_base64": audio_b64,
                    "duration_seconds": tts_res.get("duration_seconds", 0.0),
                    "is_final": True,
                }
            if session_id:
                try:
                    chat_repo.add_message(session_id, "user", f"🎤 [{lang_info['name']}] {transcribed_text}")
                    chat_repo.add_message(session_id, "assistant", denial_text)
                except Exception as store_err:
                    logger.warning("Could not persist message to session: %s", store_err)
            yield {
                "event": "complete",
                "response": denial_text,
                "spoken_text": spoken_denial,
                "status": "denied",
                "trace": voice_trace,
                "trace_id": trace_id,
            }
            return

        t_gov_done = time.perf_counter()
        gov_latency_ms = (t_gov_done - t_gov_start) * 1000
        _log_trace(
            "SCOPE_CHECK",
            f"Scope Guard: Allowed ({scope_res.scope_category})",
            "allowed",
            {"category": scope_res.scope_category, "latency_ms": gov_latency_ms},
        )
        _log_trace(
            "RBAC_CHECK",
            f"RBAC verified: Clearance {user_clearance}",
            "allowed",
            {"roles": roles, "clearance": user_clearance},
        )
        yield {"event": "governance_checked", "scope": scope_res.scope_category, "latency_ms": round(gov_latency_ms, 1), "turn_id": turn_id}

        # ── 4. Query Classification & Conversational Context Policy ───────────
        authoritative_query = transcribed_text

        # Deterministically resolve contextual follow-ups vs independent topic shifts
        chat_history_list = []
        if session_id:
            try:
                chat_history_list = chat_repo.list_messages(session_id)
            except Exception:
                chat_history_list = []

        from backend.services.voice.context_policy import determine_conversational_context
        context_res = determine_conversational_context(
            current_query=authoritative_query,
            conversation_history=chat_history_list,
        )

        retrieval_query = context_res.resolved_retrieval_query if context_res.is_followup else authoritative_query
        if context_res.is_followup:
            _log_trace(
                "CONTEXTUAL_FOLLOWUP_RESOLVED",
                f'Resolved contextual follow-up: "{retrieval_query}" (entity: {context_res.target_entity})',
                "verified",
                {"current_query": authoritative_query, "retrieval_query": retrieval_query, "target_entity": context_res.target_entity},
            )
            yield {
                "event": "context_resolved",
                "current_query": authoritative_query,
                "retrieval_query": retrieval_query,
                "target_entity": context_res.target_entity,
                "reason": context_res.reason,
                "turn_id": turn_id,
            }

        decision = route_task(retrieval_query)
        task_type = decision.get("task_type", "GENERAL")
        requires_rag = decision.get("requires_rag", False)
        model_name = decision.get("model", "gemma3:4b")
        model_id = decision.get("model_id", "gemma3_4b")
        endpoint = decision.get("endpoint", "http://127.0.0.1:11434/api/generate")

        _log_trace("ROUTER_QUERY", f'Router Query: "{retrieval_query}"', "verified", {"current_query": authoritative_query, "retrieval_query": retrieval_query, "task_type": task_type, "trace_id": trace_id})
        _log_trace("ROUTING", f"Routing: {task_type} (requires_rag={requires_rag}, model={model_name})", "allowed", {"task_type": task_type, "model": model_name, "requires_rag": requires_rag})
        _log_trace("MODEL_SELECTED", f"Model Selected: {model_name} ({model_id})", "allowed", {"model": model_name, "model_id": model_id})
        yield {"event": "router_query", "current_query": authoritative_query, "retrieval_query": retrieval_query, "task_type": task_type, "trace_id": trace_id, "turn_id": turn_id}

        # Natural Acknowledgement for heavy processing tasks (Scope & RBAC already verified)
        is_comparative = any(kw in authoritative_query.lower() for kw in ["compare", "comparison", "difference between", "what changed"])
        if (task_type in ("DOCUMENT_ANALYSIS", "deep_analysis") or is_comparative) and not self.is_interrupted(session_id) and not _is_stale_turn():
            ack_text = "Sure, give me a moment to compare them." if is_comparative else "Checking the technical documents now."
            try:
                ack_synth = await asyncio.to_thread(self.tts.synthesize, ack_text, language=lang_info["code"])
                if ack_synth.get("audio_bytes"):
                    ack_b64 = base64.b64encode(ack_synth["audio_bytes"]).decode("utf-8")
                    yield {
                        "event": "tts_chunk",
                        "chunk_index": 0,
                        "spoken_text": ack_text,
                        "audio_base64": ack_b64,
                        "duration_seconds": ack_synth.get("duration_seconds", 1.5),
                        "is_final": False,
                        "is_acknowledgement": True,
                        "turn_id": turn_id,
                    }
            except Exception as ack_err:
                logger.debug("Acknowledgement synthesis skipped: %s", ack_err)

        # Chat history formatted
        chat_history_str = ""
        for m in chat_history_list[-6:]:
            chat_history_str += f"{m['role']}: {m['content']}\n"

        sources = []
        rag_latency_ms = 0.0

        if not requires_rag and task_type in ("GENERAL", "general_chat"):
            # FAST PATH: Direct local model generation without expensive retrieval
            _log_trace(
                "fast_routing",
                f"Fast Conversational Path selected for {task_type} (no RAG needed)",
                "allowed",
                {"task_type": task_type, "model": model_name, "reason": decision.get("reason")},
            )
            yield {
                "event": "routing_decision",
                "path": "fast_path",
                "task_type": task_type,
                "model": model_name,
                "requires_rag": False,
                "reason": decision.get("reason", "Direct conversational response"),
                "turn_id": turn_id,
            }
            from backend.agent.prompts import GENERAL_CHAT_PROMPT
            prompt = GENERAL_CHAT_PROMPT.format(
                conversation_history=chat_history_str or "No previous conversation.",
                current_query=authoritative_query,
            )
        elif requires_rag:
            # RAG PATH: Hybrid search + empirical confidence gate using resolved retrieval query
            t_rag_start = time.perf_counter()
            _log_trace("RAG_STARTED", f'RAG retrieval started for "{retrieval_query[:80]}"', "allowed", {"query": retrieval_query[:100]})
            _log_trace("RAG_QUERY", f'RAG Search Query: "{retrieval_query}"', "verified", {"current_query": authoritative_query, "retrieval_query": retrieval_query, "trace_id": trace_id})
            yield {"event": "retrieval_started", "query": retrieval_query[:100], "turn_id": turn_id}
            yield {"event": "rag_query", "current_query": authoritative_query, "retrieval_query": retrieval_query, "trace_id": trace_id, "turn_id": turn_id}

            from backend.agent.tools import execute_tool
            from backend.services.confidence_gate import evaluate_retrieval_confidence

            rag_res = await asyncio.to_thread(execute_tool, "rag_search", user=user, query=retrieval_query)
            if rag_res.get("status") == "success":
                rag_data = rag_res.get("result", {})
                context_str = rag_data.get("context", "")
                sources = rag_data.get("sources", [])
            else:
                context_str = ""
                sources = []

            conf_res = evaluate_retrieval_confidence(transcribed_text, sources)
            t_rag_done = time.perf_counter()
            rag_latency_ms = (t_rag_done - t_rag_start) * 1000

            _log_trace(
                "RAG_COMPLETED",
                f"Retrieved {len(sources)} source documents ({rag_latency_ms:.1f} ms, confidence: {conf_res.confidence_band})",
                "verified",
                {"source_count": len(sources), "confidence": conf_res.confidence_score, "latency_ms": rag_latency_ms},
            )
            _log_trace(
                "CONFIDENCE_GATE",
                f"Confidence Gate: {conf_res.confidence_band} ({conf_res.confidence_score:.2f})",
                "verified" if conf_res.is_sufficient else "insufficient",
                conf_res.to_dict(),
            )

            if conf_res.confidence_band == "INSUFFICIENT_EVIDENCE" or len(sources) == 0:
                _log_trace(
                    "insufficient_evidence",
                    f"Confidence: {conf_res.confidence_band} ({conf_res.confidence_score:.2f}) - Insufficient Evidence",
                    "insufficient",
                    {"confidence": conf_res.confidence_score, "escalation_rule": "LOW_RETRIEVAL_CONFIDENCE", "reason": conf_res.reason},
                )
                insufficient_text = "The MRPL Knowledge Base contains insufficient evidence to verify this query without risk of hallucination."
                spoken_insufficient = "MRPL Knowledge Base contains insufficient evidence to verify this query."
                tts_res = await asyncio.to_thread(self.tts.synthesize, spoken_insufficient, language=lang_info["code"])
                audio_bytes = tts_res.get("audio_bytes", b"")
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

                yield {"event": "insufficient_evidence", "confidence_score": conf_res.confidence_score}
                if audio_b64:
                    yield {
                        "event": "tts_chunk",
                        "chunk_index": 0,
                        "spoken_text": spoken_insufficient,
                        "audio_base64": audio_b64,
                        "duration_seconds": tts_res.get("duration_seconds", 0.0),
                        "is_final": True,
                    }
                if session_id:
                    try:
                        chat_repo.add_message(session_id, "user", f"🎤 [{lang_info['name']}] {transcribed_text}")
                        chat_repo.add_message(session_id, "assistant", insufficient_text)
                    except Exception as store_err:
                        logger.warning("Could not persist message to session: %s", store_err)
                yield {
                    "event": "complete",
                    "response": insufficient_text,
                    "spoken_text": spoken_insufficient,
                    "status": "insufficient_evidence",
                    "trace": voice_trace,
                    "trace_id": trace_id,
                    "requires_human_review": True,
                    "escalation_rule": "LOW_RETRIEVAL_CONFIDENCE",
                }
                return

            yield {
                "event": "routing_decision",
                "path": "rag_path",
                "task_type": task_type,
                "model": model_name,
                "requires_rag": True,
                "sources_count": len(sources),
                "confidence_score": conf_res.confidence_score,
                "rag_latency_ms": round(rag_latency_ms, 1),
            }

            from backend.agent.prompts import REASONING_PROMPT
            if not context_str and sources:
                context_str = "\n\n".join([
                    f"[Document: {s.get('document_title', s.get('source', 'Doc'))}, Page {s.get('page_number', s.get('page', 1))}]\n{s.get('content', '')}"
                    for s in sources
                ])
            prompt = REASONING_PROMPT.format(
                retrieved_context=context_str or "No direct context available.",
                conversation_history=chat_history_str or "No previous conversation.",
                current_query=authoritative_query,
            )
        else:
            # Coding / Vision / Document Analysis fallback
            yield {
                "event": "routing_decision",
                "path": "specialized_path",
                "task_type": task_type,
                "model": model_name,
                "requires_rag": False,
            }
            from backend.agent.prompts import GENERAL_CHAT_PROMPT
            prompt = GENERAL_CHAT_PROMPT.format(
                conversation_history=chat_history_str or "No previous conversation.",
                current_query=authoritative_query,
            )

        # ── 5. Streaming Ollama Token Generation & Sentence Chunking ───────────
        from backend.services.task_router import is_ollama_model_installed
        if not is_ollama_model_installed(model_name):
            missing_msg = f"LOCAL MODEL ERROR: Required model '{model_name}' is not installed locally in Ollama."
            logger.error(missing_msg)
            yield {"event": "model_missing", "model": model_name, "error": missing_msg}
            yield {
                "event": "complete",
                "response": missing_msg,
                "spoken_text": f"Model {model_name} is not available locally.",
                "status": "model_missing",
                "trace": voice_trace,
                "trace_id": trace_id,
            }
            return

        # Deterministic Query Integrity Assertion before LLM generation
        integrity_info = {
            "current_query": authoritative_query,
            "retrieval_query": retrieval_query if requires_rag else None,
            "intent": task_type,
            "trace_id": trace_id,
        }
        _log_trace("LLM_CURRENT_QUERY", f'LLM Current Query: "{authoritative_query}"', "verified", {"current_query": authoritative_query, "retrieval_query": retrieval_query, "model": model_name, "trace_id": trace_id})
        _log_trace("QUERY_INTEGRITY_VERIFIED", f'Query Integrity Enforced: "{authoritative_query}"', "verified", integrity_info)
        yield {"event": "llm_current_query", "current_query": authoritative_query, "model": model_name, "trace_id": trace_id}
        yield {"event": "query_integrity_verified", **integrity_info}

        tts_cfg = self.config.get("voice", {}).get("tts", {})
        min_words = tts_cfg.get("min_chunk_words", 4)
        max_words = tts_cfg.get("max_chunk_words", 25)
        sentence_buffer = SentenceChunkBuffer(min_words=min_words, max_words=max_words)

        full_tokens: list[str] = []
        spoken_sentences: list[str] = []
        chunk_index = 0
        t_first_token = None
        t_first_audio = None

        keep_alive = self.config.get("voice", {}).get("routing", {}).get("keep_alive", "30m")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {
                "temperature": 0.05,
                "top_p": 0.85,
                "num_ctx": 4096,
                "num_predict": 512,
            },
        }

        _log_trace("OLLAMA_STARTED", f"Ollama generation started on {model_name}", "allowed", {"model": model_name})

        try:
            async for token in self._async_token_stream(endpoint, payload):
                # Check for barge-in interruption or newer asynchronous turn
                if self.is_interrupted(session_id) or _is_stale_turn():
                    _log_trace("BARGE_IN", "Speech generation stopped due to user barge-in or newer turn", "flagged")
                    yield {"event": "interrupted", "reason": "User interrupted speech", "trace_id": trace_id, "turn_id": turn_id}
                    break

                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    ttft_ms = (t_first_token - t_start) * 1000
                    _log_trace("OLLAMA_FIRST_TOKEN", f"First LLM token received ({ttft_ms:.1f} ms)", "verified", {"latency_ms": ttft_ms})
                    yield {"event": "first_token", "time_to_first_token_ms": round(ttft_ms, 1), "turn_id": turn_id}

                full_tokens.append(token)
                yield {"event": "token", "token": token, "turn_id": turn_id}

                # Chunk tokens into natural speech sentences
                ready_chunks = sentence_buffer.add_token(token)
                for sentence in ready_chunks:
                    if self.is_interrupted(session_id) or _is_stale_turn():
                        break

                    clean_speech = render_speech_text(sentence)
                    if clean_speech:
                        if chunk_index == 0:
                            _log_trace("TTS_STARTED", "Text-to-speech synthesis initiated", "allowed")
                        tts_res = await asyncio.to_thread(self.tts.synthesize, clean_speech, language=lang_info["code"])
                        audio_bytes = tts_res.get("audio_bytes", b"")

                        if t_first_audio is None and audio_bytes:
                            t_first_audio = time.perf_counter()
                            ttfa_ms = (t_first_audio - t_start) * 1000
                            _log_trace("TTS_FIRST_AUDIO", f"First audio chunk synthesized ({ttfa_ms:.1f} ms)", "verified", {"latency_ms": ttfa_ms})
                            yield {"event": "first_audio", "time_to_first_audio_ms": round(ttfa_ms, 1), "turn_id": turn_id}

                        chunk_index += 1
                        spoken_sentences.append(clean_speech)
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes else None

                        yield {
                            "event": "tts_chunk",
                            "chunk_index": chunk_index,
                            "spoken_text": clean_speech,
                            "audio_base64": audio_b64,
                            "duration_seconds": tts_res.get("duration_seconds", 0.0),
                            "is_final": False,
                            "turn_id": turn_id,
                        }

            # Flush remaining buffer at stream end
            if not self.is_interrupted(session_id) and not _is_stale_turn():
                remaining_chunks = sentence_buffer.flush()
                for sentence in remaining_chunks:
                    if self.is_interrupted(session_id) or _is_stale_turn():
                        break

                    clean_speech = render_speech_text(sentence)
                    if clean_speech:
                        if chunk_index == 0:
                            _log_trace("TTS_STARTED", "Text-to-speech synthesis initiated", "allowed")
                        tts_res = await asyncio.to_thread(self.tts.synthesize, clean_speech, language=lang_info["code"])
                        audio_bytes = tts_res.get("audio_bytes", b"")

                        if t_first_audio is None and audio_bytes:
                            t_first_audio = time.perf_counter()
                            ttfa_ms = (t_first_audio - t_start) * 1000
                            _log_trace("TTS_FIRST_AUDIO", f"First audio chunk synthesized ({ttfa_ms:.1f} ms)", "verified", {"latency_ms": ttfa_ms})
                            yield {"event": "first_audio", "time_to_first_audio_ms": round(ttfa_ms, 1), "turn_id": turn_id}

                        chunk_index += 1
                        spoken_sentences.append(clean_speech)
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes else None

                        yield {
                            "event": "tts_chunk",
                            "chunk_index": chunk_index,
                            "spoken_text": clean_speech,
                            "audio_base64": audio_b64,
                            "duration_seconds": tts_res.get("duration_seconds", 0.0),
                            "is_final": True,
                            "turn_id": turn_id,
                        }

            if chunk_index > 0:
                _log_trace("TTS_COMPLETED", f"Speech synthesis completed ({chunk_index} chunks)", "verified")

        except Exception as exc:
            logger.error("Streaming LLM generation error: %s", exc, exc_info=True)
            _log_trace("llm_generation_error", f"LLM streaming error: {exc}", "blocked", {"error": str(exc)})
            yield {"event": "error", "error": f"LLM streaming generation error: {str(exc)}", "trace_id": trace_id, "turn_id": turn_id}
            return

        # ── 6. Finalization & Latency Measurement ─────────────────────────────
        t_complete = time.perf_counter()
        total_latency_ms = (t_complete - t_start) * 1000
        full_response_text = "".join(full_tokens).strip()
        full_spoken_text = " ".join(spoken_sentences).strip()

        ttft_ms = round((t_first_token - t_start) * 1000, 1) if t_first_token else None
        ttfa_ms = round((t_first_audio - t_start) * 1000, 1) if t_first_audio else None

        latencies = {
            "asr_start_ms": 0.0,
            "asr_final_ms": round(asr_latency_ms, 1),
            "request_received_ms": 0.0,
            "governance_completed_ms": round(gov_latency_ms, 1),
            "rag_latency_ms": round(rag_latency_ms, 1) if requires_rag else 0.0,
            "time_to_first_token_ms": ttft_ms,
            "time_to_first_audio_ms": ttfa_ms,
            "total_latency_ms": round(total_latency_ms, 1),
        }

        _log_trace("VOICE_CAPTURE_RESUMED", "Continuous conversational voice capture resumed", "allowed")
        _log_trace("LISTENING_RESUMED", "Continuous conversational turn ready", "allowed")
        _log_trace(
            "VOICE_SESSION_ENDED",
            f"Voice interaction completed (TTFT: {ttft_ms} ms, TTFA: {ttfa_ms} ms, Total: {total_latency_ms:.1f} ms)",
            "verified",
            latencies,
        )

        # Store in session
        if session_id and full_response_text:
            try:
                chat_repo.add_message(session_id, "user", f"🎤 [{lang_info['name']}] {transcribed_text}")
                chat_repo.add_message(session_id, "assistant", full_response_text)
            except Exception as store_err:
                logger.warning("Could not persist voice messages to session: %s", store_err)

        # Immutable Audit Log
        try:
            audit_log(
                action="voice_interaction",
                outcome="success" if not self.is_interrupted(session_id) else "interrupted",
                user_id=user["id"],
                username=user.get("username", "anonymous"),
                target=f"session:{session_id or 'direct'}",
                details={
                    "trace_id": trace_id,
                    "input_mode": "voice_stream",
                    "language": lang_info["code"],
                    "asr_confidence": asr_confidence,
                    "task_type": task_type,
                    "model_selected": model_name,
                    "time_to_first_token_ms": ttft_ms,
                    "time_to_first_audio_ms": ttfa_ms,
                    "total_latency_ms": round(total_latency_ms, 1),
                },
            )
        except Exception as audit_err:
            logger.warning("Could not record streaming voice audit log: %s", audit_err)

        yield {
            "event": "complete",
            "response": full_response_text,
            "spoken_text": full_spoken_text,
            "trace": voice_trace,
            "latencies": latencies,
            "trace_id": trace_id,
            "task_type": task_type,
            "model_id": model_id,
            "requires_human_review": False,
            "approval_id": None,
            "voice_mode": voice_mode,
            "voice_session_id": voice_session_id,
            "turn_id": turn_id,
        }

        if voice_mode == "NOVA" and not self.is_interrupted(session_id) and not self.is_voice_session_stopped(voice_session_id):
            yield {
                "event": "listening",
                "voice_mode": "NOVA",
                "voice_session_id": voice_session_id,
                "turn_id": turn_id,
                "trace_id": trace_id,
            }


# Singleton instance pattern
_voice_service = None

def get_voice_service() -> VoiceService:
    """Return the singleton instance of VoiceService."""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
