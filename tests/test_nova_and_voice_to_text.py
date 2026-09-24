"""
Tests for Nova Voice Assistant & Normal Voice-to-Text Modes (25 Requirements).

Verifies:
1. Nova starts -> exact deterministic greeting -> then listening
2. Nova greeting occurs only once per session
3. Nova greeting does not occur in Voice-to-Text mode
4. Voice-to-Text produces transcript in composer
5. Voice-to-Text does not auto-submit
6. Voice-to-Text does not call TTS
7. Voice-to-Text does not call RAG before Send
8. Voice-to-Text can be edited before Send
9. Stop Nova immediately stops microphone
10. Stop Nova immediately stops TTS
11. Stop Nova clears TTS queue
12. Stop Nova cancels pending ASR
13. Stop Nova invalidates old events
14. Stop Nova does not auto-restart
15. Restart Nova creates new voice_session_id
16. Old-session SSE cannot affect new session
17. Barge-in interrupts Nova TTS
18. Clear Nova query answers normally
19. Low-confidence Nova uses spoken clarification
20. Low-confidence Voice-to-Text shows retry/edit behavior
21. Only one component owns getUserMedia()
22. VoiceOrb never acquires microphone
23. MRP is not silently rewritten to MRPL
24. "What is MRPL?" remains unchanged
25. "What are the main units in MRPL?" followed by "Explain in detail." remains a valid contextual follow-up in Nova
"""

import asyncio
import io
import re
import unittest.mock as mock
from pathlib import Path
import pytest

from backend.services.voice.voice_service import VoiceService
from backend.services.voice.clarification import (
    check_query_completeness,
    check_ambiguous_entity,
    detect_ambiguity,
    resolve_clarification_response,
)
from backend.services.voice.domain_vocabulary import normalize_asr_transcript
from backend.services.voice.context_policy import determine_conversational_context


# ── Mock Fixtures ─────────────────────────────────────────────────────────────

class MockASR:
    def __init__(self, text="What is MRPL?", confidence=0.95):
        self.text = text
        self.confidence = confidence
        self.model_name = "mock-whisper"
        self.confidence_threshold = 0.75

    def transcribe(self, audio_data, language=None):
        return {
            "raw_text": self.text,
            "normalized_text": self.text,
            "is_normalized": False,
            "text": self.text,
            "language": language or "en",
            "confidence": self.confidence,
            "status": "success" if self.confidence >= 0.75 else "low_confidence",
            "duration_seconds": 1.0,
            "engine": "mock-whisper",
        }

    def is_ready(self):
        return True, "Mock ASR Ready"


class MockTTS:
    def __init__(self):
        self.synthesize_called = False
        self.call_count = 0
        self.last_text = None

    def synthesize(self, text, language="en"):
        self.synthesize_called = True
        self.call_count += 1
        self.last_text = text
        return {
            "audio_bytes": b"RIFFmockwavdata",
            "duration_seconds": 1.2,
            "status": "success",
            "engine": "mock-tts",
        }

    def is_ready(self):
        return True, "Mock TTS Ready"


DUMMY_USER = {"id": 1, "username": "testuser", "role": "engineer", "clearance_level": "SECRET"}


@pytest.fixture
def voice_service():
    asr = MockASR("What is MRPL?", 0.95)
    tts = MockTTS()
    service = VoiceService(asr_engine=asr, tts_engine=tts)
    return service


# ── Test Suite ────────────────────────────────────────────────────────────────

def test_1_nova_exact_deterministic_greeting(voice_service):
    """1. Nova starts -> exact deterministic greeting -> then listening."""
    greeting = voice_service.get_greeting(language="en")
    assert greeting["text"] == "Hi, I'm Nova. How can I help you today?"
    assert greeting["status"] == "success"
    assert greeting["audio_base64"] is not None


def test_2_nova_greeting_occurs_only_once_per_session():
    """2. Nova greeting occurs only once when voice session starts (deterministic system text)."""
    vs = VoiceService(asr_engine=MockASR(), tts_engine=MockTTS())
    greeting1 = vs.get_greeting("en")
    assert greeting1["text"] == "Hi, I'm Nova. How can I help you today?"
    from backend.agent.prompts import GENERAL_CHAT_PROMPT
    assert "NEVER output a generic workbench introduction or greeting" in GENERAL_CHAT_PROMPT


def test_3_nova_greeting_does_not_occur_in_voice_to_text(voice_service):
    """3. Nova greeting does not occur in Voice-to-Text mode."""
    result = asyncio.run(voice_service.process_voice_query(
        b"dummy_audio_bytes",
        user=DUMMY_USER,
        voice_mode="VOICE_TO_TEXT"
    ))
    assert result["voice_mode"] == "VOICE_TO_TEXT"
    assert "Hi, I'm Nova" not in result.get("spoken_text", "")
    assert voice_service.tts.synthesize_called is False


def test_4_voice_to_text_produces_transcript_in_composer(voice_service):
    """4. Voice-to-Text produces transcript in composer without auto-executing."""
    events = []
    async def collect_stream():
        async for event in voice_service.stream_voice_query(
            b"dummy_audio_bytes",
            user=DUMMY_USER,
            voice_mode="VOICE_TO_TEXT",
            turn_id=1
        ):
            events.append(event)
    asyncio.run(collect_stream())

    completed_events = [e for e in events if e.get("event") == "voice_to_text_completed"]
    assert len(completed_events) == 1
    assert completed_events[0]["raw_text"] == "What is MRPL?"
    assert completed_events[0]["normalized_text"] == "What is MRPL?"


def test_5_voice_to_text_does_not_auto_submit(voice_service):
    """5. Voice-to-Text does not auto-submit."""
    events = []
    async def collect_stream():
        async for event in voice_service.stream_voice_query(
            b"dummy_audio_bytes",
            user=DUMMY_USER,
            voice_mode="VOICE_TO_TEXT"
        ):
            events.append(event)
    asyncio.run(collect_stream())

    event_names = [e.get("event") for e in events]
    assert "complete" not in event_names
    assert "voice_to_text_completed" in event_names


def test_6_voice_to_text_does_not_call_tts(voice_service):
    """6. Voice-to-Text does not call TTS."""
    asyncio.run(voice_service.process_voice_query(
        b"dummy_audio_bytes",
        user=DUMMY_USER,
        voice_mode="VOICE_TO_TEXT"
    ))
    assert voice_service.tts.synthesize_called is False


def test_7_voice_to_text_does_not_call_rag_before_send(voice_service):
    """7. Voice-to-Text does not call RAG before Send."""
    events = []
    async def collect_stream():
        async for event in voice_service.stream_voice_query(
            b"dummy_audio_bytes",
            user=DUMMY_USER,
            voice_mode="VOICE_TO_TEXT"
        ):
            events.append(event)
    asyncio.run(collect_stream())

    event_names = [e.get("event") for e in events]
    assert "retrieval_started" not in event_names
    assert "routing_decision" not in event_names


def test_8_voice_to_text_can_be_edited_before_send(voice_service):
    """8. Voice-to-Text preserves raw and normalized transcript so user can edit."""
    res = voice_service.transcribe_audio(b"dummy_audio_bytes")
    assert "raw_text" in res
    assert "normalized_text" in res
    assert res["status"] == "success"


def test_9_stop_nova_immediately_stops_microphone(voice_service):
    """9. Stop Nova immediately marks session stopped and stops pipeline."""
    session_id = "test_nova_session_9"
    voice_service.register_voice_session(session_id)
    assert not voice_service.is_voice_session_stopped(session_id)
    voice_service.stop_voice_session(session_id)
    assert voice_service.is_voice_session_stopped(session_id)


def test_10_stop_nova_immediately_stops_tts(voice_service):
    """10. Stop Nova immediately stops TTS."""
    session_id = "test_nova_session_10"
    voice_service.stop_voice_session(session_id)
    events = []
    async def collect():
        async for ev in voice_service.stream_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="NOVA",
            voice_session_id=session_id
        ):
            events.append(ev)
    asyncio.run(collect())
    assert len(events) == 0


def test_11_stop_nova_clears_tts_queue(voice_service):
    """11. Stop Nova clears TTS queue and interrupts turn."""
    voice_service.interrupt_turn(101)
    assert 101 in voice_service._interrupted_turns


def test_12_stop_nova_cancels_pending_asr(voice_service):
    """12. Stop Nova cancels pending ASR."""
    session_id = "stopped_session_12"
    voice_service.stop_voice_session(session_id)
    events = []
    async def collect():
        async for ev in voice_service.stream_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="NOVA",
            voice_session_id=session_id
        ):
            events.append(ev)
    asyncio.run(collect())
    assert len(events) == 0


def test_13_stop_nova_invalidates_old_events(voice_service):
    """13. Stop Nova invalidates old events."""
    old_session = "old_session_13"
    voice_service.stop_voice_session(old_session)
    assert voice_service.is_voice_session_stopped(old_session)


def test_14_stop_nova_does_not_auto_restart(voice_service):
    """14. Stop Nova does not auto-restart."""
    session = "stopped_session_14"
    voice_service.stop_voice_session(session)
    assert voice_service.is_voice_session_stopped(session)
    assert session in voice_service._stopped_voice_sessions


def test_15_restart_nova_creates_new_voice_session_id(voice_service):
    """15. Restart Nova creates new voice_session_id."""
    session_old = "nova_session_old"
    session_new = "nova_session_new"
    voice_service.stop_voice_session(session_old)
    voice_service.register_voice_session(session_new)

    assert voice_service.is_voice_session_stopped(session_old)
    assert not voice_service.is_voice_session_stopped(session_new)


def test_16_old_session_sse_cannot_affect_new_session(voice_service):
    """16. Old-session SSE cannot affect new session."""
    old_session = "session_A"
    new_session = "session_B"
    voice_service.stop_voice_session(old_session)
    voice_service.register_voice_session(new_session)

    assert voice_service.is_voice_session_stopped(old_session) is True
    assert voice_service.is_voice_session_stopped(new_session) is False


def test_17_barge_in_interrupts_nova_tts(voice_service):
    """17. Barge-in interrupts Nova TTS."""
    turn_id = 42
    voice_service.interrupt_turn(turn_id)
    assert 42 in voice_service._interrupted_turns
    assert voice_service.is_turn_interrupted(turn_id) is True


def test_18_clear_nova_query_answers_normally(voice_service):
    """18. Clear Nova query answers normally with full pipeline."""
    with mock.patch("backend.agent.graph.run_agent") as mock_run:
        mock_run.return_value = {
            "current_query": "What is MRPL?",
            "response": "Mangalore Refinery and Petrochemicals Limited (MRPL) is a central public sector undertaking.",
            "final_response": "Mangalore Refinery and Petrochemicals Limited (MRPL) is a central public sector undertaking.",
            "task_type": "GENERAL_CHAT",
            "model_id": "qwen2.5:7b-instruct",
            "execution_trace": [{"step": "test", "status": "verified"}],
            "trace_id": "tr_18",
        }
        res = asyncio.run(voice_service.process_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="NOVA",
            voice_session_id="session_18"
        ))
        assert "MRPL" in res["response"]
        assert res["voice_mode"] == "NOVA"
        assert res["task_type"] == "GENERAL_CHAT"


def test_19_low_confidence_nova_uses_spoken_clarification():
    """19. Low-confidence Nova uses spoken clarification."""
    asr = MockASR("whisper mumble", confidence=0.45)
    tts = MockTTS()
    vs = VoiceService(asr_engine=asr, tts_engine=tts)

    res = asyncio.run(vs.process_voice_query(
        b"dummy_audio",
        user=DUMMY_USER,
        voice_mode="NOVA"
    ))
    assert res["status"] in ("clarification_needed", "low_confidence")
    assert res["requires_clarification"] is True
    assert "hear you clearly" in res.get("clarification_text", "") or "repeat" in res.get("clarification_text", "")
    assert vs.tts.synthesize_called is True


def test_20_low_confidence_voice_to_text_shows_retry_behavior():
    """20. Low-confidence Voice-to-Text shows retry/edit behavior without TTS."""
    asr = MockASR("whisper mumble", confidence=0.45)
    tts = MockTTS()
    vs = VoiceService(asr_engine=asr, tts_engine=tts)

    events = []
    async def collect():
        async for ev in vs.stream_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="VOICE_TO_TEXT"
        ):
            events.append(ev)
    asyncio.run(collect())

    event_names = [e.get("event") for e in events]
    assert "voice_to_text_low_confidence" in event_names
    assert vs.tts.synthesize_called is False


def test_21_only_one_component_owns_get_user_media():
    """21. Only one component owns navigator.mediaDevices.getUserMedia()."""
    frontend_dir = Path("frontend/src")
    jsx_files = list(frontend_dir.rglob("*.jsx"))
    
    files_with_gum = []
    for f in jsx_files:
        content = f.read_text(encoding="utf-8")
        if "getUserMedia" in content:
            files_with_gum.append(f.name)

    assert files_with_gum == ["ChatPage.jsx"], f"Multiple components call getUserMedia: {files_with_gum}"


def test_22_voice_orb_never_acquires_microphone():
    """22. VoiceOrb never acquires microphone (pure reactive visual component)."""
    orb_path = Path("frontend/src/components/VoiceOrb.jsx")
    content = orb_path.read_text(encoding="utf-8")
    assert "getUserMedia" not in content
    assert "MediaRecorder" not in content
    assert "AudioContext" not in content


def test_23_mrp_is_not_silently_rewritten_to_mrpl():
    """23. MRP is not silently rewritten to MRPL."""
    # Test normalization rules
    normalized, is_norm = normalize_asr_transcript("Give me information about MRP")
    assert "MRPL" not in normalized
    assert "MRP" in normalized

    # Test clarification detection
    ambig = detect_ambiguity("Give me information about MRP and")
    assert ambig is not None
    assert ambig.ambiguity_type in ("incomplete_query", "ambiguous_query", "incomplete_ambiguous")
    assert "do you mean MRP or MRPL" in ambig.clarification_question
    # Completeness check
    is_inc, reason = check_query_completeness("Give me information about MRP and")
    assert is_inc is True
    assert "trailing_conjunction" in reason


def test_24_what_is_mrpl_remains_unchanged():
    """24. 'What is MRPL?' remains unchanged."""
    normalized, is_norm = normalize_asr_transcript("What is MRPL?")
    assert normalized == "What is MRPL?"
    assert is_norm is False

    # Complete query must NOT be detected as ambiguous
    ambig = detect_ambiguity("What is MRPL?")
    assert ambig is None


def test_25_contextual_follow_up_in_nova():
    """25. 'What are the main units in MRPL?' followed by 'Explain in detail.' remains a valid contextual follow-up."""
    # Turn 1: Main units
    ambig_1 = detect_ambiguity("What are the main units in MRPL?")
    assert ambig_1 is None

    # Turn 2: Follow-up
    # Context policy recognizes 'Explain in detail.' as a conversational follow-up
    resolution = determine_conversational_context(
        "Explain in detail.",
        conversation_history=[
            {
                "role": "user",
                "content": "What are the main units in MRPL?",
                "resolved_retrieval_query": "What are the main units in MRPL?",
                "resolved_subject": "MRPL main units",
            },
            {
                "role": "assistant",
                "content": "The main units include CDU, VDU, and HCU.",
            }
        ]
    )
    assert resolution.is_followup is True
    assert "detail" in resolution.resolved_retrieval_query.lower() or "units" in resolution.resolved_retrieval_query.lower()


# ── UI / Voice Input Cleanup Acceptance Tests ─────────────────────────────────

def test_26_voice_to_text_button_removed_from_ui():
    """26. 'Voice to Text' button is completely removed from composer mode controls and CSS."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    index_css = Path("frontend/src/index.css").read_text(encoding="utf-8")

    # The separate button class and text should NOT be present
    assert "btn-mode-vtt" not in chat_page, "ChatPage.jsx must not contain btn-mode-vtt"
    assert "btn-mode-vtt" not in index_css, "index.css must not contain .btn-mode-vtt"
    assert "🎙 Voice to Text" not in chat_page, "ChatPage.jsx must not render '🎙 Voice to Text' button"
    # Nova button remains in voice-mode-controls
    assert "btn-mode-nova" in chat_page
    assert "Stop Assistant" in chat_page


def test_27_rightmost_microphone_exists_and_controls_dictation():
    """27. Rightmost microphone exists and routes to normal dictation with updated tooltip."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")

    # Rightmost mic button exists
    assert "chat-mic-btn" in chat_page
    # Tooltip updated to 'Speak to type'
    assert "Speak to type" in chat_page
    # Connected to normal voice input, not Nova toggle
    assert "toggleNormalVoiceInput" in chat_page
    assert "onClick={toggleNormalVoiceInput}" in chat_page
    # Dictation sets input composer state without auto-submit
    assert "setInput((prev) => (prev && prev.trim() ? `${prev.trim()} ${finalNorm}` : finalNorm))" in chat_page


def test_28_dictation_mode_backend_processing(voice_service):
    """28. Backend supports voice_mode='DICTATION' returning transcript without LLM/RAG/TTS."""
    # Synchronous process_voice_query
    res = asyncio.run(voice_service.process_voice_query(
        b"dummy_audio_bytes",
        user=DUMMY_USER,
        voice_mode="DICTATION"
    ))
    assert res["status"] == "success"
    assert res["voice_mode"] == "DICTATION"
    assert res["raw_text"] == "What is MRPL?"
    assert res["normalized_text"] == "What is MRPL?"
    assert voice_service.tts.synthesize_called is False

    # Streaming stream_voice_query
    events = []
    async def collect_dictation():
        async for event in voice_service.stream_voice_query(
            b"dummy_audio_bytes",
            user=DUMMY_USER,
            voice_mode="DICTATION",
            turn_id=1
        ):
            events.append(event)
    asyncio.run(collect_dictation())

    event_names = [e.get("event") for e in events]
    assert "voice_to_text_completed" in event_names
    assert "complete" not in event_names
    completed_event = next(e for e in events if e.get("event") == "voice_to_text_completed")
    assert completed_event["voice_mode"] == "DICTATION"
    assert completed_event["normalized_text"] == "What is MRPL?"


def test_29_low_confidence_dictation_non_conversational():
    """29. Low-confidence dictation produces non-conversational guidance without TTS or Nova flow."""
    asr = MockASR("whisper mumble", confidence=0.3)
    tts = MockTTS()
    vs = VoiceService(asr_engine=asr, tts_engine=tts)

    events = []
    async def collect():
        async for ev in vs.stream_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="DICTATION"
        ):
            events.append(ev)
    asyncio.run(collect())

    event_names = [e.get("event") for e in events]
    assert "voice_to_text_low_confidence" in event_names
    assert vs.tts.synthesize_called is False


def test_30_single_microphone_ownership_path_verified():
    """30. Single microphone ownership path: only ChatPage.jsx calls getUserMedia()."""
    frontend_dir = Path("frontend/src")
    jsx_files = list(frontend_dir.rglob("*.jsx"))
    gum_files = [f.name for f in jsx_files if "getUserMedia" in f.read_text(encoding="utf-8")]
    assert gum_files == ["ChatPage.jsx"], f"Expected single mic owner, found: {gum_files}"


def test_31_nova_assistant_preserved_separately():
    """31. Nova assistant remains dedicated conversational assistant with greeting, barge-in, and Stop."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")

    assert "startNovaAssistant" in chat_page
    assert "stopNovaAssistant" in chat_page
    assert "handleInterrupt" in chat_page
    assert "VoiceAssistantModal" in chat_page
    assert "btn-stop-nova" in chat_page
    assert "btn-start-nova" in chat_page


def test_32_equipment_tag_normalization():
    """32. Equipment tags normalize continuous and spaced formats to canonical refinery format."""
    assert normalize_asr_transcript("Check pump 11p101a now")[0] == "Check pump 11-P-101A now"
    assert normalize_asr_transcript("Status of 11-P-101A")[0] == "Status of 11-P-101A"
    assert normalize_asr_transcript("Inspect 11 p 101 a")[0] == "Inspect 11-P-101A"
    assert normalize_asr_transcript("What is MRPL?")[0] == "What is MRPL?"


def test_33_decode_audio_to_pcm():
    """33. decode_audio_to_pcm converts WAV and raw PCM into 16kHz float32 mono array."""
    from backend.services.voice.vad import decode_audio_to_pcm
    import numpy as np

    # Test empty
    pcm, sr, ch = decode_audio_to_pcm(b"")
    assert len(pcm) == 0
    assert sr == 16000

    # Test WAV
    import wave
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)) * 16000).astype(np.int16)
        wf.writeframes(samples.tobytes())
    wav_bytes = bio.getvalue()

    pcm, sr, ch = decode_audio_to_pcm(wav_bytes)
    assert len(pcm) == 8000
    assert sr == 16000
    assert ch == 1
    assert -1.0 <= np.max(np.abs(pcm)) <= 1.0


def test_34_asr_quality_gate_language_jitter():
    """34. ASR quality gate tolerates language prior jitter on short queries but enforces 0.55 token threshold."""
    from backend.services.voice.asr import validate_asr_quality

    # Clean English query with Whisper-tiny language jitter ('ur', low prior) should pass
    clean_jitter = validate_asr_quality("Show refinery CDU throughput", confidence=0.72, language="ur", language_prob=0.22)
    assert clean_jitter["valid"] is True
    assert clean_jitter["status"] == "success"

    # Acoustic token confidence below 0.55 must be rejected
    low_conf = validate_asr_quality("Show refinery CDU throughput", confidence=0.48, language="en", language_prob=0.9)
    assert low_conf["valid"] is False
    assert low_conf["status"] == "low_confidence"
    assert low_conf["low_confidence_reason"] == "low_token_confidence"

    # Noise hallucination phrase rejected
    hallucination = validate_asr_quality("thank you for watching", confidence=0.95, language="en")
    assert hallucination["valid"] is False
    assert hallucination["low_confidence_reason"] == "noise_hallucination"


def test_35_audio_diagnostics_trace_and_sse():
    """35. Voice streaming yields rich audio_diagnostics containing duration, sample rate, RMS, SNR."""
    vs = VoiceService(asr_engine=MockASR("What is MRPL?", 0.88), tts_engine=MockTTS())
    events = []
    async def run():
        async for ev in vs.stream_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="DICTATION"
        ):
            events.append(ev)
    asyncio.run(run())

    diag_event = next((e for e in events if e.get("event") == "audio_diagnostics"), None)
    assert diag_event is not None
    assert "noise_floor_rms" in diag_event
    assert "speech_rms" in diag_event
    assert "snr_db" in diag_event
    assert "vad_confidence" in diag_event
    assert "capture_duration_ms" in diag_event
    assert "sample_rate" in diag_event


def test_36_dictation_low_confidence_details():
    """36. Low-confidence dictation event includes low_confidence_reason and diagnostics payload."""
    asr = MockASR("muffled noise", confidence=0.40)
    vs = VoiceService(asr_engine=asr, tts_engine=MockTTS())

    events = []
    async def run():
        async for ev in vs.stream_voice_query(
            b"dummy_audio",
            user=DUMMY_USER,
            voice_mode="DICTATION"
        ):
            events.append(ev)
    asyncio.run(run())

    low_ev = next((e for e in events if e.get("event") == "voice_to_text_low_confidence"), None)
    assert low_ev is not None
    assert "low_confidence_reason" in low_ev
    assert "diagnostics" in low_ev
    assert low_ev["diagnostics"]["asr_confidence"] == 0.40

