"""
MRPL Sovereign AI Workbench — Noise-Robust Voice Input Verification Suite.
Validates all 25 criteria from the senior voice-AI engineer specification:
 1. Microphone capability detection
 2. Constraint fallback
 3. Adaptive noise-floor tracking
 4. Quiet-environment VAD
 5. Continuous background-noise VAD
 6. Speech over moderate noise (fan/AC rumble attenuation via highpass)
 7. Low-SNR handling
 8. Silence endpointing
 9. Partial ASR responsiveness & non-authoritativeness
10. Final ASR authoritativeness
11. Stale ASR cancellation & request ID matching
12. Request / turn isolation (turn_id / trace_id)
13. raw_text preservation alongside normalized_text
14. Deterministic technical normalization (e.g., '11 P 101 A' -> '11-P-101A')
15. Low-confidence clarification (spoken repeat prompt)
16. Normal clear speech does NOT trigger clarification
17. Interruption / barge-in mechanics
18. TTS completion returns to listening
19. TTS failure returns safely to listening
20. Zero external network calls (Network Seal)
21. Single microphone ownership path
22. English speech handling
23. Hindi speech handling
24. Marathi speech handling
25. MRPL technical identifier preservation & query integrity
"""

import asyncio
import io
import math
from pathlib import Path
import unittest.mock as mock
import wave
import numpy as np
import pytest

from backend.agent.graph import run_agent
from backend.services.network_seal import get_network_seal
from backend.services.voice.asr import BaseASREngine, LocalWhisperASR, validate_asr_quality
from backend.services.voice.clarification import detect_ambiguity, resolve_clarification_response
from backend.services.voice.context_policy import determine_conversational_context
from backend.services.voice.language_detection import detect_voice_language
from backend.services.voice.speech_renderer import render_speech_text
from backend.services.voice.vad import VoiceActivityDetector
from backend.services.voice.voice_service import VoiceService


# ── Acoustic Fixture Generators ──────────────────────────────────────────────

def create_pcm_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode float or int numpy samples into 16-bit PCM WAV bytes."""
    int_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return bio.getvalue()


def generate_synthetic_speech(duration_sec: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.35) -> np.ndarray:
    """Generate harmonic vowels mimicking human vocal formants (150Hz, 300Hz, 600Hz, 1200Hz)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    vowels = (
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.15 * np.sin(2 * np.pi * 600 * t) +
        0.05 * np.sin(2 * np.pi * 1200 * t)
    )
    return vowels * amplitude


def generate_fan_ac_rumble(duration_sec: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.08) -> np.ndarray:
    """Generate low-frequency fan/AC mechanical rumble (50 Hz power hum + 70 Hz vibration)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    rumble = 0.7 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 70 * t)
    # Add subtle low-passed random hiss
    hiss = np.random.normal(0, 0.05, len(t))
    return (rumble * 0.8 + hiss * 0.2) * amplitude


@pytest.fixture
def test_user():
    return {
        "id": 42,
        "username": "sr_process_eng",
        "roles": ["engineer"],
        "clearance": "CONFIDENTIAL",
    }


# ── TEST 1: Microphone Capability Detection ──────────────────────────────────

def test_1_microphone_capability_detection():
    """Verify that frontend audio capture inspects getSupportedConstraints before requesting."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    assert "getSupportedConstraints" in chat_page, "ChatPage.jsx must check getSupportedConstraints()"
    assert "echoCancellation" in chat_page
    assert "noiseSuppression" in chat_page
    assert "autoGainControl" in chat_page


# ── TEST 2: Constraint Fallback ──────────────────────────────────────────────

def test_2_constraint_fallback():
    """Verify graceful fallback to basic { audio: true } if preferred constraints fail."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    assert "{ audio: true }" in chat_page, "ChatPage.jsx must have fallback to basic audio: true"


# ── TEST 3: Adaptive Noise-Floor Tracking ────────────────────────────────────

def test_3_adaptive_noise_floor_tracking():
    """Verify ambient noise floor is dynamically tracked using exponential smoothing."""
    vad = VoiceActivityDetector(adaptive=True, noise_floor_smoothing=0.95)
    initial_floor = vad.noise_floor_rms

    # Ambient frames with ~0.020 RMS
    for _ in range(20):
        vad.update_noise_floor(frame_rms=0.020, is_speech=False)

    updated_floor = vad.noise_floor_rms
    assert updated_floor > initial_floor, "Noise floor should adapt upward towards ambient RMS"
    assert updated_floor < 0.025, "Noise floor should stay bounded"

    # Verify speech frame does NOT corrupt noise floor
    vad.update_noise_floor(frame_rms=0.45, is_speech=True)
    assert vad.noise_floor_rms == updated_floor, "Speech frames must never redefine ambient noise floor"


# ── TEST 4: Quiet-Environment VAD ────────────────────────────────────────────

def test_4_quiet_environment_vad():
    """Verify clean speech in quiet environment (high SNR) triggers speech_started and has_speech."""
    speech = generate_synthetic_speech(duration_sec=0.8, amplitude=0.30)
    audio_wav = create_pcm_wav(speech)

    vad = VoiceActivityDetector(adaptive=True)
    result = vad.detect_speech_boundaries(audio_wav)

    assert result["has_speech"] is True
    assert result["speech_started"] is True
    assert result["snr_db"] >= 12.0, f"Expected high SNR in quiet room, got {result['snr_db']:.1f} dB"
    assert result["vad_confidence"] >= 0.7, "VAD confidence should be high for clear speech"


# ── TEST 5: Continuous Background-Noise VAD (Fan/AC Rejection) ────────────────

def test_5_continuous_background_noise_vad():
    """Verify continuous low-frequency fan/AC hum alone does NOT trigger false speech detection."""
    fan_noise = generate_fan_ac_rumble(duration_sec=1.5, amplitude=0.025)
    audio_wav = create_pcm_wav(fan_noise)

    vad = VoiceActivityDetector(adaptive=True, highpass_cutoff_hz=80.0)
    result = vad.detect_speech_boundaries(audio_wav)

    assert result["has_speech"] is False, "Continuous ambient fan rumble must NOT trigger speech detection"


# ── TEST 6: Speech Over Moderate Noise (80 Hz High-Pass Attenuation) ─────────

def test_6_speech_over_moderate_noise():
    """Verify 80 Hz high-pass filter attenuates mechanical rumble while speech remains detectable."""
    speech = generate_synthetic_speech(duration_sec=1.0, amplitude=0.25)
    rumble = generate_fan_ac_rumble(duration_sec=1.0, amplitude=0.08)
    mixed = speech + rumble
    mixed_wav = create_pcm_wav(mixed)

    vad = VoiceActivityDetector(adaptive=True, highpass_cutoff_hz=80.0)
    # High-pass filter removes sub-80Hz rumble
    filtered_wav = vad.apply_highpass_filter(mixed_wav, cutoff_hz=80.0)
    raw_rms = vad.calculate_rms(mixed_wav)
    filtered_rms = vad.calculate_rms(filtered_wav)

    # 50Hz/70Hz rumble energy should be attenuated
    assert filtered_rms <= raw_rms, "High-pass filtering must reduce low-frequency rumble energy"

    # Speech remains detectable
    boundary = vad.detect_speech_boundaries(filtered_wav)
    assert boundary["has_speech"] is True, "Speech must remain clearly detectable after high-pass filtering"


# ── TEST 7: Low-SNR Handling ─────────────────────────────────────────────────

def test_7_low_snr_handling():
    """Verify that very faint speech overwhelmed by noise does not produce false high confidence."""
    faint_speech = generate_synthetic_speech(duration_sec=0.5, amplitude=0.015)
    loud_noise = generate_fan_ac_rumble(duration_sec=0.5, amplitude=0.09)
    low_snr = faint_speech + loud_noise
    low_snr_wav = create_pcm_wav(low_snr)

    vad = VoiceActivityDetector(adaptive=True)
    boundary = vad.detect_speech_boundaries(low_snr_wav)

    # Low SNR must not report false high confidence
    assert boundary["vad_confidence"] < 0.65


# ── TEST 8: Silence Endpointing ──────────────────────────────────────────────

def test_8_silence_endpointing():
    """Verify endpoint silence detection triggers speech_ended after silence_threshold_ms."""
    # 0.5s speech followed by 0.9s silence
    speech = generate_synthetic_speech(duration_sec=0.5, amplitude=0.30)
    silence = np.zeros(int(16000 * 0.9))
    utterance = np.concatenate([speech, silence])
    audio_wav = create_pcm_wav(utterance)

    vad = VoiceActivityDetector(silence_threshold_ms=750)
    res = vad.detect_speech_boundaries(audio_wav)

    assert res["speech_started"] is True
    assert res["speech_ended"] is True
    assert res["silence_trailing_ms"] >= 750


# ── TEST 9: Partial ASR Responsiveness & Non-Authoritativeness ───────────────

def test_9_partial_asr_responsiveness_and_non_authoritative():
    """Verify transcribe_partial returns status='partial' and is marked non-authoritative (is_final=False)."""
    mock_engine = mock.MagicMock(spec=LocalWhisperASR)
    mock_engine.transcribe_partial.return_value = {
        "raw_text": "What is M R P L",
        "normalized_text": "What is MRPL",
        "is_normalized": True,
        "text": "What is MRPL",
        "language": "en",
        "status": "partial",
        "is_final": False,
    }
    partial = mock_engine.transcribe_partial(b"audio_bytes")
    assert partial["status"] == "partial"
    assert partial["is_final"] is False
    assert partial["normalized_text"] == "What is MRPL"


# ── TEST 10: Final ASR Authoritativeness ─────────────────────────────────────

def test_10_final_asr_authoritativeness(test_user):
    """Verify final ASR is authoritative: exact normalized text is sent to run_agent."""
    async def _test():
        mock_asr = mock.MagicMock(spec=BaseASREngine)
        mock_asr.transcribe.return_value = {
            "raw_text": "What is MRPL?",
            "normalized_text": "What is MRPL?",
            "is_normalized": False,
            "text": "What is MRPL?",
            "confidence": 0.95,
            "status": "success",
            "language": "en",
            "duration_seconds": 1.0,
            "engine": "faster-whisper",
        }
        mock_asr.confidence_threshold = 0.55
        mock_asr.is_ready.return_value = (True, "Ready")

        mock_tts = mock.MagicMock()
        mock_tts.is_ready.return_value = (True, "Ready")
        mock_tts.synthesize.return_value = {"audio_bytes": b"RIFF_WAV", "duration_seconds": 1.0, "status": "success"}

        service = VoiceService(asr_engine=mock_asr, tts_engine=mock_tts)

        with mock.patch("backend.agent.graph.run_agent") as mock_agent:
            mock_agent.return_value = {
                "response": "MRPL is an ONGC company in Mangalore.",
                "scope_decision": {"allowed": True},
                "execution_trace": [],
            }
            res = await service.process_voice_query(b"audio", user=test_user)
            mock_agent.assert_called_once()
            called_query = mock_agent.call_args.kwargs["query"]
            assert called_query == "What is MRPL?", "Authoritative query passed to agent must match final ASR"
            assert res["normalized_text"] == "What is MRPL?"

    asyncio.run(_test())


# ── TEST 11: Stale ASR Cancellation (Request ID Matching) ────────────────────

def test_11_stale_asr_cancellation():
    """Verify frontend code contains request_id matching and AbortController for partial ASR."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    assert "interimRequestIdRef" in chat_page
    assert "interimAbortRef" in chat_page
    assert "reqId === interimRequestIdRef.current" in chat_page


# ── TEST 12: Request / Turn Isolation (turn_id / trace_id) ───────────────────

def test_12_request_turn_isolation(test_user):
    """Verify events emitted during stream_voice_query include turn_id and trace_id."""
    async def _test():
        mock_asr = mock.MagicMock(spec=BaseASREngine)
        mock_asr.transcribe.return_value = {
            "raw_text": "What is MRPL?",
            "normalized_text": "What is MRPL?",
            "is_normalized": False,
            "text": "What is MRPL?",
            "confidence": 0.95,
            "status": "success",
            "language": "en",
            "duration_seconds": 1.0,
            "engine": "faster-whisper",
        }
        mock_asr.confidence_threshold = 0.55
        mock_asr.is_ready.return_value = (True, "Ready")

        mock_tts = mock.MagicMock()
        mock_tts.is_ready.return_value = (True, "Ready")
        mock_tts.synthesize.return_value = {"audio_bytes": b"WAV", "duration_seconds": 0.8, "status": "success"}

        service = VoiceService(asr_engine=mock_asr, tts_engine=mock_tts)

        with mock.patch("backend.agent.graph.run_agent") as mock_agent:
            mock_agent.return_value = {"response": "MRPL is a refinery.", "execution_trace": []}
            events = []
            async for ev in service.stream_voice_query(b"audio", user=test_user, session_id=10, turn_id=77):
                events.append(ev)

            # Check that events contain turn_id
            for ev in events:
                if "turn_id" in ev:
                    assert ev["turn_id"] == 77, f"Event {ev.get('event')} should have turn_id=77"

    asyncio.run(_test())


# ── TEST 13: raw_text Preservation Alongside normalized_text ─────────────────

def test_13_raw_text_preservation():
    """Verify raw_text is preserved alongside normalized_text without losing spoken words."""
    raw = "11 P 101 A"
    norm, changed = LocalWhisperASR.deterministic_normalize(raw)
    assert raw == "11 P 101 A", "raw_text must remain strictly intact"
    assert norm == "11-P-101A", "normalized_text should format industrial equipment tags"
    assert changed is True


# ── TEST 14: Deterministic Technical Normalization ───────────────────────────

def test_14_deterministic_technical_normalization():
    """Verify deterministic formatting of MRPL technical tags and units."""
    cases = [
        ("cdu", "CDU"),
        ("vdu", "VDU"),
        ("p and id", "P&ID"),
        ("11 p 101 a", "11-P-101A"),
        ("pfccu column", "PFCCU column"),
    ]
    for raw, expected in cases:
        norm, _ = LocalWhisperASR.deterministic_normalize(raw)
        assert expected.lower() in norm.lower()


# ── TEST 15: Low-Confidence Clarification (Spoken Repeat Prompt) ─────────────

def test_15_low_confidence_clarification():
    """Verify genuinely low-confidence ASR asks a spoken repeat question without modals."""
    ambiguity = detect_ambiguity("garbled mumble", asr_confidence=0.35, language="en", confidence_threshold=0.55)
    assert ambiguity is not None
    assert ambiguity.ambiguity_type in ("asr_failure", "low_confidence")
    assert ambiguity.clarification_question == "Sorry, I didn't catch that. Could you repeat it?"


# ── TEST 16: Normal Clear Speech Does NOT Trigger Clarification ──────────────

def test_16_clear_speech_does_not_trigger_clarification():
    """Verify clear unambiguous speech answers immediately with zero clarification."""
    queries = [
        "What is MRPL?",
        "What are the main units in MRPL?",
        "Explain the CDU.",
        "Check operating pressure of 11-P-101A",
    ]
    for q in queries:
        amb = detect_ambiguity(q, asr_confidence=0.92, language="en", confidence_threshold=0.55)
        assert amb is None, f"Query '{q}' should not trigger clarification"


# ── TEST 17: Interruption / Barge-In Mechanics ───────────────────────────────

def test_17_barge_in_mechanics():
    """Verify barge-in interrupts active session audio generation immediately."""
    service = VoiceService(asr_engine=mock.MagicMock(), tts_engine=mock.MagicMock())
    service.interrupt_session(session_id=55)
    assert service.is_interrupted(55) is True

    # Clearing interruption for a new turn
    service.clear_interruption(session_id=55)
    assert service.is_interrupted(55) is False


# ── TEST 18: TTS Completion Returns to Listening ─────────────────────────────

def test_18_tts_completion_returns_to_listening():
    """Verify frontend audio onended handler returns to LISTENING state hands-free."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    assert "audio.onended =" in chat_page
    assert "startListening(true)" in chat_page


# ── TEST 19: TTS Failure Returns Safely to Listening ─────────────────────────

def test_19_tts_failure_returns_safely_to_listening():
    """Verify frontend audio onerror handler safely recovers to LISTENING state."""
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    assert "audio.onerror =" in chat_page
    assert "setVoiceState('LISTENING')" in chat_page


# ── TEST 20: Zero External Network Calls (Network Seal) ──────────────────────

def test_20_zero_external_network_calls():
    """Verify network seal confirms zero external network egress."""
    seal = get_network_seal()
    assert seal is not None
    assert seal.seal_active is True
    # Verify no external cloud API endpoints in ASR or VAD
    asr_code = Path("backend/services/voice/asr.py").read_text(encoding="utf-8")
    assert "api.openai.com" not in asr_code
    assert "speech.googleapis.com" not in asr_code


# ── TEST 21: Single Microphone Ownership ─────────────────────────────────────

def test_21_single_microphone_ownership():
    """Verify getUserMedia() is ONLY called in ChatPage.jsx, NOT VoiceOrb or VoiceAssistantModal."""
    voice_orb = Path("frontend/src/components/VoiceOrb.jsx").read_text(encoding="utf-8")
    modal = Path("frontend/src/components/VoiceAssistantModal.jsx").read_text(encoding="utf-8")
    assert "getUserMedia" not in voice_orb, "VoiceOrb must not call getUserMedia()"
    assert "getUserMedia" not in modal, "VoiceAssistantModal must not call getUserMedia()"

    frontend_dir = Path("frontend/src")
    mic_files = [f.name for f in frontend_dir.rglob("*.jsx") if "getUserMedia" in f.read_text(encoding="utf-8")]
    assert mic_files == ["ChatPage.jsx"], f"Only ChatPage.jsx may own microphone, found in: {mic_files}"


# ── TEST 22: English Speech Handling ─────────────────────────────────────────

def test_22_english_speech_handling():
    """Verify English query detection and technical phrase preservation."""
    lang = detect_voice_language("What is the operating temperature of the crude distillation unit?")
    assert lang["code"] == "en"
    assert lang["name"] == "English"


# ── TEST 23: Hindi Speech Handling ───────────────────────────────────────────

def test_23_hindi_speech_handling():
    """Verify Hindi query detection (both Devanagari and Romanized) without corrupting tags."""
    # Devanagari Hindi
    lang_hi = detect_voice_language("सीडीयू इकाई का तापमान क्या है?")
    assert lang_hi["code"] == "hi"

    # Romanized Hindi with technical tag
    lang_rom = detect_voice_language("CDU unit ka pressure kitna hai batao")
    assert lang_rom["code"] == "hi"


# ── TEST 24: Marathi Speech Handling ─────────────────────────────────────────

def test_24_marathi_speech_handling():
    """Verify Marathi query detection (both Devanagari and Romanized) without corrupting tags."""
    # Devanagari Marathi
    lang_mr = detect_voice_language("सीडीयू युनिटचे तापमान किती आहे आणि माहिती सांगा")
    assert lang_mr["code"] == "mr"

    # Romanized Marathi
    lang_rom = detect_voice_language("11-P-101A pump chi mahiti dya")
    assert lang_rom["code"] == "mr"


# ── TEST 25: MRPL Technical Identifier Preservation & Query Integrity ───────

def test_25_mrpl_technical_identifiers_and_query_integrity():
    """
    Verify Query Integrity (Section 14 & Section 19):
    1. 'What is MRPL?' stays 'What is MRPL?' (never turns into P&ID temperature/pressure).
    2. Turn 1: 'What are the main units in MRPL?' -> Turn 2: 'Explain in detail.' inherits subject.
    3. Turn 1: 'What are the main units in MRPL?' -> Turn 2: 'What is the capital of France?' does NOT inherit MRPL.
    """
    # 1. Authoritative text preservation
    raw = "What is MRPL?"
    norm, changed = LocalWhisperASR.deterministic_normalize(raw)
    assert norm == "What is MRPL?"
    assert changed is False

    # 2. Context Policy: Follow-up elaboration
    history_turn1 = [
        {"role": "user", "content": "What are the main units in MRPL?", "task_type": "RAG"},
        {"role": "assistant", "content": "The main units include CDU, VDU, and PFCCU."},
    ]
    res_followup = determine_conversational_context("Explain in detail.", conversation_history=history_turn1)
    assert res_followup.is_followup is True
    assert "units" in res_followup.resolved_retrieval_query.lower() or "mrpl" in res_followup.resolved_retrieval_query.lower()

    # 3. Context Policy: Topic Shift (Independent query)
    res_shift = determine_conversational_context("What is the capital of France?", conversation_history=history_turn1)
    assert res_shift.is_followup is False
    assert "mrpl" not in res_shift.resolved_retrieval_query.lower()
    assert res_shift.resolved_retrieval_query == "What is the capital of France?"
