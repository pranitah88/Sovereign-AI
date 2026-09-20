"""
MRPL Sovereign AI Workbench — Conversational Voice Agent Integration Test Suite.
Verifies natural conversational voice turn-taking:
TEST A: Standalone query 'What is MRPL?' -> Immediate spoken answer, no clarification.
TEST B: Ambiguous equipment tag 'What is 11-P-101?' -> Speaks: 'Do you mean 11-P-101A or 11-P-101B?', then listens.
TEST C: User answers clarification '11-P-101A' -> Continues answering without buttons or mic restart.
TEST D: Follow-up 'What does it do?' -> Resolves referent to 11-P-101A without mutating current query.
TEST E: During clarification, user says 'What is MRPL?' -> Cancels clarification, processes new topic cleanly.
TEST F: User interrupts assistant speaking ('Wait, I meant B') -> Immediately stops speaking, processes correction.
TEST G: Unclear speech -> Speaks: 'Sorry, I didn't catch that. Could you repeat it?', then listens.
TEST H: Pathological ASR repetition -> Rejects transcript, speaks retry prompt, skips RAG and LLM.
"""

import asyncio
import io
import json
import unittest.mock as mock
import pytest

from backend.agent.graph import run_agent
from backend.services.voice.asr import BaseASREngine, validate_asr_quality
from backend.services.voice.clarification import (
    ClarificationContext,
    detect_ambiguity,
    resolve_clarification_response,
)
from backend.services.voice.context_policy import determine_conversational_context
from backend.services.voice.voice_service import VoiceService


# ── Mocks & Test Fixtures ─────────────────────────────────────────────────────

class MockASREngine(BaseASREngine):
    def __init__(self, text="What is MRPL?", confidence=0.95, status="success"):
        self._text = text
        self._confidence = confidence
        self._status = status
        self.model_name = "mock-whisper"
        self.confidence_threshold = 0.55

    def set_output(self, text, confidence=0.95, status="success"):
        self._text = text
        self._confidence = confidence
        self._status = status

    def transcribe(self, audio_data, language=None):
        quality = validate_asr_quality(self._text, self._confidence)
        status = quality["status"] if not quality["valid"] else self._status
        return {
            "raw_text": self._text,
            "normalized_text": self._text,
            "is_normalized": False,
            "text": self._text,
            "language": language or "en",
            "confidence": self._confidence if quality["valid"] else 0.0,
            "status": status,
            "duration_seconds": 1.0,
            "engine": self.model_name,
            "quality": quality,
        }

    def is_ready(self):
        return True, "Mock ASR Ready"


class MockTTSEngine:
    def __init__(self):
        self.spoken_phrases = []

    def synthesize(self, text, language="en"):
        self.spoken_phrases.append(text)
        return {
            "audio_bytes": b"RIFF_MOCK_WAV_CHUNK",
            "duration_seconds": 1.2,
            "status": "success",
            "engine": "mock-tts",
        }

    def is_ready(self):
        return True, "Mock TTS Ready"


@pytest.fixture
def test_user():
    return {
        "id": 101,
        "username": "process_engineer",
        "roles": ["engineer"],
        "clearance": "CONFIDENTIAL",
    }


# ── TEST A: Standalone Query 'What is MRPL?' ─────────────────────────────────

def test_a_standalone_query_immediate_spoken_answer(test_user):
    """
    TEST A:
    User: 'What is MRPL?'
    Expected: Immediate spoken answer, no clarification requested.
    """
    async def _run():
        asr = MockASREngine("What is MRPL?", confidence=0.95)
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)

        events = []
        async for ev in service.stream_voice_query(b"dummy_audio", user=test_user):
            events.append(ev)
        return events, tts

    events, tts = asyncio.run(_run())

    # 1. Verify NO clarification was requested (immediate answer)
    assert not any(e.get("event") == "clarification_needed" for e in events)
    assert not any(e.get("event") == "waiting_for_clarification" for e in events)

    # 2. Verify authoritative query is 'What is MRPL?'
    req_evt = next((e for e in events if e.get("event") == "chat_request_query"), None)
    assert req_evt is not None
    assert req_evt["current_query"] == "What is MRPL?"

    # 3. Verify spoken response completed with MRPL topic
    complete_evt = next((e for e in events if e.get("event") == "complete"), None)
    assert complete_evt is not None
    response_text = complete_evt.get("response", "").lower()
    assert "safety bar" not in response_text
    assert "temperature" not in response_text or "mrpl" in response_text


# ── TEST B: Ambiguous Tag 'What is 11-P-101?' ─────────────────────────────────

def test_b_ambiguous_tag_speaks_clarification_and_listens(test_user):
    """
    TEST B:
    User: 'What is 11-P-101?'
    Expected:
    - Detects ambiguity (11-P-101A vs 11-P-101B exist).
    - Assistant SPEAKS: 'Do you mean 11-P-101A or 11-P-101B?'
    - Yields waiting_for_clarification so client automatically listens hands-free.
    - Does NOT call RAG or LLM.
    """
    async def _run():
        asr = MockASREngine("What is 11-P-101?", confidence=0.95)
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)

        events = []
        async for ev in service.stream_voice_query(b"dummy_audio", user=test_user):
            events.append(ev)
        return events, tts

    events, tts = asyncio.run(_run())

    # 1. Verify clarification event was generated
    clarif_evt = next((e for e in events if e.get("event") == "clarification_needed"), None)
    assert clarif_evt is not None
    assert clarif_evt["ambiguity_type"] == "ambiguous_tag"
    assert "11-P-101A" in clarif_evt["clarification_text"]
    assert "11-P-101B" in clarif_evt["clarification_text"]

    # 2. Verify clarification question was synthesized into audio
    tts_chunk = next((e for e in events if e.get("event") == "tts_chunk"), None)
    assert tts_chunk is not None
    assert "11-P-101A or 11-P-101B" in tts_chunk["spoken_text"]

    # 3. Verify waiting_for_clarification event triggers automatic listening on client
    wait_evt = next((e for e in events if e.get("event") == "waiting_for_clarification"), None)
    assert wait_evt is not None
    assert wait_evt["has_audio"] is True

    # 4. Verify RAG and LLM were NOT called
    assert not any(e.get("event") == "rag_query" for e in events)
    assert not any(e.get("event") == "token" for e in events)


# ── TEST C: Clarification Answer Resolved Naturally ──────────────────────────

def test_c_clarification_reply_resolves_and_continues(test_user):
    """
    TEST C:
    Turn 1: 'What is 11-P-101?' -> Assistant asks: 'Do you mean 11-P-101A or 11-P-101B?'
    Turn 2: User says '11-P-101A' (or 'the A one')
    Expected:
    - Resolves referent to 'What is 11-P-101A?'
    - Continues answering without buttons or manual mic restart.
    """
    clarification_context = {
        "original_query": "What is 11-P-101?",
        "ambiguity_type": "ambiguous_tag",
        "candidate_tags": ["11-P-101A", "11-P-101B"],
        "base_tag": "11-P-101",
        "language": "en",
    }

    # Test both exact tag and spoken colloquial 'the A one'
    for user_reply in ["11-P-101A", "The A one."]:
        async def _run(reply):
            asr = MockASREngine(reply, confidence=0.92)
            tts = MockTTSEngine()
            service = VoiceService(asr_engine=asr, tts_engine=tts)

            events = []
            async for ev in service.stream_voice_query(
                b"dummy_audio",
                user=test_user,
                clarification_context=clarification_context,
            ):
                events.append(ev)
            return events

        events = asyncio.run(_run(user_reply))

        # 1. Verify clarification was resolved
        resolved_evt = next((e for e in events if e.get("event") == "clarification_resolved"), None)
        assert resolved_evt is not None
        assert resolved_evt["resolved_query"] == "What is 11-P-101A?"

        # 2. Verify router and RAG received the resolved technical query
        router_evt = next((e for e in events if e.get("event") == "router_query"), None)
        assert router_evt is not None
        assert "11-P-101A" in router_evt["retrieval_query"]


# ── TEST D: Contextual Follow-up ('What does it do?') ────────────────────────

def test_d_contextual_followup_anaphora_resolution(test_user):
    """
    TEST D:
    Turn 1: User asks 'What is 11-P-101A?' -> answered.
    Turn 2: User asks 'What does it do?'
    Expected:
    - Context policy resolves 'it' -> 11-P-101A.
    - retrieval_query = 'What does 11-P-101A do?' (or 'What does 11-P-101A do of 11-P-101A')
    - current_user_query remains 'What does it do?' (immutable).
    """
    # Simulate conversation history with Turn 1
    mock_history = [
        {"role": "user", "content": "What is 11-P-101A?"},
        {"role": "assistant", "content": "11-P-101A is the crude charge pump for the CDU unit."},
    ]

    context_res = determine_conversational_context(
        current_query="What does it do?",
        conversation_history=mock_history,
    )

    assert context_res.is_followup is True
    assert context_res.target_entity == "11-P-101A"
    assert "11-P-101A" in context_res.resolved_retrieval_query

    # Test via VoiceService stream with mocked chat_repo
    async def _run():
        asr = MockASREngine("What does it do?", confidence=0.95)
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)

        with mock.patch("backend.database.repositories.chat.list_messages", return_value=mock_history):
            events = []
            async for ev in service.stream_voice_query(b"dummy_audio", user=test_user, session_id=42):
                events.append(ev)
            return events

    events = asyncio.run(_run())

    # Verify context_resolved event emitted
    ctx_evt = next((e for e in events if e.get("event") == "context_resolved"), None)
    assert ctx_evt is not None
    assert ctx_evt["current_query"] == "What does it do?"
    assert "11-P-101A" in ctx_evt["retrieval_query"]

    # Verify authoritative user query was NOT mutated
    chat_req_evt = next((e for e in events if e.get("event") == "chat_request_query"), None)
    assert chat_req_evt["current_query"] == "What does it do?"


# ── TEST E: Topic Override During Clarification ───────────────────────────────

def test_e_topic_override_cancels_clarification(test_user):
    """
    TEST E:
    Assistant asks: 'Do you mean 11-P-101A or 11-P-101B?'
    User says: 'What is MRPL?'
    Expected:
    - Clarification is cancelled.
    - 'What is MRPL?' is processed as a fresh, independent turn.
    - No P&ID or 11-P-101 context overrides MRPL company information.
    """
    clarification_context = {
        "original_query": "What is 11-P-101?",
        "ambiguity_type": "ambiguous_tag",
        "candidate_tags": ["11-P-101A", "11-P-101B"],
        "base_tag": "11-P-101",
        "language": "en",
    }

    resolved = resolve_clarification_response("What is MRPL?", clarification_context, "en")
    assert resolved == "What is MRPL?"

    async def _run():
        asr = MockASREngine("What is MRPL?", confidence=0.96)
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)

        events = []
        async for ev in service.stream_voice_query(
            b"dummy_audio",
            user=test_user,
            clarification_context=clarification_context,
        ):
            events.append(ev)
        return events

    events = asyncio.run(_run())

    # 1. Verify clarification was cancelled and authoritative query is 'What is MRPL?'
    assert not any(e.get("event") == "clarification_needed" for e in events)
    req_evt = next((e for e in events if e.get("event") == "chat_request_query"), None)
    assert req_evt is not None
    assert req_evt["current_query"] == "What is MRPL?"

    # 2. Verify router query is clean and unpolluted
    router_evt = next((e for e in events if e.get("event") == "router_query"), None)
    assert router_evt is not None
    assert "MRPL" in router_evt["current_query"]
    assert "11-P-101" not in router_evt["current_query"]


# ── TEST F: User Interruption (Barge-In) ──────────────────────────────────────

def test_f_user_barge_in_stops_speech_generation(test_user):
    """
    TEST F:
    Assistant is speaking.
    User interrupts ('Wait, I meant 11-P-101B').
    Expected:
    - Server-side generation stops immediately upon interrupt flag.
    - Yields 'interrupted' event.
    """
    async def _run():
        asr = MockASREngine("What is 11-P-101A?", confidence=0.95)
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)
        session_id = 999

        events = []
        async for ev in service.stream_voice_query(b"dummy_audio", user=test_user, session_id=session_id):
            events.append(ev)
            # Simulate barge-in as soon as token/audio generation starts
            if ev.get("event") in ("first_token", "first_audio", "token", "routing_decision"):
                service.interrupt_session(session_id)
        return events

    events = asyncio.run(_run())

    # Verify interrupted event received
    interrupted_evt = next((e for e in events if e.get("event") == "interrupted"), None)
    assert interrupted_evt is not None
    assert "interrupted" in interrupted_evt["reason"].lower()


# ── TEST G: Unclear Speech Speaks Retry Prompt ────────────────────────────────

def test_g_unclear_speech_speaks_retry_and_listens(test_user):
    """
    TEST G:
    User produces unclear/muffled speech (low confidence < 0.55).
    Expected:
    - Assistant speaks: 'Sorry, I didn't catch that. Could you repeat it?'
    - Yields waiting_for_clarification.
    - Does not call RAG or LLM.
    """
    async def _run():
        asr = MockASREngine("muffled sound", confidence=0.35, status="low_confidence")
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)

        events = []
        async for ev in service.stream_voice_query(b"dummy_audio", user=test_user):
            events.append(ev)
        return events, tts

    events, tts = asyncio.run(_run())

    # 1. Clarification needed for ASR failure
    clarif_evt = next((e for e in events if e.get("event") == "clarification_needed"), None)
    assert clarif_evt is not None
    assert clarif_evt["ambiguity_type"] == "asr_failure"
    assert "didn't catch that" in clarif_evt["clarification_text"].lower()

    # 2. Spoken retry audio chunk
    tts_chunk = next((e for e in events if e.get("event") == "tts_chunk"), None)
    assert tts_chunk is not None
    assert "repeat it" in tts_chunk["spoken_text"].lower()

    # 3. No RAG or LLM
    assert not any(e.get("event") == "rag_query" for e in events)
    assert not any(e.get("event") == "token" for e in events)


# ── TEST H: Pathological ASR Repetition Rejected ─────────────────────────────

def test_h_pathological_repetition_rejected_without_rag_or_llm(test_user):
    """
    TEST H:
    ASR produces pathological hallucination: 'I'm going to do a little bit of a little bit of a little bit...'
    Expected:
    - validate_asr_quality flags repetition_hallucination.
    - Transcript is rejected.
    - Assistant speaks retry prompt: 'Sorry, I didn't catch that. Could you say it again?'
    - Zero RAG, zero LLM calls.
    """
    pathological_text = "I'm going to do a little bit of a little bit of a little bit of work"
    quality = validate_asr_quality(pathological_text, confidence=0.90)

    assert quality["valid"] is False
    assert quality["status"] == "repetition_hallucination"
    assert "repetition" in quality["reason"].lower()

    async def _run():
        asr = MockASREngine(pathological_text, confidence=0.90)
        tts = MockTTSEngine()
        service = VoiceService(asr_engine=asr, tts_engine=tts)

        events = []
        async for ev in service.stream_voice_query(b"dummy_audio", user=test_user):
            events.append(ev)
        return events, tts

    events, tts = asyncio.run(_run())

    # 1. Verify repetition was rejected
    clarif_evt = next((e for e in events if e.get("event") == "clarification_needed"), None)
    assert clarif_evt is not None
    assert clarif_evt["ambiguity_type"] == "repetition_hallucination"

    # 2. Verify retry question was synthesized
    tts_chunk = next((e for e in events if e.get("event") == "tts_chunk"), None)
    assert tts_chunk is not None
    assert "say it again" in tts_chunk["spoken_text"].lower()

    # 3. Verify zero RAG or LLM calls
    assert not any(e.get("event") == "rag_query" for e in events)
    assert not any(e.get("event") == "token" for e in events)
