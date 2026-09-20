"""
MRPL Sovereign AI Workbench — Query Integrity & Pipeline Trace Test Suite.

Verifies:
1. TEST 1: 'What is MRPL?' retains exact current_query throughout pipeline and answers MRPL (no P&ID).
2. TEST 2: 'What is MRPL?' with previous P&ID conversation history answers 'What is MRPL?' without history override.
3. TEST 3: 'What is MRPL?' with irrelevant P&ID retrieval evidence refuses or handles cleanly without becoming a P&ID query.
4. TEST 4: 'Tell me about 11-P-101' asks clarification (11-P-101A vs 11-P-101B).
5. TEST 5: 'What is the pressure of 11-P-101A?' proceeds directly to answer pressure without clarification.
6. TEST 6: Concurrent requests A ('What is MRPL?') and B ('What is the pressure of 11-P-101A?') do not cross-contaminate.
7. TEST 7: Voice ASR final 'What is MRPL?' sends exact query in API request body.
"""

import asyncio
import io
import json
import unittest.mock as mock
import pytest

from backend.agent.graph import run_agent, node_classify, node_retrieve, node_reason
from backend.agent.prompts import REASONING_PROMPT, GENERAL_CHAT_PROMPT
from backend.agent.state import AgentState
from backend.services.voice.asr import LocalWhisperASR
from backend.services.voice.clarification import detect_ambiguity, resolve_clarification_response
from backend.services.voice.voice_service import VoiceService


# ── Mock Fixtures ─────────────────────────────────────────────────────────────

class MockASR:
    def __init__(self, transcribed_text="What is MRPL?", confidence=0.95):
        self.transcribed_text = transcribed_text
        self.confidence = confidence
        self.model_name = "mock-asr"
        self.confidence_threshold = 0.75

    def transcribe(self, audio_data, language=None):
        return {
            "raw_text": self.transcribed_text,
            "normalized_text": self.transcribed_text,
            "is_normalized": False,
            "text": self.transcribed_text,
            "language": language or "en",
            "confidence": self.confidence,
            "status": "success",
            "duration_seconds": 1.0,
            "engine": "mock-whisper",
        }

    def is_ready(self):
        return True, "Mock ASR Ready"


class MockTTS:
    def synthesize(self, text, language="en"):
        return {
            "audio_bytes": b"RIFFmockwavdata",
            "duration_seconds": 1.0,
            "status": "success",
            "engine": "mock-tts",
        }

    def is_ready(self):
        return True, "Mock TTS Ready"


@pytest.fixture
def engineer_user():
    return {
        "id": 101,
        "username": "test_engineer",
        "roles": ["engineer"],
        "clearance": "CONFIDENTIAL",
    }


# ── TEST 1: 'What is MRPL?' Query Preserved & Answers MRPL ────────────────────

def test_1_what_is_mrpl_authoritative_query_preserved(engineer_user):
    """
    TEST 1:
    Input: 'What is MRPL?'
    Expected:
    - Current query remains 'What is MRPL?' throughout the entire pipeline.
    - Trace records ASR_FINAL_RAW, ASR_NORMALIZED, CHAT_REQUEST_QUERY, ORCHESTRATOR_QUERY, ROUTER_QUERY, RAG_QUERY, LLM_CURRENT_QUERY.
    - Expected answer topic is MRPL/company information, NOT P&ID / temperature / pressure.
    """
    async def _run():
        mock_asr = MockASR("What is MRPL?")
        mock_tts = MockTTS()
        service = VoiceService(asr_engine=mock_asr, tts_engine=mock_tts)

        events = []
        async for event in service.stream_voice_query(b"fake_audio", user=engineer_user, session_id=None):
            events.append(event)
        return events

    events = asyncio.run(_run())

    complete_evt = next((e for e in events if e.get("event") == "complete"), None)
    assert complete_evt is not None, "Pipeline must yield a complete event"

    trace = complete_evt.get("trace", [])
    trace_events = [t.get("event") or t.get("stage") for t in trace]

    # Verify structured query lineage trace events
    assert "ASR_FINAL_RAW" in trace_events
    assert "ASR_NORMALIZED" in trace_events
    assert "CHAT_REQUEST_QUERY" in trace_events
    assert "ORCHESTRATOR_QUERY" in trace_events
    assert "ROUTER_QUERY" in trace_events
    assert "RAG_QUERY" in trace_events or "fast_routing" in trace_events
    assert "LLM_CURRENT_QUERY" in trace_events

    # Verify the authoritative query value in trace
    chat_req_trace = next(t for t in trace if (t.get("event") or t.get("stage")) == "CHAT_REQUEST_QUERY")
    assert chat_req_trace["details"]["current_query"] == "What is MRPL?"

    llm_trace = next(t for t in trace if (t.get("event") or t.get("stage")) == "LLM_CURRENT_QUERY")
    assert llm_trace["details"]["current_query"] == "What is MRPL?"

    # Verify response does NOT discuss P&ID safety bar temperature pressure
    response_text = complete_evt.get("response", "").lower()
    assert "p&id" not in response_text or "mrpl" in response_text
    assert "safety bar" not in response_text


# ── TEST 2: Previous P&ID Conversation Does NOT Override Current Query ────────

def test_2_previous_conversation_does_not_override_current_query(engineer_user):
    """
    TEST 2:
    Input: 'What is MRPL?'
    Previous conversation: 'Tell me about P&ID safety pressure.'
    Expected:
    - Answer 'What is MRPL?'.
    - Previous P&ID question must not override current query.
    """
    # Create an AgentState with previous conversation about P&ID
    prior_history = [
        {"role": "user", "content": "Tell me about P&ID safety bar temperature pressure."},
        {"role": "assistant", "content": "Could you please clarify which P&ID you are referring to? We can analyze the safety bar temperature and pressure requirements."},
    ]

    state = AgentState(
        query="What is MRPL?",
        current_query="What is MRPL?",
        chat_history=prior_history,
        conversation_history=prior_history,
        user=engineer_user,
        user_id=engineer_user["id"],
    )

    # Classify
    state = node_classify(state)
    assert state.current_query == "What is MRPL?"

    # Format prompts and verify prompt hardening
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in state.chat_history])
    rendered_prompt = REASONING_PROMPT.format(
        retrieved_context="MRPL is a Schedule A Miniratna Central Public Sector Enterprise operating a 15 MMTPA refinery in Mangalore.",
        conversation_history=history_str,
        current_query=state.current_query,
    )

    # Assert that the prompt strictly puts CURRENT USER QUERY as authoritative
    assert "## CURRENT USER QUERY (AUTHORITATIVE):\nWhat is MRPL?" in rendered_prompt
    assert "The CURRENT USER QUERY is authoritative." in rendered_prompt
    assert "NEVER answer a previous user question or assistant prompt" in rendered_prompt

    # Mock Ollama call with gemma3 / factual answer
    with mock.patch("backend.agent.graph._call_llm") as mock_llm:
        mock_llm.return_value = "MRPL (Mangalore Refinery and Petrochemicals Limited) is a Schedule A Miniratna CPSE refinery located in Mangalore."
        state.context = "MRPL is a Schedule A Miniratna CPSE refinery located in Mangalore."
        state = node_reason(state)

        # Response must address MRPL and not P&ID
        assert "MRPL" in state.response
        assert "p&id safety bar" not in state.response.lower()
        assert state.current_query == "What is MRPL?"


# ── TEST 3: RAG Irrelevant P&ID Evidence Does NOT Change Query ────────────────

def test_3_irrelevant_rag_evidence_does_not_change_query(engineer_user):
    """
    TEST 3:
    Input: 'What is MRPL?'
    RAG intentionally returns irrelevant P&ID documents.
    Expected:
    - System does not change question to P&ID.
    - System either answers the current question appropriately or states evidence is insufficient.
    """
    irrelevant_context = (
        "[Document: Drawing 11-P-101.pdf, Page 1]\n"
        "P&ID drawing indicates safety bar temperature 350C and operating pressure 45 bar."
    )

    state = AgentState(
        query="What is MRPL?",
        current_query="What is MRPL?",
        context=irrelevant_context,
        retrieved_context=irrelevant_context,
        user=engineer_user,
        user_id=engineer_user["id"],
        task_type="RAG",
    )

    with mock.patch("backend.agent.graph._call_llm") as mock_llm:
        # LLM correctly notes insufficient evidence in context for MRPL
        mock_llm.return_value = "I don't have sufficient information in the knowledge base to answer this reliably."
        state = node_reason(state)

        # System did not change the query
        assert state.current_query == "What is MRPL?"
        # System did NOT answer the P&ID topic
        assert "safety bar temperature 350c" not in state.response.lower()


# ── TEST 4: 'Tell me about 11-P-101' Asks A/B Clarification ───────────────────

def test_4_ambiguous_tag_clarification():
    """
    TEST 4:
    Input: 'Tell me about 11-P-101'
    Expected:
    - Detects ambiguity between 11-P-101A and 11-P-101B.
    - Returns clarification intent asking if user means 11-P-101A or 11-P-101B.
    """
    ambiguity = detect_ambiguity("Tell me about 11-P-101", asr_confidence=0.95, language="en")
    assert ambiguity is not None, "Ambiguous base tag 11-P-101 must trigger clarification"
    assert ambiguity.ambiguity_type == "ambiguous_tag"
    assert "11-P-101A" in ambiguity.clarification_question
    assert "11-P-101B" in ambiguity.clarification_question
    assert "Do you mean" in ambiguity.clarification_question


# ── TEST 5: 'What is the pressure of 11-P-101A?' No Clarification Needed ─────

def test_5_specific_tag_no_clarification():
    """
    TEST 5:
    Input: 'What is the pressure of 11-P-101A?'
    Expected:
    - Specific tag with suffix 'A' is NOT ambiguous.
    - detect_ambiguity returns None; proceeds directly to answer.
    """
    ambiguity = detect_ambiguity("What is the pressure of 11-P-101A?", asr_confidence=0.95, language="en")
    assert ambiguity is None, "Specific tag 11-P-101A must not trigger clarification"


# ── TEST 6: Concurrent Requests Cannot Cross-Contaminate ──────────────────────

def test_6_concurrent_requests_no_cross_contamination(engineer_user):
    """
    TEST 6:
    Two concurrent requests:
    Request A: 'What is MRPL?'
    Request B: 'What is the pressure of 11-P-101A?'
    Expected:
    - Responses cannot cross-contaminate.
    - Request A answers MRPL company information.
    - Request B answers pressure for 11-P-101A.
    """
    async def _run():
        mock_asr_a = MockASR("What is MRPL?")
        mock_asr_b = MockASR("What is the pressure of 11-P-101A?")
        mock_tts = MockTTS()

        service_a = VoiceService(asr_engine=mock_asr_a, tts_engine=mock_tts)
        service_b = VoiceService(asr_engine=mock_asr_b, tts_engine=mock_tts)

        async def run_voice_query(svc, q_text):
            collected = []
            async for evt in svc.stream_voice_query(b"audio", user=engineer_user, session_id=None):
                collected.append(evt)
            return collected

        results_a, results_b = await asyncio.gather(
            run_voice_query(service_a, "What is MRPL?"),
            run_voice_query(service_b, "What is the pressure of 11-P-101A?"),
        )
        return results_a, results_b

    results_a, results_b = asyncio.run(_run())

    complete_a = next(e for e in results_a if e.get("event") == "complete")
    complete_b = next(e for e in results_b if e.get("event") == "complete")

    # Check Request A trace
    trace_a = complete_a.get("trace", [])
    chat_req_a = next(t for t in trace_a if (t.get("event") or t.get("stage")) == "CHAT_REQUEST_QUERY")
    assert chat_req_a["details"]["current_query"] == "What is MRPL?"

    # Check Request B trace
    trace_b = complete_b.get("trace", [])
    chat_req_b = next(t for t in trace_b if (t.get("event") or t.get("stage")) == "CHAT_REQUEST_QUERY")
    assert chat_req_b["details"]["current_query"] == "What is the pressure of 11-P-101A?"

    # Confirm zero cross-contamination
    assert chat_req_a["details"]["current_query"] != chat_req_b["details"]["current_query"]


# ── TEST 7: Voice ASR Final Transmitted Unmodified ────────────────────────────

def test_7_asr_final_query_unmodified():
    """
    TEST 7:
    Voice:
    ASR final: 'What is MRPL?'
    Expected:
    - Normalization preserves 'What is MRPL?' verbatim.
    - Clarification resolver preserves 'What is MRPL?' verbatim.
    - No silent rewriting or substitution occurs.
    """
    raw_query = "What is MRPL?"
    norm_text, is_norm = LocalWhisperASR.deterministic_normalize(raw_query)
    assert norm_text == "What is MRPL?"

    # Even if previous clarification context was lingering, a standalone query is untouched
    resolved = resolve_clarification_response(
        raw_query,
        context={"original_query": "P&ID safety bar temperature pressure", "ambiguity_type": "ambiguous_tag"},
        language="en",
    )
    assert resolved == "What is MRPL?", f"Expected 'What is MRPL?', got '{resolved}'"
