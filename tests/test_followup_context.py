"""
Regression test suite for conversational follow-up context resolution and RAG promotion.

Validates:
1. "explain in detail" after MRPL units turn resolves subject and promotes to RAG.
2. "tell me more" after specific refinery unit query resolves and promotes to RAG.
3. Follow-up after a greeting or non-knowledge turn does NOT promote to RAG (prose scanning rejected).
4. Follow-up after a coding turn does NOT promote to RAG.
5. Independent conceptual query ("explain what python is") stays GENERAL.
6. Standalone MRPL query remains RAG.
7. Structured context resolution respects the 4-tier preference hierarchy:
   Tier 1: Previous turn resolved subject/entity
   Tier 2: Previous turn resolved retrieval query
   Tier 3: Previous turn task/topic metadata
   Tier 4: Previous retrieved evidence metadata
8. Anaphora resolution on industrial tags ("What is its function?") continues working.
9. current_user_query / current_query is NEVER mutated; only retrieval_query is resolved.
"""

import pytest
from backend.services.task_router import route_task, classify_task_type
from backend.services.voice.context_policy import (
    determine_conversational_context,
    _resolve_structured_subject,
    extract_query_subject,
    extract_referenced_entities,
)
from backend.agent.state import AgentState
from backend.agent.graph import node_classify


def test_followup_explain_in_detail_after_mrpl_units():
    """Turn 1: MRPL units -> Turn 2: 'explain in detail' must promote to RAG."""
    history = [
        {"role": "user", "content": "what are the main units in mrpl"},
        {
            "role": "assistant",
            "content": "MRPL main units include Crude Distillation Unit (CDU), Hydrocracker Unit (HCU), PFCCU, Delayed Coker Unit, and Visbreaker.",
            "sources": [{"source": "mrpl_overview_process_units.pdf", "page": 1}],
            "task_type": "RAG",
            "retrieval_query": "what are the main units in mrpl",
        },
    ]

    decision = route_task("explain in detail", conversation_history=history)

    assert decision["task_type"] == "RAG"
    assert decision["requires_rag"] is True
    assert decision.get("target_entity") in ("MRPL", "main units in mrpl")
    retrieval_q = decision.get("retrieval_query", "").lower()
    assert "explain in detail" in retrieval_q
    assert "mrpl" in retrieval_q or "main units" in retrieval_q


def test_followup_tell_me_more_after_unit_query():
    """Turn 1: CDU query -> Turn 2: 'tell me more' must promote to RAG with CDU."""
    history = [
        {"role": "user", "content": "tell me about CDU"},
        {
            "role": "assistant",
            "content": "CDU is the Crude Distillation Unit designed to separate crude oil into fractions.",
            "task_type": "RAG",
            "retrieval_query": "tell me about CDU",
        },
    ]

    decision = route_task("tell me more", conversation_history=history)

    assert decision["task_type"] == "RAG"
    assert decision["requires_rag"] is True
    assert decision.get("target_entity") == "CDU"
    assert "cdu" in decision.get("retrieval_query", "").lower()


def test_followup_after_greeting_does_not_promote_to_rag():
    """
    Turn 1: Greeting -> Turn 2: 'explain in detail'
    Must NOT promote to RAG because no reliable structured knowledge subject exists.
    Assistant greeting prose ('MRPL operations') must not be treated as authoritative.
    """
    history = [
        {"role": "user", "content": "hello good morning"},
        {
            "role": "assistant",
            "content": "Good morning! Welcome to MRPL Sovereign AI Workbench. How can I assist you today with MRPL refinery operations?",
            "task_type": "GENERAL",
        },
    ]

    decision = route_task("explain in detail", conversation_history=history)

    assert decision["task_type"] == "GENERAL"
    assert decision["requires_rag"] is False


def test_followup_after_coding_does_not_promote_to_rag():
    """Turn 1: Coding task -> Turn 2: 'explain in detail' stays GENERAL (no RAG subject)."""
    history = [
        {"role": "user", "content": "write a python function to add two numbers"},
        {
            "role": "assistant",
            "content": "def add(a, b):\n    return a + b",
            "task_type": "CODING",
        },
    ]

    decision = route_task("explain in detail", conversation_history=history)

    assert decision["task_type"] == "GENERAL"
    assert decision["requires_rag"] is False


def test_conceptual_query_without_history_stays_general():
    """Independent conceptual query 'explain what python is' stays GENERAL."""
    decision = route_task("explain what python is", conversation_history=[])
    assert decision["task_type"] == "GENERAL"
    assert decision["requires_rag"] is False

    decision_no_hist = route_task("explain what python is")
    assert decision_no_hist["task_type"] == "GENERAL"
    assert decision_no_hist["requires_rag"] is False


def test_standalone_mrpl_query_is_rag():
    """Standalone MRPL organization query is RAG regardless of history."""
    decision = route_task("what are the main units in mrpl", conversation_history=[])
    assert decision["task_type"] == "RAG"
    assert decision["requires_rag"] is True


def test_preference_hierarchy_tier_1_explicit_entity():
    """Tier 1: Explicit target_entity metadata is preferred."""
    history = [
        {"role": "user", "content": "Tell me about the primary units"},
        {
            "role": "assistant",
            "content": "Here are the units...",
            "target_entity": "Hydrocracker",
            "task_type": "RAG",
        },
    ]

    subj = _resolve_structured_subject(history)
    assert subj is not None
    assert subj["entity"] == "Hydrocracker"
    assert subj["source"] == "explicit_entity_metadata"


def test_preference_hierarchy_tier_1_user_query_entity():
    """Tier 1: Entity extracted from previous user query."""
    history = [
        {"role": "user", "content": "What is the capacity of PFCCU?"},
        {
            "role": "assistant",
            "content": "PFCCU capacity is 2.2 MMTPA.",
        },
    ]

    subj = _resolve_structured_subject(history)
    assert subj is not None
    assert subj["entity"] == "PFCCU"
    assert subj["source"] == "prev_user_query_entity"


def test_preference_hierarchy_tier_2_retrieval_query():
    """Tier 2: Retrieval query metadata is used when no explicit entity is given."""
    history = [
        {"role": "user", "content": "Check process details"},
        {
            "role": "assistant",
            "content": "Found process details.",
            "retrieval_query": "operating conditions of CDU-101 column",
        },
    ]

    subj = _resolve_structured_subject(history)
    assert subj is not None
    assert subj["entity"] == "CDU-101"
    assert subj["source"] == "prev_retrieval_query"


def test_preference_hierarchy_tier_3_target_document():
    """Tier 3: Task target document is resolved."""
    history = [
        {"role": "user", "content": "Analyze report"},
        {
            "role": "assistant",
            "content": "Analysis complete.",
            "target_document": "TranscriptSd.pdf",
            "task_type": "DOCUMENT_ANALYSIS",
        },
    ]

    subj = _resolve_structured_subject(history)
    assert subj is not None
    assert subj["entity"] == "TranscriptSd.pdf"
    assert subj["source"] == "task_target_document"


def test_preference_hierarchy_tier_4_sources_metadata():
    """Tier 4: Evidence metadata from sources list is used."""
    history = [
        {"role": "user", "content": "Show me the report sections"},
        {
            "role": "assistant",
            "content": "Sections are listed.",
            "sources": [{"source": "MRPL Annual Report 2023-24.pdf", "page": 12}],
        },
    ]

    subj = _resolve_structured_subject(history)
    assert subj is not None
    assert "MRPL" in str(subj["entity"])
    assert subj["source"] == "retrieved_evidence_metadata"


def test_anaphora_equipment_tag_resolution_preserved():
    """Existing equipment tag anaphora resolution continues working deterministically."""
    history = [
        {"role": "user", "content": "What is 11-P-101A?"},
        {"role": "assistant", "content": "11-P-101A is the crude charge pump for CDU."},
    ]

    ctx = determine_conversational_context(
        current_query="What is its function?",
        conversation_history=history,
    )

    assert ctx.is_followup is True
    assert ctx.target_entity == "11-P-101A"
    assert "11-P-101A" in ctx.resolved_retrieval_query


def test_current_query_not_mutated_in_node_classify():
    """
    When 'explain in detail' is processed in node_classify, state.current_query
    must remain 'explain in detail' (literal query), while task_type is promoted to RAG.
    """
    history = [
        {"role": "user", "content": "what are the main units in mrpl"},
        {
            "role": "assistant",
            "content": "The main units in MRPL are CDU, HCU, PFCCU, Delayed Coker...",
            "sources": [{"source": "mrpl_units.pdf", "page": 1}],
            "task_type": "RAG",
        },
    ]

    state = AgentState(
        query="explain in detail",
        chat_history=history,
    )

    state = node_classify(state)

    # Authoritative current_query is NEVER mutated
    assert state.current_query == "explain in detail"
    assert state.task_type == "RAG"
    assert state.requires_rag is True
