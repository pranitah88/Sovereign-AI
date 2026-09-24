"""
Test suite verifying incomplete / ambiguous query handling regression fix:
"Give me information about MRP and" -> must ask concise conversational clarification:
"Could you complete your question? Also, do you mean MRP or MRPL?"
Strictly preserving "MRP" verbatim without fuzzy rewriting to "MRPL".
"""

import pytest
import asyncio
from backend.services.voice.clarification import (
    check_query_completeness,
    check_ambiguous_entity,
    detect_ambiguity,
    resolve_clarification_response,
)
from backend.services.task_router import classify_task_type
from backend.agent.graph import node_classify, node_reason
from backend.agent.state import AgentState


def test_incomplete_query_completeness_detection():
    """Verify check_query_completeness identifies trailing conjunctions, prepositions, articles, etc."""
    # Trailing conjunction 'and'
    is_inc, reason = check_query_completeness("Give me information about MRP and")
    assert is_inc is True
    assert "trailing_conjunction" in reason

    # Trailing preposition
    is_inc, reason = check_query_completeness("Tell me about the unit with")
    assert is_inc is True
    assert "trailing_preposition" in reason

    # Trailing article
    is_inc, reason = check_query_completeness("What is the")
    assert is_inc is True
    assert "trailing_article" in reason

    # Complete factual query
    is_inc, reason = check_query_completeness("What is MRP?")
    assert is_inc is False

    is_inc, reason = check_query_completeness("What is MRPL?")
    assert is_inc is False


def test_ambiguous_entity_mrp_detection():
    """Verify check_ambiguous_entity identifies MRP vs MRPL ambiguity."""
    has_ambig, entity_type, candidates = check_ambiguous_entity("Give me information about MRP and")
    assert has_ambig is True
    assert entity_type in ("ambiguous_acronym_mrp", "mrp_vs_mrpl")
    assert "MRP" in candidates
    assert "MRPL" in candidates

    # Complete MRPL query has no ambiguity
    has_ambig, entity_type, candidates = check_ambiguous_entity("What is MRPL?")
    assert has_ambig is False


def test_detect_ambiguity_returns_expected_clarification():
    """Verify detect_ambiguity produces the exact expected clarification prompt."""
    query = "Give me information about MRP and"
    ambig = detect_ambiguity(query)
    assert ambig is not None
    assert ambig.ambiguity_type == "incomplete_ambiguous"
    assert "Could you complete your question? Also, do you mean MRP or MRPL?" in ambig.clarification_question
    assert "MRP" in ambig.candidate_tags
    assert "MRPL" in ambig.candidate_tags


def test_task_router_routes_incomplete_query_to_clarification():
    """Verify task_router routes incomplete query to CLARIFICATION before LLM fallback."""
    decision = classify_task_type("Give me information about MRP and")
    assert decision["task_type"] in ("CLARIFICATION", "INCOMPLETE_QUERY")
    assert decision["requires_rag"] is False


def test_agent_graph_fast_path_clarification_without_generic_greeting():
    """Verify AgentState handles clarification fast-path without invoking LLM or greeting."""
    state = AgentState(
        query="Give me information about MRP and",
        current_query="Give me information about MRP and",
        user_id=1,
        user={"id": 1, "username": "testuser", "role": "engineer"},
        session_id=1,
    )
    
    # Run node_classify
    classified_state = node_classify(state)
    assert classified_state.task_type in ("CLARIFICATION", "INCOMPLETE_QUERY")
    assert classified_state.clarification_question != ""
    assert "do you mean MRP or MRPL" in classified_state.clarification_question

    # Run node_reason (should return clarification question directly)
    reasoned_state = node_reason(classified_state)
    response = reasoned_state.response

    # Must NOT contain the generic greeting
    assert "I am the sovereign on-premise AI workbench for MRPL" not in response
    assert "How may I assist you with refinery operations" not in response
    # Must contain clarification
    assert "Could you complete your question? Also, do you mean MRP or MRPL?" in response
