"""
Regression tests for MRPL unit retrieval and deterministic confidence gate.

Verifies:
1. "what are the main units in mrpl" retrieves authoritative refinery documentation
   and achieves sufficient confidence without human escalation.
2. Major process units and secondary processing unit queries retrieve valid evidence.
3. Irrelevant queries remain strictly gated as INSUFFICIENT_EVIDENCE.
4. Unit-normalized Chroma L2 distance converts properly to cosine distance.
5. BM25-only candidates do not receive synthetic distances.
6. Refusal reasons accurately reflect numeric comparisons.
"""

import numpy as np
import pytest

from backend.services.rag_engine import query_knowledge_base
from backend.services.confidence_gate import (
    evaluate_retrieval_confidence,
    MAX_ACCEPTABLE_DISTANCE,
    STRONG_DISTANCE_THRESHOLD,
)
from backend.services.temporal_guard import (
    detect_temporal_intent,
    validate_temporal_suitability,
)


def test_main_units_in_mrpl_retrieval_and_confidence():
    """
    Test A: 'what are the main units in mrpl' retrieves authoritative
    manufacturing/refining sources and passes the confidence gate.
    """
    query = "what are the main units in mrpl"
    rag_res = query_knowledge_base(query)
    sources = rag_res.get("sources", [])

    assert len(sources) >= 1, "At least one authoritative MRPL source must be retrieved"
    top_source = sources[0]
    assert any(
        term in top_source["document_title"].lower()
        for term in ["refining", "manufacturing", "technology"]
    ), f"Expected refinery/manufacturing document, got: {top_source['document_title']}"

    # Verify distance fields
    if top_source.get("distance") is not None:
        assert top_source["distance"] <= MAX_ACCEPTABLE_DISTANCE
        assert top_source["distance"] <= STRONG_DISTANCE_THRESHOLD
        assert "cosine_distance" in top_source

    # Check that no candidate has synthetic 1.0 distance
    for s in sources:
        if s.get("distance") is not None:
            assert s["distance"] < 1.0, f"Candidate {s['document_title']} had unexpected distance {s['distance']}"

    # Evaluate confidence gate
    t_intent = detect_temporal_intent(query)
    t_val = validate_temporal_suitability(t_intent, sources)
    conf = evaluate_retrieval_confidence(query, sources, temporal_validation=t_val)

    assert conf.is_sufficient is True, f"Confidence gate failed: {conf.reason}"
    assert conf.escalation_required is False
    assert conf.status in ("HIGH_CONFIDENCE", "MODERATE_CONFIDENCE")
    assert conf.confidence_score >= 0.35


def test_major_process_units_query():
    """
    Test B: 'What are the major process units in MRPL's refinery?'
    retrieves refinery evidence and passes confidence gate.
    """
    query = "What are the major process units in MRPL's refinery?"
    rag_res = query_knowledge_base(query)
    sources = rag_res.get("sources", [])

    assert len(sources) >= 1
    t_intent = detect_temporal_intent(query)
    t_val = validate_temporal_suitability(t_intent, sources)
    conf = evaluate_retrieval_confidence(query, sources, temporal_validation=t_val)

    assert conf.is_sufficient is True
    assert conf.escalation_required is False


def test_secondary_processing_units_query():
    """
    Test C: 'What are the main secondary processing units in MRPL?'
    retrieves secondary processing evidence and passes confidence gate.
    """
    query = "What are the main secondary processing units in MRPL?"
    rag_res = query_knowledge_base(query)
    sources = rag_res.get("sources", [])

    assert len(sources) >= 1
    t_intent = detect_temporal_intent(query)
    t_val = validate_temporal_suitability(t_intent, sources)
    conf = evaluate_retrieval_confidence(query, sources, temporal_validation=t_val)

    assert conf.is_sufficient is True
    assert conf.escalation_required is False


def test_irrelevant_query_insufficient_evidence():
    """
    Test D: Irrelevant query correctly rejected with INSUFFICIENT_EVIDENCE
    and triggers escalation.
    """
    query = "What is the secret formula of the moon refinery?"
    rag_res = query_knowledge_base(query)
    sources = rag_res.get("sources", [])
    conf = evaluate_retrieval_confidence(query, sources)

    assert conf.is_sufficient is False
    assert conf.escalation_required is True
    assert conf.status == "INSUFFICIENT_EVIDENCE"


def test_chroma_l2_to_cosine_distance_semantics():
    """
    Test E: Unit-normalized embedding pair with Chroma squared L2 distance 0.7548
    mathematically converts to cosine distance 0.3774.
    """
    chroma_l2_dist = 0.7548
    expected_cosine = chroma_l2_dist / 2.0
    assert abs(expected_cosine - 0.3774) < 1e-4

    # Verify vector math with simulated normalized vectors
    rng = np.random.default_rng(42)
    u = rng.standard_normal(384)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(384)
    v /= np.linalg.norm(v)

    l2_sq = float(np.sum((u - v) ** 2))
    cos_dist = float(1.0 - np.dot(u, v))
    assert np.isclose(l2_sq / 2.0, cos_dist, atol=1e-6)


def test_bm25_only_candidates_have_none_distance():
    """
    Test F: A BM25-only candidate must have distance == None and must not
    cause artificial distance-threshold failure or inflate avg_distance.
    """
    bm25_only_sources = [
        {
            "source": "MRPL Engineering Notes.txt",
            "page": 1,
            "distance": None,
            "cosine_distance": None,
            "bm25_score": 14.5,
            "rerank_score": 5.2,
            "text": "Specific process unit configurations.",
        }
    ]

    conf = evaluate_retrieval_confidence("process configuration", bm25_only_sources)
    assert conf.metrics["distance_available"] is False
    assert conf.metrics["best_distance"] is None
    assert conf.metrics["avg_distance"] is None
    assert conf.is_sufficient is True
    assert conf.escalation_required is False


def test_misleading_refusal_reason_regression():
    """
    Test G: If distance = 0.755 and MAX_ACCEPTABLE_DISTANCE = 0.82,
    the refusal reason must NOT claim distance exceeds the threshold.
    """
    borderline_sources = [
        {
            "source": "Marginal Document.pdf",
            "page": 1,
            "distance": 0.755,
            "rerank_score": -4.0,  # Negative rerank score drives confidence below 0.35
            "text": "Tangentially mentioned text.",
        }
    ]

    conf = evaluate_retrieval_confidence("test query", borderline_sources)
    assert conf.is_sufficient is False
    # The refusal reason must accurately cite low confidence/relevance, not distance > 0.82
    assert "exceeds acceptable threshold" not in conf.reason
    assert "below the minimum sufficiency threshold" in conf.reason
