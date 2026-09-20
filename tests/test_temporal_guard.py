"""
Deterministic Test Suite for Temporal / Recency Grounding in MRPL Sovereign AI Workbench.

Covers the 10 mandatory requirements:
1. "What is MRPL's financial status right now?" -> CURRENT intent
2. "What are MRPL's latest financial results?" -> RECENT intent
3. "What was MRPL's PAT in FY2016-17?" -> HISTORICAL / DATE_SPECIFIC intent
4. "Explain the 29th Annual Report." -> HISTORICAL intent
5. Current query + only FY2016-17 evidence -> INSUFFICIENT_RECENT_EVIDENCE
6. Historical FY2016-17 query + FY2016-17 evidence -> VALID
7. FY2016-17 query + FY2019-only evidence -> TEMPORAL_MISMATCH
8. Generated answer claims FY2019 while citation source is FY2016-17 -> GROUNDING FAILURE (flagged)
9. "What is MRPL?" -> Not classified as CURRENT (UNSPECIFIED)
10. Previous query asks about P&ID, Current query asks "What is MRPL?" -> Query isolation preserved, no contamination.
"""

import pytest

from backend.agent.state import AgentState
from backend.services.citation_validator import validate_citations
from backend.services.confidence_gate import evaluate_retrieval_confidence
from backend.services.temporal_guard import (
    CURRENT_DEPLOYMENT_YEAR,
    detect_temporal_intent,
    extract_temporal_metadata,
    generate_evidence_limited_response,
    validate_temporal_claims,
    validate_temporal_suitability,
)


# ── Test 1: "What is MRPL's financial status right now?" -> CURRENT ─────────

def test_01_current_intent_detection():
    """Verify that 'right now' queries are classified as CURRENT with recency requirement."""
    intent = detect_temporal_intent("What is MRPL's financial status right now?")
    assert intent.temporal_intent == "CURRENT"
    assert intent.requires_recent_evidence is True
    assert intent.requested_year == CURRENT_DEPLOYMENT_YEAR


# ── Test 2: "What are MRPL's latest financial results?" -> RECENT ───────────

def test_02_recent_intent_detection():
    """Verify that 'latest' queries are classified as RECENT with recency requirement."""
    intent = detect_temporal_intent("What are MRPL's latest financial results?")
    assert intent.temporal_intent == "RECENT"
    assert intent.requires_recent_evidence is True
    assert intent.relative_time == "latest"


# ── Test 3: "What was MRPL's PAT in FY2016-17?" -> HISTORICAL ───────────────

def test_03_historical_pat_query():
    """Verify that queries for past fiscal years are classified as HISTORICAL/DATE_SPECIFIC."""
    intent = detect_temporal_intent("What was MRPL's PAT in FY2016-17?")
    assert intent.temporal_intent in ("HISTORICAL", "DATE_SPECIFIC")
    assert intent.requested_period == "FY2016-17"
    assert intent.requested_year == 2017
    assert intent.requires_recent_evidence is False


# ── Test 4: "Explain the 29th Annual Report." -> HISTORICAL ─────────────────

def test_04_annual_report_explanation_historical():
    """Verify that requests to explain specific numbered reports map to HISTORICAL with mapped period."""
    intent = detect_temporal_intent("Explain the 29th Annual Report.")
    assert intent.temporal_intent == "HISTORICAL"
    assert intent.requested_period == "FY2016-17"
    assert intent.requires_recent_evidence is False


# ── Test 5: Current query + only FY2016-17 evidence -> INSUFFICIENT_RECENT ──

def test_05_current_query_with_stale_evidence():
    """Verify that CURRENT query with only FY2016-17 documents triggers INSUFFICIENT_RECENT_EVIDENCE."""
    intent = detect_temporal_intent("What is MRPL's financial status right now?")
    sources = [{
        "source": "29th Annual Report for 2016-17.pdf",
        "document_title": "29th Annual Report for 2016-17",
        "page": 42,
        "text": "During FY2016-17, the company recorded total turnover of Rs. 59,415 Crore."
    }]
    val = validate_temporal_suitability(intent, sources, current_year=2026)
    assert val.temporal_status == "INSUFFICIENT_RECENT_EVIDENCE"
    assert val.is_valid is False
    assert val.answer_allowed_as_current is False
    assert val.latest_reporting_period == "FY2016-17"
    assert "historical" in val.reason.lower()


# ── Test 6: Historical FY2016-17 query + FY2016-17 evidence -> VALID ─────────

def test_06_historical_query_with_matching_evidence():
    """Verify that historical query matching retrieved document period passes validation normally."""
    intent = detect_temporal_intent("What was MRPL's PAT in FY2016-17?")
    sources = [{
        "source": "29th Annual Report for 2016-17.pdf",
        "document_title": "29th Annual Report for 2016-17",
        "page": 42,
        "text": "For FY2016-17, MRPL recorded Profit After Tax of Rs. 3,644 Crore."
    }]
    val = validate_temporal_suitability(intent, sources, current_year=2026)
    assert val.temporal_status == "VALID"
    assert val.is_valid is True
    assert len(val.matched_sources) >= 1
    assert val.requested_period == "FY2016-17"


# ── Test 7: FY2016-17 query + FY2019-only evidence -> TEMPORAL_MISMATCH ──────

def test_07_temporal_mismatch_detection():
    """Verify that requesting FY2016-17 when only FY2018-19 is retrieved yields TEMPORAL_MISMATCH."""
    intent = detect_temporal_intent("What was MRPL's turnover in FY2016-17?")
    sources = [{
        "source": "31st Annual Report for 2018-19.pdf",
        "document_title": "31st Annual Report for 2018-19",
        "page": 10,
        "text": "In FY2018-19, turnover was higher."
    }]
    val = validate_temporal_suitability(intent, sources, current_year=2026)
    assert val.temporal_status == "TEMPORAL_MISMATCH"
    assert val.is_valid is False
    assert "not supported" in val.reason.lower()


# ── Test 8: Hallucinated FY2019 claimed against FY2016-17 source -> FLAGGED ─

def test_08_hallucinated_fiscal_year_flagged():
    """Verify that asserting 'as of FY2019' when citing a 2016-17 document is flagged as grounding failure."""
    answer = "MRPL's financial status as of FY2019 is strong with high turnover (Source: 29th Annual Report for 2016-17, p. 42)."
    sources = [{
        "source": "29th Annual Report for 2016-17.pdf",
        "document_title": "29th Annual Report for 2016-17",
        "page": 42,
        "text": "The company's performance for the financial year 2016-17 showed solid operational recovery."
    }]
    # 1. Test via temporal guard directly
    res = validate_temporal_claims(answer, sources)
    assert res.valid is False
    assert res.flagged is True
    assert any("FY2019" in claim for claim in res.unsupported_claims)

    # 2. Test via integrated citation validator
    cit_res = validate_citations(answer, sources)
    assert cit_res.is_grounded is False
    assert cit_res.flagged is True
    assert any("FY2019" in str(c) for c in cit_res.invalid_citations)


# ── Test 9: "What is MRPL?" -> UNSPECIFIED (not CURRENT) ────────────────────

def test_09_general_query_not_classified_as_current():
    """Verify negative boundary: general query without temporal markers is UNSPECIFIED."""
    intent = detect_temporal_intent("What is MRPL?")
    assert intent.temporal_intent == "UNSPECIFIED"
    assert intent.requires_recent_evidence is False

    intent2 = detect_temporal_intent("Tell me about the refinery units.")
    assert intent2.temporal_intent == "UNSPECIFIED"
    assert intent2.requires_recent_evidence is False


# ── Test 10: Query Isolation & No Temporal Contamination ────────────────────

def test_10_query_isolation_and_no_contamination():
    """Verify multi-turn query isolation: previous P&ID context does not pollute current temporal query."""
    state = AgentState(
        query="What is MRPL?",
        chat_history=[
            {"role": "user", "content": "Explain the P&ID for Crude Distillation Unit in FY2016-17."},
            {"role": "assistant", "content": "Here is the P&ID flow diagram breakdown from the 2016-17 report..."},
        ],
    )
    # Current query is strictly isolated
    assert state.current_query == "What is MRPL?"
    intent = detect_temporal_intent(state.current_query)
    assert intent.temporal_intent == "UNSPECIFIED"
    assert intent.requested_period is None
    assert "p&id" not in state.current_query.lower()


# ── Additional Verification: Confidence Gate & Safe Refusal ─────────────────

def test_confidence_gate_scales_down_on_stale_evidence():
    """Verify confidence gate reduces score to <= 0.20 and requires escalation on stale evidence."""
    intent = detect_temporal_intent("What is the financial status of MRPL right now?")
    sources = [{
        "source": "29th Annual Report for 2016-17.pdf",
        "document_title": "29th Annual Report for 2016-17",
        "page": 42,
        "distance": 0.35,  # Strong vector distance
        "text": "During FY2016-17, the company recorded total turnover of Rs. 59,415 Crore."
    }]
    val = validate_temporal_suitability(intent, sources, current_year=2026)
    conf = evaluate_retrieval_confidence("financial status of MRPL right now", sources, temporal_validation=val)

    assert conf.is_sufficient is False
    assert conf.escalation_required is True
    assert conf.status == "INSUFFICIENT_EVIDENCE"
    assert conf.confidence_score <= 0.20
    assert "temporal_status" in conf.metrics
    assert conf.metrics["temporal_status"] == "INSUFFICIENT_RECENT_EVIDENCE"


def test_evidence_limited_response_generation():
    """Verify deterministic evidence-limited refusal text matches requirements."""
    intent = detect_temporal_intent("What is the financial status of MRPL right now?")
    sources = [{
        "source": "29th Annual Report for 2016-17.pdf",
        "document_title": "29th Annual Report for 2016-17",
        "page": 45,
        "text": "During the year under review, MRPL recorded highest ever turnover of Rs. 59,415 Crore. Profit after tax stood at Rs. 3,644 Crore."
    }]
    val = validate_temporal_suitability(intent, sources, current_year=2026)
    resp = generate_evidence_limited_response("What is the financial status of MRPL right now?", val, sources)

    assert "reliably determine MRPL's current financial status" in resp
    assert "29th Annual Report for 2016-17" in resp
    assert "historical" in resp.lower()
    assert "FY2016-17" in resp
