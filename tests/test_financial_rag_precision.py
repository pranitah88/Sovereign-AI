"""
Automated Test Suite for Financial & Quantitative RAG Precision.

Tests:
1. Query metric detection (revenue, exports, profit, fiscal year expansion)
2. Mismatch penalty calculation (penalizing export chunks for revenue queries)
3. Revenue retrieval precision (top result is revenue statement, not export %)
4. Export retrieval precision (top result is export % chunk)
5. Answer validation guard (rejecting hallucinated/substituted export percentages for revenue queries)
6. Debug scoring breakdown validation (all scoring components present and logged)
"""

import pytest
from backend.services.query_analyzer import QueryAnalysis, analyze_query
from backend.services.rag_engine import compute_metric_adjustments, query_knowledge_base
from backend.agent.graph import node_validate
from backend.agent.state import AgentState


# ── Test 1: Query Metric Detection ──────────────────────────────────────────

def test_query_metric_detection_revenue():
    analysis = analyze_query("What was MRPL's revenue in FY2025?")
    assert analysis.is_metric_query is True
    assert analysis.primary_metric == "revenue"
    assert "exports" in analysis.competing_metrics
    assert any("2024-25" in v or "March 31, 2025" in v for v in analysis.year_variants)
    assert analysis.requires_exact_value is True
    assert analysis.requires_percentage is False


def test_query_metric_detection_exports():
    analysis = analyze_query("What percentage of MRPL's turnover came from exports?")
    assert analysis.is_metric_query is True
    assert analysis.primary_metric == "exports"
    assert analysis.requires_percentage is True


def test_query_metric_detection_profit():
    analysis = analyze_query("What was MRPL's profit after tax (PAT) in 2023-24?")
    assert analysis.is_metric_query is True
    assert analysis.primary_metric == "profit"
    assert any("2023-24" in v for v in analysis.year_variants)


# ── Test 2: Mismatch Penalty ────────────────────────────────────────────────

def test_mismatch_penalty_applied_to_export_chunk_for_revenue_query():
    revenue_analysis = analyze_query("What was MRPL's revenue in FY2025?")
    export_chunk = {
        "text": "What is the contribution of exports as a percentage of the total turnover of the entity? Total exports of MRPL for FY2025-26 was 28.43% of the total turnover.",
        "metadata": {
            "source": "38th Annual Report 2025–26 Size 41.37 MB Language English Format PDF.pdf",
            "page": 110,
            "document_type": "annual_report",
        },
    }
    adj = compute_metric_adjustments(export_chunk, revenue_analysis)
    # The export chunk must receive a heavy mismatch penalty when query asks for revenue
    assert adj["mismatch_penalty"] >= 3.0
    # And year mismatch penalty because it specifies FY2025-26 while query asks for FY2025 (FY2024-25)
    assert adj["year_score"] < 0.0
    assert adj["total_adjustment"] < 0.0


def test_no_mismatch_penalty_for_export_query():
    export_analysis = analyze_query("What percentage of MRPL's turnover came from exports?")
    export_chunk = {
        "text": "What is the contribution of exports as a percentage of the total turnover of the entity? Total exports of MRPL for FY2025-26 was 28.43% of the total turnover.",
        "metadata": {
            "source": "38th Annual Report 2025–26 Size 41.37 MB Language English Format PDF.pdf",
            "page": 110,
            "document_type": "annual_report",
        },
    }
    adj = compute_metric_adjustments(export_chunk, export_analysis)
    assert adj["mismatch_penalty"] == 0.0
    assert adj["metric_score"] >= 1.5
    assert adj["total_adjustment"] > 0.0


# ── Test 3: Revenue Retrieval Precision ─────────────────────────────────────

def test_revenue_retrieval_does_not_return_export_chunk_as_top():
    res = query_knowledge_base("What was MRPL's revenue in FY2025?", top_k=3, debug=True)
    assert res["refusal"] is False
    assert res["result_count"] > 0
    top_source = res["sources"][0]
    
    # Top source must NOT be Page 110 (the export percentage chunk)
    assert not (top_source.get("page") == 110 and "28.43%" in top_source.get("text", ""))
    
    # Top source must contain Revenue / Turnover details
    top_text = top_source.get("text", "").lower()
    assert any(term in top_text for term in ("revenue", "turnover", "income"))
    assert top_source.get("score") > 0.0


# ── Test 4: Export Retrieval Precision ──────────────────────────────────────

def test_export_retrieval_returns_export_percentage():
    res = query_knowledge_base("What percentage of MRPL's turnover came from exports?", top_k=3, debug=True)
    assert res["refusal"] is False
    assert res["result_count"] > 0
    top_source = res["sources"][0]
    
    # Top source must be the export contribution chunk (Page 110)
    assert top_source.get("page") == 110
    assert "28.43%" in top_source.get("text", "")


# ── Test 5: Answer Validation Guard ─────────────────────────────────────────

def test_answer_validation_catches_export_substitution_for_revenue():
    state = AgentState(
        query="What was MRPL's revenue in FY2025?",
        user_id=1,
    )
    state.context = "Total exports of MRPL for FY2025-26 was 28.43% of the total turnover."
    state.sources = [{"source": "38th Annual Report", "page": 110}]
    # Simulated model hallucination asserting 28.43% is revenue
    state.response = "MRPL's revenue for FY2025 was 28.43% of the total turnover."
    
    validated_state = node_validate(state)
    # The validator must override or clarify rather than outputting 28.43% as revenue
    assert "28.43% of the total turnover" not in validated_state.response or "could not find the exact revenue figure" in validated_state.response


def test_answer_validation_passes_legitimate_revenue_response():
    state = AgentState(
        query="What was MRPL's revenue in FY2025?",
        user_id=1,
    )
    state.context = "The Company recorded Revenue from Operations of ₹1,05,155 crore during FY 2025-26 as against ₹1,09,280 crore during FY 2024-25."
    state.sources = [{"source": "38th Annual Report 2025-26", "page": 7}]
    state.response = "MRPL recorded Revenue from Operations of ₹1,09,280 crore in FY 2024-25 (Source: 38th Annual Report 2025-26, p. 7)."
    
    validated_state = node_validate(state)
    assert "₹1,09,280 crore" in validated_state.response
    assert "could not find the exact revenue figure" not in validated_state.response


# ── Test 6: Debug Scoring Breakdown ─────────────────────────────────────────

def test_debug_scoring_breakdown_structure():
    res = query_knowledge_base("What was MRPL's revenue in FY2025?", top_k=3, debug=True)
    assert "debug_info" in res
    debug = res["debug_info"]
    assert debug["is_metric_query"] is True
    assert debug["primary_metric"] == "revenue"
    assert "scoring_breakdown" in debug
    assert len(debug["scoring_breakdown"]) > 0
    first_item = debug["scoring_breakdown"][0]
    for required_key in (
        "rank", "source", "page", "base_score", "metric_score",
        "year_score", "doc_type_score", "numeric_score", "mismatch_penalty", "final_score"
    ):
        assert required_key in first_item
