"""
Comprehensive tests for HYBRID Data + Code Execution Workflow.

Verifies:
1. Task Router Classification:
   - Failing query from prompt -> HYBRID
   - "Calculate MRPL's revenue growth using the annual reports in the knowledge base." -> HYBRID
   - "Write Python code to calculate financial status." -> CODING (requires_rag=False)
   - "What was MRPL's revenue in FY2025?" -> RAG (requires_sandbox=False, model=gemma3:4b)
   - "What is Python?" -> GENERAL
2. RAG Query Normalization & Cleansing (stripping sandbox meta-instructions).
3. Structured Metric Extraction (exact values, fiscal years, units, currencies, evidence).
4. Financial Metric Safety (Revenue != Exports != PAT != GRM).
5. Required Input Validation (sufficient data vs controlled refusal).
6. Qwen 2.5 Coder Structured Handoff Prompt.
7. Docker Sandbox Execution & Bounded Self-Correction.
8. Fail-Closed Security (Docker unavailable -> never runs on host).
9. Final Grounded Response Format (Task Type, Model, RAG: VERIFIED, Execution: VERIFIED, Citations).
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.agent.graph import (
    MAX_CORRECTION_ATTEMPTS,
    node_retrieve,
    node_tool_call,
    node_reason,
    node_validate,
)
from backend.agent.state import AgentState
from backend.services.hybrid_executor import (
    extract_hybrid_retrieval_query,
    extract_structured_metrics,
    validate_required_inputs,
    build_qwen_hybrid_prompt,
    format_hybrid_final_response,
    make_safe_variable_name,
    sanitize_fiscal_year_labels,
)
from backend.services.task_router import route_task


# ── 1. Task Router Classification Tests ─────────────────────────────────

def test_router_hybrid_failing_query_from_prompt():
    """The exact query that previously produced false refusal must route to HYBRID."""
    q = (
        "Using the financial data available in the knowledge base, calculate the relevant "
        "financial metrics, show the Python calculation, execute and verify the code in the "
        "sandbox, and provide the final results with source citations."
    )
    decision = route_task(q)
    assert decision["task_type"] == "HYBRID"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["model_id"] == "qwen25_coder_3b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is True
    assert "rag_search" in decision["tools"]
    assert "sandbox_execute" in decision["tools"]


def test_router_hybrid_revenue_growth_report():
    """'Calculate MRPL's revenue growth using the annual reports in the knowledge base.' -> HYBRID."""
    q = "Calculate MRPL's revenue growth using the annual reports in the knowledge base."
    decision = route_task(q)
    assert decision["task_type"] == "HYBRID"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is True
    assert "rag_search" in decision["tools"]
    assert "sandbox_execute" in decision["tools"]


def test_router_pure_coding_financial_status():
    """'Write Python code to calculate financial status.' -> CODING."""
    q = "Write Python code to calculate financial status."
    decision = route_task(q)
    assert decision["task_type"] == "CODING"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["requires_rag"] is False
    assert decision["requires_sandbox"] is True
    assert "sandbox_execute" in decision["tools"]
    assert "rag_search" not in decision["tools"]


def test_router_pure_factual_lookup_revenue():
    """'What was MRPL's revenue in FY2025?' -> RAG -> Gemma."""
    q = "What was MRPL's revenue in FY2025?"
    decision = route_task(q)
    assert decision["task_type"] == "RAG"
    assert decision["model"] == "gemma3:4b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is False
    assert "rag_search" in decision["tools"]
    assert "sandbox_execute" not in decision["tools"]


def test_router_general_question():
    """'What is Python?' -> GENERAL -> Gemma."""
    q = "What is Python?"
    decision = route_task(q)
    assert decision["task_type"] == "GENERAL"
    assert decision["model"] == "gemma3:4b"
    assert decision["requires_rag"] is False
    assert decision["requires_sandbox"] is False


def test_router_vision_request():
    """'Analyze this image of an inspection drawing.' -> VISION / DOCUMENT_ANALYSIS."""
    q = "Analyze this image of an inspection drawing."
    decision = route_task(q)
    assert decision["task_type"] in ("VISION", "DOCUMENT_ANALYSIS")
    assert decision["requires_rag"] is False
    assert decision["requires_sandbox"] is False


# ── 2. Query Cleansing & Retrieval Normalization ────────────────────────

def test_extract_hybrid_retrieval_query():
    """Meta-instructions such as sandbox and code execution must be removed for RAG."""
    raw_query = (
        "Using the financial data available in the knowledge base, calculate the relevant "
        "financial metrics, show the Python calculation, execute and verify the code in the "
        "sandbox, and provide the final results with source citations."
    )
    clean = extract_hybrid_retrieval_query(raw_query)
    assert "sandbox" not in clean.lower()
    assert "python" not in clean.lower()
    assert "execute" not in clean.lower()
    assert "MRPL" in clean
    assert "revenue" in clean.lower() or "financial" in clean.lower()


# ── 3. Structured Metric Extraction & Financial Safety ──────────────────

SAMPLE_CONTEXT = """[Document: 38Th Annual Report (2025–26), Page 7, Class: INTERNAL]
<untrusted_document_context source="38Th Annual Report (2025–26)" page="7" classification="INTERNAL">
ANNUAL REPORT 2025-26 7 Standalone Financial Performance g The Company recorded Revenue from Operations of ₹1,05,155 crore during FY 2025-26 as against ₹1,09,280 crore during FY 2024-25. g The Company’s Export stood at ₹29,880 crore during FY 2025-26 as against ₹36,466 crore during FY 2024-25. g Profit Before Tax stood at ₹4,022 crore during FY 2025-26 as against ₹113 crore during FY 2024-25. g Profit After Tax stood at ₹1,931 crore during FY 2025-26 as against ₹51 crore during FY 2024-25. g Gross Refining Margin (GRM) improved significantly to US$ 9.22 per barrel during FY 2025-26 as compared to US$ 4.45 per barrel during FY 2024-25, supported by stronger product cracks and a favourable crude basket.
</untrusted_document_context>"""

SAMPLE_SOURCES = [
    {"source": "38Th Annual Report (2025–26)", "page": 7, "document_title": "38th Annual Report 2025–26"}
]


def test_extract_structured_metrics():
    """Verify deterministic extraction of metrics with exact values, fiscal years, and units."""
    metrics = extract_structured_metrics(SAMPLE_CONTEXT, SAMPLE_SOURCES)
    assert len(metrics) >= 8

    # Revenue
    rev_26 = next((m for m in metrics if m["metric"] == "Revenue from Operations" and "2025-26" in m["fiscal_year"]), None)
    rev_25 = next((m for m in metrics if m["metric"] == "Revenue from Operations" and "2024-25" in m["fiscal_year"]), None)
    assert rev_26 is not None
    assert rev_26["value"] == 105155
    assert rev_26["unit"] == "₹ crore"
    assert rev_25 is not None
    assert rev_25["value"] == 109280
    assert rev_25["unit"] == "₹ crore"

    # PAT
    pat_26 = next((m for m in metrics if m["metric"] == "Profit After Tax (PAT)" and "2025-26" in m["fiscal_year"]), None)
    pat_25 = next((m for m in metrics if m["metric"] == "Profit After Tax (PAT)" and "2024-25" in m["fiscal_year"]), None)
    assert pat_26 is not None
    assert pat_26["value"] == 1931
    assert pat_25 is not None
    assert pat_25["value"] == 51

    # GRM
    grm_26 = next((m for m in metrics if m["metric"] == "Gross Refining Margin (GRM)" and "2025-26" in m["fiscal_year"]), None)
    grm_25 = next((m for m in metrics if m["metric"] == "Gross Refining Margin (GRM)" and "2024-25" in m["fiscal_year"]), None)
    assert grm_26 is not None
    assert grm_26["value"] == 9.22
    assert grm_25 is not None
    assert grm_25["value"] == 4.45


def test_financial_metric_safety_no_conflation():
    """Verify that Revenue is strictly isolated from Exports, PAT, and GRM."""
    metrics = extract_structured_metrics(SAMPLE_CONTEXT, SAMPLE_SOURCES, query="calculate revenue growth")
    # Should only return Revenue metrics
    assert all(m["metric"] == "Revenue from Operations" for m in metrics)
    assert not any(m["metric"] == "Exports" for m in metrics)


# ── 4. Required Input Validation Tests ──────────────────────────────────

def test_validate_required_inputs_sufficient():
    """Multi-period metrics present must validate successfully."""
    metrics = extract_structured_metrics(SAMPLE_CONTEXT, SAMPLE_SOURCES)
    valid, msg = validate_required_inputs(metrics, "Calculate revenue growth")
    assert valid is True
    assert msg == ""


def test_validate_required_inputs_insufficient_returns_controlled_refusal():
    """If multi-period data is missing for growth calculation, return controlled refusal."""
    single_metric = [
        {
            "metric": "Revenue from Operations",
            "value": 105155,
            "fiscal_year": "FY 2025-26",
            "unit": "₹ crore",
        }
    ]
    valid, msg = validate_required_inputs(single_metric, "Calculate revenue growth")
    assert valid is False
    assert "The knowledge base does not contain sufficient verified data to calculate this metric." in msg


# ── 5. Qwen 2.5 Coder Structured Handoff Prompt ──────────────────────────

def test_build_qwen_hybrid_prompt():
    """Prompt must include VERIFIED INPUT DATA and strict deterministic instructions."""
    metrics = extract_structured_metrics(SAMPLE_CONTEXT, SAMPLE_SOURCES)
    prompt = build_qwen_hybrid_prompt("calculate relevant metrics", metrics, SAMPLE_CONTEXT)

    assert "VERIFIED INPUT DATA:" in prompt
    assert "Revenue from Operations" in prompt
    assert "105155" in prompt
    assert "109280" in prompt
    assert "QWEN INSTRUCTIONS:" in prompt
    assert "Use ONLY the supplied verified values" in prompt
    assert "Do not invent values" in prompt
    assert "Do not access the filesystem" in prompt
    assert "Docker will execute the code separately" in prompt


# ── 6. End-to-End HYBRID Agent Pipeline ─────────────────────────────────

def test_hybrid_node_tool_call_successful_execution():
    """Verify HYBRID flow: RAG data -> Qwen Coder prompt -> Docker execution -> state."""
    state = AgentState(
        query="Using the financial data available in the knowledge base, calculate the relevant financial metrics, show the Python calculation, execute and verify the code in the sandbox, and provide the final results with source citations.",
        task_type="HYBRID",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_rag=True,
        requires_sandbox=True,
        required_tools=["rag_search", "sandbox_execute"],
        context=SAMPLE_CONTEXT,
        sources=SAMPLE_SOURCES,
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    mock_qwen_code = (
        "```python\n"
        "rev_2025_26 = 105155\n"
        "rev_2024_25 = 109280\n"
        "growth = ((rev_2025_26 - rev_2024_25) / rev_2024_25) * 100\n"
        "print(f'Revenue Growth: {growth:.2f}%')\n"
        "```"
    )

    mock_sandbox_success = {
        "tool": "sandbox_execute",
        "status": "success",
        "result": {
            "status": "success",
            "stdout": "Revenue Growth: -3.78%",
            "stderr": "",
            "exit_code": 0,
            "execution_time_ms": 50,
        },
    }

    with patch("backend.agent.graph._call_llm", return_value=mock_qwen_code) as mock_llm:
        with patch("backend.agent.graph.execute_tool", return_value=mock_sandbox_success) as mock_exec:
            out_state = node_tool_call(state)

            assert mock_llm.call_count == 1
            assert mock_exec.call_count == 1
            assert len(out_state.tool_results) == 1
            res = out_state.tool_results[0]
            assert res["status"] == "verified"
            assert res["exit_code"] == 0
            assert "Revenue Growth: -3.78%" in res["stdout"]

            # Now test node_reason formatting
            reasoned_state = node_reason(out_state)
            assert "**Task Type:** HYBRID" in reasoned_state.response
            assert "**Model:** qwen2.5-coder:3b" in reasoned_state.response
            assert "**RAG:** VERIFIED" in reasoned_state.response
            assert "**Execution:** VERIFIED" in reasoned_state.response
            assert "Revenue from Operations" in reasoned_state.response
            assert "Revenue Growth: -3.78%" in reasoned_state.response
            assert "Sources" in reasoned_state.response


def test_hybrid_node_tool_call_fails_closed_when_docker_unavailable():
    """Verify that when Docker is unavailable, the system fails closed and never executes on host."""
    state = AgentState(
        query="Calculate MRPL revenue growth using the annual report",
        task_type="HYBRID",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_rag=True,
        requires_sandbox=True,
        required_tools=["rag_search", "sandbox_execute"],
        context=SAMPLE_CONTEXT,
        sources=SAMPLE_SOURCES,
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    mock_qwen_code = "```python\nprint('testing')\n```"
    mock_sandbox_error = {
        "tool": "sandbox_execute",
        "status": "error",
        "result": {
            "status": "error",
            "stdout": "",
            "stderr": "Code sandbox is unavailable: Docker daemon is not running.",
            "exit_code": None,
            "execution_time_ms": 0,
        },
    }

    with patch("backend.agent.graph._call_llm", return_value=mock_qwen_code):
        with patch("backend.agent.graph.execute_tool", return_value=mock_sandbox_error):
            out_state = node_tool_call(state)
            assert len(out_state.tool_results) == 1
            assert out_state.tool_results[0]["status"] == "error"

            reasoned_state = node_reason(out_state)
            assert "**Task Type:** HYBRID" in reasoned_state.response
            assert "BLOCKED (Docker Sandbox Unavailable)" in reasoned_state.response
            assert "failed closed and did not execute the code on the host" in reasoned_state.response


def test_hybrid_node_tool_call_refusal_when_insufficient_data():
    """Verify that missing required data returns controlled refusal without invoking Qwen."""
    state = AgentState(
        query="Calculate MRPL revenue growth using the annual report",
        task_type="HYBRID",
        model_id="qwen25_coder_3b",
        ollama_model_name="qwen2.5-coder:3b",
        requires_rag=True,
        requires_sandbox=True,
        required_tools=["rag_search", "sandbox_execute"],
        context="No financial figures available in this context chunk.",
        sources=[],
        user={"id": 1, "username": "admin", "roles": ["administrator"]},
    )

    with patch("backend.agent.graph._call_llm") as mock_llm:
        out_state = node_tool_call(state)
        # LLM must NOT be called to invent fake numbers
        assert mock_llm.call_count == 0
        assert "The knowledge base does not contain sufficient verified data" in out_state.response

        reasoned_state = node_reason(out_state)
        assert "The knowledge base does not contain sufficient verified data" in reasoned_state.response


# ── 7. Safe Fiscal Year Variable Names & Regression Tests ───────────────

def test_make_safe_variable_name():
    """Verify that variable names are deterministic, safe, and follow revenue_fy2025_26 convention."""
    assert make_safe_variable_name("Revenue from Operations", "FY 2025-26") == "revenue_fy2025_26"
    assert make_safe_variable_name("Revenue from Operations", "FY 2024-25") == "revenue_fy2024_25"
    assert make_safe_variable_name("Exports", "FY 2025-26") == "exports_fy2025_26"
    assert make_safe_variable_name("Exports", "FY 2024-25") == "exports_fy2024_25"
    assert make_safe_variable_name("Profit Before Tax (PBT)", "FY 2025-26") == "pbt_fy2025_26"
    assert make_safe_variable_name("Profit Before Tax (PBT)", "FY 2024-25") == "pbt_fy2024_25"
    assert make_safe_variable_name("Profit After Tax (PAT)", "FY 2025-26") == "pat_fy2025_26"
    assert make_safe_variable_name("Profit After Tax (PAT)", "FY 2024-25") == "pat_fy2024_25"
    assert make_safe_variable_name("Gross Refining Margin (GRM)", "FY 2025-26") == "grm_fy2025_26"
    assert make_safe_variable_name("Gross Refining Margin (GRM)", "FY 2024-25") == "grm_fy2024_25"


def test_grm_retains_actual_prior_fiscal_year():
    """
    Regression Test for Requirement 7:
    In source data Page 7, GRM had a typographical repetition:
    'improved significantly to US$ 9.22 per barrel during FY 2025-26 as compared to US$ 4.45 per barrel during FY 2025-26'
    While Page 20 explicitly had:
    'The Gross Refining Margin (GRM) for Financial Year 2025-26 was US$ 9.22/bbl as against US$ 4.45/bbl during the Financial Year 2024-25.'
    The system must correctly retain FY 2024-25 for 4.45 and FY 2025-26 for 9.22.
    """
    # Test with Page 7 alone (repetition in comparative clause resolved via surrounding block years)
    context_page_7 = """[Document: 38Th Annual Report (2025–26), Page 7, Class: INTERNAL]
ANNUAL REPORT 2025-26 7 Standalone Financial Performance g The Company recorded Revenue from Operations of ₹1,05,155 crore during FY 2025-26 as against ₹1,09,280 crore during FY 2024-25. g Gross Refining Margin (GRM) improved significantly to US$ 9.22 per barrel during FY 2025-26 as compared to US$ 4.45 per barrel during FY 2025-26, supported by stronger product cracks and a favourable crude basket."""
    metrics_p7 = extract_structured_metrics(context_page_7, SAMPLE_SOURCES, query="grm")
    grm_26 = next((m for m in metrics_p7 if m["value"] == 9.22), None)
    grm_25 = next((m for m in metrics_p7 if m["value"] == 4.45), None)
    assert grm_26 is not None
    assert grm_26["fiscal_year"] == "FY 2025-26"
    assert grm_26["variable_name"] == "grm_fy2025_26"
    assert grm_25 is not None
    assert grm_25["fiscal_year"] == "FY 2024-25"
    assert grm_25["variable_name"] == "grm_fy2024_25"
    assert not any(m["value"] == 4.45 and m["fiscal_year"] == "FY 2025-26" for m in metrics_p7)

    # Test with Page 20 alone (Order B with 'Financial Year')
    context_page_20 = """[Document: 38Th Annual Report (2025–26), Page 20, Class: INTERNAL]
The Gross Refining Margin (GRM) for Financial Year 2025-26 was US$ 9.22/bbl as against US$ 4.45/bbl during the Financial Year 2024-25."""
    metrics_p20 = extract_structured_metrics(context_page_20, SAMPLE_SOURCES, query="grm")
    grm_26_p20 = next((m for m in metrics_p20 if m["value"] == 9.22), None)
    grm_25_p20 = next((m for m in metrics_p20 if m["value"] == 4.45), None)
    assert grm_26_p20 is not None
    assert grm_26_p20["fiscal_year"] == "FY 2025-26"
    assert grm_25_p20 is not None
    assert grm_25_p20["fiscal_year"] == "FY 2024-25"


def test_fiscal_year_labels_never_blend_2024_26():
    """
    Regression Test for Requirements 1, 2, 4, 6:
    Ensure that FY 2024-25 is never accidentally rendered as FY 2024-26 or 2024_26.
    """
    # 1. Test prompt instructions
    metrics = extract_structured_metrics(SAMPLE_CONTEXT, SAMPLE_SOURCES)
    prompt = build_qwen_hybrid_prompt("calculate relevant metrics", metrics)
    assert "revenue_fy2025_26" in prompt
    assert "revenue_fy2024_25" in prompt
    assert "2024-26" in prompt  # Mentioned in NEVER alter instructions
    assert "NEVER alter, infer, or blend fiscal-year labels" in prompt
    assert "The prior comparison year is strictly 'FY 2024-25'" in prompt

    # 2. Test sanitizer function on accidental code outputs
    accidental_code = (
        "revenue_operations_2024_26 = 109280\n"
        "grm_2024_26 = 4.45\n"
        "print('Revenue FY 2024-26: ' + str(revenue_operations_2024_26))\n"
        "print('GRM (2024-26): ' + str(grm_2024_26))\n"
    )
    sanitized_code = sanitize_fiscal_year_labels(accidental_code)
    assert "2024_26" not in sanitized_code
    assert "2024-26" not in sanitized_code
    assert "revenue_operations_fy2024_25" in sanitized_code
    assert "grm_fy2024_25" in sanitized_code
    assert "Revenue FY 2024-25:" in sanitized_code
    assert "GRM (FY 2024-25):" in sanitized_code

    # 3. Test final formatted response cleans up any accidental 2024-26
    final_output = format_hybrid_final_response(
        query="calculate metrics",
        model_name="qwen2.5-coder:3b",
        metrics=metrics,
        sources=SAMPLE_SOURCES,
        code=accidental_code,
        stdout="Revenue in FY 2024-26: 109280\nRevenue in FY 2025-26: 105155",
        stderr="",
        exit_code=0,
        status="verified",
        correction_attempts=0,
    )
    assert "2024-26" not in final_output
    assert "2024_26" not in final_output
    assert "FY 2024-25" in final_output
    assert "FY 2025-26" in final_output
