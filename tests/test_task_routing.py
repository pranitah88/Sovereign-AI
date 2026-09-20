"""
Tests for Task Router and Model Routing.

Verifies:
1. "write python code to calculate financial status" -> CODING, qwen2.5-coder:3b, requires_rag=False, sandbox=True
2. "What is MRPL's revenue according to the annual report?" -> RAG, gemma3:4b, requires_rag=True, sandbox=False
3. "Write Python code to calculate MRPL revenue growth using the annual report" -> HYBRID, qwen2.5-coder:3b, requires_rag=True, sandbox=True
4. "What is the purpose of the hydrocracker?" -> RAG, gemma3:4b, requires_rag=True, sandbox=False
5. "Write a Python function to sort an array" -> CODING, qwen2.5-coder:3b, requires_rag=False, sandbox=True
6. "Explain what Python is" -> GENERAL, gemma3:4b, requires_rag=False, sandbox=False
7. Security: only approved models in registry can be selected, unapproved models raise ValueError.
"""

import pytest

from backend.services.task_router import (
    route_task,
    classify_task_type,
    validate_model_selection,
    select_model_and_tools,
    get_model_entry,
)


# ── 1. The 6 Required Test Cases ─────────────────────────────────────────

def test_case_1_coding_financial_status():
    """1. 'write python code to calculate financial status' -> CODING -> qwen2.5-coder:3b -> sandbox."""
    decision = route_task("write python code to calculate financial status")
    assert decision["task_type"] == "CODING"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["requires_rag"] is False
    assert decision["requires_sandbox"] is True
    assert "sandbox_execute" in decision["tools"]
    assert "rag_search" not in decision["tools"]


def test_case_2_rag_revenue_annual_report():
    """2. 'What is MRPL's revenue according to the annual report?' -> RAG -> appropriate local model."""
    decision = route_task("What is MRPL's revenue according to the annual report?")
    assert decision["task_type"] == "RAG"
    assert decision["model"] == "gemma3:4b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is False
    assert "rag_search" in decision["tools"]


def test_case_3_hybrid_revenue_growth_report():
    """3. 'Write Python code to calculate MRPL revenue growth using the annual report' -> HYBRID -> RAG -> Qwen Coder -> Docker Sandbox."""
    decision = route_task("Write Python code to calculate MRPL revenue growth using the annual report")
    assert decision["task_type"] == "HYBRID"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is True
    assert "rag_search" in decision["tools"]
    assert "sandbox_execute" in decision["tools"]


def test_case_4_rag_purpose_of_hydrocracker():
    """4. 'What is the purpose of the hydrocracker?' -> RAG -> appropriate local model."""
    decision = route_task("What is the purpose of the hydrocracker?")
    assert decision["task_type"] == "RAG"
    assert decision["model"] == "gemma3:4b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is False
    assert "rag_search" in decision["tools"]


def test_case_5_coding_sort_array():
    """5. 'Write a Python function to sort an array' -> CODING -> Qwen Coder -> sandbox."""
    decision = route_task("Write a Python function to sort an array")
    assert decision["task_type"] == "CODING"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["requires_rag"] is False
    assert decision["requires_sandbox"] is True
    assert "sandbox_execute" in decision["tools"]
    assert "rag_search" not in decision["tools"]


def test_case_6_general_explain_python():
    """6. 'Explain what Python is' -> GENERAL -> appropriate local model, without unnecessary RAG."""
    decision = route_task("Explain what Python is")
    assert decision["task_type"] == "GENERAL"
    assert decision["model"] == "gemma3:4b"
    assert decision["requires_rag"] is False
    assert decision["requires_sandbox"] is False
    assert "rag_search" not in decision["tools"]


# ── 2. Structured Information Contract ───────────────────────────────────

def test_router_structured_output_keys():
    """Verify router returns all required structured fields."""
    decision = route_task("write python code to calculate financial status")
    required_keys = {"task_type", "model", "requires_rag", "requires_sandbox", "reason"}
    assert required_keys.issubset(decision.keys()), f"Missing keys: {required_keys - set(decision.keys())}"


def test_hybrid_structured_output_keys():
    """Verify hybrid tasks return all required structured fields."""
    decision = route_task("Create Python code that calculates MRPL's EBITDA from the annual report")
    assert decision["task_type"] == "HYBRID"
    assert decision["model"] == "qwen2.5-coder:3b"
    assert decision["requires_rag"] is True
    assert decision["requires_sandbox"] is True


# ── 3. Security Requirements ─────────────────────────────────────────────

def test_security_only_approved_models_selected():
    """Verify that only models in the approved model registry can be selected."""
    approved_models = {"gemma3_4b", "qwen25_coder_3b"}
    entry_gemma = validate_model_selection("gemma3_4b")
    entry_qwen = validate_model_selection("qwen25_coder_3b")
    assert entry_gemma["approved"] is True
    assert entry_qwen["approved"] is True


def test_security_unapproved_models_rejected():
    """Verify that attempting to select unapproved or external models raises an error."""
    with pytest.raises(ValueError):
        validate_model_selection("gpt-4o")

    with pytest.raises(ValueError):
        validate_model_selection("claude-3-opus")

    with pytest.raises(ValueError):
        validate_model_selection("deepseek-r1:671b")


def test_security_backward_compatible_wrapper():
    """Verify classify_task_type wrapper works as expected."""
    res = classify_task_type("write python code to calculate financial status")
    assert res["task_type"] == "CODING"
    assert res["model"] == "qwen2.5-coder:3b"
    assert res["requires_rag"] is False
    assert res["requires_sandbox"] is True
