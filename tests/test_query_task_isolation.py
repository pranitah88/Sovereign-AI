"""
Regression tests for Query-Task Isolation and Document Analysis Contamination Prevention.
Verifies that previous turn state, attachments, and generated deliverables do not
contaminate independent queries.
"""

import pytest
from unittest.mock import patch

from backend.agent.state import AgentState
from backend.agent.graph import node_classify, node_reason, node_respond
from backend.services.task_router import route_task, check_document_analysis_intent


PREV_DOC_HISTORY = [
    {
        "role": "user",
        "content": "Analyze TranscriptSd.pdf",
        "task_type": "DOCUMENT_ANALYSIS",
        "target_document": "TranscriptSd.pdf",
    },
    {
        "role": "assistant",
        "content": "The analysis report from TranscriptSd.pdf has been successfully generated in PDF format.\n\n**Deliverable:** `TranscriptSd_Analysis.pdf` (12400 bytes)",
        "task_type": "DOCUMENT_ANALYSIS",
        "target_document": "TranscriptSd.pdf",
        "deliverable": {"filename": "TranscriptSd_Analysis.pdf", "file_size_bytes": 12400, "status": "success"},
        "sources": [{"source": "TranscriptSd.pdf", "page": 1}],
    }
]


def test_01_query_with_previous_doc_analysis_not_contaminated():
    """
    1. Current query: "give information about engineering machinary"
       Previous turn: "Analyze TranscriptSd.pdf"
       Expected: NOT DOCUMENT_ANALYSIS (must be GENERAL or RAG, without docgen_pdf).
    """
    query = "give information about engineering machinary"
    decision = route_task(query, conversation_history=PREV_DOC_HISTORY)
    assert decision["task_type"] != "DOCUMENT_ANALYSIS", f"Expected not DOCUMENT_ANALYSIS, got {decision['task_type']}"
    assert decision["task_type"] in ("GENERAL", "RAG")
    assert "docgen_pdf" not in decision.get("tools", [])

    state = AgentState(query=query, current_query=query, chat_history=PREV_DOC_HISTORY)
    state = node_classify(state)
    assert state.task_type != "DOCUMENT_ANALYSIS"
    assert state.target_document is None
    assert "docgen_pdf" not in state.required_tools


def test_02_refinery_query_after_doc_analysis_routes_to_rag():
    """
    2. Current query: "What are the main units in MRPL?"
       Previous turn: "Analyze TranscriptSd.pdf"
       Expected: RAG/technical, not DOCUMENT_ANALYSIS.
    """
    query = "What are the main units in MRPL?"
    decision = route_task(query, conversation_history=PREV_DOC_HISTORY)
    assert decision["task_type"] == "RAG"
    assert decision["task_type"] != "DOCUMENT_ANALYSIS"
    assert "rag_search" in decision["tools"]
    assert "docgen_pdf" not in decision["tools"]

    state = AgentState(query=query, current_query=query, chat_history=PREV_DOC_HISTORY)
    state = node_classify(state)
    assert state.task_type == "RAG"
    assert state.target_document is None


def test_03_explicit_doc_analysis_routes_to_document_analysis():
    """
    3. Current query: "Analyze TranscriptSd.pdf"
       Expected: DOCUMENT_ANALYSIS.
    """
    query = "Analyze TranscriptSd.pdf"
    decision = route_task(query)
    assert decision["task_type"] == "DOCUMENT_ANALYSIS"
    assert decision.get("target_document") == "TranscriptSd.pdf"

    state = AgentState(query=query, current_query=query)
    state = node_classify(state)
    assert state.task_type == "DOCUMENT_ANALYSIS"
    assert state.target_document == "TranscriptSd.pdf"


def test_04_explicit_report_generation_routes_to_docgen_pdf():
    """
    4. Current query: "Generate a report from TranscriptSd.pdf"
       Expected: DOCUMENT_ANALYSIS + PDF generation (docgen_pdf tool).
    """
    query = "Generate a report from TranscriptSd.pdf"
    decision = route_task(query)
    assert decision["task_type"] == "DOCUMENT_ANALYSIS"
    assert "docgen_pdf" in decision.get("tools", [])
    assert decision.get("deliverable_type") == "pdf"


def test_05_valid_document_followup_resolves_context():
    """
    5. Current query: "Explain the document"
       ONLY if the previous turn clearly established one unique active document context:
       -> valid document follow-up
    """
    query = "Explain the document"
    # Router recognizes document reference intent
    decision = route_task(query, conversation_history=PREV_DOC_HISTORY)
    assert decision["task_type"] == "DOCUMENT_ANALYSIS"

    state = AgentState(query=query, current_query=query, chat_history=PREV_DOC_HISTORY)
    state = node_classify(state)
    assert state.task_type == "DOCUMENT_ANALYSIS"
    assert state.target_document == "TranscriptSd.pdf"


def test_06_attachment_presence_does_not_force_document_analysis():
    """
    6. Current query: "Give information about engineering machinery" with a PDF attached:
       -> NOT DOCUMENT_ANALYSIS unless the query explicitly refers to the PDF.
    """
    query = "Give information about engineering machinery"
    decision = route_task(query, has_scanned_pdf=True)
    assert decision["task_type"] != "DOCUMENT_ANALYSIS"
    assert decision["task_type"] in ("GENERAL", "RAG")
    assert "docgen_pdf" not in decision.get("tools", [])

    state = AgentState(query=query, current_query=query, has_scanned_pdf=True)
    state = node_classify(state)
    assert state.task_type != "DOCUMENT_ANALYSIS"
    assert state.target_document is None


def test_07_previous_generated_pdf_never_returned_for_unrelated_query():
    """
    7. Current query: "Give information about engineering machinery" with previous generated PDF:
       -> do NOT return previous PDF.
    """
    query = "Give information about engineering machinery"
    state = AgentState(
        query=query,
        current_query=query,
        chat_history=PREV_DOC_HISTORY,
        deliverable={"filename": "TranscriptSd_Analysis.pdf", "file_size_bytes": 12400, "status": "success"},
    )
    state = node_classify(state)
    assert state.task_type != "DOCUMENT_ANALYSIS"
    assert state.target_document is None

    # In node_respond, deliverable is cleared for non-document tasks (Response Type Integrity)
    state = node_respond(state)
    assert state.deliverable is None
    assert "TranscriptSd_Analysis.pdf" not in state.response
    assert "successfully generated in PDF format" not in state.response


def test_08_different_turn_id_no_task_type_inheritance():
    """
    8. Different turn_id:
       -> no task-type inheritance.
    """
    query = "give information about engineering machinary"
    state = AgentState(
        query=query,
        current_query=query,
        turn_id=2,
        chat_history=PREV_DOC_HISTORY,
    )
    state = node_classify(state)
    assert state.task_type != "DOCUMENT_ANALYSIS"
    assert state.task_type in ("GENERAL", "RAG")


def test_09_previous_has_scanned_pdf_does_not_force_document_analysis():
    """
    9. Previous has_scanned_pdf=true with new independent query:
       -> must not force DOCUMENT_ANALYSIS.
    """
    query = "What is MRPL?"
    # Turn 2 has no new scanned pdf, but history had one
    decision = route_task(query, has_scanned_pdf=False, conversation_history=PREV_DOC_HISTORY)
    assert decision["task_type"] == "RAG"
    assert decision["task_type"] != "DOCUMENT_ANALYSIS"


def test_10_previous_document_name_does_not_force_document_analysis():
    """
    10. Previous document_name="TranscriptSd.pdf" with new independent query:
        -> must not force DOCUMENT_ANALYSIS.
    """
    query = "Give Python code for area of a circle"
    decision = route_task(query, conversation_history=PREV_DOC_HISTORY)
    assert decision["task_type"] == "CODING"
    assert decision["task_type"] != "DOCUMENT_ANALYSIS"
    assert "sandbox_execute" in decision["tools"]

    state = AgentState(query=query, current_query=query, chat_history=PREV_DOC_HISTORY)
    state = node_classify(state)
    assert state.task_type == "CODING"
    assert state.target_document is None


def test_11_critical_cross_turn_isolation():
    """
    16. CRITICAL CROSS-TURN TEST
    Run exactly:
    TURN 1: "Analyze TranscriptSd.pdf" -> DOCUMENT_ANALYSIS
    TURN 2: "give information about engineering machinary" -> GENERAL/RAG, NOT DOCUMENT_ANALYSIS, NOT TranscriptSd_Analysis.pdf
    TURN 3: "Analyze TranscriptSd.pdf" -> DOCUMENT_ANALYSIS again.
    """
    # Turn 1
    t1_query = "Analyze TranscriptSd.pdf"
    t1_state = AgentState(query=t1_query, current_query=t1_query, turn_id=1)
    t1_state = node_classify(t1_state)
    assert t1_state.task_type == "DOCUMENT_ANALYSIS"
    assert t1_state.target_document == "TranscriptSd.pdf"

    # Simulate history with Turn 1 completed
    turn_1_history = [
        {"role": "user", "content": t1_query, "task_type": "DOCUMENT_ANALYSIS", "target_document": "TranscriptSd.pdf"},
        {
            "role": "assistant",
            "content": "Analysis of TranscriptSd.pdf complete.",
            "task_type": "DOCUMENT_ANALYSIS",
            "target_document": "TranscriptSd.pdf",
            "deliverable": {"filename": "TranscriptSd_Analysis.pdf", "status": "success"},
            "sources": [{"source": "TranscriptSd.pdf", "page": 1}],
        },
    ]

    # Turn 2: Bug reproduction query
    t2_query = "give information about engineering machinary"
    t2_state = AgentState(
        query=t2_query,
        current_query=t2_query,
        turn_id=2,
        chat_history=turn_1_history,
        deliverable={"filename": "TranscriptSd_Analysis.pdf"},  # stale artifact candidate
    )
    t2_state = node_classify(t2_state)
    assert t2_state.task_type != "DOCUMENT_ANALYSIS"
    assert t2_state.task_type in ("GENERAL", "RAG")
    assert t2_state.target_document is None
    assert "docgen_pdf" not in t2_state.required_tools

    t2_state = node_respond(t2_state)
    assert t2_state.deliverable is None
    assert "TranscriptSd_Analysis.pdf" not in t2_state.response
    assert "successfully generated in PDF format" not in t2_state.response

    # Verify Turn 2 trace events
    trace_events = [e["event"] for e in t2_state.execution_trace]
    assert "CURRENT_QUERY_RECEIVED" in trace_events
    assert "ATTACHMENT_CONTEXT_DETECTED" in trace_events
    assert "TASK_INTENT_CLASSIFIED" in trace_events
    assert "DOCUMENT_ANALYSIS_GATE" in trace_events
    assert "TOOL_SELECTION" in trace_events
    assert "RESPONSE_GENERATION" in trace_events

    # Verify DOCUMENT_ANALYSIS_GATE was NOT_REQUIRED for Turn 2
    gate_event = next(e for e in t2_state.execution_trace if e["event"] == "DOCUMENT_ANALYSIS_GATE")
    assert gate_event["details"]["gate_status"] == "NOT_REQUIRED"

    # Turn 3: User requests document analysis again
    turn_2_history = turn_1_history + [
        {"role": "user", "content": t2_query, "task_type": "GENERAL"},
        {"role": "assistant", "content": "Engineering machinery refers to...", "task_type": "GENERAL"},
    ]
    t3_query = "Analyze TranscriptSd.pdf"
    t3_state = AgentState(query=t3_query, current_query=t3_query, turn_id=3, chat_history=turn_2_history)
    t3_state = node_classify(t3_state)
    assert t3_state.task_type == "DOCUMENT_ANALYSIS"
    assert t3_state.target_document == "TranscriptSd.pdf"

    t3_gate_event = next(e for e in t3_state.execution_trace if e["event"] == "DOCUMENT_ANALYSIS_GATE")
    assert t3_gate_event["details"]["gate_status"] == "PASSED"


def test_12_machinary_typo_preserves_visible_query():
    """
    11. ENGINEERING MACHINERY TYPO:
    Input contains "machinary".
    - Do not let typo trigger document analysis.
    - Do not rewrite visible user query.
    """
    query = "give information about engineering machinary"
    state = AgentState(query=query, current_query=query)
    state = node_classify(state)
    assert state.query == "give information about engineering machinary"
    assert state.current_query == "give information about engineering machinary"
    assert state.task_type != "DOCUMENT_ANALYSIS"
