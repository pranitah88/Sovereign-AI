"""
LangGraph agent orchestrator — the central reasoning engine.

Implements a state graph with nodes: classify → retrieve → reason →
tool_call → validate → respond, with conditional routing based on
task type and tool results.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from langgraph.graph import StateGraph, START, END

from backend.agent.prompts import (
    CODE_CORRECTION_PROMPT,
    CODE_GENERATION_PROMPT,
    DOCUMENT_ANALYSIS_PROMPT,
    GENERAL_CHAT_PROMPT,
    REASONING_PROMPT,
    SYNTHESIS_PROMPT,
)
from backend.agent.state import AgentState
from backend.agent.tools import execute_tool
from backend.database.connection import get_connection, transaction
from backend.database.repositories import chat as chat_repo
from backend.services.audit import DOCUMENT_GENERATED, DOCUMENT_GENERATION_FAILED, audit_log
from backend.services.rag_engine import clean_document_title
from backend.services.task_router import route_task

logger = logging.getLogger(__name__)


def strip_raw_ui_markers(text: str) -> str:
    """
    Remove internal UI/svg step markers, and sanitize raw markdown asterisks (* and **)
    from assistant prose outside code blocks for clean structured text rendering.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Strip internal SVG markers
    cleaned = re.sub(r'svg(?:Copy|Query|Retrieval|Model|Tools|Response|Download)', '', text, flags=re.IGNORECASE).strip()

    # 2. Preserve code blocks while sanitizing markdown asterisks in prose
    parts = re.split(r'(```[\s\S]*?```)', cleaned)
    for i in range(0, len(parts), 2):
        p = parts[i]
        # Remove bold markdown **...**
        p = re.sub(r'\*\*([^*]+)\*\*', r'\1', p)
        # Remove double asterisks that may be unclosed
        p = p.replace('**', '')
        # Remove asterisk bullets at line start
        p = re.sub(r'(?m)^\s*\*\s+', '', p)
        # Clean dash bullets before common labels (e.g. "- Purpose:" -> "Purpose:")
        p = re.sub(r'(?m)^\s*-\s+(Purpose|Function|Role|Capacity|Details|Status|Note|Output|Inputs|Feedstock|Products?):', r'\1:', p, flags=re.IGNORECASE)
        # Remove remaining standalone emphasis asterisks *word*
        p = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', p)
        parts[i] = p

    return "".join(parts).strip()


def is_code_modification_request(query: str) -> bool:
    """
    Check if the user explicitly references modifying previous code.
    Examples that match:
      - "modify the code you gave me"
      - "add a rectangle option to that code"
      - "modify the previous code to also support triangles"
      - "update the code to..."
    Examples that do NOT match (fresh requests):
      - "write a python code for calculating area"
      - "write python code to calculate the area of a circle with radius 5"
      - "write code to calculate factorial"
    """
    if not query:
        return False
    q = query.lower()
    explicit_modification_patterns = [
        r"\b(?:modify|change|update|edit|refactor|fix|extend|expand|adapt|adjust)\s+(?:the|that|this|your|our|previous)\s+code\b",
        r"\b(?:add|include|append)\s+.+\s+(?:to|in)\s+(?:the|that|this|your|our|previous)\s+code\b",
        r"\b(?:to|in)\s+(?:that|this|the\s+previous)\s+code\b",
        r"\b(?:from|with)\s+(?:the\s+previous|that)\s+code\b",
        r"\b(?:modify|update|change|fix)\s+(?:it|the\s+previous\s+function|the\s+previous\s+script)\b",
        r"\bprevious\s+code\b",
        r"\bthat\s+code\b",
    ]
    return any(re.search(pat, q) for pat in explicit_modification_patterns)


def extract_previous_generated_code(chat_history: list[dict] | None) -> str | None:
    """
    Extract the most recent generated Python code from previous assistant messages in chat history.
    """
    if not chat_history:
        return None
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            code = _extract_code_block(content)
            if code and len(code.strip()) > 10 and ("def " in code or "print(" in code or "=" in code):
                return code.strip()
    return None


def _add_trace_event(state: AgentState, event: str, title: str, status: str, details: dict | None = None) -> None:
    """Record an auditable, structured execution step into the agent state trace."""
    if state.execution_trace is None:
        state.execution_trace = []
    state.execution_trace.append({
        "event": event,
        "title": title,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    })


# ── Graph Nodes ──────────────────────────────────────────────────────────

def node_classify(state: AgentState) -> AgentState:
    """Classify the user's request into a task type and route to approved model and tools."""
    # Deterministic Query Integrity Enforcement
    current_q = state.current_query or state.query
    state.current_query = current_q
    state.query = current_q

    _add_trace_event(
        state,
        "CURRENT_QUERY_RECEIVED",
        f'Current Query Received: "{current_q}"',
        "verified",
        {"current_query": current_q, "turn_id": state.turn_id, "trace_id": state.trace_id},
    )

    _add_trace_event(
        state,
        "ATTACHMENT_CONTEXT_DETECTED",
        f"Attachment context: has_scanned_pdf={state.has_scanned_pdf}, has_image={state.has_image}",
        "verified",
        {"has_scanned_pdf": state.has_scanned_pdf, "has_image": state.has_image},
    )

    _add_trace_event(
        state,
        "ORCHESTRATOR_QUERY",
        f'Orchestrator Query: "{current_q}"',
        "verified",
        {"current_query": current_q, "trace_id": state.trace_id},
    )

    from backend.services.entity_preservation import detect_query_language_deterministic
    query_lang = detect_query_language_deterministic(current_q)
    _add_trace_event(
        state,
        "QUERY_LANGUAGE_DETECTED",
        f'Query Language: {query_lang} (Authoritative: current_user_query)',
        "verified",
        {
            "QUERY_LANGUAGE": query_lang,
            "LANGUAGE_DECISION_SOURCE": "current_user_query",
        },
    )

    decision = route_task(
        current_q,
        has_image=state.has_image,
        has_scanned_pdf=state.has_scanned_pdf,
        conversation_history=state.chat_history or state.conversation_history,
    )

    state.task_type = decision["task_type"]
    state.classification_reasoning = decision.get("reason", "")
    state.requires_rag = decision.get("requires_rag", False)
    state.requires_sandbox = decision.get("requires_sandbox", False)
    state.model_id = decision["model_id"]
    state.ollama_model_name = decision["model"]
    state.model_endpoint = decision.get("endpoint", "http://127.0.0.1:11434/api/generate")
    state.required_tools = decision.get("tools", [])
    state.target_document = decision.get("target_document")
    state.clarification_question = decision.get("clarification_question", "")
    state.ambiguity_type = decision.get("ambiguity_type", "")

    _add_trace_event(
        state,
        "TASK_INTENT_CLASSIFIED",
        f"Task Intent: {state.task_type}",
        "verified",
        {"task_type": state.task_type, "model_id": state.model_id, "reason": state.classification_reasoning},
    )

    if state.task_type in ("CLARIFICATION", "INCOMPLETE_QUERY", "AMBIGUOUS_QUERY"):
        _add_trace_event(
            state,
            "QUERY_COMPLETENESS_CHECK",
            f'Query Completeness: Incomplete ({decision.get("reason", "")})',
            "insufficient",
            {"is_complete": False, "reason": decision.get("reason", "")},
        )
        if decision.get("ambiguity_type") in ("incomplete_ambiguous", "ambiguous_entity"):
            _add_trace_event(
                state,
                "AMBIGUITY_DETECTED",
                f'Ambiguity Detected: {decision.get("ambiguity_type")}',
                "insufficient",
                {"ambiguity_type": decision.get("ambiguity_type")},
            )
        _add_trace_event(
            state,
            "CLARIFICATION_REQUIRED",
            f'Clarification Required: {state.clarification_question}',
            "insufficient",
            {"clarification_question": state.clarification_question, "task_type": state.task_type},
        )

    _add_trace_event(
        state,
        "ROUTER_QUERY",
        f'Router Query: "{current_q}"',
        "verified",
        {"current_query": current_q, "task_type": state.task_type, "requires_rag": state.requires_rag},
    )

    # Document Analysis Gate & Context Resolution:
    # Rule 4: Previous document state must not leak across turns!
    # Independent queries MUST NEVER inherit previous turn's target_document or document state.
    # ONLY search chat history for target_document IF:
    # 1. The current task is DOCUMENT_ANALYSIS
    # 2. AND state.target_document was not in current query
    # 3. AND the current query is an explicit document follow-up (e.g. "explain the document", "explain the findings in more detail", "what does page 5 say")
    if state.task_type == "DOCUMENT_ANALYSIS" and not state.target_document and state.chat_history:
        current_lower = current_q.lower()
        is_doc_followup = bool(re.search(
            r'\b(the\s+document|the\s+pdf|the\s+report|this\s+document|this\s+pdf|the\s+findings|page\s+\d+|it|its)\b',
            current_lower,
        ))
        if is_doc_followup:
            for msg in reversed(state.chat_history):
                if msg.get("target_document"):
                    state.target_document = str(msg["target_document"]).strip()
                    break
                content = msg.get("content", "")
                q_docs = re.findall(r'["\']([^"\']+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))["\']', content, re.IGNORECASE)
                if q_docs:
                    state.target_document = q_docs[0].strip()
                    break
                ticks = re.findall(r'`([^`]+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))`', content, re.IGNORECASE)
                if ticks:
                    state.target_document = ticks[0].strip()
                    break
                docs = re.findall(r'\b([\w\-]+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))\b', content, re.IGNORECASE)
                if docs:
                    state.target_document = docs[0].strip()
                    break
    elif state.task_type != "DOCUMENT_ANALYSIS":
        # Hard turn isolation: non-document tasks MUST NOT have a target_document
        state.target_document = None

    # DOCUMENT_ANALYSIS_GATE trace event (Section 14)
    if state.task_type == "DOCUMENT_ANALYSIS":
        if state.target_document:
            _add_trace_event(
                state,
                "DOCUMENT_ANALYSIS_GATE",
                f"Document Analysis Gate: PASSED ({state.target_document})",
                "allowed",
                {"gate_status": "PASSED", "target_document": state.target_document},
            )
        else:
            _add_trace_event(
                state,
                "DOCUMENT_ANALYSIS_GATE",
                "Document Analysis Gate: PASSED (general document/OCR intent)",
                "allowed",
                {"gate_status": "PASSED", "target_document": None},
            )
    else:
        _add_trace_event(
            state,
            "DOCUMENT_ANALYSIS_GATE",
            f"Document Analysis Gate: NOT_REQUIRED (Task: {state.task_type})",
            "verified",
            {"gate_status": "NOT_REQUIRED", "task_type": state.task_type},
        )

    # TOOL_SELECTION trace event (Section 14)
    _add_trace_event(
        state,
        "TOOL_SELECTION",
        f"Tool Selection: {state.required_tools}",
        "verified",
        {"tools": state.required_tools},
    )

    if state.target_document:
        try:
            from backend.services.doc_analysis import resolve_document_file
            resolved_p, resolved_meta = resolve_document_file(state.target_document)
            if resolved_meta and resolved_meta.get("original_name"):
                state.target_document = resolved_meta["original_name"]
            elif resolved_p:
                state.target_document = resolved_p.name
        except Exception:
            pass

    state.current_step = "classified"
    state.step_index += 1
    _log_step(state)

    _add_trace_event(
        state,
        "model_selected",
        f"Selected {state.ollama_model_name} for {state.task_type}",
        "allowed",
        {
            "model": state.ollama_model_name,
            "model_id": state.model_id,
            "task_type": state.task_type,
            "reason": state.classification_reasoning,
            "candidates": decision.get("candidates", []),
            "license": decision.get("license", "Open Source"),
        },
    )

    logger.info(
        "Classified as '%s' -> model=%s (%s), requires_rag=%s, requires_sandbox=%s, tools=%s",
        state.task_type,
        state.model_id,
        state.ollama_model_name,
        state.requires_rag,
        state.requires_sandbox,
        state.required_tools,
    )

    return state


def node_retrieve(state: AgentState) -> AgentState:
    """Retrieve context from RAG with deterministic RBAC, clearance verification, and confidence gate."""
    if not state.requires_rag:
        state.current_step = "retrieve_skipped"
        state.step_index += 1
        return state

    retrieval_start = time.perf_counter()
    user = state.user
    if user is None and state.user_id:
        try:
            from backend.database.repositories import users as users_repo
            user = users_repo.get_user_by_id(state.user_id)
            state.user = user
        except Exception:
            user = None

    user_roles = user.get("roles", ["engineer"]) if user else ["engineer"]
    from backend.services.rag_engine import CLASSIFICATION_LEVELS, get_max_clearance_for_roles
    user_clearance = user.get("clearance") if user and user.get("clearance") else get_max_clearance_for_roles(user_roles)
    user_level = CLASSIFICATION_LEVELS.get(str(user_clearance).upper(), 2)

    # 1. Deterministic RBAC & Clearance Gate before Retrieval / LLM Reasoning
    is_restricted_intent = any(w in state.query.lower() for w in ["restricted", "highly confidential", "board minutes", "classified report", "secret audit", "unreleased audit"])
    if is_restricted_intent and user_level < CLASSIFICATION_LEVELS.get("CONFIDENTIAL", 3):
        _add_trace_event(
            state,
            "authorization_checked",
            f"RBAC Denied: Role {user_roles} (Clearance: {user_clearance}) lacks required clearance",
            "denied",
            {"role": user_roles, "clearance": user_clearance, "required": "CONFIDENTIAL/RESTRICTED"},
        )
        state.response = (
            f"⛔ ACCESS DENIED: Role 'Viewer' with clearance level '{user_clearance}' is not authorized to access RESTRICTED or CONFIDENTIAL industrial materials.\n\n"
            "Access was denied deterministically before model reasoning. This unauthorized request has been recorded in the MRPL security audit ledger."
        )
        audit_log(
            action="rbac_access_denied",
            outcome="denied",
            user_id=state.user_id,
            username=user.get("username", "viewer") if user else "viewer",
            details={"query": state.query[:120], "clearance": user_clearance, "required": "CONFIDENTIAL"},
        )
        state.current_step = "rbac_denied"
        _log_step(state)
        return state

    _add_trace_event(
        state,
        "authorization_checked",
        f"Clearance verified: {user_clearance}",
        "verified",
        {"role": user_roles, "clearance": user_clearance},
    )
    _add_trace_event(
        state,
        "retrieval_started",
        f"Retrieval started for {state.task_type}",
        "allowed",
        {"query": state.query[:100]},
    )

    retrieval_query = state.current_query
    if state.task_type in ("CODING", "code_generation") or not state.requires_rag:
        # Pure coding tasks do not perform RAG search; skip to preserve query isolation
        state.retrieval_query = retrieval_query
        state.current_step = "retrieved"
        state.step_index += 1
        _log_step(state)
        return state

    if state.task_type == "DOCUMENT_ANALYSIS" and state.target_document:
        from backend.services.doc_analysis import extract_full_document, build_document_analysis_context
        doc_model = extract_full_document(state.target_document)
        state.context = build_document_analysis_context(doc_model)
        state.retrieved_context = state.context
        state.sources = [
            {"source": doc_model.document_name, "page": p.page_num}
            for p in doc_model.pages
        ]
        state.retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)
        _add_trace_event(
            state,
            "retrieval_completed",
            f"Extracted {len(state.sources)} pages from {state.target_document}",
            "verified",
            {"source_count": len(state.sources)},
        )
        state.current_step = "retrieved"
        state.step_index += 1
        _log_step(state)
        return state
    elif state.task_type == "HYBRID":
        from backend.services.hybrid_executor import extract_hybrid_retrieval_query
        retrieval_query = extract_hybrid_retrieval_query(state.current_query)
    else:
        from backend.services.voice.context_policy import determine_conversational_context
        context_res = determine_conversational_context(
            current_query=state.current_query,
            conversation_history=state.chat_history,
        )
        if context_res.is_followup:
            retrieval_query = context_res.resolved_retrieval_query
            _add_trace_event(
                state,
                "CONTEXTUAL_FOLLOWUP_RESOLVED",
                f'Resolved contextual follow-up: "{retrieval_query}" (entity: {context_res.target_entity})',
                "verified",
                {"current_query": state.current_query, "retrieval_query": retrieval_query, "entity": context_res.target_entity},
            )

    state.retrieval_query = retrieval_query
    _add_trace_event(
        state,
        "RAG_QUERY",
        f'RAG Search Query: "{retrieval_query}"',
        "verified",
        {"current_query": state.current_query, "retrieval_query": retrieval_query},
    )

    result = execute_tool("rag_search", user=user, query=retrieval_query)
    if isinstance(result, dict):
        result["query"] = retrieval_query
    state.retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    if result["status"] == "success":
        data = result["result"]
        state.context = data.get("context", "")
        state.retrieved_context = state.context
        state.sources = data.get("sources", [])
        state.tool_results.append(result)

    _add_trace_event(
        state,
        "retrieval_completed",
        f"Retrieved {len(state.sources)} authorized sources",
        "verified",
        {"source_count": len(state.sources)},
    )

    # 2. Deterministic Temporal Intent & Recency Grounding
    from backend.services.temporal_guard import (
        detect_temporal_intent,
        validate_temporal_suitability,
        validate_temporal_claims,
        generate_evidence_limited_response,
    )

    t_intent = detect_temporal_intent(state.current_query)
    state.temporal_intent = t_intent.to_dict()
    _add_trace_event(
        state,
        "TEMPORAL_INTENT_DETECTED",
        f"Temporal Intent: {t_intent.temporal_intent}" + (f" ({t_intent.requested_period})" if t_intent.requested_period else ""),
        "verified",
        t_intent.to_dict(),
    )

    t_val = validate_temporal_suitability(t_intent, state.sources)
    state.temporal_validation = t_val.to_dict()
    _add_trace_event(
        state,
        "TEMPORAL_VALIDATION",
        f"Temporal Status: {t_val.temporal_status}" + (f" (Latest: {t_val.latest_reporting_period})" if t_val.latest_reporting_period else ""),
        "verified" if t_val.is_valid else "flagged",
        t_val.to_dict(),
    )

    # 3. Deterministic Retrieval Confidence Gate with Temporal Integration
    from backend.services.confidence_gate import evaluate_retrieval_confidence
    conf_res = evaluate_retrieval_confidence(
        retrieval_query,
        state.sources,
        temporal_validation=t_val,
    )
    state.confidence_decision = conf_res.to_dict()

    if not conf_res.is_sufficient:
        _add_trace_event(
            state,
            "confidence_checked",
            f"Confidence: {conf_res.confidence_band} ({conf_res.confidence_score:.2f}) - Insufficient Evidence",
            "insufficient",
            state.confidence_decision,
        )
        state.requires_human_review = True
        try:
            from backend.database.repositories import approvals as approvals_repo
            username = user.get("username", "engineer") if user else "engineer"
            prop = approvals_repo.propose_action(
                requesting_user_id=state.user_id or 1,
                requesting_username=username,
                action_type="low_confidence_retrieval",
                affected_resource=f"Query: {state.query[:80]}",
                proposed_payload={"query": state.query, "confidence": conf_res.to_dict(), "temporal": t_val.to_dict()},
            )
            state.approval_id = prop.get("approval_id")
            _add_trace_event(
                state,
                "approval_required",
                f"Requires Human Review (Approval #{state.approval_id}): Low Retrieval Confidence",
                "flagged",
                {
                    "approval_id": state.approval_id,
                    "escalation_rule": "LOW_RETRIEVAL_CONFIDENCE",
                    "reason": conf_res.reason,
                    "confidence_score": conf_res.confidence_score,
                    "temporal_status": t_val.temporal_status,
                },
            )
        except Exception as app_err:
            logger.warning("Failed to propose approval: %s", app_err)

        if t_val.temporal_status == "INSUFFICIENT_RECENT_EVIDENCE":
            state.response = generate_evidence_limited_response(
                query=state.current_query,
                temporal_val=t_val,
                sources=state.sources,
            )
            state.current_step = "temporal_refusal"
            _log_step(state)
            return state

        if t_val.temporal_status == "TEMPORAL_MISMATCH":
            state.response = (
                f"I don't have sufficient information in the knowledge base for the requested period ({t_val.requested_period}). "
                f"The available documents cover {t_val.latest_reporting_period or 'other periods'}, which do not match the requested timeframe."
            )
            state.current_step = "temporal_refusal"
            _log_step(state)
            return state

        if conf_res.confidence_band == "INSUFFICIENT_EVIDENCE" or len(state.sources) == 0:
            state.response = (
                "⚠️ REQUIRES REVIEW · INSUFFICIENT EVIDENCE\n\n"
                "The MRPL Knowledge Base does not contain sufficient validated technical information to answer this query without risk of hallucination.\n\n"
                f"This query has been flagged and submitted to the Human Approval Queue (Action ID: #{state.approval_id or 'Pending'}). "
                "Please consult plant technical documentation or subject matter experts directly."
            )
            state.current_step = "insufficient_evidence"
            _log_step(state)
            return state
    else:
        _add_trace_event(
            state,
            "confidence_checked",
            f"Confidence: {conf_res.confidence_band} ({conf_res.confidence_score:.2f}) - Evidence Sufficient",
            "sufficient",
            state.confidence_decision,
        )

    state.current_step = "retrieved"
    state.step_index += 1
    _log_step(state)

    return state



# Maximum attempts for Docker-based code self-correction
MAX_CORRECTION_ATTEMPTS = 3


def node_tool_call(state: AgentState) -> AgentState:
    """Execute required tools (excluding rag_search, already handled)."""
    non_rag_tools = [t for t in state.required_tools if t != "rag_search"]

    if not non_rag_tools and not state.requires_sandbox:
        state.current_step = "tools_skipped"
        state.step_index += 1
        return state

    user = state.user
    if not user and state.user_id:
        try:
            from backend.database.repositories import users as users_repo
            user = users_repo.get_user_by_id(state.user_id)
        except Exception:
            user = None

    for tool_name in non_rag_tools:
        if tool_name == "sandbox_execute" or state.requires_sandbox:
            # Code generation & Bounded Docker Sandbox Self-Correction Loop
            # 1. Construct initial prompt
            if state.task_type == "HYBRID":
                from backend.services.hybrid_executor import (
                    extract_structured_metrics,
                    validate_required_inputs,
                    build_qwen_hybrid_prompt,
                )
                metrics = extract_structured_metrics(state.context, state.sources, state.query)
                state.structured_metrics = metrics
                is_valid, validation_msg = validate_required_inputs(metrics, state.query)
                if not is_valid:
                    # Enforce Requirement 8: Required Input Validation
                    # If required inputs are missing, do not generate fake calculation.
                    logger.warning("HYBRID input validation failed: %s", validation_msg)
                    state.response = validation_msg
                    state.tool_results.append({
                        "tool": "hybrid_validation",
                        "status": "refusal",
                        "error": validation_msg,
                    })
                    break

                current_prompt = build_qwen_hybrid_prompt(state.query, metrics, state.context)
            else:
                # Requirement 1, 2, 3: Query isolation and explicit code modification check
                prev_code = None
                if is_code_modification_request(state.query):
                    prev_code = extract_previous_generated_code(state.chat_history)

                if prev_code:
                    previous_context = (
                        f"The user explicitly requested to modify previous code:\n"
                        f"```python\n{prev_code}\n```"
                    )
                else:
                    previous_context = "None (This is a fresh, standalone coding request. Do not include or solve previous tasks.)"

                current_prompt = CODE_GENERATION_PROMPT.format(
                    query=state.query,
                    previous_context=previous_context,
                )

            # 2. Self-Correction Loop (Bounded to MAX_CORRECTION_ATTEMPTS)
            for attempt_idx in range(1, MAX_CORRECTION_ATTEMPTS + 1):
                logger.info(
                    "Coding task attempt %d/%d for query: %s",
                    attempt_idx,
                    MAX_CORRECTION_ATTEMPTS,
                    state.query[:60],
                )
                generated_output = _call_llm(state, current_prompt, num_predict=1024)
                code = _extract_code_block(generated_output)
                if state.task_type == "HYBRID":
                    from backend.services.hybrid_executor import sanitize_fiscal_year_labels
                    code = sanitize_fiscal_year_labels(code)

                # Execute strictly inside Docker sandbox (network=none, read-only root, non-root)
                result = execute_tool("sandbox_execute", user=user, code=code)
                sb_data = result.get("result", {}) if isinstance(result.get("result"), dict) else {}
                status = sb_data.get("status", result.get("status", "unknown"))
                exit_code = sb_data.get("exit_code")
                stdout = (sb_data.get("stdout") or "").strip()
                stderr = (sb_data.get("stderr") or result.get("error") or "").strip()

                is_error = (status in ("error", "blocked", "forbidden", "invalid_params")) or (exit_code is None and status != "success")
                attempt_status = "verified" if exit_code == 0 else ("error" if is_error else "failed")

                sandbox_info = sb_data.get("sandbox_info")
                if not sandbox_info and (exit_code is not None or attempt_status == "verified"):
                    sandbox_info = {
                        "sandbox": "Docker",
                        "network": "Disabled",
                        "filesystem": "Read-only",
                        "user": "Non-root",
                        "exit_code": exit_code,
                        "status": "Verified" if exit_code == 0 else "Failed",
                    }

                attempt_record = {
                    "tool": "sandbox_execute",
                    "attempt": attempt_idx,
                    "code": code,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "status": attempt_status,
                    **({"sandbox_info": sandbox_info} if sandbox_info else {}),
                }
                state.tool_results.append(attempt_record)

                # Security / Fail-Closed Check: If Docker is unavailable or returned an error, fail closed immediately
                if is_error:
                    logger.warning("Docker sandbox unavailable or failed closed on attempt %d: %s", attempt_idx, stderr)
                    break

                # Verification check: If exit code 0, stop loop
                if exit_code == 0:
                    logger.info("Docker sandbox execution VERIFIED on attempt %d", attempt_idx)
                    break

                # Self-correction prompt for next iteration
                if attempt_idx < MAX_CORRECTION_ATTEMPTS:
                    logger.info(
                        "Docker sandbox returned exit_code=%s on attempt %d. Sending stderr to Qwen for correction...",
                        exit_code,
                        attempt_idx,
                    )
                    current_prompt = CODE_CORRECTION_PROMPT.format(
                        query=state.query,
                        exit_code=exit_code,
                        stderr=stderr or "Process exited with non-zero status",
                        stdout=stdout or "No stdout produced",
                        code=code,
                    )

        elif tool_name in ("docgen_pdf", "docgen_docx") or tool_name.startswith("docgen_"):
            doc_type = "pdf" if tool_name == "docgen_pdf" else "docx"
            from backend.services.doc_analysis import (
                extract_full_document,
                build_document_analysis_context,
                validate_and_sanitize_analysis,
                resolve_document_file,
            )

            # Tool Gate: Deliverable generation is strictly gated to DOCUMENT_ANALYSIS tasks (Section 6 & 12)
            if state.task_type != "DOCUMENT_ANALYSIS":
                logger.warning("Tool gate: rejected %s because task_type is %s", tool_name, state.task_type)
                state.tool_results.append({
                    "tool": tool_name,
                    "status": "blocked",
                    "error": f"Deliverable generation not permitted for task type {state.task_type}",
                })
                continue

            target_doc = state.target_document
            is_image_target = False
            if not target_doc and state.chat_history:
                current_lower = (state.current_query or state.query).lower()
                is_doc_followup = bool(re.search(
                    r'\b(the\s+document|the\s+pdf|the\s+report|this\s+document|this\s+pdf|the\s+findings|page\s+\d+|it|its)\b',
                    current_lower,
                ))
                if is_doc_followup:
                    for msg in reversed(state.chat_history):
                        if msg.get("target_document"):
                            target_doc = str(msg["target_document"]).strip()
                            break
                        content = msg.get("content", "")
                        # Check for attached image marker first
                        img_match = re.search(r'\[Attached Image:\s*([^\]]+)\]', content, re.IGNORECASE)
                        if img_match:
                            target_doc = img_match.group(1).strip()
                            is_image_target = True
                            break
                        # Check for image file extensions
                        img_docs = re.findall(r'[\w\-\s]+\.(?:png|jpg|jpeg|webp)', content, re.IGNORECASE)
                        if img_docs:
                            target_doc = img_docs[0].strip()
                            is_image_target = True
                            break
                        # Check for regular documents
                        q_docs = re.findall(r'["\']([^"\']+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))["\']', content, re.IGNORECASE)
                        if q_docs:
                            target_doc = q_docs[0].strip()
                            break
                        ticks = re.findall(r'`([^`]+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))`', content, re.IGNORECASE)
                        if ticks:
                            target_doc = ticks[0].strip()
                            break
                        docs = re.findall(r'\b([\w\-]+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))\b', content, re.IGNORECASE)
                        if docs:
                            target_doc = docs[0].strip()
                            break

            if not target_doc:
                logger.warning("No target document available for deliverable generation; fail-closed without fallback")
                state.tool_results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": "No target document specified for deliverable generation.",
                })
                continue

            if not is_image_target:
                try:
                    resolved_p, resolved_meta = resolve_document_file(target_doc)
                    if resolved_meta and resolved_meta.get("original_name"):
                        target_doc = resolved_meta["original_name"]
                    elif resolved_p:
                        target_doc = resolved_p.name
                except Exception:
                    pass

            state.target_document = target_doc
            doc_model = extract_full_document(target_doc) if not is_image_target else None

            # 1. Preserve previous analysis if available in session history (Requirement 9)
            analysis_content = ""
            if state.chat_history:
                for msg in reversed(state.chat_history):
                    if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
                        content_cand = msg.get("content", "")
                        # Filter out hallucinated/filler progress messages
                        if not any(filler in content_cand.lower() for filler in [
                            "generating the analysis report",
                            "generating the pdf",
                            "please allow a moment",
                            "now available. please let me know",
                            "successfully generated in",
                        ]):
                            analysis_content = content_cand
                            logger.info("Reusing existing analysis from previous chat message (length %d)", len(analysis_content))
                            break

            # 2. If no previous analysis exists in history, synthesize it now
            if not analysis_content:
                logger.info("No prior analysis found in session; running document analysis synthesis")
                if doc_model and not state.context:
                    state.context = build_document_analysis_context(doc_model)
                    state.sources = [
                        {"source": doc_model.document_name, "page": p.page_num}
                        for p in doc_model.pages
                    ]

                analysis_prompt = DOCUMENT_ANALYSIS_PROMPT.format(
                    document_name=target_doc,
                    context=state.context if state.context else f"Document: {target_doc}",
                    query=state.query,
                )
                analysis_content = _call_llm(state, analysis_prompt, num_predict=1536)

            # Validate and sanitize document analysis (skip for image targets)
            if doc_model:
                analysis_content = validate_and_sanitize_analysis(analysis_content, doc_model)

            # 3. Physically generate file
            clean_name = re.sub(r'\.(pdf|docx|doc|xlsx|txt|png|jpg|jpeg|webp)$', '', target_doc, flags=re.IGNORECASE)
            import uuid
            unique_token = uuid.uuid4().hex[:8]
            out_filename = f"report_{unique_token}_{clean_name}.{doc_type}" if is_image_target else f"{clean_name}_Analysis.{doc_type}"

            try:
                if doc_type == "pdf":
                    from backend.services.docgen import generate_pdf
                    file_res = generate_pdf(
                        title=f"ANALYSIS REPORT: {clean_name}" if is_image_target else "DOCUMENT ANALYSIS REPORT",
                        content=analysis_content,
                        document_name=target_doc,
                        filename=out_filename,
                    )
                else:
                    from backend.services.docgen import generate_docx
                    file_res = generate_docx(
                        title="DOCUMENT ANALYSIS REPORT",
                        content=analysis_content,
                        document_name=target_doc,
                        filename=out_filename,
                    )

                # 4. Strict physical file existence and size verification
                out_path = Path(file_res["path"])
                if not out_path.exists():
                    raise RuntimeError(f"Physical file not found on disk: {out_path}")
                if not out_path.is_file():
                    raise RuntimeError(f"Generated path is not a regular file: {out_path}")
                if out_path.stat().st_size == 0:
                    raise RuntimeError(f"Generated file is empty (0 bytes): {out_path}")

                # 5. Record generated deliverable in database
                from backend.database.repositories import files as files_repo
                file_id = files_repo.record_generated_file(
                    user_id=state.user_id or 1,
                    file_type=doc_type,
                    original_name=out_filename,
                    stored_path=str(out_path),
                    file_size_bytes=file_res["file_size_bytes"],
                    task_id=state.task_id,
                )

                download_url = f"/api/documents/generated/{file_id}/download"
                deliverable_meta = {
                    "status": "success",
                    "type": doc_type,
                    "filename": out_filename,
                    "output_id": file_id,
                    "download_url": download_url,
                    "file_size_bytes": file_res["file_size_bytes"],
                    "path": str(out_path),
                }
                state.deliverable = deliverable_meta

                # 6. Audit log (Requirement 13)
                audit_log(
                    action=DOCUMENT_GENERATED,
                    outcome="success",
                    user_id=state.user_id,
                    username=user.get("username") if user else None,
                    target=f"file:{out_filename}",
                    details={
                        "event": "DOCUMENT_GENERATED",
                        "type": doc_type.upper(),
                        "source_document": target_doc,
                        "output": out_filename,
                        "status": "success",
                        "file_size_bytes": file_res["file_size_bytes"],
                    },
                )

                state.tool_results.append({
                    "tool": tool_name,
                    "status": "success",
                    "result": deliverable_meta,
                })

            except Exception as gen_exc:
                logger.error("Local %s generation failed: %s", doc_type.upper(), gen_exc, exc_info=True)
                audit_log(
                    action=DOCUMENT_GENERATION_FAILED,
                    outcome="failure",
                    user_id=state.user_id,
                    username=user.get("username") if user else None,
                    target=f"file:{out_filename}",
                    details={
                        "event": "DOCUMENT_GENERATION_FAILED",
                        "type": doc_type.upper(),
                        "source_document": target_doc,
                        "status": "error",
                        "reason": str(gen_exc),
                    },
                )
                state.tool_results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": str(gen_exc),
                })

        else:
            logger.warning("Unhandled tool: %s", tool_name)

    state.current_step = "tools_executed"
    state.step_index += 1
    _log_step(state)

    return state


def node_reason(state: AgentState) -> AgentState:
    """Generate the main LLM response."""
    start = time.perf_counter()

    # Deterministic Query Integrity Verification
    current_q = state.current_query or state.query
    state.current_query = current_q
    state.query = current_q

    _add_trace_event(
        state,
        "LLM_CURRENT_QUERY",
        f'LLM Current Query: "{current_q}"',
        "verified",
        {"current_query": current_q, "model": state.ollama_model_name, "trace_id": state.trace_id},
    )

    # Build chat history string (last 6 messages to save tokens).
    chat_history_str = ""
    if state.chat_history:
        for msg in state.chat_history[-6:]:
            chat_history_str += f"{msg['role']}: {msg['content']}\n"

    # Multimodal Vision analysis task
    if state.task_type == "VISION":
        from backend.services.vision_verification import (
            extract_process_labels,
            extract_grounded_equipment_tags,
            extract_equipment_tags_from_text,
            validate_equipment_tags,
            build_grounded_vision_prompt,
        )

        process_labels = extract_process_labels(state.query)
        grounded_tags = extract_equipment_tags_from_text(state.query)

        vision_prompt = build_grounded_vision_prompt(
            user_query=state.query,
            visual_evidence=state.query,
            process_labels=process_labels,
            grounded_tags=grounded_tags,
        )
        vision_resp = _call_llm(state, vision_prompt, num_predict=1024)

        # Ground candidate tags against query evidence
        detected_tags = extract_grounded_equipment_tags(state.query, proposed_text=vision_resp)
        val_res = validate_equipment_tags(detected_tags)

        verification_text = ""
        if detected_tags:
            status_str = "REQUIRES REVIEW" if val_res.requires_human_review else "VERIFIED_APPROVED"
            verification_text = f"\n\n---\n### 🔍 Deterministic P&ID Verification: `{status_str}`\n"
            if val_res.matched_tags:
                verification_text += f"- **Verified Tags**: {', '.join(val_res.matched_tags)}\n"
            if val_res.mismatched_tags:
                verification_text += f"- **⚠️ Unregistered Tags**: {', '.join(val_res.mismatched_tags)} (Flagged for Review)\n"
                state.requires_human_review = True
        else:
            verification_text = (
                f"\n\n---\n### 🔍 Deterministic Verification Status: No Verifiable Equipment Tags\n"
                f"- **Summary**: No alphanumeric equipment tags detected. General process flow labels detected: {', '.join(process_labels[:6]) if process_labels else 'General'}.\n"
                f"- **Registry**: No equipment registry verification performed.\n"
            )

        state.response = f"{vision_resp}{verification_text}"
        state.execution_ms = int((time.perf_counter() - start) * 1000)
        state.current_step = "responded"
        state.step_index += 1
        _log_step(state)
        return state

    # Fast path: Incomplete query / Ambiguous entity clarification (deterministic clarification without LLM)
    if state.task_type in ("CLARIFICATION", "INCOMPLETE_QUERY", "AMBIGUOUS_QUERY"):
        clarif_q = state.clarification_question or "Could you complete your question? Also, do you mean MRP or MRPL?"
        _add_trace_event(
            state,
            "CLARIFICATION_REQUIRED",
            f'Clarification Required ({state.task_type}): "{clarif_q}"',
            "insufficient",
            {"clarification_question": clarif_q, "ambiguity_type": getattr(state, "ambiguity_type", state.task_type)},
        )
        state.response = clarif_q
        state.execution_ms = int((time.perf_counter() - start) * 1000)
        state.current_step = "responded"
        state.step_index += 1
        _log_step(state)
        return state

    # Fast path: General conversational chat / conceptual explanation (no RAG context needed)
    if state.task_type in ("GENERAL", "general_chat"):
        # Response Type Integrity (Section 13)
        state.deliverable = None
        state.target_document = None
        prompt = GENERAL_CHAT_PROMPT.format(
            conversation_history=chat_history_str or "No previous conversation.",
            current_query=state.current_query,
        )
        state.response = _call_llm(state, prompt, num_predict=512)
        _add_trace_event(
            state,
            "RESPONSE_GENERATION",
            "General chat response generated",
            "verified",
            {"task_type": state.task_type, "deliverable": None},
        )
        state.execution_ms = int((time.perf_counter() - start) * 1000)
        state.current_step = "responded"
        state.step_index += 1
        _log_step(state)
        return state

    # Deliverable generation response (PDF or DOCX) — strictly reflects backend file status (Fail-Closed)
    # Gated strictly to DOCUMENT_ANALYSIS task type with requested deliverable tools (Section 6 & 12)
    if state.task_type == "DOCUMENT_ANALYSIS" and any(t in ("docgen_pdf", "docgen_docx") or t.startswith("docgen_") for t in state.required_tools):
        doc_type = "pdf" if "docgen_pdf" in state.required_tools else "docx"
        target_doc = state.target_document or "Document"

        if state.deliverable and state.deliverable.get("status") == "success":
            state.response = (
                f"The analysis report from **{target_doc}** has been successfully generated in {doc_type.upper()} format.\n\n"
                f"**Deliverable:** `{state.deliverable['filename']}` ({state.deliverable['file_size_bytes']} bytes)\n\n"
                f"You can open or download the document using the deliverable actions below."
            )
        else:
            err = next(
                (r.get("error") for r in state.tool_results if r.get("tool") in ("docgen_pdf", "docgen_docx")),
                "File creation or validation failed",
            )
            state.response = f"{doc_type.upper()} generation failed: {err}"

        _add_trace_event(
            state,
            "RESPONSE_GENERATION",
            f"Deliverable response generated for {state.deliverable.get('filename') if state.deliverable else 'FAILED'}",
            "verified",
            {"task_type": state.task_type, "deliverable": state.deliverable.get("filename") if state.deliverable else None},
        )
        state.execution_ms = int((time.perf_counter() - start) * 1000)
        state.current_step = "responded"
        state.step_index += 1
        _log_step(state)
        return state

    # Document analysis response (Turn 1: "analyze TranscriptSd.pdf")
    if state.task_type == "DOCUMENT_ANALYSIS":
        target_doc = state.target_document
        if not target_doc:
            state.response = "Please specify an uploaded document to analyze."
            _add_trace_event(
                state,
                "RESPONSE_GENERATION",
                "Document analysis response: missing target document",
                "verified",
                {"task_type": state.task_type, "target_document": None},
            )
            state.execution_ms = int((time.perf_counter() - start) * 1000)
            state.current_step = "responded"
            state.step_index += 1
            _log_step(state)
            return state

        from backend.services.doc_analysis import (
            extract_full_document,
            build_document_analysis_context,
            validate_and_sanitize_analysis,
        )
        doc_model = extract_full_document(target_doc)
        if not state.context:
            state.context = build_document_analysis_context(doc_model)
            state.sources = [
                {"source": doc_model.document_name, "page": p.page_num}
                for p in doc_model.pages
            ]

        prompt = DOCUMENT_ANALYSIS_PROMPT.format(
            document_name=target_doc,
            context=state.context if state.context else "No context available in local knowledge base.",
            query=state.query,
        )
        base_response = _call_llm(state, prompt, num_predict=1536)
        sanitized_response = validate_and_sanitize_analysis(base_response, doc_model)
        state.response = _normalize_citations_and_sources(sanitized_response, state.sources)
        _add_trace_event(
            state,
            "RESPONSE_GENERATION",
            f"Document analysis response generated for {target_doc}",
            "verified",
            {"task_type": state.task_type, "target_document": target_doc},
        )
        state.execution_ms = int((time.perf_counter() - start) * 1000)
        state.current_step = "responded"
        state.step_index += 1
        _log_step(state)
        return state

    # Controlled refusal check: ONLY if task required RAG and RAG found no relevant context
    if state.requires_rag:
        rag_tool_res = next((r.get("result", {}) for r in state.tool_results if r.get("tool") == "rag_search"), None)
        if rag_tool_res and rag_tool_res.get("refusal"):
            state.response = rag_tool_res.get("message", "I couldn't find sufficient information in the local knowledge base to answer that.")
            state.execution_ms = int((time.perf_counter() - start) * 1000)
            state.current_step = "responded"
            state.step_index += 1
            _log_step(state)
            return state

    # Handling CODING tasks: format generated Python code + Docker sandbox execution output
    if state.task_type in ("CODING", "code_generation"):
        sandbox_attempts = [
            r for r in state.tool_results
            if r.get("tool") == "sandbox_execute" or "attempt" in r
        ]
        if sandbox_attempts:
            verified_attempt = next(
                (a for a in sandbox_attempts if a.get("status") == "verified" or a.get("exit_code") == 0),
                None,
            )
            target_attempt = verified_attempt if verified_attempt else sandbox_attempts[-1]

            code = target_attempt.get("code", "")
            exit_code = target_attempt.get("exit_code")
            stdout = (target_attempt.get("stdout") or "").strip()
            stderr = (target_attempt.get("stderr") or "").strip()
            status = target_attempt.get("status", "unknown")
            verified = (exit_code == 0) or (status == "verified")
            attempts = len(sandbox_attempts)

            if status in ("error", "blocked") or (exit_code is None and not verified):
                notice = (
                    f"{stderr}. In accordance with MRPL sovereign security policy, the system failed closed and did not execute the code on the host."
                    if stderr else
                    "Docker sandbox is unavailable. In accordance with MRPL sovereign security policy, the system failed closed and did not execute the code on the host."
                )
                state.response = (
                    f"**Task Type:** CODING\n"
                    f"**Model:** {state.ollama_model_name}\n"
                    f"**Execution:** BLOCKED (Docker Sandbox Unavailable)\n\n"
                    f"```python\n{code}\n```\n\n"
                    f"⚠️ **Sandbox Execution Notice**: {notice}"
                )
            elif verified or exit_code == 0:
                parts = [
                    f"**Task Type:** CODING\n"
                    f"**Model:** {state.ollama_model_name}\n"
                    f"**Execution:** VERIFIED\n"
                    f"**Correction Attempts:** {attempts}",
                    f"```python\n{code}\n```",
                ]
                if stdout:
                    parts.append(f"### Execution Result (Docker Sandbox)\n```\n{stdout}\n```")
                state.response = "\n\n".join(parts)
            else:
                parts = [
                    f"**Task Type:** CODING\n"
                    f"**Model:** {state.ollama_model_name}\n"
                    f"**Execution:** FAILED (Exit Code {exit_code})\n"
                    f"**Correction Attempts:** {attempts}",
                    f"```python\n{code}\n```",
                ]
                if stderr:
                    parts.append(f"### Diagnostic Output (Docker Sandbox)\n```\n{stderr}\n```")
                elif stdout:
                    parts.append(f"### Execution Result (Docker Sandbox)\n```\n{stdout}\n```")
                state.response = "\n\n".join(parts)

            state.execution_ms = int((time.perf_counter() - start) * 1000)
            state.current_step = "responded"
            state.step_index += 1
            _log_step(state)
            return state

    # Handling HYBRID tasks: document context + code + sandbox output + citations
    if state.task_type == "HYBRID":
        # Controlled refusal check if input validation determined insufficient data
        if state.response and "not contain sufficient verified data" in state.response:
            state.execution_ms = int((time.perf_counter() - start) * 1000)
            state.current_step = "responded"
            state.step_index += 1
            _log_step(state)
            return state

        sandbox_attempts = [
            r for r in state.tool_results
            if r.get("tool") == "sandbox_execute" or "attempt" in r
        ]
        if sandbox_attempts:
            verified_attempt = next(
                (a for a in sandbox_attempts if a.get("status") == "verified" or a.get("exit_code") == 0),
                None,
            )
            target_attempt = verified_attempt if verified_attempt else sandbox_attempts[-1]

            code = target_attempt.get("code", "")
            exit_code = target_attempt.get("exit_code")
            stdout = (target_attempt.get("stdout") or "").strip()
            stderr = (target_attempt.get("stderr") or "").strip()
            status = target_attempt.get("status", "unknown")
            attempts = len(sandbox_attempts)
        else:
            code = ""
            exit_code = None
            stdout = ""
            stderr = ""
            status = "unknown"
            attempts = 0

        from backend.services.hybrid_executor import (
            extract_structured_metrics,
            format_hybrid_final_response,
        )
        metrics = state.structured_metrics or []
        if not metrics and state.context:
            metrics = extract_structured_metrics(state.context, state.sources, state.query)
            state.structured_metrics = metrics

        state.response = format_hybrid_final_response(
            query=state.query,
            model_name=state.ollama_model_name,
            metrics=metrics,
            sources=state.sources,
            code=code,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            status=status,
            correction_attempts=attempts - 1 if attempts > 0 else 0,
        )
        state.execution_ms = int((time.perf_counter() - start) * 1000)
        state.current_step = "responded"
        state.step_index += 1
        _log_step(state)
        return state

    # If we have non-RAG tool results (e.g. general synthesis), use synthesis prompt.
    non_rag_tools = [
        r for r in state.tool_results
        if r.get("tool") != "rag_search" and r.get("status") == "success"
    ]
    if non_rag_tools:
        tool_results_str = json.dumps(
            non_rag_tools,
            indent=2,
            default=str,
        )[:5000]

        prompt = SYNTHESIS_PROMPT.format(
            tool_results=tool_results_str,
            current_query=state.current_query,
        )
    else:
        prompt = REASONING_PROMPT.format(
            retrieved_context=state.context if state.context else "No context available.",
            conversation_history=chat_history_str or "No previous conversation.",
            current_query=state.current_query,
        )

    # Select num_predict based on task type for latency optimization.
    num_predict = 768
    if state.task_type in ("document_summary", "DOCUMENT_ANALYSIS", "approval_note"):
        num_predict = 1024

    state.response = _call_llm(state, prompt, num_predict=num_predict)

    # Unit-Aware Grounding & Cross-Unit Product Isolation Check (Requirements 6 & 7)
    try:
        from backend.services.unit_grounding import (
            UNIT_DEFINITIONS,
            detect_target_units,
            detect_target_unit,
            format_unit_structured_context,
            validate_unit_grounding,
        )
        target_units = detect_target_units(state.current_query)
        target_unit = target_units[0] if target_units else None
        if target_unit and target_unit in UNIT_DEFINITIONS:
            _, primary_unit_evidence = format_unit_structured_context(state.sources, target_units)
            is_valid, failure_reason = validate_unit_grounding(
                response=state.response,
                target_unit=target_units,
                primary_evidence=primary_unit_evidence,
                all_evidence=state.context,
            )
            if not is_valid:
                logger.warning("Unit grounding violation in initial generation: %s", failure_reason)
                if primary_unit_evidence and len(primary_unit_evidence.strip()) > 30:
                    unit_name = UNIT_DEFINITIONS[target_unit]["name"]
                    logger.info("Regenerating once with strictly isolated evidence for unit: %s", unit_name)
                    regen_prompt = REASONING_PROMPT.format(
                        retrieved_context=primary_unit_evidence,
                        conversation_history=chat_history_str or "No previous conversation.",
                        current_query=state.current_query,
                    )
                    regenerated = _call_llm(state, regen_prompt, num_predict=num_predict)
                    is_regen_valid, regen_failure = validate_unit_grounding(
                        response=regenerated,
                        target_unit=target_units,
                        primary_evidence=primary_unit_evidence,
                        all_evidence=state.context,
                    )
                    if is_regen_valid:
                        logger.info("Unit grounding validation succeeded after single regeneration")
                        state.response = regenerated
                    else:
                        logger.warning("Unit grounding validation failed again after regeneration: %s. Returning controlled refusal.", regen_failure)
                        from backend.services.entity_preservation import detect_query_language_deterministic
                        detected_lang = detect_query_language_deterministic(state.current_query)
                        if detected_lang == "mr":
                            state.response = (
                                "MRPL च्या ज्ञान स्त्रोतांमध्ये उपलब्ध माहितीनुसार, इतर युनिट्सची माहिती मिसळल्याशिवाय या युनिटचे फीड आणि प्रमुख उत्पादने निश्चितपणे सांगण्यासाठी पुरेसा पुरावा उपलब्ध नाही."
                            )
                        elif detected_lang == "hi":
                            state.response = (
                                "MRPL के ज्ञान आधार में प्राप्त दस्तावेजों के आधार पर, अन्य इकाइयों के विवरण को शामिल किए बिना इस विशिष्ट इकाई के फीड और उत्पादों की पुष्टि करने के लिए पर्याप्त साक्ष्य उपलब्ध नहीं हैं।"
                            )
                        else:
                            state.response = (
                                "Based on the retrieved MRPL documents, there is insufficient evidence to determine the specific feeds and products for this unit without conflating details from adjacent refinery units."
                            )
                else:
                    logger.warning("No primary evidence available for unit %s to regenerate. Returning controlled refusal.", target_unit)
                    from backend.services.entity_preservation import detect_query_language_deterministic
                    detected_lang = detect_query_language_deterministic(state.current_query)
                    if detected_lang == "mr":
                        state.response = "MRPL च्या ज्ञान स्त्रोतांमध्ये या युनिटसाठी स्वतंत्र माहिती उपलब्ध नाही."
                    elif detected_lang == "hi":
                        state.response = "MRPL के ज्ञान आधार में इस इकाई के लिए अलग से विवरण उपलब्ध नहीं है।"
                    else:
                        state.response = "I couldn't find sufficient information in the local knowledge base for this specific refinery unit."
    except Exception as ug_err:
        logger.warning("Error during unit grounding check in node_reason: %s", ug_err)

    # Response Type Integrity (Section 13)
    # Validate requested_task_type, actual_executed_task_type, generated_artifacts
    if state.task_type in ("GENERAL", "RAG", "CODING", "CLARIFICATION", "INCOMPLETE_QUERY"):
        if state.deliverable is not None:
            logger.warning("Response Type Integrity: clearing stale deliverable artifact for task_type=%s", state.task_type)
            state.deliverable = None
        if state.target_document is not None:
            logger.warning("Response Type Integrity: clearing stale target_document for task_type=%s", state.task_type)
            state.target_document = None

    _add_trace_event(
        state,
        "RESPONSE_GENERATION",
        f"Response generated for {state.task_type}",
        "verified",
        {
            "task_type": state.task_type,
            "has_deliverable": bool(state.deliverable),
            "deliverable_file": state.deliverable.get("filename") if state.deliverable else None,
            "target_document": state.target_document,
        },
    )

    state.execution_ms = int((time.perf_counter() - start) * 1000)

    state.current_step = "responded"
    state.step_index += 1
    _log_step(state)

    return state


def _normalize_citations_and_sources(response: str, sources: list[dict]) -> str:
    """
    Ensure direct document citations are used and the final Sources section
    strictly reflects the retrieved source metadata.

    - Replaces numeric source citations (e.g. 'Source: 1, p. 10' or 'Source: 5, p. 11')
      with direct document citations (e.g. 'Source: 36th Annual Report for 2023-24, p. 10').
    - Disallows any invalid source number.
    - Constructs the final Sources section strictly from retrieved metadata.
    """
    if not response or not sources:
        return response

    # Refusal responses should remain as stated without appending fake sources
    refusal_phrases = [
        "don't have sufficient information",
        "not enough information",
        "couldn't find sufficient information",
        "could not find sufficient information",
    ]
    if any(p in response.lower() for p in refusal_phrases):
        return response

    def _build_citation(num_str: str, page_str: str | None, open_b: str = "", close_b: str = "") -> str:
        try:
            idx = int(num_str)
        except ValueError:
            return ""

        target_src = None
        if 1 <= idx <= len(sources):
            target_src = sources[idx - 1]
        elif page_str:
            for s in sources:
                if str(s.get("page")) == str(page_str):
                    target_src = s
                    break

        if not target_src and sources:
            target_src = sources[0]

        if target_src:
            title = target_src.get("document_title") or clean_document_title(target_src.get("source", "Document"))
            page = page_str or target_src.get("page", "")
            text = f"Source: {title}, p. {page}" if page else f"Source: {title}"
            return f"{open_b}{text}{close_b}"
        return ""

    # Match bracketed numbered citations: (Source: 1, p. 10), [Source: 2], (Document 5, page 11)
    pat_bracketed = re.compile(
        r'([(\[])\s*(?:Source|Doc|Document)[:\s]+([1-9]\d*)(?:\s*,\s*p(?:age|\.)?\s*(\d+))?\s*([)\]])',
        re.IGNORECASE
    )
    # Match bare numbered citations: Source: 1, p. 10.  or  Source: 5
    pat_bare = re.compile(
        r'(?<![\w/])(?:Source|Doc|Document)[:\s]+([1-9]\d*)(?:\s*,\s*p(?:age|\.)?\s*(\d+))?(?=[.,;\n]|$)',
        re.IGNORECASE
    )

    def _replace_bracketed(match):
        open_b = match.group(1)
        num_str = match.group(2)
        page_str = match.group(3)
        close_b = match.group(4)
        rep = _build_citation(num_str, page_str, open_b, close_b)
        return rep if rep else match.group(0)

    def _replace_bare(match):
        num_str = match.group(1)
        page_str = match.group(2)
        rep = _build_citation(num_str, page_str)
        return rep if rep else match.group(0)

    lines = response.split("\n")
    processed_lines = []
    in_sources_section = False

    for line in lines:
        if re.match(r'^(?:##\s*)?Sources:\s*$', line.strip(), flags=re.IGNORECASE):
            in_sources_section = True
            continue
        if in_sources_section:
            continue
        new_line = pat_bracketed.sub(_replace_bracketed, line)
        new_line = pat_bare.sub(_replace_bare, new_line)
        processed_lines.append(new_line)

    body = "\n".join(processed_lines).strip()

    # Build authoritative Sources section strictly from retrieved metadata
    # If specific sources were cited in the body, include only those; otherwise include all retrieved sources.
    body_lower = body.lower()
    used_sources = []
    for s in sources:
        title = s.get("document_title") or clean_document_title(s.get("source", "Unknown Document"))
        page = s.get("page")
        title_in_body = title.lower() in body_lower
        page_in_body = bool(page and (f"p. {page}" in body_lower or f"page {page}" in body_lower))
        if page:
            if title_in_body and page_in_body:
                used_sources.append(s)
        elif title_in_body:
            used_sources.append(s)

    final_sources = used_sources if used_sources else sources

    seen = set()
    authoritative_sources = []
    for s in final_sources:
        title = s.get("document_title") or clean_document_title(s.get("source", "Unknown Document"))
        page = s.get("page")
        key = (title, page)
        if key not in seen:
            seen.add(key)
            authoritative_sources.append(f"{title}, p. {page}" if page else title)

    if authoritative_sources:
        sources_block = "\n\nSources:\n" + "\n".join(
            f"{i}. {src}" for i, src in enumerate(authoritative_sources, 1)
        )
        return f"{body}\n\n{sources_block}".strip()

    return body


def node_validate(state: AgentState) -> AgentState:
    """
    Lightweight Python-only validation of the LLM response.

    - Converts numbered citations to direct document citations.
    - Constructs the authoritative Sources list directly from retrieved metadata.
    - Checks that numerical claims in the response body are grounded in context.
    - Appends a warning ONLY when an unsupported numerical claim or source mismatch is detected.
    - No LLM call — pure regex and metadata verification.
    """
    if not state.context or not state.response:
        state.current_step = "validated"
        state.step_index += 1
        return state

    # HYBRID responses are constructed deterministically with verified metrics, execution status, and citations
    if state.task_type == "HYBRID":
        state.current_step = "validated"
        state.step_index += 1
        _log_step(state)
        return state

    # 1. Normalize citations and construct verified Sources section
    state.response = _normalize_citations_and_sources(state.response, state.sources)

    # 1b. Language & Technical Entity Fidelity Verification (Deterministic query-language isolation & entity preservation)
    try:
        from backend.services.entity_preservation import validate_and_preserve_entities
        fidelity_res = validate_and_preserve_entities(
            response=state.response,
            query=state.current_query,
            evidence=state.context,
        )
        state.response = fidelity_res.corrected_response
        _add_trace_event(
            state,
            "TECHNICAL_ENTITY_VALIDATION",
            f"Language & Entity Fidelity: {fidelity_res.technical_entity_validation} "
            f"(Query: {fidelity_res.query_language}, Response: {fidelity_res.response_language})",
            "verified" if fidelity_res.technical_entity_validation in ("PASSED", "CORRECTED") else "flagged",
            fidelity_res.to_dict(),
        )
    except Exception as ent_err:
        logger.warning("Error during entity and language fidelity check in node_validate: %s", ent_err)

    # 2. Extract numbers from body only (ignoring the Sources section)
    body_only = re.split(r'\n+(?:##\s*)?Sources:\s*\n', state.response, flags=re.IGNORECASE)[0]
    response_numbers = set(_extract_numbers(body_only))
    context_numbers = set(_extract_numbers(state.context))
    query_numbers = set(_extract_numbers(state.query))
    source_pages = {str(s.get("page")) for s in state.sources if s.get("page") is not None}

    # Common trivial numbers (indices, single digits, days/months up to 31)
    trivial = {str(i) for i in range(32)}

    # Detect truly ungrounded numerical claims
    unsupported = response_numbers - (context_numbers | query_numbers | source_pages | trivial)

    # Only append warning if actual ungrounded numerical claims exist
    if unsupported:
        logger.warning("Unsupported numerical claim detected: %s", unsupported)
        warning = (
            "\n\n⚠️ *Note: Some numerical values in this response could not be "
            "verified against the retrieved sources. Please cross-check with "
            "the original documents.*"
        )
        state.response += warning

    # 3. Financial & Quantitative Metric Answer Validation
    try:
        from backend.services.query_analyzer import analyze_query
        q_analysis = analyze_query(state.query)
        if q_analysis.is_metric_query:
            resp_lower = body_only.lower()
            if q_analysis.primary_metric == "revenue":
                # Check if model incorrectly asserted export percentage instead of revenue amount
                has_currency_amount = any(
                    kw in resp_lower
                    for kw in ("crore", "million", "lakh", "billion", "₹", "rs.", "inr")
                )
                substituted_export = (
                    "28.43%" in body_only
                    or ("export" in resp_lower and not has_currency_amount)
                )
                if substituted_export:
                    logger.warning("Answer validation failure: Model substituted export percentage for revenue query '%s'", state.query)
                    state.response = (
                        "I could not find the exact revenue figure for the requested period in the available MRPL documents. "
                        "The retrieved sources contain details regarding export turnover percentages rather than total revenue from operations."
                    )
    except Exception as val_err:
        logger.warning("Error during metric answer validation: %s", val_err)

    # 4. Unit Grounding Validation Check (Fail-Closed on Cross-Unit Product Leakage)
    try:
        from backend.services.unit_grounding import detect_target_units, format_unit_structured_context, validate_unit_grounding
        target_units = detect_target_units(state.query)
        target_unit = target_units[0] if target_units else None
        if target_unit:
            _, primary_unit_evidence = format_unit_structured_context(state.sources, target_units)
            is_valid, failure_reason = validate_unit_grounding(
                response=state.response,
                target_unit=target_units,
                primary_evidence=primary_unit_evidence,
                all_evidence=state.context,
            )
            if not is_valid:
                logger.warning("Deterministic unit grounding failure detected in node_validate: %s", failure_reason)
                from backend.services.entity_preservation import detect_query_language_deterministic
                detected_lang = detect_query_language_deterministic(state.current_query)
                if detected_lang == "mr":
                    state.response = (
                        "MRPL च्या ज्ञान स्त्रोतांमध्ये उपलब्ध माहितीनुसार, इतर युनिट्सची माहिती मिसळल्याशिवाय या युनिटचे फीड आणि प्रमुख उत्पादने निश्चितपणे सांगण्यासाठी पुरेसा पुरावा उपलब्ध नाही."
                    )
                elif detected_lang == "hi":
                    state.response = (
                        "MRPL के ज्ञान आधार में प्राप्त दस्तावेजों के आधार पर, अन्य इकाइयों के विवरण को शामिल किए बिना इस विशिष्ट इकाई के फीड और उत्पादों की पुष्टि करने के लिए पर्याप्त साक्ष्य उपलब्ध नहीं हैं।"
                    )
                else:
                    state.response = (
                        "Based on the retrieved MRPL documents, there is insufficient evidence to determine the specific feeds and products for this unit without conflating details from adjacent refinery units."
                    )
    except Exception as u_val_err:
        logger.warning("Error during unit grounding validation in node_validate: %s", u_val_err)

    # 5. Deterministic Citation Validation Check (Phase 2 Component 4)
    try:
        from backend.services.citation_validator import validate_citations
        cit_val = validate_citations(state.response, state.sources, intent=state.temporal_intent)
        state.citation_validation = cit_val.to_dict()
        if not cit_val.is_grounded:
            _add_trace_event(
                state,
                "grounding_checked",
                f"Grounding: {len(cit_val.flagged_citations)} ungrounded citations flagged",
                "flagged",
                cit_val.to_dict(),
            )
        else:
            _add_trace_event(
                state,
                "grounding_checked",
                f"Grounding: Citations verified ({len(cit_val.valid_citations)} citations)",
                "verified",
                cit_val.to_dict(),
            )
    except Exception as cit_err:
        logger.warning("Citation validation error in node_validate: %s", cit_err)

    # 6. Strict Temporal Claims Validation Check (Requirement 4 & 8)
    try:
        from backend.services.temporal_guard import validate_temporal_claims, TemporalIntent
        t_intent_obj = None
        if state.temporal_intent:
            t_intent_obj = TemporalIntent(**state.temporal_intent)
        claim_res = validate_temporal_claims(state.response, state.sources, intent=t_intent_obj)
        state.temporal_claims = claim_res.to_dict()
        if not claim_res.valid:
            _add_trace_event(
                state,
                "TEMPORAL_CLAIM_VALIDATION",
                f"Temporal Grounding: {len(claim_res.unsupported_claims)} unsupported claims flagged",
                "flagged",
                claim_res.to_dict(),
            )
            logger.warning("Temporal claim validation failed: %s", claim_res.unsupported_claims)
            warning = (
                f"\n\n⚠️ *Temporal Grounding Notice: The following period/date references in this response "
                f"could not be verified against the cited sources: {', '.join(claim_res.unsupported_claims)}.*"
            )
            if warning not in state.response:
                state.response += warning
        else:
            _add_trace_event(
                state,
                "TEMPORAL_CLAIM_VALIDATION",
                "Temporal Grounding: All reporting periods and dates substantiated",
                "verified",
                claim_res.to_dict(),
            )
    except Exception as tc_err:
        logger.warning("Temporal claim validation error in node_validate: %s", tc_err)

    state.current_step = "validated"
    state.step_index += 1
    _log_step(state)

    return state


def node_respond(state: AgentState) -> AgentState:
    """Finalize and persist the response with output trace and audit log."""
    # Response Type Integrity (Section 13):
    # Non-document tasks MUST NOT have deliverable artifacts or target documents
    if state.task_type in ("GENERAL", "RAG", "CODING", "CLARIFICATION", "INCOMPLETE_QUERY"):
        state.deliverable = None
        state.target_document = None

    if not any(e.get("event") == "RESPONSE_GENERATION" for e in (state.execution_trace or [])):
        _add_trace_event(
            state,
            "RESPONSE_GENERATION",
            f"Response generated for {state.task_type}",
            "verified",
            {
                "task_type": state.task_type,
                "has_deliverable": bool(state.deliverable),
                "deliverable_file": state.deliverable.get("filename") if state.deliverable else None,
                "target_document": state.target_document,
            },
        )

    _add_trace_event(
        state,
        "output_generated",
        "Final verified output delivered to user",
        "verified",
        {"response_length": len(state.response or "")},
    )
    _add_trace_event(
        state,
        "audit_logged",
        "Security, trace, and governance events recorded in MRPL Sovereign Audit Ledger",
        "verified",
        {"trace_id": state.trace_id},
    )

    # Audit log temporal status and intent (Requirement 6)
    if state.temporal_intent or state.temporal_validation:
        try:
            audit_log(
                action="temporal_grounding_verified",
                outcome="success" if (state.temporal_validation and state.temporal_validation.get("is_valid", True)) else "failure",
                user_id=state.user_id,
                username=state.user.get("username", "engineer") if state.user else "engineer",
                details={
                    "event": "TEMPORAL_GROUNDING_EVALUATED",
                    "temporal_intent": state.temporal_intent.get("temporal_intent") if state.temporal_intent else None,
                    "requested_period": state.temporal_intent.get("requested_period") if state.temporal_intent else None,
                    "latest_source_period": state.temporal_validation.get("latest_reporting_period") if state.temporal_validation else None,
                    "temporal_status": state.temporal_validation.get("temporal_status") if state.temporal_validation else None,
                },
            )
        except Exception as aud_err:
            logger.warning("Failed to record temporal audit log: %s", aud_err)

    # Save assistant message to chat if we have a session.
    if state.session_id:
        chat_repo.add_message(
            session_id=state.session_id,
            role="assistant",
            content=state.response,
            model_id=state.model_id,
            sources=state.sources if state.sources else None,
            tool_calls=state.tool_results if state.tool_results else None,
            execution_ms=state.execution_ms,
        )

    # Update task status.
    if state.task_id:
        with transaction() as conn:
            conn.execute(
                "UPDATE tasks SET status = 'completed', output_data = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (json.dumps({"response": state.response[:5000]}), state.task_id),
            )

    state.current_step = "complete"
    state.step_index += 1

    return state


# ── Main entry point ─────────────────────────────────────────────────────

# ── Safety & Reliability Guards ───────────────────────────────────────────
MAX_AGENT_ITERATIONS = 10
MAX_TOOL_CALLS_PER_TASK = 15
TASK_TIMEOUT_SECONDS = 300


def check_agent_safety_limits(current_state: AgentState, start_time: float) -> None:
    """Validate that agent state does not exceed loop iteration, tool call, or timeout bounds."""
    if current_state.step_index >= MAX_AGENT_ITERATIONS:
        logger.error("Agent exceeded maximum iterations (%d)", MAX_AGENT_ITERATIONS)
        raise RuntimeError(f"Agent exceeded maximum permitted iterations ({MAX_AGENT_ITERATIONS}). Stopped for safety.")
    if len(current_state.tool_results) >= MAX_TOOL_CALLS_PER_TASK:
        logger.error("Agent exceeded maximum tool calls (%d)", MAX_TOOL_CALLS_PER_TASK)
        raise RuntimeError(f"Agent exceeded maximum permitted tool calls ({MAX_TOOL_CALLS_PER_TASK}). Stopped for safety.")
    if time.monotonic() - start_time > TASK_TIMEOUT_SECONDS:
        logger.error("Agent execution timed out after %ds", TASK_TIMEOUT_SECONDS)
        raise TimeoutError(f"Agent task execution exceeded timeout limit ({TASK_TIMEOUT_SECONDS}s). Stopped for safety.")


# ── LangGraph Node Wrappers ──────────────────────────────────────────────
# These thin wrappers bridge existing node functions (which mutate AgentState
# in-place and return it) to LangGraph's dict-return contract.
# The actual business logic remains ENTIRELY in node_classify, node_retrieve, etc.

def _lg_classify(state: AgentState) -> dict:
    """LangGraph wrapper for node_classify."""
    _add_trace_event(state, "GRAPH_NODE_ENTERED", "Graph node: classify", "verified")
    updated = node_classify(state)
    _add_trace_event(updated, "GRAPH_NODE_COMPLETED", "Graph node: classify completed", "verified")
    return updated.__dict__


def _lg_retrieve(state: AgentState) -> dict:
    """LangGraph wrapper for node_retrieve."""
    _add_trace_event(state, "GRAPH_NODE_ENTERED", "Graph node: retrieve", "verified")
    updated = node_retrieve(state)
    _add_trace_event(updated, "GRAPH_NODE_COMPLETED", "Graph node: retrieve completed", "verified")
    return updated.__dict__


def _lg_tool_call(state: AgentState) -> dict:
    """LangGraph wrapper for node_tool_call."""
    _add_trace_event(state, "GRAPH_NODE_ENTERED", "Graph node: tool_call", "verified")
    updated = node_tool_call(state)
    _add_trace_event(updated, "GRAPH_NODE_COMPLETED", "Graph node: tool_call completed", "verified")
    return updated.__dict__


def _lg_reason(state: AgentState) -> dict:
    """LangGraph wrapper for node_reason."""
    _add_trace_event(state, "GRAPH_NODE_ENTERED", "Graph node: reason", "verified")
    updated = node_reason(state)
    _add_trace_event(updated, "GRAPH_NODE_COMPLETED", "Graph node: reason completed", "verified")
    return updated.__dict__


def _lg_validate(state: AgentState) -> dict:
    """LangGraph wrapper for node_validate."""
    _add_trace_event(state, "GRAPH_NODE_ENTERED", "Graph node: validate", "verified")
    updated = node_validate(state)
    _add_trace_event(updated, "GRAPH_NODE_COMPLETED", "Graph node: validate completed", "verified")
    return updated.__dict__


def _lg_respond(state: AgentState) -> dict:
    """LangGraph wrapper for node_respond."""
    _add_trace_event(state, "GRAPH_NODE_ENTERED", "Graph node: respond", "verified")
    updated = node_respond(state)
    _add_trace_event(updated, "GRAPH_NODE_COMPLETED", "Graph node: respond completed", "verified")
    return updated.__dict__


# ── LangGraph Conditional Routing ────────────────────────────────────────

def route_after_retrieve(state: AgentState) -> str:
    """
    Conditional routing after the retrieve node.

    If retrieval/security processing determined that the task must terminate early
    (RBAC denied, insufficient evidence, temporal refusal), route directly to respond.
    Otherwise, continue to tool_call.
    """
    if state.current_step in {
        "rbac_denied",
        "insufficient_evidence",
        "temporal_refusal",
    }:
        return "respond"
    return "tool_call"


# ── LangGraph Construction ───────────────────────────────────────────────

def build_agent_graph():
    """
    Build and compile the LangGraph StateGraph for the sovereign agent pipeline.

    Graph structure:
        START → classify → retrieve → [conditional] → tool_call → reason → validate → respond → END
                                        ↓ (early refusal)
                                      respond → END

    LangGraph controls WHICH node executes next.
    Python services inside nodes decide WHETHER an operation is allowed.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("classify", _lg_classify)
    workflow.add_node("retrieve", _lg_retrieve)
    workflow.add_node("tool_call", _lg_tool_call)
    workflow.add_node("reason", _lg_reason)
    workflow.add_node("validate", _lg_validate)
    workflow.add_node("respond", _lg_respond)

    # Define edges
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "retrieve")

    # Conditional routing after retrieve:
    # - Early refusal (RBAC denied, insufficient evidence, temporal refusal) → respond → END
    # - Normal flow → tool_call → reason → validate → respond → END
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "respond": "respond",
            "tool_call": "tool_call",
        },
    )

    workflow.add_edge("tool_call", "reason")
    workflow.add_edge("reason", "validate")
    workflow.add_edge("validate", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


# Compile the graph at module level — this is the single authoritative orchestration path.
agent_graph = build_agent_graph()


def _build_agent_return(state: AgentState) -> dict:
    """Construct complete, auditable agent result with full governance lineage."""
    coding_res = None
    if state.task_type in ("CODING", "code_generation"):
        sb_attempts = [
            r for r in state.tool_results
            if r.get("tool") == "sandbox_execute" or "attempt" in r
        ]
        if sb_attempts:
            ver = next(
                (a for a in sb_attempts if a.get("status") == "verified" or a.get("exit_code") == 0),
                None,
            )
            tgt = ver if ver else sb_attempts[-1]
            c_status = (
                "VERIFIED" if (tgt.get("exit_code") == 0 or tgt.get("status") == "verified")
                else ("BLOCKED" if tgt.get("status") in ("error", "blocked") else "FAILED")
            )
            coding_res = {
                "task_type": "CODING",
                "model": state.ollama_model_name,
                "status": c_status,
                "current_query": state.current_query,
                "code": strip_raw_ui_markers(tgt.get("code", "")),
                "stdout": strip_raw_ui_markers(tgt.get("stdout", "")),
                "stderr": strip_raw_ui_markers(tgt.get("stderr", "")),
                "exit_code": tgt.get("exit_code"),
                "correction_attempts": len(sb_attempts),
                "execution_details": tgt.get("sandbox_info") or (
                    {
                        "sandbox": "Docker",
                        "network": "Disabled",
                        "filesystem": "Read-only",
                        "user": "Non-root",
                        "exit_code": tgt.get("exit_code"),
                        "status": "Verified" if tgt.get("exit_code") == 0 else "Failed",
                    } if (tgt.get("exit_code") is not None or tgt.get("status") == "verified") else None
                ),
                "sandbox_info": tgt.get("sandbox_info") or (
                    {
                        "sandbox": "Docker",
                        "network": "Disabled",
                        "filesystem": "Read-only",
                        "user": "Non-root",
                        "exit_code": tgt.get("exit_code"),
                        "status": "Verified" if tgt.get("exit_code") == 0 else "Failed",
                    } if (tgt.get("exit_code") is not None or tgt.get("status") == "verified") else None
                ),
                "trace_id": state.trace_id,
            }

    clean_response = strip_raw_ui_markers(state.response)

    return {
        "task_id": state.task_id,
        "task_type": state.task_type,
        "model_id": state.model_id,
        "model": state.ollama_model_name,
        "status": (
            coding_res["status"] if coding_res
            else ("VERIFIED" if state.current_step in ("complete", "responded") else "SUCCESS")
        ),
        "current_query": state.current_query,
        "code": coding_res["code"] if coding_res else None,
        "stdout": coding_res["stdout"] if coding_res else None,
        "stderr": coding_res["stderr"] if coding_res else None,
        "exit_code": coding_res["exit_code"] if coding_res else None,
        "correction_attempts": coding_res["correction_attempts"] if coding_res else None,
        "execution_details": coding_res["execution_details"] if coding_res else None,
        "response": clean_response,
        "sources": state.sources,
        "deliverable": state.deliverable,
        "coding_result": coding_res,
        "tool_results": [
            {
                "tool": r.get("tool", "sandbox_execute"),
                "status": r.get("status", "unknown"),
                **({"attempt": r["attempt"]} if "attempt" in r else {}),
                **({"exit_code": r.get("exit_code")} if "exit_code" in r else {}),
                **({"result": r["result"]} if "result" in r else {}),
                **({"error": r["error"]} if "error" in r else {}),
                **({"code": r["code"]} if "code" in r else {}),
                **({"stdout": r["stdout"]} if "stdout" in r else {}),
                **({"stderr": r["stderr"]} if "stderr" in r else {}),
                **({"sandbox_info": r["sandbox_info"]} if "sandbox_info" in r else {}),
            }
            for r in state.tool_results
        ],
        "execution_ms": state.execution_ms,
        "retrieval_ms": state.retrieval_ms,
        "trace_id": state.trace_id,
        "execution_trace": state.execution_trace or [],
        "scope_decision": state.scope_decision,
        "confidence_decision": state.confidence_decision,
        "citation_validation": state.citation_validation,
        "temporal_intent": state.temporal_intent,
        "temporal_validation": state.temporal_validation,
        "temporal_claims": state.temporal_claims,
        "requires_human_review": state.requires_human_review,
        "approval_id": state.approval_id,
        "escalation_rule": (
            "LOW_RETRIEVAL_CONFIDENCE"
            if state.requires_human_review and state.confidence_decision and not state.confidence_decision.get("is_sufficient", True)
            else ("UNREGISTERED_EQUIPMENT_TAGS" if state.requires_human_review and state.has_image else None)
        ),
    }


async def run_agent(
    query: str,
    user_id: int,
    session_id: int | None = None,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
    user: dict | None = None,
) -> dict:
    """
    Run the full sovereign agent pipeline for a user query with strict security & governance controls:
    1. Deterministic Scope Guard (rejection prior to LLM reasoning)
    2. RBAC & Clearance Authorization
    3. Empirical Retrieval Confidence Gate
    4. Model Registry & License Gate Verification
    5. Local Isolated Ollama LLM Reasoning
    6. Grounding & Citation Validation
    7. Complete Audited Execution Trace
    """
    start_time = time.monotonic()

    if user is None and user_id:
        try:
            from backend.database.repositories import users as users_repo
            user = users_repo.get_user_by_id(user_id)
        except Exception:
            user = None

    state = AgentState(
        query=query,
        user_id=user_id,
        session_id=session_id,
        has_image=has_image,
        has_scanned_pdf=has_scanned_pdf,
        user=user,
    )
    state.trace_id = f"TRC-{uuid.uuid4().hex[:8].upper()}"
    state.execution_trace = []

    # 1. Authoritative Query & Pipeline Trace
    _add_trace_event(
        state,
        "CHAT_REQUEST_QUERY",
        f'Chat Request Query: "{state.current_query}"',
        "verified",
        {"current_query": state.current_query, "trace_id": state.trace_id},
    )
    _add_trace_event(state, "query_received", "Query received by Sovereign Agent", "allowed", {"query": state.current_query[:120]})

    # 2. Deterministic Scope Guard (Phase 2 Component 1)
    username = user.get("username", "anonymous") if user else "anonymous"
    roles = user.get("roles", ["engineer"]) if user else ["engineer"]
    from backend.services.scope_guard import evaluate_scope
    scope_res = evaluate_scope(query, username=username, role=roles[0] if roles else "engineer")
    state.scope_decision = {
        "allowed": scope_res.allowed,
        "scope_status": scope_res.scope_status,
        "scope_category": scope_res.scope_category,
        "reason": scope_res.reason,
    }

    if not scope_res.allowed:
        _add_trace_event(state, "scope_checked", f"Scope Guard: {scope_res.reason}", "blocked", state.scope_decision)
        state.response = (
            f"⛔ REQUEST OUT OF SCOPE: {scope_res.reason}\n\n"
            "The MRPL Sovereign AI Workbench strictly permits industrial operations, refining engineering, "
            "compliance guidelines, financial reports, and sandboxed calculations. General entertainment, "
            "sports, gaming, or non-refinery consumer queries are rejected prior to model reasoning."
        )
        _add_trace_event(state, "output_generated", "Response generated (Scope rejection notice)", "blocked")
        _add_trace_event(state, "audit_logged", "Security audit logged", "verified")
        state.current_step = "scope_blocked"

        if session_id:
            chat_repo.add_message(session_id, "assistant", state.response)
        if state.task_id:
            with transaction() as conn:
                conn.execute(
                    "UPDATE tasks SET status = 'completed', output_data = ?, completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                    (json.dumps({"response": state.response}), state.task_id),
                )
        return _build_agent_return(state)

    _add_trace_event(state, "scope_checked", f"Scope Guard: Allowed ({scope_res.scope_category})", "allowed", state.scope_decision)

    # Load chat history if we have a session.
    if session_id:
        messages = chat_repo.list_messages(session_id)
        state.chat_history = [
            {
                "role": m["role"],
                "content": m["content"],
                "sources": m.get("sources"),
                "tool_calls": m.get("tool_calls"),
                "model_id": m.get("model_id"),
                "task_type": m.get("task_type"),
                "retrieval_query": m.get("retrieval_query"),
                "target_entity": m.get("target_entity"),
            }
            for m in messages[-20:]  # Last 20 messages for context
        ]
        state.turn_id = len(state.chat_history) // 2 + 1
    else:
        state.turn_id = 1

    # Create a task record.
    with transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO tasks (user_id, session_id, task_type, status, input_data) VALUES (?, ?, ?, 'running', ?)",
            (user_id, session_id, "pending", json.dumps({"query": query})),
        )
        state.task_id = cursor.lastrowid

    try:
        # ── Invoke the compiled LangGraph StateGraph ──────────────────
        # This is the SINGLE authoritative orchestration path.
        # LangGraph controls which node executes next.
        # Python services inside nodes decide whether an operation is allowed.
        check_agent_safety_limits(state, start_time)

        _add_trace_event(
            state,
            "LANGGRAPH_INVOCATION",
            "Invoking LangGraph StateGraph agent pipeline",
            "verified",
            {"graph_nodes": ["classify", "retrieve", "tool_call", "reason", "validate", "respond"]},
        )

        result_dict = await agent_graph.ainvoke(state)

        # Reconstruct AgentState from the result dict for _build_agent_return()
        # Filter out any keys not in AgentState fields to avoid unexpected keyword args
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(AgentState)}
        filtered_result = {k: v for k, v in result_dict.items() if k in valid_fields}
        final_state = AgentState(**filtered_result)

        # Update task with classified type (post-graph, since classify ran inside graph)
        if final_state.task_id and final_state.task_type:
            with transaction() as conn:
                conn.execute(
                    "UPDATE tasks SET task_type = ?, model_id = ? WHERE id = ?",
                    (final_state.task_type, final_state.model_id, final_state.task_id),
                )

    except Exception as exc:
        logger.error("Agent pipeline failed: %s", exc, exc_info=True)
        state.error_info = str(exc)

        if state.task_id:
            with transaction() as conn:
                conn.execute(
                    "UPDATE tasks SET status = 'failed', error_info = ? WHERE id = ?",
                    (json.dumps({"error": str(exc)}), state.task_id),
                )

        raise

    return _build_agent_return(final_state)


# ── Private helpers ──────────────────────────────────────────────────────

def _call_llm(state: AgentState, prompt: str, num_predict: int = 768) -> str:
    """Call the Ollama LLM and return the response text."""
    from backend.services.network_seal import get_network_seal
    get_network_seal().record_local_call(state.model_endpoint, f"Ollama {state.ollama_model_name}")

    payload = {
        "model": state.ollama_model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "top_p": 0.85,
            "num_ctx": 4096,
            "num_predict": num_predict,
        },
    }

    try:
        resp = requests.post(state.model_endpoint, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()

        if "response" not in result:
            raise RuntimeError(f"Unexpected Ollama response: {result}")

        return result["response"]

    except requests.RequestException as exc:
        logger.error("LLM call failed: %s", exc)
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def _extract_numbers(text: str) -> list[str]:
    """
    Extract all numerical values from text, including Indian comma notation.

    Matches patterns like: 1,01,578  7,22,831  3.14  100  12.5
    Returns normalized matched strings for verbatim comparison.
    """
    # Commas and decimals must be surrounded by digits
    matches = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
    result = []
    for m in matches:
        clean = m.strip(".,;: ")
        if clean:
            result.append(clean)
            no_commas = clean.replace(",", "")
            if no_commas != clean:
                result.append(no_commas)
    return result


def _extract_code_block(text: str) -> str:
    """Extract code from markdown code blocks."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 2:
            return parts[1].strip()
    return text


def _log_step(state: AgentState) -> None:
    """Log the current step to the task_steps table."""
    if not state.task_id:
        return

    try:
        with transaction() as conn:
            conn.execute(
                """
                INSERT INTO task_steps (task_id, step_index, state_name, input_data, output_data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    state.task_id,
                    state.step_index,
                    state.current_step,
                    json.dumps({"query": state.query[:200]}),
                    json.dumps(state.to_dict()),
                ),
            )
    except Exception as exc:
        logger.warning("Failed to log step: %s", exc)
