"""
Agent state definition — the data structure flowing through the LangGraph.

Serializable, logged to the task_steps table at each transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """
    Immutable-ish state object passed through the LangGraph nodes.

    Each field is updated by the node that produces it, and the full
    state is logged to task_steps for auditability.
    """

    # ── Authoritative Query & State Integrity ─────────────────────────
    query: str = ""
    current_query: str = ""  # Authoritative user query after deterministic normalization
    current_user_query: str = ""  # Authoritative current turn query alias
    raw_user_query: str = ""
    normalized_user_query: str = ""
    retrieval_query: str = ""  # Search query representation for RAG only
    retrieved_context: str = ""  # Evidence returned by RAG
    conversation_history: list[dict] = field(default_factory=list)  # Prior turns only

    user_id: int | None = None
    session_id: int | None = None
    conversation_id: str | int | None = None
    turn_id: int | None = None
    has_image: bool = False
    has_scanned_pdf: bool = False
    user: dict | None = None

    def __post_init__(self) -> None:
        if self.current_user_query:
            self.current_query = self.current_user_query
            self.query = self.current_user_query
        elif self.current_query:
            self.current_user_query = self.current_query
            self.query = self.current_query
        elif self.query:
            self.current_user_query = self.query
            self.current_query = self.query

        if self.session_id and not self.conversation_id:
            self.conversation_id = self.session_id
        elif self.conversation_id and not self.session_id and isinstance(self.conversation_id, int):
            self.session_id = self.conversation_id

        if self.chat_history and not self.conversation_history:
            self.conversation_history = self.chat_history
        elif self.conversation_history and not self.chat_history:
            self.chat_history = self.conversation_history
        if self.context and not self.retrieved_context:
            self.retrieved_context = self.context
        elif self.retrieved_context and not self.context:
            self.context = self.retrieved_context

    # ── Classification ───────────────────────────────────────────────
    task_type: str = ""
    classification_reasoning: str = ""
    requires_rag: bool = False
    requires_sandbox: bool = False

    # ── Model selection ──────────────────────────────────────────────
    model_id: str = ""
    ollama_model_name: str = ""
    model_endpoint: str = ""
    required_tools: list[str] = field(default_factory=list)

    # ── RAG / Context ────────────────────────────────────────────────
    context: str = ""
    sources: list[dict] = field(default_factory=list)

    # ── Tool results ─────────────────────────────────────────────────
    tool_results: list[dict] = field(default_factory=list)

    # ── LLM response ─────────────────────────────────────────────────
    response: str = ""
    execution_ms: int = 0
    retrieval_ms: int = 0

    # ── Chat history ─────────────────────────────────────────────────
    chat_history: list[dict] = field(default_factory=list)

    # ── Task tracking ────────────────────────────────────────────────
    task_id: int | None = None
    current_step: str = "start"
    step_index: int = 0
    error_info: str | None = None
    target_document: str | None = None
    deliverable: dict | None = None
    # ── Governance & Execution Trace ─────────────────────────────────
    trace_id: str = ""
    execution_trace: list[dict] = field(default_factory=list)
    scope_decision: dict | None = None
    confidence_decision: dict | None = None
    citation_validation: dict | None = None
    temporal_intent: dict | None = None
    temporal_validation: dict | None = None
    temporal_claims: dict | None = None
    requires_human_review: bool = False
    approval_id: int | None = None

    def add_trace(self, stage: str, status: str, detail: str = "") -> None:
        """Add a structured execution trace checkpoint."""
        self.execution_trace.append({
            "stage": stage,
            "status": status,
            "detail": detail,
            "step": len(self.execution_trace) + 1,
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-compatible dict."""
        return {
            "query": self.query,
            "current_query": self.current_query or self.query,
            "current_user_query": self.current_user_query or self.current_query or self.query,
            "raw_user_query": self.raw_user_query,
            "normalized_user_query": self.normalized_user_query,
            "retrieval_query": self.retrieval_query,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id or self.session_id,
            "turn_id": self.turn_id,
            "trace_id": self.trace_id,
            "task_type": self.task_type,
            "classification_reasoning": self.classification_reasoning,
            "requires_rag": self.requires_rag,
            "requires_sandbox": self.requires_sandbox,
            "model_id": self.model_id,
            "required_tools": self.required_tools,
            "context": self.context[:500] if self.context else "",
            "sources_count": len(self.sources),
            "tool_results_count": len(self.tool_results),
            "response": self.response[:500] if self.response else "",
            "execution_ms": self.execution_ms,
            "retrieval_ms": self.retrieval_ms,
            "current_step": self.current_step,
            "step_index": self.step_index,
            "error_info": self.error_info,
            "target_document": self.target_document,
            "deliverable": self.deliverable,
            "execution_trace": self.execution_trace,
            "scope_decision": self.scope_decision,
            "confidence_decision": self.confidence_decision,
            "citation_validation": self.citation_validation,
            "temporal_intent": self.temporal_intent,
            "temporal_validation": self.temporal_validation,
            "temporal_claims": self.temporal_claims,
            "requires_human_review": self.requires_human_review,
            "approval_id": self.approval_id,
        }

