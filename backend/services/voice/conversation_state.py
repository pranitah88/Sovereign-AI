"""
MRPL Sovereign AI Workbench — First-Class Conversational State & Turn Lifecycle.
Enforces explicit turn management, query lineage integrity, and state isolation.
"""

from __future__ import annotations

import datetime
from enum import Enum
import threading
import time
from typing import Any
from pydantic import BaseModel, Field


class TurnStatus(str, Enum):
    """Explicit conversational turn state machine states."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    UNDERSTANDING = "UNDERSTANDING"
    RETRIEVING = "RETRIEVING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    SPEAKING_CLARIFICATION = "SPEAKING_CLARIFICATION"
    WAITING_FOR_USER_REPLY = "WAITING_FOR_USER_REPLY"
    USER_INTERRUPTED = "USER_INTERRUPTED"
    ERROR_RECOVERY = "ERROR_RECOVERY"


class TurnState(BaseModel):
    """
    State representing a single conversational interaction turn.
    Strictly preserves authoritative user query and separates retrieval evidence.
    """
    conversation_id: str | int
    turn_id: int
    trace_id: str

    # Query Lineage (Strict Hierarchy)
    raw_user_query: str = ""
    normalized_user_query: str = ""
    current_user_query: str = ""  # Authoritative user query
    retrieval_query: str = ""     # Internal search representation only
    retrieved_context: str = ""   # Evidence returned by RAG only

    # Context & History
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    active_skill: str = "CHAT"
    clarification_state: dict[str, Any] | None = None

    # Lifecycle State
    status: TurnStatus = TurnStatus.IDLE
    assistant_state: str = "IDLE"
    asr_confidence: float = 1.0
    retrieval_confidence: float = 1.0
    asr_status: str = "success"
    task_type: str = "GENERAL"
    model_id: str = "gemma3_4b"
    sources: list[dict[str, Any]] = Field(default_factory=list)

    # Execution & Synthesis
    response_text: str = ""
    spoken_text: str = ""
    is_interrupted: bool = False
    last_error: str | None = None

    # Observability & Latency (Real measured ms)
    timestamps: dict[str, str] = Field(default_factory=dict)
    latencies: dict[str, float] = Field(default_factory=dict)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)

    def set_current_user_query(self, query: str) -> None:
        """
        Set the authoritative current user query.
        INVARIANT: Once established and non-empty, it becomes strictly immutable.
        It must NEVER be overwritten by retrieved text, previous turns, or rewriting.
        """
        clean_q = (query or "").strip()
        if not clean_q:
            return
        if self.current_user_query and self.current_user_query != clean_q:
            # Enforce immutability: preserve the original established query
            return
        self.current_user_query = clean_q

    def record_timestamp(self, checkpoint: str) -> None:
        """Record ISO UTC timestamp for a turn lifecycle checkpoint."""
        self.timestamps[checkpoint] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def record_latency(self, metric_name: str, latency_ms: float) -> None:
        """Record genuine measured latency in milliseconds."""
        self.latencies[metric_name] = round(latency_ms, 2)

    def add_trace(self, stage: str, status: str, detail: str = "", metadata: dict | None = None) -> dict[str, Any]:
        """Append a structured trace event."""
        evt = {
            "event": stage,
            "stage": stage,
            "status": status,
            "detail": detail,
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "turn_id": self.turn_id,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
        }
        self.trace_events.append(evt)
        return evt

    def add_propagation_trace(self, source: str, destination: str, query: str | None = None) -> dict[str, Any]:
        """Add a structured QUERY_PROPAGATION lineage trace event."""
        q = query if query is not None else self.current_user_query
        return self.add_trace(
            stage="QUERY_PROPAGATION",
            status="verified",
            detail=f"Lineage: {source} -> {destination}",
            metadata={
                "event": "QUERY_PROPAGATION",
                "conversation_id": self.conversation_id,
                "turn_id": self.turn_id,
                "trace_id": self.trace_id,
                "source": source,
                "destination": destination,
                "query": q,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )


class ConversationState:
    """
    Session-level conversational context tracker.
    Maintains active turn, topic memory, active equipment tags, and thread safety.
    """
    def __init__(self, conversation_id: str | int):
        self.conversation_id = conversation_id
        self._lock = threading.RLock()
        self.active_turn_id: int = 0
        self.active_turn: TurnState | None = None
        self.turns: list[TurnState] = []
        self.active_equipment_tag: str | None = None
        self.active_topic: str | None = None
        self.active_clarification: dict[str, Any] | None = None
        self.is_interrupted: bool = False

    def new_turn(self, trace_id: str, raw_query: str = "") -> TurnState:
        """Create and activate a new isolated turn."""
        with self._lock:
            self.active_turn_id += 1
            turn = TurnState(
                conversation_id=self.conversation_id,
                turn_id=self.active_turn_id,
                trace_id=trace_id,
                raw_user_query=raw_query,
                current_user_query=raw_query,
                status=TurnStatus.LISTENING,
                assistant_state="LISTENING",
            )
            turn.record_timestamp("turn_started")
            self.active_turn = turn
            self.turns.append(turn)
            self.is_interrupted = False
            return turn

    def get_turn(self, turn_id: int) -> TurnState | None:
        """Get turn by turn ID."""
        with self._lock:
            for t in self.turns:
                if t.turn_id == turn_id:
                    return t
            return None

    def is_stale_turn(self, turn_id: int) -> bool:
        """Return True if a newer turn has already superseded this turn."""
        with self._lock:
            if turn_id < self.active_turn_id:
                return True
            target = self.get_turn(turn_id)
            if target and target.is_interrupted:
                return True
            return self.is_interrupted

    def mark_interrupted(self, turn_id: int | None = None) -> None:
        """Mark turn as interrupted by barge-in."""
        with self._lock:
            self.is_interrupted = True
            target = self.active_turn if turn_id is None else self.get_turn(turn_id)
            if target:
                target.is_interrupted = True
                target.status = TurnStatus.USER_INTERRUPTED
                target.assistant_state = "USER_INTERRUPTED"
                target.record_timestamp("interrupted")
                target.add_trace("BARGE_IN_TRIGGERED", "interrupted", "User speech interrupted active TTS playback")

    def update_topic(self, topic: str | None, tag: str | None = None) -> None:
        """Update ongoing conversation subject or equipment tag."""
        with self._lock:
            if topic:
                self.active_topic = topic
            if tag:
                self.active_equipment_tag = tag


# ── Global In-Memory Conversation State Store ──────────────────────────────────
_conversation_registry: dict[str | int, ConversationState] = {}
_registry_lock = threading.Lock()


def get_conversation_state(conversation_id: str | int) -> ConversationState:
    """Retrieve or initialize thread-safe ConversationState."""
    key = str(conversation_id)
    with _registry_lock:
        if key not in _conversation_registry:
            _conversation_registry[key] = ConversationState(conversation_id)
        return _conversation_registry[key]


def clear_conversation_state(conversation_id: str | int) -> None:
    """Evict conversation state on session close/logout."""
    key = str(conversation_id)
    with _registry_lock:
        _conversation_registry.pop(key, None)
