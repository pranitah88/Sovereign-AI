"""
MRPL Sovereign AI Workbench — Conversational Intent Detection & Local Response.

Lightweight, deterministic layer that intercepts simple conversational inputs
(greetings, farewells, "how are you") and returns curated natural responses
WITHOUT invoking RAG, Ollama, ChromaDB, or any heavy pipeline.

Design:
- Pure regex/pattern matching — no LLM, no embeddings.
- Deterministic rotating index per intent — no uncontrolled randomness.
- Last-response tracking prevents immediate consecutive duplicates.
- Phrases containing substantive questions ("Hi, what is MRPL?") are
  correctly classified as NORMAL_QUERY, not greetings.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Intent Categories ────────────────────────────────────────────────────────

class ConversationalIntent(str, Enum):
    GREETING = "GREETING"
    HOW_ARE_YOU = "HOW_ARE_YOU"
    TIME_GREETING = "TIME_GREETING"
    FAREWELL = "FAREWELL"
    NORMAL_QUERY = "NORMAL_QUERY"


# ── Curated Response Pools ───────────────────────────────────────────────────

RESPONSE_POOLS: dict[str, list[str]] = {
    "GREETING": [
        "Hey! How can I help?",
        "Hello! What can I help you with?",
        "Hey there! What are you working on?",
        "Hi! What can I help you with today?",
        "Hello! Ready when you are.",
    ],
    "HOW_ARE_YOU": [
        "I'm doing well! What can I help you with?",
        "Doing good! What would you like to work on?",
        "I'm ready to help! What's on your mind?",
        "All good here! What can I help you with?",
    ],
    "TIME_GREETING_MORNING": [
        "Good morning! How can I help?",
        "Good morning! What can I help you with today?",
        "Good morning! Ready when you are.",
    ],
    "TIME_GREETING_AFTERNOON": [
        "Good afternoon! What can I help you with?",
        "Good afternoon! How can I help?",
        "Good afternoon! Ready when you are.",
    ],
    "TIME_GREETING_EVENING": [
        "Good evening! What would you like to work on?",
        "Good evening! How can I help?",
        "Good evening! Ready when you are.",
    ],
    "FAREWELL": [
        "Goodbye!",
        "See you later!",
        "Alright, take care!",
        "Bye! I'll be here when you need me.",
    ],
}


# ── Pattern Definitions ──────────────────────────────────────────────────────
# Patterns are tested against the *entire* normalized input.
# Ordering matters: more specific patterns are checked first.

# Detectors for substantive content that disqualifies a greeting classification.
# If any of these match, the input is NORMAL_QUERY regardless of greeting words.
_SUBSTANTIVE_TAIL = re.compile(
    r"""
    (?:
        \bwhat\b | \bwho\b | \bwhere\b | \bwhen\b | \bwhy\b | \bhow\s+(?!are\s+you)
        | \btell\s+me\b | \bexplain\b | \bshow\b | \bdescribe\b
        | \bcalculate\b | \banalyze\b | \bcompare\b | \bsummarize\b
        | \blist\b | \bfind\b | \bsearch\b | \bcheck\b | \bstatus\b
        | \bpressure\b | \btemperature\b | \bflow\b | \bpump\b | \bvessel\b
        | \breport\b | \bdocument\b | \baudit\b
        | \?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Phrases that look like greetings but carry contextual meaning as normal queries.
_FALSE_POSITIVE_GUARDS = [
    re.compile(r"\bsaid\s+(?:goodbye|bye|hello|hi)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:goodbye|bye|hello|hi)\b", re.IGNORECASE),
    re.compile(r"\bwhy\s+(?:did|does|do)\b", re.IGNORECASE),
]


# Greeting patterns (pure greeting, no trailing question content)
_GREETING_PATTERNS = [
    re.compile(r"^(?:hi|hey|hello|hiya|yo|heya|howdy)[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^(?:hi|hey|hello|hiya)\s+(?:there|nova)[\s!.,;:]*$", re.IGNORECASE),
]

# "How are you" patterns
_HOW_ARE_YOU_PATTERNS = [
    re.compile(r"^how\s+are\s+you[\s!?.,;:]*$", re.IGNORECASE),
    re.compile(r"^how\s+are\s+you\s+(?:doing|today|nova)[\s!?.,;:]*$", re.IGNORECASE),
    re.compile(r"^how(?:'s|\s+is)\s+it\s+going[\s!?.,;:]*$", re.IGNORECASE),
    re.compile(r"^how\s+do\s+you\s+do[\s!?.,;:]*$", re.IGNORECASE),
    re.compile(r"^you\s+(?:good|okay|ok|alright|fine)[\s!?.,;:]*$", re.IGNORECASE),
]

# Time-of-day greetings
_TIME_MORNING_PATTERNS = [
    re.compile(r"^good\s+morning[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^good\s+morning\s+(?:nova)[\s!.,;:]*$", re.IGNORECASE),
]
_TIME_AFTERNOON_PATTERNS = [
    re.compile(r"^good\s+afternoon[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^good\s+afternoon\s+(?:nova)[\s!.,;:]*$", re.IGNORECASE),
]
_TIME_EVENING_PATTERNS = [
    re.compile(r"^good\s+evening[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^good\s+evening\s+(?:nova)[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^good\s+night[\s!.,;:]*$", re.IGNORECASE),
]

# Farewell patterns
_FAREWELL_PATTERNS = [
    re.compile(r"^(?:bye|goodbye|good\s*bye)[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^(?:see\s+you|see\s+ya|later|take\s+care|cya)[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^(?:bye|goodbye)\s+(?:nova|for\s+now)[\s!.,;:]*$", re.IGNORECASE),
    re.compile(r"^(?:thanks|thank\s+you)[\s!.,;:]*(?:bye|goodbye)[\s!.,;:]*$", re.IGNORECASE),
]


# ── Intent Detection Result ─────────────────────────────────────────────────

@dataclass
class ConversationalIntentResult:
    """Result of conversational intent classification."""
    intent: ConversationalIntent
    response_text: str | None = None
    response_key: str | None = None  # Pool key used for response selection
    is_conversational: bool = False   # True if handled locally (no pipeline needed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "response_text": self.response_text,
            "response_key": self.response_key,
            "is_conversational": self.is_conversational,
        }


# ── Deterministic Response Selector ──────────────────────────────────────────

class _ResponseRotator:
    """
    Thread-safe deterministic response rotator.
    Maintains a rotating index per pool key and tracks last-served response
    to prevent immediate consecutive duplicates.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Current rotation index per pool key
        self._indices: dict[str, int] = {}
        # Last served response text per pool key
        self._last_response: dict[str, str] = {}

    def select(self, pool_key: str) -> str:
        """
        Select the next response from the pool identified by pool_key.
        Guarantees no immediate consecutive duplicate for the same pool_key.
        """
        pool = RESPONSE_POOLS.get(pool_key)
        if not pool:
            return "Hello!"

        with self._lock:
            idx = self._indices.get(pool_key, 0)
            candidate = pool[idx % len(pool)]

            # Advance index
            next_idx = (idx + 1) % len(pool)

            # Check for consecutive duplicate
            last = self._last_response.get(pool_key)
            if candidate == last and len(pool) > 1:
                # Skip to next response
                candidate = pool[next_idx % len(pool)]
                next_idx = (next_idx + 1) % len(pool)

            self._indices[pool_key] = next_idx
            self._last_response[pool_key] = candidate
            return candidate

    def reset(self) -> None:
        """Reset all rotation state (for testing)."""
        with self._lock:
            self._indices.clear()
            self._last_response.clear()


# Module-level singleton rotator
_rotator = _ResponseRotator()


def get_rotator() -> _ResponseRotator:
    """Return the module-level response rotator (for testing access)."""
    return _rotator


# ── Public API ───────────────────────────────────────────────────────────────

def detect_conversational_intent(text: str) -> ConversationalIntentResult:
    """
    Classify user input into a conversational intent.

    If the input is a simple greeting, farewell, or casual exchange,
    returns is_conversational=True with a locally selected response.

    If the input contains substantive content (questions, technical terms),
    returns NORMAL_QUERY with is_conversational=False so the full
    governed pipeline handles it.

    Args:
        text: Normalized ASR transcription text.

    Returns:
        ConversationalIntentResult with intent classification and optional
        local response.
    """
    normalized = (text or "").strip()

    if not normalized:
        return ConversationalIntentResult(intent=ConversationalIntent.NORMAL_QUERY)

    # ── Guard: false-positive phrases that mention greetings in context ────
    for guard_pattern in _FALSE_POSITIVE_GUARDS:
        if guard_pattern.search(normalized):
            return ConversationalIntentResult(intent=ConversationalIntent.NORMAL_QUERY)

    # ── Guard: substantive content disqualifies greeting classification ────
    # Split on greeting prefix and check if remainder has substance
    greeting_prefix = re.match(
        r"^(?:hi|hey|hello|hiya|good\s+(?:morning|afternoon|evening))\s*[,;:!.]?\s*",
        normalized,
        re.IGNORECASE,
    )
    if greeting_prefix:
        remainder = normalized[greeting_prefix.end():].strip()
        if remainder and _SUBSTANTIVE_TAIL.search(remainder):
            # e.g. "Hi, what is MRPL?" → NORMAL_QUERY
            return ConversationalIntentResult(intent=ConversationalIntent.NORMAL_QUERY)

    # ── Check farewell first (before greeting, since "goodbye" contains greeting-like words) ──
    for pattern in _FAREWELL_PATTERNS:
        if pattern.match(normalized):
            response = _rotator.select("FAREWELL")
            logger.info("CONVERSATION_INTENT_DETECTED: FAREWELL → %r", response)
            return ConversationalIntentResult(
                intent=ConversationalIntent.FAREWELL,
                response_text=response,
                response_key="FAREWELL",
                is_conversational=True,
            )

    # ── Check "how are you" ──
    for pattern in _HOW_ARE_YOU_PATTERNS:
        if pattern.match(normalized):
            response = _rotator.select("HOW_ARE_YOU")
            logger.info("CONVERSATION_INTENT_DETECTED: HOW_ARE_YOU → %r", response)
            return ConversationalIntentResult(
                intent=ConversationalIntent.HOW_ARE_YOU,
                response_text=response,
                response_key="HOW_ARE_YOU",
                is_conversational=True,
            )

    # ── Check time-of-day greetings ──
    for pattern in _TIME_MORNING_PATTERNS:
        if pattern.match(normalized):
            response = _rotator.select("TIME_GREETING_MORNING")
            logger.info("CONVERSATION_INTENT_DETECTED: TIME_GREETING (morning) → %r", response)
            return ConversationalIntentResult(
                intent=ConversationalIntent.TIME_GREETING,
                response_text=response,
                response_key="TIME_GREETING_MORNING",
                is_conversational=True,
            )

    for pattern in _TIME_AFTERNOON_PATTERNS:
        if pattern.match(normalized):
            response = _rotator.select("TIME_GREETING_AFTERNOON")
            logger.info("CONVERSATION_INTENT_DETECTED: TIME_GREETING (afternoon) → %r", response)
            return ConversationalIntentResult(
                intent=ConversationalIntent.TIME_GREETING,
                response_text=response,
                response_key="TIME_GREETING_AFTERNOON",
                is_conversational=True,
            )

    for pattern in _TIME_EVENING_PATTERNS:
        if pattern.match(normalized):
            response = _rotator.select("TIME_GREETING_EVENING")
            logger.info("CONVERSATION_INTENT_DETECTED: TIME_GREETING (evening) → %r", response)
            return ConversationalIntentResult(
                intent=ConversationalIntent.TIME_GREETING,
                response_text=response,
                response_key="TIME_GREETING_EVENING",
                is_conversational=True,
            )

    # ── Check greeting ──
    for pattern in _GREETING_PATTERNS:
        if pattern.match(normalized):
            response = _rotator.select("GREETING")
            logger.info("CONVERSATION_INTENT_DETECTED: GREETING → %r", response)
            return ConversationalIntentResult(
                intent=ConversationalIntent.GREETING,
                response_text=response,
                response_key="GREETING",
                is_conversational=True,
            )

    # ── Default: NORMAL_QUERY ──
    return ConversationalIntentResult(intent=ConversationalIntent.NORMAL_QUERY)
