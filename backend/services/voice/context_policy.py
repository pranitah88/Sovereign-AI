"""
MRPL Sovereign AI Workbench — Conversational Context Policy.
Distinguishes contextual follow-ups from independent topic shifts.
Resolves anaphora deterministically for retrieval without mutating current_user_query.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel


ANAPHORIC_PRONOUN_PATTERNS = [
    r"\b(its|it|this|that|these|those|the\s+same|the\s+equipment|the\s+pump|the\s+unit)\b",
    r"\b(what\s+about|how\s+about|and\s+its|tell\s+me\s+more|explain\s+further)\b",
]

TAG_PATTERN = re.compile(r"\b(\d{1,3}[- ]+[A-Za-z]{1,3}[- ]+\d{1,4}[A-Za-z]?)\b")


class ContextResolution(BaseModel):
    is_followup: bool
    resolved_retrieval_query: str
    target_entity: str | None = None
    reason: str = ""


KNOWN_UNITS_AND_ENTITIES = [
    ("CDU-101", "CDU-101"),
    ("CDU", "CDU"),
    ("VDU", "VDU"),
    ("HCU", "HCU"),
    ("PFCCU", "PFCCU"),
    ("PFCC", "PFCCU"),
    ("FCCU", "FCCU"),
    ("DHDT", "DHDT"),
    ("DHT", "DHT"),
    ("HYDROCRACKER", "Hydrocracker"),
    ("DESALTER", "Desalter"),
    ("MRPL", "MRPL"),
    ("ONGC", "ONGC"),
]


def extract_referenced_entities(text: str) -> list[str]:
    """Extract industrial tags, refinery units, and corporate entities from text."""
    if not text:
        return []
    tags = TAG_PATTERN.findall(text)
    # Standardize tags
    std_tags = []
    for t in tags:
        clean = re.sub(r"\s+", "-", t).upper()
        if clean not in std_tags:
            std_tags.append(clean)
    
    # Check known refinery units and corporate entities if no tag matched
    text_upper = text.upper()
    for pattern, canonical in KNOWN_UNITS_AND_ENTITIES:
        if re.search(rf"\b{re.escape(pattern)}\b", text_upper):
            if canonical not in std_tags:
                std_tags.append(canonical)

    return std_tags


def determine_conversational_context(
    current_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
    active_equipment_tag: str | None = None,
) -> ContextResolution:
    """
    Deterministically analyze query structure against conversation history.

    Rules:
    1. If the current query has complete standalone subjects and zero anaphoric pronouns
       (e.g., 'What is MRPL?', 'Who owns MRPL?', 'Explain CDU column'), it is an INDEPENDENT query.
       -> is_followup = False, retrieval_query = current_query.
       -> History is NOT injected into retrieval.

    2. If the current query contains anaphoric pronouns or elliptical phrasing
       (e.g., 'What is its function?', 'What is its pressure?', 'Is it running?'),
       it is a CONTEXTUAL FOLLOW-UP.
       -> Extracts target entity from active_equipment_tag or latest turns.
       -> Formulates retrieval_query (e.g. 'What is the function of 11-P-101A?').
       -> current_user_query remains 'What is its function?'.
    """
    text = (current_query or "").strip()
    text_lower = text.lower()

    if not text:
        return ContextResolution(
            is_followup=False,
            resolved_retrieval_query="",
            reason="Empty query",
        )

    # 1. Check for explicit tags in the current query
    current_tags = extract_referenced_entities(text)
    if current_tags:
        # User explicitly mentioned a new or specific tag -> standalone focus
        return ContextResolution(
            is_followup=False,
            resolved_retrieval_query=text,
            target_entity=current_tags[0],
            reason=f"Explicit entity in current query: {current_tags[0]}",
        )

    # 2. Check for clear company / general corporate queries that are always independent
    standalone_topics = [
        "mrpl", "ongc", "mangalore refinery", "refinery overview", "annual report",
        "vision", "mission", "vigilance", "complaint portal", "what is mrpl",
        "who is the md", "who is the chairman", "history of mrpl",
    ]
    if any(st in text_lower for st in standalone_topics):
        return ContextResolution(
            is_followup=False,
            resolved_retrieval_query=text,
            target_entity=None,
            reason="Clear independent standalone corporate topic",
        )

    # 3. Check for anaphoric pronouns or elliptical phrasing
    has_anaphora = any(bool(re.search(p, text_lower)) for p in ANAPHORIC_PRONOUN_PATTERNS)
    is_short_fragment = len(text.split()) <= 4 and not any(text_lower.startswith(w) for w in ["what is mrpl", "who is", "where is mrpl"])

    if has_anaphora or is_short_fragment:
        # Find candidate entity from active_equipment_tag or conversation history
        candidate_entity = active_equipment_tag
        if not candidate_entity and conversation_history:
            # Search backwards through prior turns for an equipment tag or entity
            for msg in reversed(conversation_history[-4:]):
                content = msg.get("content", "")
                found = extract_referenced_entities(content)
                if found:
                    candidate_entity = found[0]
                    break

        if candidate_entity:
            # Construct resolved retrieval query deterministically
            resolved_query = text
            # Replace possessive "its <property>" -> "the \1 of <candidate_entity>"
            resolved_query = re.sub(
                r"\bits\s+([a-zA-Z]+)\b",
                rf"the \1 of {candidate_entity}",
                resolved_query,
                flags=re.IGNORECASE,
            )
            # Replace remaining standalone pronouns
            resolved_query = re.sub(
                r"\b(its|it|this|that|the\s+pump|the\s+equipment|the\s+unit)\b",
                candidate_entity,
                resolved_query,
                flags=re.IGNORECASE,
            )
            if candidate_entity.lower() not in resolved_query.lower():
                resolved_query = f"{resolved_query} of {candidate_entity}"

            return ContextResolution(
                is_followup=True,
                resolved_retrieval_query=resolved_query,
                target_entity=candidate_entity,
                reason=f"Resolved anaphora using previous entity {candidate_entity}",
            )

    # 4. Default: Treat as independent query
    return ContextResolution(
        is_followup=False,
        resolved_retrieval_query=text,
        target_entity=None,
        reason="No anaphoric dependency detected; treated as independent turn",
    )
