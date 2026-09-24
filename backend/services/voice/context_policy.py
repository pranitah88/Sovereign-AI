"""
MRPL Sovereign AI Workbench — Conversational Context Policy.
Distinguishes contextual follow-ups from independent topic shifts.
Resolves follow-ups and anaphora deterministically for retrieval without mutating current_user_query.
Strictly prefers structured previous-turn context (entities, retrieval queries, task metadata, sources)
over scanning unstructured generated assistant prose.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel


ANAPHORIC_PRONOUN_PATTERNS = [
    r"\b(its|it|this|that|these|those|the\s+same|the\s+equipment|the\s+pump|the\s+unit)\b",
    r"\b(what\s+about|how\s+about|and\s+its)\b",
]

FOLLOWUP_ELABORATION_PATTERNS = [
    r"^\s*(explain\s+in\s+detail|explain\s+further|tell\s+me\s+more|more\s+details|elaborate|go\s+deeper|expand\s+on\s+that|continue|go\s+on|what\s+else|explain\s+each\s+one|in\s+more\s+detail|can\s+you\s+explain|give\s+details|details)\b",
    r"\b(explain\s+in\s+detail|in\s+more\s+detail|give\s+more\s+details|explain\s+each\s+one|elaborate\s+further)\b",
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
    ("DCU", "Delayed Coker Unit"),
    ("DELAYED COKER", "Delayed Coker Unit"),
    ("VBU", "Visbreaker Unit"),
    ("VISBREAKER", "Visbreaker Unit"),
    ("BITUMEN", "Bitumen Unit"),
    ("ISOM", "Isomerisation Unit"),
    ("ISOMERISATION", "Isomerisation Unit"),
    ("ISOMERIZATION", "Isomerisation Unit"),
    ("CCR", "Continuous Catalytic Reforming"),
    ("PLATFORMING", "Platforming Unit"),
    ("HGU", "Hydrogen Generation Unit"),
    ("SRU", "Sulphur Recovery Unit"),
    ("MEROX", "Merox Treating Unit"),
    ("TREATING UNIT", "Treating Unit"),
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


def extract_query_subject(query: str) -> str:
    """
    Extract the core subject phrase from a user query by stripping
    common leading question frames (e.g. 'what are the', 'tell me about').
    """
    text = (query or "").strip()
    clean = re.sub(
        r"^(what\s+(is|are|was|were|does|do)(\s+the)?|tell\s+me\s+about(\s+the)?|can\s+you\s+explain(\s+the)?|explain(\s+the)?|give\s+me(\s+the)?|list(\s+the)?|show\s+me(\s+the)?|describe(\s+the)?)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    clean = clean.rstrip("?.!")
    return clean or text


def _resolve_structured_subject(
    conversation_history: list[dict[str, Any]] | None,
    active_equipment_tag: str | None = None,
) -> dict[str, Any] | None:
    """
    Resolve the subject from structured previous-turn context whenever available.

    Preference order:
    1. Previous turn's resolved subject/entity (explicit target_entity or tag/unit in prev user query)
    2. Previous turn's resolved retrieval query (retrieval_query metadata)
    3. Previous turn's task/topic metadata (task_type == 'RAG' / 'DOCUMENT_ANALYSIS', target_document, topic)
    4. Previous retrieved evidence metadata (sources list with document/unit metadata)

    Strict rule: Do NOT infer the subject primarily by scanning generated assistant prose.
    The assistant response may only be used as supplementary context, not as authoritative.
    If no reliable structured subject exists, returns None.
    """
    if active_equipment_tag:
        return {
            "entity": active_equipment_tag,
            "query_subject": active_equipment_tag,
            "retrieval_query": None,
            "source": "active_equipment_tag",
        }

    if not conversation_history:
        return None

    prev_user_msg: dict[str, Any] | None = None
    prev_asst_msg: dict[str, Any] | None = None

    for msg in reversed(conversation_history[-4:]):
        role = msg.get("role")
        if role == "user" and not prev_user_msg:
            prev_user_msg = msg
        elif role == "assistant" and not prev_asst_msg:
            prev_asst_msg = msg

    prev_user_content = (prev_user_msg.get("content") or "").strip() if prev_user_msg else ""

    # =========================================================================
    # 1. Previous turn's resolved subject/entity
    # =========================================================================
    # 1a. Explicit structured entity from turn metadata
    explicit_entity = (
        (prev_asst_msg and (prev_asst_msg.get("target_entity") or prev_asst_msg.get("resolved_entity") or prev_asst_msg.get("entity")))
        or (prev_user_msg and (prev_user_msg.get("target_entity") or prev_user_msg.get("entity")))
    )
    if explicit_entity:
        query_subject = extract_query_subject(prev_user_content) if prev_user_content else str(explicit_entity)
        return {
            "entity": str(explicit_entity),
            "query_subject": query_subject,
            "retrieval_query": (prev_asst_msg or {}).get("retrieval_query"),
            "source": "explicit_entity_metadata",
        }

    # 1b. Entity extracted from previous USER query (authoritative user intent, not assistant prose)
    if prev_user_content:
        user_entities = extract_referenced_entities(prev_user_content)
        if user_entities:
            query_subject = extract_query_subject(prev_user_content)
            return {
                "entity": user_entities[0],
                "query_subject": query_subject,
                "retrieval_query": (prev_asst_msg or {}).get("retrieval_query"),
                "source": "prev_user_query_entity",
            }

    # =========================================================================
    # 2. Previous turn's resolved retrieval query
    # =========================================================================
    retrieval_q = (
        (prev_asst_msg and (prev_asst_msg.get("retrieval_query") or prev_asst_msg.get("resolved_retrieval_query")))
        or (prev_user_msg and (prev_user_msg.get("retrieval_query") or prev_user_msg.get("resolved_retrieval_query")))
    )
    # Also check tool_calls for rag_search query
    if not retrieval_q and prev_asst_msg and prev_asst_msg.get("tool_calls"):
        tool_calls = prev_asst_msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("tool") in ("rag_search", "search_kb") and tc.get("query"):
                    retrieval_q = tc.get("query")
                    break

    if retrieval_q and isinstance(retrieval_q, str) and retrieval_q.strip():
        query_subject = extract_query_subject(retrieval_q)
        q_entities = extract_referenced_entities(retrieval_q)
        return {
            "entity": q_entities[0] if q_entities else None,
            "query_subject": query_subject,
            "retrieval_query": retrieval_q,
            "source": "prev_retrieval_query",
        }

    # =========================================================================
    # 3. Previous turn's task/topic metadata
    # =========================================================================
    task_type = (prev_asst_msg and prev_asst_msg.get("task_type")) or (prev_user_msg and prev_user_msg.get("task_type"))
    target_doc = (prev_asst_msg and prev_asst_msg.get("target_document")) or (prev_user_msg and prev_user_msg.get("target_document"))
    topic = (prev_asst_msg and prev_asst_msg.get("topic")) or (prev_user_msg and prev_user_msg.get("topic"))

    if target_doc:
        return {
            "entity": target_doc,
            "query_subject": target_doc,
            "retrieval_query": None,
            "source": "task_target_document",
        }

    if topic:
        return {
            "entity": topic,
            "query_subject": topic,
            "retrieval_query": None,
            "source": "task_topic",
        }

    # If previous turn was explicitly marked as RAG and had a user query
    if task_type == "RAG" and prev_user_content:
        query_subject = extract_query_subject(prev_user_content)
        entities = extract_referenced_entities(prev_user_content)
        return {
            "entity": entities[0] if entities else None,
            "query_subject": query_subject,
            "retrieval_query": prev_user_content,
            "source": "task_type_rag",
        }

    # =========================================================================
    # 4. Previous retrieved evidence metadata
    # =========================================================================
    sources = (prev_asst_msg and prev_asst_msg.get("sources")) or (prev_user_msg and prev_user_msg.get("sources"))
    if sources and isinstance(sources, list) and len(sources) > 0:
        first_src = sources[0]
        if isinstance(first_src, dict):
            src_doc = first_src.get("source") or first_src.get("title") or first_src.get("document_name")
            if src_doc:
                query_subject = extract_query_subject(prev_user_content) if prev_user_content else str(src_doc)
                doc_entities = extract_referenced_entities(str(src_doc))
                return {
                    "entity": doc_entities[0] if doc_entities else str(src_doc),
                    "query_subject": query_subject,
                    "retrieval_query": None,
                    "source": "retrieved_evidence_metadata",
                }

    # If no reliable structured subject exists, return None (do NOT scan prose)
    return None


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

    2. If the current query contains anaphoric pronouns, elliptical phrasing, or elaboration
       patterns (e.g., 'What is its function?', 'explain in detail', 'tell me more'),
       it is a candidate CONTEXTUAL FOLLOW-UP.
       -> Resolves the subject strictly from structured previous-turn context:
          1. Previous turn's resolved subject/entity
          2. Previous turn's resolved retrieval query
          3. Previous turn's task/topic metadata
          4. Previous retrieved evidence metadata
       -> Does NOT infer subject primarily by scanning generated assistant prose.
       -> If no reliable structured subject exists, does NOT promote to follow-up.
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

    # 1b. Check if query is incomplete or ambiguous — must NEVER inherit prior turn context
    from backend.services.voice.clarification import check_query_completeness, check_ambiguous_entity
    is_incomplete, _ = check_query_completeness(text)
    is_ambig, _, _ = check_ambiguous_entity(text)
    if is_incomplete or (is_ambig and "mrp" in text_lower):
        return ContextResolution(
            is_followup=False,
            resolved_retrieval_query=text,
            target_entity=None,
            reason="Incomplete or ambiguous query; cannot inherit prior context",
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

    # 3. Check for anaphoric pronouns, elaboration patterns, or short fragments
    has_anaphora = any(bool(re.search(p, text_lower)) for p in ANAPHORIC_PRONOUN_PATTERNS)
    has_elaboration = any(bool(re.search(p, text_lower)) for p in FOLLOWUP_ELABORATION_PATTERNS)
    is_short_fragment = len(text.split()) <= 4 and not any(
        text_lower.startswith(w) for w in ["what is mrpl", "who is", "where is mrpl"]
    )

    if has_anaphora or has_elaboration or is_short_fragment:
        # Resolve subject from structured previous-turn context (Order of preference 1-4)
        structured = _resolve_structured_subject(
            conversation_history=conversation_history,
            active_equipment_tag=active_equipment_tag,
        )

        if structured is not None:
            candidate_entity = structured.get("entity")
            query_subject = structured.get("query_subject")
            prev_retrieval_q = structured.get("retrieval_query")
            source_tier = structured.get("source", "structured_context")

            # Formulate resolved retrieval query
            resolved_query = text

            if has_anaphora:
                # Replace possessive "its <property>" -> "the \1 of <candidate_entity>"
                subject_label = candidate_entity or query_subject or ""
                resolved_query = re.sub(
                    r"\bits\s+([a-zA-Z]+)\b",
                    rf"the \1 of {subject_label}",
                    resolved_query,
                    flags=re.IGNORECASE,
                )
                # Replace remaining standalone pronouns
                resolved_query = re.sub(
                    r"\b(its|it|this|that|the\s+pump|the\s+equipment|the\s+unit)\b",
                    subject_label,
                    resolved_query,
                    flags=re.IGNORECASE,
                )
                if subject_label and subject_label.lower() not in resolved_query.lower():
                    resolved_query = f"{resolved_query} of {subject_label}"
            else:
                # Elaboration / fragment: combine current intent with the previous subject
                target_sub = query_subject or candidate_entity or prev_retrieval_q
                if target_sub and target_sub.lower() not in text_lower:
                    resolved_query = f"{text} {target_sub}"

            target_entity_result = candidate_entity or query_subject

            return ContextResolution(
                is_followup=True,
                resolved_retrieval_query=resolved_query.strip(),
                target_entity=target_entity_result,
                reason=f"Resolved follow-up using structured previous-turn context ({source_tier}): {target_entity_result}",
            )

    # 4. Default: Treat as independent query
    return ContextResolution(
        is_followup=False,
        resolved_retrieval_query=text,
        target_entity=None,
        reason="No anaphoric or contextual follow-up dependency detected; treated as independent turn",
    )
