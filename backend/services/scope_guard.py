"""
Deterministic Python Scope Guard — Pre-LLM Boundary Protection.

Enforces strict domain boundaries BEFORE any LLM inference, vector retrieval,
or code execution occurs. Python controls the security boundary deterministically;
models never make authorization or scope decisions.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import yaml

from backend.services.audit import audit_log
from backend.services.security_guards import detect_prompt_injection

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "scope_rules.yaml"

_rules_cache: dict[str, Any] | None = None


def _load_scope_rules() -> dict[str, Any]:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    if _RULES_PATH.exists():
        try:
            with open(_RULES_PATH, "r", encoding="utf-8") as f:
                _rules_cache = yaml.safe_load(f) or {}
            logger.info("Loaded Scope Guard rules from %s", _RULES_PATH)
            return _rules_cache
        except Exception as exc:
            logger.error("Failed to load scope rules from %s: %s", _RULES_PATH, exc)

    _rules_cache = {}
    return _rules_cache


@dataclass
class ScopeDecision:
    allowed: bool
    scope_status: str  # "ALLOWED" | "REJECTED"
    scope_category: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "scope_status": self.scope_status,
            "scope_category": self.scope_category,
            "reason": self.reason,
        }


def check_scope(
    query: str,
    user: dict | None = None,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
    has_uploaded_doc: bool = False,
) -> ScopeDecision:
    """
    Deterministic evaluation of user query scope before expensive RAG or LLM calls.

    Returns ScopeDecision.
    """
    text = (query or "").strip()
    if not text and not has_image and not has_scanned_pdf and not has_uploaded_doc:
        return ScopeDecision(
            allowed=False,
            scope_status="REJECTED",
            scope_category="empty_query",
            reason="Query is empty. Please enter an operational question, engineering task, or document command.",
        )

    # 1. Baseline Prompt Injection Check
    is_injection, matched_pattern = detect_prompt_injection(text)
    if is_injection:
        username = user.get("username", "anonymous") if user else "anonymous"
        user_id = user.get("id") if user else None
        audit_log(
            action="PROMPT_INJECTION_DETECTED",
            outcome="denied",
            user_id=user_id,
            username=username,
            target=f"query_text:{text[:100]}",
            details={"pattern": matched_pattern, "full_query": text[:250]},
        )
        return ScopeDecision(
            allowed=False,
            scope_status="REJECTED",
            scope_category="prompt_injection",
            reason=f"Security violation: Query matched forbidden prompt override pattern '{matched_pattern}'.",
        )

    # 2. Multimodal / Image Upload Tasks are always in-scope
    if has_image:
        return ScopeDecision(
            allowed=True,
            scope_status="ALLOWED",
            scope_category="vision_and_drawings",
            reason="Multimodal drawing/image inspection task.",
        )

    # 3. Document Analysis Tasks are always in-scope
    if has_scanned_pdf or has_uploaded_doc:
        return ScopeDecision(
            allowed=True,
            scope_status="ALLOWED",
            scope_category="document_analysis",
            reason="Document upload analysis task.",
        )

    rules = _load_scope_rules()
    text_lower = text.lower()

    # 4. Check Prohibited Domains & Out-of-Scope Patterns
    for prohibited in rules.get("prohibited_domains", []):
        for pattern in prohibited.get("patterns", []):
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    reason = prohibited.get("reason", "This topic is outside the MRPL sovereign engineering scope.")
                    username = user.get("username", "anonymous") if user else "anonymous"
                    user_id = user.get("id") if user else None
                    audit_log(
                        action="SCOPE_GUARD_BLOCKED",
                        outcome="denied",
                        user_id=user_id,
                        username=username,
                        target=f"query:{text[:100]}",
                        details={"category": prohibited.get("id"), "matched_pattern": pattern, "query": text[:250]},
                    )
                    return ScopeDecision(
                        allowed=False,
                        scope_status="REJECTED",
                        scope_category=prohibited.get("id", "out_of_scope"),
                        reason=f"Request Out of Scope ({prohibited.get('name', 'General')}): {reason}",
                    )
            except Exception as e:
                logger.warning("Error evaluating scope regex '%s': %s", pattern, e)

    # 5. Check Supported Industrial Categories
    matched_categories = []
    for cat in rules.get("supported_categories", []):
        cat_id = cat.get("id")
        for kw in cat.get("keywords", []):
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                matched_categories.append(cat_id)
                break

    if matched_categories:
        primary_cat = matched_categories[0]
        return ScopeDecision(
            allowed=True,
            scope_status="ALLOWED",
            scope_category=primary_cat,
            reason=f"Matched supported industrial domain: {primary_cat}",
        )

    # 6. Check for Technical / Code / Engineering Calculation patterns
    has_math_ops = bool(re.search(r"[\+\-\*\/=><]\s*\d+|\d+\s*[\+\-\*\/=><]", text))
    has_engineering_units = bool(re.search(r"\b\d+\s*(?:bar|psi|kpa|mpa|degc|deg\s*c|celsius|fahrenheit|kg|mt|tph|m3|bpd|rpm|mw|kw|kv|mm|cm|m)\b", text_lower))
    has_code_indicators = bool(re.search(r"\b(def|import|class|print|return|lambda|select|from|where|docker|bash|pip|curl|python|script|pandas|numpy|dataframe)\b", text_lower))
    has_engineering_terms = bool(re.search(r"\b(unit|plant|steam|flow|pressure|temp|temperature|heat|pipeline|loss|gain|audit|iso|valve|pump|heater|refinery)\b", text_lower))

    if has_math_ops or has_engineering_units or has_code_indicators or has_engineering_terms:
        return ScopeDecision(
            allowed=True,
            scope_status="ALLOWED",
            scope_category="technical_calculation_and_code",
            reason="Technical, quantitative, or script-related engineering query.",
        )

    # 7. Conversational greetings and brief navigational questions are allowed under general support
    words = text_lower.split()
    if len(words) <= 4:
        greetings = {"hi", "hello", "hey", "help", "good morning", "good afternoon", "thanks", "thank you", "status"}
        if any(text_lower.startswith(g) for g in greetings) or text_lower in greetings:
            return ScopeDecision(
                allowed=True,
                scope_status="ALLOWED",
                scope_category="system_and_general",
                reason="Standard navigational or conversational query.",
            )

    # 8. Unclassified / Distant Query - Default to rejection if purely non-industrial
    # For ambiguous queries, if they are long narrative texts with zero industrial keywords, reject
    if len(words) > 8 and not any(k in text_lower for k in ["mrpl", "oil", "gas", "refin", "fuel", "energy", "petro", "plant"]):
        username = user.get("username", "anonymous") if user else "anonymous"
        user_id = user.get("id") if user else None
        audit_log(
            action="SCOPE_GUARD_BLOCKED",
            outcome="denied",
            user_id=user_id,
            username=username,
            target=f"query:{text[:100]}",
            details={"reason": "No industrial or refinery context detected", "query": text[:250]},
        )
        return ScopeDecision(
            allowed=False,
            scope_status="REJECTED",
            scope_category="unsupported_domain",
            reason="Request Out of Scope: No industrial, refinery, or technical context was detected. Please ask about MRPL refinery operations, technical manuals, plant safety, calculations, or engineering documents.",
        )

    # Default: allow as general RAG query against the local knowledge base
    return ScopeDecision(
        allowed=True,
        scope_status="ALLOWED",
        scope_category="general_industrial_query",
        reason="General industrial query routed to local knowledge retrieval.",
    )


def is_scope_guard_enabled() -> bool:
    """Return True if Scope Guard is active according to configuration."""
    rules = _load_scope_rules()
    return rules.get("enabled", True)


def evaluate_scope(
    query: str,
    username: str = "anonymous",
    role: str = "engineer",
    user: dict | None = None,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
    has_uploaded_doc: bool = False,
) -> ScopeDecision:
    """Convenience alias for check_scope accepting username/role directly."""
    resolved_user = user or {"username": username, "role": role}
    return check_scope(
        query=query,
        user=resolved_user,
        has_image=has_image,
        has_scanned_pdf=has_scanned_pdf,
        has_uploaded_doc=has_uploaded_doc,
    )

