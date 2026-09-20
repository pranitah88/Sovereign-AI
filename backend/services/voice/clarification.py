"""
Conversational Voice Clarification Service for MRPL Sovereign Voice Assistant.

Enforces natural, spoken voice-first clarification:
1. Detects low ASR confidence, ambiguous technical terms (e.g., '11-P-101' -> '11-P-101A' vs '11-P-101B'), or unclear intent.
2. Generates short, conversational clarification questions in en/hi/mr.
3. Merges the user's spoken answer with prior context to continue the governance pipeline.
4. Zero separate text confirmation modals or button clicks — 100% voice conversation.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from backend.services.voice.speech_renderer import render_speech_text

logger = logging.getLogger(__name__)


@dataclass
class ClarificationIntent:
    ambiguity_type: str  # "ambiguous_tag" | "missing_unit" | "low_confidence"
    clarification_question: str
    original_query: str
    candidate_tags: list[str] = field(default_factory=list)
    base_tag: str = ""
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguity_type": self.ambiguity_type,
            "clarification_question": self.clarification_question,
            "original_query": self.original_query,
            "candidate_tags": self.candidate_tags,
            "base_tag": self.base_tag,
            "language": self.language,
        }


@dataclass
class ClarificationContext:
    original_query: str
    ambiguity_type: str
    candidate_tags: list[str] = field(default_factory=list)
    base_tag: str = ""
    language: str = "en"

    def resolve_reply(self, reply: str) -> tuple[str, bool]:
        resolved = resolve_clarification_response(reply, {
            "original_query": self.original_query,
            "ambiguity_type": self.ambiguity_type,
            "candidate_tags": self.candidate_tags,
            "base_tag": self.base_tag,
            "language": self.language,
        }, self.language)
        is_resolved = (resolved != reply) or (self.ambiguity_type in ("asr_failure", "low_confidence"))
        return resolved, is_resolved


def detect_ambiguity(
    query: str,
    asr_confidence: float = 1.0,
    language: str = "en",
    confidence_threshold: float = 0.55,
) -> ClarificationIntent | None:
    """
    Detect genuine ambiguity or ASR failure in user speech.
    There are ONLY three reasons for spoken voice clarification:
    1. REPETITIVE ASR HALLUCINATION: Pathological repetition or runaway loops -> asks to repeat.
    2. ASR FAILURE: Speech cannot be understood (confidence < threshold) -> asks to repeat.
    3. SEMANTIC AMBIGUITY: Speech is clear, but equipment tag lacks suffix (e.g. 11-P-101 matching 11-P-101A and 11-P-101B).
    
    All clear queries (e.g. 'What is MRPL?', 'What is CDU?', 'What is the pressure of 11-P-101A?')
    proceed directly without confirmation.
    """
    text = (query or "").strip()
    if not text:
        return None

    lang_code = (language or "en").lower()
    if lang_code not in ("en", "hi", "mr"):
        lang_code = "en"

    # ── 1. REPETITIVE ASR HALLUCINATION / RUNAWAY REPETITION ─────────────────
    from backend.services.voice.asr import validate_asr_quality
    quality = validate_asr_quality(text, asr_confidence)
    if not quality["valid"] and quality["status"] == "repetition_hallucination":
        if lang_code == "hi":
            q = "क्षमा करें, मुझे समझ नहीं आया। क्या आप कृपया दोबारा बोल सकते हैं?"
        elif lang_code == "mr":
            q = "क्षमस्व, मला स्पष्ट ऐकू आले नाही. कृपया पुन्हा सांगाल का?"
        else:
            q = "Sorry, I didn't catch that. Could you say it again?"

        return ClarificationIntent(
            ambiguity_type="repetition_hallucination",
            clarification_question=q,
            original_query=text,
            language=lang_code,
        )

    # ── 2. ASR FAILURE (Low Confidence) ───────────────────────────────────────
    # NEVER guess missing words, reconstruct sentences, or echo 'Did you mean: ...'
    if asr_confidence < confidence_threshold or (not quality["valid"] and quality["status"] == "empty"):
        if lang_code == "hi":
            q = "क्षमा करें, मुझे समझ नहीं आया। क्या आप कृपया दोबारा बोल सकते हैं?"
        elif lang_code == "mr":
            q = "क्षमस्व, मला स्पष्ट ऐकू आले नाही. कृपया पुन्हा सांगाल का?"
        else:
            q = "Sorry, I didn't catch that. Could you repeat it?"

        return ClarificationIntent(
            ambiguity_type="asr_failure",
            clarification_question=q,
            original_query=text,
            language=lang_code,
        )

    # ── 3. SEMANTIC AMBIGUITY (Ambiguous Equipment Tag) ────────────────────────
    # Only when ASR is clear, but meaning has multiple valid registered equipment assets
    base_tag_match = re.search(r"\b(\d{2}-[A-Z]-\d{3})(?![A-Z0-9])\b", text.upper())
    if base_tag_match:
        base_tag = base_tag_match.group(1)
        try:
            from backend.services.vision_verification import KNOWN_EQUIPMENT_REGISTER
            candidates = sorted([
                tag for tag in KNOWN_EQUIPMENT_REGISTER.keys()
                if tag.startswith(base_tag) and tag != base_tag
            ])
            if len(candidates) > 1:
                options_en = " or ".join(candidates)
                if lang_code == "hi":
                    q = f"क्या आपका मतलब {' या '.join(candidates)} से है?"
                elif lang_code == "mr":
                    q = f"आपल्याला {' की '.join(candidates)} बद्दल माहिती हवी आहे?"
                else:
                    q = f"Do you mean {options_en}?"

                return ClarificationIntent(
                    ambiguity_type="ambiguous_tag",
                    clarification_question=q,
                    original_query=text,
                    candidate_tags=candidates,
                    base_tag=base_tag,
                    language=lang_code,
                )
        except Exception as err:
            logger.warning("Error checking known equipment register: %s", err)

    # ── Clear Query: Proceed directly to governance & processing ─────────────
    return None


def resolve_clarification_response(
    user_response: str,
    context: dict[str, Any],
    language: str = "en",
) -> str:
    """
    Merge the user's spoken answer with prior clarification context.
    If user initiates a new standalone query (e.g. 'What is MRPL?'), cancels the clarification
    and processes the new query independently.
    """
    resp = (user_response or "").strip()
    if not resp:
        return context.get("original_query", "")

    resp_lower = resp.lower().rstrip(".,!?")
    original_query = context.get("original_query", "")
    ambiguity_type = context.get("ambiguity_type", "")
    candidate_tags = context.get("candidate_tags", [])
    base_tag = context.get("base_tag", "")

    # ASR Failure or Repetition Hallucination repeat: user spoke a new/repeated query
    if ambiguity_type in ("asr_failure", "low_confidence", "repetition_hallucination"):
        return resp

    # Check if user query is an independent new standalone question (topic shift)
    standalone_patterns = [
        r"^(what|who|where|when|why|how|which|explain|tell\s+me|give\s+me|show|describe|define|status|analyze|is\s+there|are\s+there|can\s+you)\b",
        r"\?$",
    ]
    is_standalone_query = any(bool(re.search(p, resp_lower)) for p in standalone_patterns)

    # Specific check: if user asked "What is MRPL?" or another topic while clarification was active,
    # immediately treat as standalone query overriding clarification
    if is_standalone_query and not any(c.lower() in resp_lower for c in candidate_tags):
        logger.info("Clarification overridden by standalone query: '%s'", resp)
        return resp

    # Check for affirmation ("yes", "yeah", "correct", "haan", "ho", etc.)
    is_affirmative = bool(re.search(r"^(yes|yeah|yep|correct|right|sure|ha|haan|ho|barobar)\b", resp_lower))

    # Semantic Ambiguity for Equipment Tag
    if ambiguity_type in ("ambiguous_tag", "semantic_ambiguity") and candidate_tags and base_tag:
        # Check explicit candidate tag match (e.g. "11-P-101A", "11-p-101a", "11 P 101 A")
        for cand in candidate_tags:
            cand_norm = cand.lower().replace("-", "")
            cand_clean = cand.lower()
            resp_norm = resp_lower.replace("-", "").replace(" ", "")
            if cand_clean in resp_lower or cand_norm in resp_norm:
                return re.sub(re.escape(base_tag), cand, original_query, flags=re.IGNORECASE)

        # Check spoken suffix matches (e.g., "the A one", "A", "pump A", "the B one", "B", "pump B")
        for cand in candidate_tags:
            cand_suffix = cand[-1].lower()  # 'a' or 'b'
            suffix_patterns = [
                rf"\b(the\s+)?{cand_suffix}(\s+one)?\b",
                rf"\bpump\s+{cand_suffix}\b",
                rf"\b{cand_suffix}\b",
            ]
            for pat in suffix_patterns:
                if re.search(pat, resp_lower):
                    return re.sub(re.escape(base_tag), cand, original_query, flags=re.IGNORECASE)

        # If user affirmed ("Yes", "the first one"), default to primary candidate
        if is_affirmative or "first" in resp_lower or "motor" in resp_lower:
            return re.sub(re.escape(base_tag), candidate_tags[0], original_query, flags=re.IGNORECASE)
        elif ("second" in resp_lower or "turbine" in resp_lower) and len(candidate_tags) > 1:
            return re.sub(re.escape(base_tag), candidate_tags[1], original_query, flags=re.IGNORECASE)

    # If length >= 4 words and no candidate tag mentioned, treat as new independent query
    if len(resp.split()) >= 4:
        return resp

    return resp
