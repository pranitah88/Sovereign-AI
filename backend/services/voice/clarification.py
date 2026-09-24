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


def check_query_completeness(query: str) -> tuple[bool, str]:
    """
    Deterministically check if an utterance appears incomplete.
    Signals:
    1. Trailing conjunctions: and, or, but, because, if
    2. Trailing prepositions/articles without object: about, with, for, to, at, by, from, into, through, as, the, a, an
       (except legitimate complete forms like 'what is this for?', 'used for')
    3. Truncated question starters: 'what is the', 'what are the', 'who is the', 'where is the', 'how does the', 'can you tell me about'
    4. Missing object after action verbs: 'explain', 'tell me', 'show me', 'describe', 'list', 'define'
    
    Returns (is_incomplete, reason).
    """
    text = (query or "").strip()
    if not text:
        return True, "empty_query"

    clean = re.sub(r"[\s\.\,\?\!\:\;\-]+$", "", text).strip()
    if not clean:
        return True, "empty_query"

    clean_lower = clean.lower()
    words = clean_lower.split()

    # Rule 1: Trailing conjunctions (unquestionably incomplete)
    # Examples: "Give me information about MRP and", "Tell me about the CDU and"
    trailing_conjunctions = {"and", "or", "but", "because", "if"}
    if words and words[-1] in trailing_conjunctions:
        return True, f"trailing_conjunction:{words[-1]}"

    # Rule 2: Trailing articles / determiners
    # Examples: "What is the", "Tell me about a"
    trailing_articles = {"the", "a", "an"}
    if words and words[-1] in trailing_articles:
        return True, f"trailing_article:{words[-1]}"

    # Rule 3: Trailing prepositions without valid question completion
    # Valid endings: "what is this for", "what is it for", "what is that used for", "what is it about"
    legitimate_preposition_patterns = [
        r"^(what|who)\s+(is|are)\s+(this|that|it)\s+(for|about)$",
        r"\bused\s+for$",
        r"^(what|which)\s+.*?\s+used\s+for$",
    ]
    is_legitimate_prep_ending = any(bool(re.search(p, clean_lower)) for p in legitimate_preposition_patterns)

    trailing_prepositions = {"about", "with", "to", "at", "by", "from", "into", "through", "as"}
    if words and words[-1] in trailing_prepositions and not is_legitimate_prep_ending:
        return True, f"trailing_preposition:{words[-1]}"
    if words and words[-1] == "for" and not is_legitimate_prep_ending:
        return True, "trailing_preposition:for"

    # Rule 4: Bare action verbs lacking an object or elaboration
    # Examples: "Explain", "Can you explain", "Please explain", "Tell me", "Show me", "Describe", "List", "Define"
    # But NOT: "Explain in detail.", "Explain further.", "Explain CDU.", "Tell me more."
    bare_verbs_pattern = r"^\s*(can\s+you\s+)?(could\s+you\s+)?(please\s+)?(explain|tell\s+me|show\s+me|describe|list|define)\s*$"
    if re.match(bare_verbs_pattern, clean_lower):
        return True, "missing_object_after_verb"

    # Rule 5: Truncated question structure
    # Examples: "What is the", "Where is the", "How does the", "Can you tell me about", "What is"
    truncated_question_patterns = [
        r"^\s*(what|who|where|when|why|how)\s+(is|are|was|were|do|does|did)\s+(the|a|an)\s*$",
        r"^\s*(can\s+you|could\s+you)\s+tell\s+me\s+about\s*$",
        r"^\s*(what|who|where|when|why|how)\s+(is|are|was|were|do|does|did)\s*$",
    ]
    if any(bool(re.match(p, clean_lower)) for p in truncated_question_patterns):
        return True, "truncated_question"

    return False, ""


def check_ambiguous_entity(query: str) -> tuple[bool, str, list[str]]:
    """
    Check if query contains ambiguous entities such as 'MRP' (which could mean MRP or MRPL).
    Preserves 'MRP' verbatim (never fuzzy matches or converts to MRPL).
    Returns (is_ambiguous, reason, candidate_list).
    """
    text = (query or "").strip()
    if not text:
        return False, "", []

    # Check for standalone token 'MRP'
    if re.search(r"\bMRP\b", text):
        return True, "ambiguous_acronym_mrp", ["MRP", "MRPL"]
    if re.search(r"\bmrp\b", text, re.IGNORECASE) and not re.search(r"\bmrpl\b", text, re.IGNORECASE):
        return True, "ambiguous_acronym_mrp", ["MRP", "MRPL"]

    return False, "", []


def detect_ambiguity(
    query: str,
    asr_confidence: float = 1.0,
    language: str = "en",
    confidence_threshold: float = 0.55,
) -> ClarificationIntent | None:
    """
    Detect genuine ambiguity or ASR failure in user speech.
    Reasons for spoken voice clarification:
    1. REPETITIVE ASR HALLUCINATION: Pathological repetition or runaway loops -> asks to repeat.
    2. ASR FAILURE: Speech cannot be understood (confidence < threshold) -> asks to repeat.
    3. INCOMPLETE & AMBIGUOUS QUERY: Trailing conjunction/sentence ending with ambiguous entity (e.g., 'Give me information about MRP and') -> asks concise clarification.
    4. INCOMPLETE QUERY: Dangling sentence structure (e.g., 'Tell me about the CDU and', 'What is the', 'Explain') -> asks to complete.
    5. SEMANTIC AMBIGUITY: Speech is clear, but equipment tag lacks suffix (e.g. 11-P-101 matching 11-P-101A and 11-P-101B).
    
    All clear queries (e.g. 'What is MRPL?', 'What is MRP?', 'What is CDU?', 'Tell me about CDU.', 'What are the main units in MRPL?')
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

    # ── 3. INCOMPLETE AND AMBIGUOUS QUERY DETECTION ──────────────────────────
    is_incomplete, inc_reason = check_query_completeness(text)
    is_ambig_ent, amb_reason, candidate_entities = check_ambiguous_entity(text)

    if is_incomplete and is_ambig_ent:
        # Example: "Give me information about MRP and"
        if lang_code == "hi":
            q = "क्या आप कृपया अपना प्रश्न पूरा कर सकते हैं? और क्या आपका मतलब MRP से है या MRPL से?"
        elif lang_code == "mr":
            q = "कृपया आपला प्रश्न पूर्ण कराल का? तसेच, आपल्याला MRP बद्दल माहिती हवी आहे की MRPL बद्दल?"
        else:
            q = "Could you complete your question? Also, do you mean MRP or MRPL?"

        return ClarificationIntent(
            ambiguity_type="incomplete_ambiguous",
            clarification_question=q,
            original_query=text,
            candidate_tags=candidate_entities,
            language=lang_code,
        )

    if is_incomplete:
        # Example: "Tell me about the CDU and", "What is the", "Explain", "Can you tell me about"
        if lang_code == "hi":
            q = "क्या आप कृपया अपना प्रश्न पूरा कर सकते हैं?"
        elif lang_code == "mr":
            q = "कृपया आपला प्रश्न पूर्ण कराल का?"
        else:
            q = "Could you please complete your question?"

        return ClarificationIntent(
            ambiguity_type="incomplete_query",
            clarification_question=q,
            original_query=text,
            language=lang_code,
        )

    # ── 4. SEMANTIC AMBIGUITY (Ambiguous Equipment Tag) ────────────────────────
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

    # Incomplete & Ambiguous Entity Resolution (e.g. user answered "MRPL" or "MRP")
    if ambiguity_type in ("incomplete_ambiguous", "incomplete_query", "ambiguous_entity"):
        # Check if user specified MRPL vs MRP
        if re.search(r"\bMRPL\b", resp, re.IGNORECASE):
            resolved = re.sub(r"\bMRP\b", "MRPL", original_query, flags=re.IGNORECASE)
            # Remove trailing conjunction
            resolved = re.sub(r"\s+(and|or|but|because|if|about)\s*[\.\,\?\!]*$", "", resolved, flags=re.IGNORECASE).strip()
            # If user provided continuation beyond "MRPL", append it
            extra = re.sub(r"^(i\s+mean\s+|it\s+is\s+)?MRPL\s*(and\s+)?", "", resp, flags=re.IGNORECASE).strip()
            if extra:
                resolved = f"{resolved} and {extra}".strip()
            return resolved
        elif re.search(r"\bMRP\b", resp, re.IGNORECASE):
            resolved = re.sub(r"\s+(and|or|but|because|if|about)\s*[\.\,\?\!]*$", "", original_query, flags=re.IGNORECASE).strip()
            extra = re.sub(r"^(i\s+mean\s+|it\s+is\s+)?MRP\s*(and\s+)?", "", resp, flags=re.IGNORECASE).strip()
            if extra:
                resolved = f"{resolved} and {extra}".strip()
            return resolved
        elif len(resp.split()) >= 3:
            # User spoke an entirely new query or continuation
            return resp

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
