"""
Technical Entity Extraction, Language Fidelity, and Transliteration Prevention
for the MRPL Sovereign AI Workbench.

Ensures:
1. Response language strictly follows CURRENT_USER_QUERY.
2. Technical entities, process units, equipment tags, and acronyms from retrieved evidence
   are preserved verbatim in English/source terminology without transliteration or corruption.
3. Transliterated Devanagari technical terms are detected and repaired.
4. Language and entity fidelity metrics are recorded in execution traces.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Common English stopwords to exclude when extracting technical capitalized entities
STOPWORDS = {
    "A", "An", "The", "And", "Or", "In", "On", "At", "By", "For", "With",
    "About", "Against", "Between", "Into", "Through", "During", "Before",
    "After", "Above", "Below", "To", "From", "Up", "Down", "These", "Those",
    "This", "That", "There", "Here", "Where", "When", "Why", "How", "All",
    "Any", "Both", "Each", "Few", "More", "Most", "Other", "Some", "Such",
    "No", "Nor", "Not", "Only", "Own", "Same", "So", "Than", "Too", "Very",
    "Can", "Will", "Just", "Should", "Now", "Under", "Over", "Source", "Page",
    "Document", "MRPL", "Table", "Total", "Figure", "Report", "Annual",
}

# Domain-specific transliteration map: Devanagari transliteration -> Official English term
TRANSLITERATION_REPAIR_MAP: dict[str, str] = {
    # Process units
    "प्लॅटफासिंग": "Platforming",
    "प्लॅटफासिंग युनिट": "Platforming Unit",
    "प्लॅटफॉर्मिंग": "Platforming",
    "प्लॅटफॉर्मिंग युनिट": "Platforming Unit",
    "प्लेटफॉर्मिंग": "Platforming",
    "प्लेटफॉर्मिंग यूनिट": "Platforming Unit",
    "डिलेस्ड कोकर": "Delayed Coker",
    "डिलेस्ड कोकर युनिट": "Delayed Coker Unit",
    "डिलेड कोकर": "Delayed Coker",
    "डिलेड कोकर युनिट": "Delayed Coker Unit",
    "डीलेड कोकर": "Delayed Coker",
    "डीलेड कोकर यूनिट": "Delayed Coker Unit",
    "बिटमॅन": "Bitumen",
    "बिटमॅन युनिट": "Bitumen Unit",
    "बिटुमेन": "Bitumen",
    "बिटुमेन युनिट": "Bitumen Unit",
    "बिटूमेन": "Bitumen",
    "बिटूमेन यूनिट": "Bitumen Unit",
    "हायड्रोक्रॅकर": "Hydrocracker",
    "हायड्रोक्रॅकर युनिट": "Hydrocracker Unit",
    "हाइड्रोक्रैकर": "Hydrocracker",
    "हाइड्रोक्रैकर यूनिट": "Hydrocracker Unit",
    "व्हिसब्रेकर": "Visbreaker",
    "व्हिसब्रेकर युनिट": "Visbreaker Unit",
    "विस्ब्रेकर": "Visbreaker",
    "विस्ब्रेकर यूनिट": "Visbreaker Unit",
    "आइसोमेरायझेशन": "Isomerisation",
    "आइसोमेरायझेशन युनिट": "Isomerisation Unit",
    "आइसोमराइजेशन": "Isomerisation",
    "आइसोमराइजेशन यूनिट": "Isomerisation Unit",
    "क्रूड युनिट": "Crude Distillation Unit (CDU)",
    "क्रूड यूनिट": "Crude Distillation Unit (CDU)",
    "व्हॅक्यूम युनिट": "Vacuum Distillation Unit (VDU)",
    "वैक्यूम यूनिट": "Vacuum Distillation Unit (VDU)",
    "ट्रीटिंग युनिट": "Treating Unit",
    "ट्रीटिंग यूनिट": "Treating Unit",
    "हायड्रोजन युनिट": "Hydrogen Generation Unit (HGU)",
    "हाइड्रोजन यूनिट": "Hydrogen Generation Unit (HGU)",

    # Acronyms
    "पीएफसीयू": "PFCCU",
    "पीएफसीसीयू": "PFCCU",
    "सीडीयू": "CDU",
    "वीडीयू": "VDU",
    "सीसीआर": "CCR",
    "डीएचडीटी": "DHDT",
    "एचसीयू": "HCU",
    "एचजीयू": "HGU",
    "डीसीयू": "DCU",
    "एसआरयू": "SRU",
    "पीपीयू": "PPU",
    "मेरोक्स": "MEROX",
    "मॅरोक्स": "MEROX",
}


@dataclass
class EntityFidelityResult:
    query_language: str
    response_language: str
    language_decision_source: str
    technical_entities_detected: list[str]
    technical_entities_preserved: list[str]
    technical_entity_validation: str  # "PASSED" | "CORRECTED" | "FLAGGED"
    corrected_response: str
    was_repaired: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "QUERY_LANGUAGE": self.query_language,
            "RESPONSE_LANGUAGE": self.response_language,
            "LANGUAGE_DECISION_SOURCE": self.language_decision_source,
            "TECHNICAL_ENTITIES_DETECTED": self.technical_entities_detected,
            "TECHNICAL_ENTITIES_PRESERVED": self.technical_entities_preserved,
            "TECHNICAL_ENTITY_VALIDATION": self.technical_entity_validation,
            "was_repaired": self.was_repaired,
            "details": self.details,
        }


def detect_query_language_deterministic(query: str) -> str:
    """
    Determine the query language deterministically based ONLY on CURRENT_USER_QUERY.
    Returns: 'en' | 'hi' | 'mr'

    Rules:
    1. Pure Latin / English query -> 'en'
    2. Devanagari script:
       - Uses Marathi marker scoring vs Hindi marker scoring
       - Returns 'mr' if Marathi markers dominate
       - Returns 'hi' if Hindi markers dominate or default for Devanagari
    3. Never inspect conversation history or retrieved documents for query language.
    """
    if not query or not query.strip():
        return "en"

    clean_q = query.strip()
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", clean_q))
    if not has_devanagari:
        return "en"

    from backend.services.multilingual import MARATHI_MARKERS, HINDI_MARKERS

    # Additional high-precision Marathi and Hindi markers
    extended_marathi = MARATHI_MARKERS + [
        r"\bमधील\b", r"\bकोणते\b", r"\bकोणती\b", r"\bआहेत\b", r"\bयांचे\b",
        r"\bसांगा\b", r"\bकसे\b", r"\bकाय\b", r"\bयेथे\b", r"\bदिलेली\b",
        r"\bमाहिती\b", r"\bकरा\b", r"\bहोते\b", r"\bझाली\b", r"\bयुनिट्स\b",
    ]
    extended_hindi = HINDI_MARKERS + [
        r"\bकी\b", r"\bके\b", r"\bमें\b", r"\bहैं\b", r"\bहै\b", r"\bबताइए\b",
        r"\bकौन\b", r"\bसी\b", r"\bक्या\b", r"\bहोते\b", r"\bदी\b", r"\bगई\b",
        r"\bयूनिट्स\b", r"\bइकाइयां\b", r"\bइकाइयाँ\b",
    ]

    q_lower = clean_q.lower()
    marathi_score = sum(1 for m in extended_marathi if re.search(m, q_lower))
    hindi_score = sum(1 for m in extended_hindi if re.search(m, q_lower))

    if marathi_score > hindi_score:
        return "mr"
    if hindi_score > marathi_score:
        return "hi"

    # Secondary check: specific Marathi suffixes (e.g. -तील, -मध्ये, -च्या)
    if re.search(r"[\u0900-\u097F]+(तील|मध्ये|च्या|साठी|बद्दल|वरून)", clean_q):
        return "mr"

    return "hi"


def extract_technical_entities(evidence_text: str) -> list[str]:
    """
    Extract domain-specific technical entities, unit names, process names,
    and uppercase acronyms from retrieved evidence.
    General mechanism — not limited to a static hardcoded list.
    """
    if not evidence_text:
        return []

    entities = set()

    # 1. Standalone uppercase acronyms (2 to 8 chars, optionally with numbers/hyphens)
    acronym_pattern = re.compile(r'\b[A-Z]{2,8}(?:-[A-Z0-9]+)?\b')
    for match in acronym_pattern.finditer(evidence_text):
        word = match.group(0)
        if word not in STOPWORDS and len(word) >= 2:
            entities.add(word)

    # 2. Multi-word technical process/unit phrases:
    # Capitalized sequences ending in Unit, Plant, System, Treater, Cracker, Reformer, etc.
    unit_phrase_pattern = re.compile(
        r'\b(?:[A-Z][a-z0-9\-/]+|[A-Z]{2,6})(?:\s+(?:[A-Z][a-z0-9\-/]+|[A-Z]{2,6}|and|of|the|for|in))*\s+'
        r'(?:Unit|Units|Plant|Plants|Section|Complex|Treater|Cracker|Reformer|Distillation|'
        r'Hydrotreater|Desulphurisation|Platforming|Merox|Coker|Visbreaker|Bitumen|'
        r'Isomerisation|Hydrocracker|Gasoline|Kerosene|Gas-oil|Treating|Facility|Facilities)\b'
    )
    for match in unit_phrase_pattern.finditer(evidence_text):
        phrase = match.group(0).strip()
        # Discard phrases that are mostly common sentence starters
        words = phrase.split()
        if words[0] in STOPWORDS and len(words) > 1:
            phrase = " ".join(words[1:])
        if len(phrase) > 3 and not all(w in STOPWORDS for w in phrase.split()):
            entities.add(phrase)

    # 3. Known technical refining processes mentioned in text
    known_refinery_terms = [
        "Crude Unit", "Crude Distillation Unit", "Vacuum Distillation Unit",
        "Hydrocracker Unit", "Continuous Catalytic Regeneration", "Platforming Unit",
        "Platforming", "CCR Platforming", "Isomerisation Unit",
        "Petro Fluidised Catalytic Cracking Unit", "PFCCU", "Delayed Coker Unit",
        "Delayed Coker", "Visbreaker Unit", "Visbreaker", "Bitumen Unit", "Bitumen",
        "Treating Unit", "LPG/Naphtha/Kerosene Merox Unit", "Merox Unit", "MEROX",
        "Gas Oil Hydro-Desulphurisation Unit", "GOHDS", "Diesel Hydrotreater Unit",
        "DHDT", "Coker Gas-oil Hydro-Treating Unit", "CHDT", "FCC Gasoline",
        "Hydrogen Generation Unit", "HGU", "Sulfur Recovery Unit", "SRU",
        "Polypropylene Unit", "PPU", "Captive Power Plant", "CPP",
    ]
    for term in known_refinery_terms:
        # Check case-insensitive boundary match in evidence
        if re.search(rf'\b{re.escape(term)}\b', evidence_text, re.IGNORECASE):
            entities.add(term)

    # 4. Equipment tags (e.g., 11-C-001, P-101A)
    tag_pattern = re.compile(r'\b[0-9]{1,3}-[A-Z]{1,4}-[0-9]{1,5}[A-Z]?\b|\b[A-Z]{1,4}-[0-9]{1,5}[A-Z]?\b')
    for match in tag_pattern.finditer(evidence_text):
        entities.add(match.group(0))

    # Clean and filter
    clean_entities = []
    for e in entities:
        c = e.strip(" ,.;:()[]{}'\"")
        if c and len(c) >= 2 and c not in STOPWORDS:
            clean_entities.append(c)

    # Sort by length descending so longer phrases match first
    clean_entities.sort(key=lambda x: len(x), reverse=True)
    return clean_entities


def repair_transliterated_technical_terms(text: str) -> tuple[str, list[str]]:
    """
    Search for transliterated or corrupted Devanagari technical terms
    and replace them with official English/source terminology.
    Returns: (repaired_text, list_of_repaired_terms)
    """
    if not text:
        return text, []

    repaired = text
    repaired_terms = []

    # Apply transliteration repair map (longest matches first)
    sorted_map = sorted(TRANSLITERATION_REPAIR_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for devanagari_term, official_term in sorted_map:
        if devanagari_term in repaired:
            repaired = repaired.replace(devanagari_term, official_term)
            repaired_terms.append(f"{devanagari_term} -> {official_term}")

    return repaired, repaired_terms


def validate_and_preserve_entities(
    response: str,
    query: str,
    evidence: str,
) -> EntityFidelityResult:
    """
    Full pipeline step for language & entity fidelity:
    1. Detect authoritative query language from `query` alone.
    2. Extract technical entities from `evidence`.
    3. Verify that response matches query language.
    4. If response is English, ensure zero Devanagari script.
    5. If response is Marathi/Hindi, repair any corrupted transliterations.
    6. Verify which technical entities are preserved.
    """
    target_lang = detect_query_language_deterministic(query)
    detected_entities = extract_technical_entities(evidence)

    has_devanagari = bool(re.search(r"[\u0900-\u097F]", response))

    # Detect response language
    if has_devanagari:
        from backend.services.multilingual import MARATHI_MARKERS, HINDI_MARKERS
        resp_lower = response.lower()
        m_score = sum(1 for m in MARATHI_MARKERS if re.search(m, resp_lower))
        h_score = sum(1 for m in HINDI_MARKERS if re.search(m, resp_lower))
        resp_lang = "mr" if m_score >= h_score else "hi"
    else:
        resp_lang = "en"

    repaired_response = response
    was_repaired = False
    validation_status = "PASSED"

    # Case 1: Query was English, but model hallucinated Marathi/Hindi
    if target_lang == "en" and has_devanagari:
        temp_rep, reps = repair_transliterated_technical_terms(repaired_response)

        # Convert Marathi framing phrases to clean English
        marathi_frame_pattern = re.compile(
            r'MRPL\s+मध्ये\s+मुख्य\s+युनिट्स\s+खालीलप्रमाणे\s+आहेत:?',
            re.IGNORECASE
        )
        if marathi_frame_pattern.search(temp_rep):
            temp_rep = marathi_frame_pattern.sub("MRPL's main units include:", temp_rep)

        # Hindi framing phrases to clean English
        hindi_frame_pattern = re.compile(
            r'MRPL\s+की\s+मुख्य\s+यूनिट्स\s+निम्नलिखित\s+हैं:?',
            re.IGNORECASE
        )
        if hindi_frame_pattern.search(temp_rep):
            temp_rep = hindi_frame_pattern.sub("MRPL's main units include:", temp_rep)

        # Replace common Devanagari connective words with English
        conjunction_replacements = [
            (r'\bआणि\b', 'and'),
            (r'\bऔर\b', 'and'),
            (r'\bतसेच\b', 'as well as'),
            (r'\bकिंवा\b', 'or'),
            (r'\bया\b', 'or'),
            (r'\bयुनिट्स\b', 'units'),
            (r'\bयूनिट्स\b', 'units'),
            (r'\bयुनिट\b', 'Unit'),
            (r'\bयूनिट\b', 'Unit'),
            (r'\bइकाई\b', 'Unit'),
            (r'\bइकाइयाँ\b', 'Units'),
        ]
        for pat, rep in conjunction_replacements:
            temp_rep = re.sub(pat, rep, temp_rep)

        # Remove any remaining stray Devanagari words if query is English
        # while keeping punctuation and Latin characters intact
        clean_latin_lines = []
        for line in temp_rep.split("\n"):
            # If line has remaining Devanagari, clean up
            if re.search(r"[\u0900-\u097F]", line):
                cleaned_line = re.sub(r'[\u0900-\u097F]+', '', line)
                # Normalize double spaces
                cleaned_line = re.sub(r'\s{2,}', ' ', cleaned_line).strip()
                if cleaned_line:
                    clean_latin_lines.append(cleaned_line)
            else:
                clean_latin_lines.append(line)

        repaired_response = "\n".join(clean_latin_lines).strip()
        was_repaired = True
        validation_status = "CORRECTED"
        resp_lang = "en"

    # Case 2: Query was Marathi or Hindi, check and repair transliterations
    elif target_lang in ("mr", "hi"):
        temp_rep, reps = repair_transliterated_technical_terms(repaired_response)
        if reps:
            repaired_response = temp_rep
            was_repaired = True
            validation_status = "CORRECTED"

    # Check preserved entities in final text
    preserved = [e for e in detected_entities if e in repaired_response]

    return EntityFidelityResult(
        query_language=target_lang,
        response_language=resp_lang,
        language_decision_source="current_user_query",
        technical_entities_detected=detected_entities,
        technical_entities_preserved=preserved,
        technical_entity_validation=validation_status,
        corrected_response=repaired_response,
        was_repaired=was_repaired,
        details={
            "initial_has_devanagari": has_devanagari,
            "target_language": target_lang,
            "detected_response_language": resp_lang,
        },
    )

