"""
MRPL Sovereign AI Workbench — Voice Language Detection Service.
Supports English, Hindi, and Marathi with Auto-Detection.
Combines ASR acoustic probability with deterministic script and phonetic marker detection.
"""

import logging
import re
from typing import Any

from backend.services.multilingual import detect_language as detect_script_language

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

# Romanized / Transliterated phonetic markers for Indian languages in Latin script
# Note: 'me' is deliberately excluded as it collides with the English pronoun 'me'.
ROMANIZED_HINDI_MARKERS = [
    r"\bka\b", r"\bki\b", r"\bke\b", r"\bhai\b", r"\bhain\b", r"\bkya\b",
    r"\bkyu\b", r"\bkyun\b", r"\bkaise\b", r"\bkitna\b", r"\bkitni\b",
    r"\bbatao\b", r"\bbataiye\b", r"\bsamjhao\b", r"\bmein\b",
    r"\bse\b", r"\bpar\b", r"\btha\b", r"\bthi\b", r"\bhoga\b", r"\bkaro\b",
    r"\bkijiye\b", r"\bjaankari\b", r"\bchahiye\b",
]

ROMANIZED_MARATHI_MARKERS = [
    r"\bcha\b", r"\bchi\b", r"\bche\b", r"\baahe\b", r"\baahit\b", r"\bkiti\b",
    r"\bkasa\b", r"\bkase\b", r"\bsaanga\b", r"\bkara\b", r"\bmadhye\b",
    r"\bbaddal\b", r"\bkaay\b", r"\bnaahi\b", r"\bzaale\b", r"\bhota\b",
    r"\bdya\b", r"\bmaahiti\b",
]

ENGLISH_LEXICAL_MARKERS = [
    r"\bwhat\b", r"\bwho\b", r"\bwhich\b", r"\bwhere\b", r"\bwhen\b", r"\bwhy\b",
    r"\bhow\b", r"\btell\b", r"\bshow\b", r"\bcheck\b", r"\bgive\b", r"\bexplain\b",
    r"\bdescribe\b", r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b", r"\bthe\b",
    r"\babout\b", r"\boperating\b", r"\bpressure\b", r"\btemperature\b", r"\bflow\b",
]


def detect_voice_language(
    text: str,
    asr_detected_language: str | None = None,
    user_override: str | None = None,
) -> dict[str, Any]:
    """
    Determine the query language across English, Hindi, and Marathi.

    Args:
        text: Transcribed query text.
        asr_detected_language: Language code from speech recognition (e.g. 'hi', 'mr', 'en').
        user_override: User-selected language code or 'auto'.

    Returns:
        {
            "code": "en" | "hi" | "mr",
            "name": "English" | "Hindi" | "Marathi",
            "method": "user_override" | "script_marker" | "romanized_marker" | "english_lexical" | "asr_prob",
            "is_auto": bool,
        }
    """
    # 1. User manual selection (if not auto)
    if user_override and user_override.lower() not in ("auto", "auto detect", "detect", "none"):
        clean_override = user_override.lower().strip()
        if clean_override in SUPPORTED_LANGUAGES:
            return {
                "code": clean_override,
                "name": SUPPORTED_LANGUAGES[clean_override],
                "method": "user_override",
                "is_auto": False,
            }
        # Fallback if code or name passed
        for code, name in SUPPORTED_LANGUAGES.items():
            if clean_override in (code, name.lower()):
                return {
                    "code": code,
                    "name": name,
                    "method": "user_override",
                    "is_auto": False,
                }
        # If an unsupported language was explicitly requested, report honestly and fallback safely
        return {
            "code": "en",
            "name": "English",
            "method": "unsupported_fallback",
            "unsupported_requested": clean_override,
            "warning": f"Requested language '{clean_override}' is not supported by on-premise voice models. Supported: English, Hindi, Marathi.",
            "is_auto": False,
        }

    if not text or not text.strip():
        # Default to ASR detected or English
        code = asr_detected_language if asr_detected_language in SUPPORTED_LANGUAGES else "en"
        return {
            "code": code,
            "name": SUPPORTED_LANGUAGES.get(code, "English"),
            "method": "default",
            "is_auto": True,
        }

    # 2. Devanagari Script Analysis (Strict multilingual service)
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    if has_devanagari:
        script_lang = detect_script_language(text)
        code = "mr" if script_lang == "marathi" else "hi"
        return {
            "code": code,
            "name": SUPPORTED_LANGUAGES[code],
            "method": "script_marker",
            "is_auto": True,
        }

    # 3. Romanized / Transliterated Marker Check (e.g. 'CDU ka function kya hai')
    text_lower = text.lower()
    marathi_hits = sum(1 for m in ROMANIZED_MARATHI_MARKERS if re.search(m, text_lower))
    hindi_hits = sum(1 for m in ROMANIZED_HINDI_MARKERS if re.search(m, text_lower))

    if marathi_hits > 0 and marathi_hits >= hindi_hits:
        return {
            "code": "mr",
            "name": "Marathi",
            "method": "romanized_marker",
            "is_auto": True,
        }
    elif hindi_hits > 0:
        return {
            "code": "hi",
            "name": "Hindi",
            "method": "romanized_marker",
            "is_auto": True,
        }

    # 4. English Lexical Check for Latin script (e.g. 'What is MRPL?', 'Tell me about...')
    english_hits = sum(1 for m in ENGLISH_LEXICAL_MARKERS if re.search(m, text_lower))
    if english_hits > 0:
        return {
            "code": "en",
            "name": "English",
            "method": "english_lexical",
            "is_auto": True,
        }

    # 5. ASR Engine Prior (Whisper acoustic classification if no lexical markers matched)
    if asr_detected_language:
        asr_clean = asr_detected_language.lower().strip()
        if asr_clean in SUPPORTED_LANGUAGES:
            # If ASR flagged an Indian language like 'hi' or 'mr', verify that the text isn't pure Latin English
            if asr_clean in ("hi", "mr"):
                if has_devanagari or hindi_hits > 0 or marathi_hits > 0:
                    return {
                        "code": asr_clean,
                        "name": SUPPORTED_LANGUAGES[asr_clean],
                        "method": "asr_prob",
                        "is_auto": True,
                    }
                # Pure Latin text without Hindi/Marathi markers — do not hallucinate Hindi on Indian English accent
            else:
                return {
                    "code": asr_clean,
                    "name": SUPPORTED_LANGUAGES[asr_clean],
                    "method": "asr_prob",
                    "is_auto": True,
                }

    # 6. Default to English
    return {
        "code": "en",
        "name": "English",
        "method": "default_english",
        "is_auto": True,
    }
