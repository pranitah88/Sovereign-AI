"""
MRPL Sovereign AI Workbench — Deterministic Domain Vocabulary & ASR Normalization.
Provides local industrial vocabulary and deterministic phonetic corrections.
Strictly Rule-Based & Vocabulary-Backed — Zero Cloud Egress, No Hallucinated Rewrites.
"""

from __future__ import annotations

import re
from typing import Any

# ── MRPL Entities & Technical Process Units ───────────────────────────────────

COMPANY_ENTITIES = [
    "MRPL",
    "Mangalore Refinery and Petrochemicals Limited",
    "ONGC",
    "OMPL",
]

PROCESS_UNITS = [
    "CDU",    # Crude Distillation Unit
    "VDU",    # Vacuum Distillation Unit
    "HCU",    # Hydrocracker Unit
    "PFCCU",  # Petrochemical Fluidized Catalytic Cracking Unit
    "PFCC",   # Petrochemical Fluidized Catalytic Cracking
    "FCCU",   # Fluidized Catalytic Cracking Unit
    "CCR",    # Continuous Catalytic Reforming
    "OHCU",   # Once Through Hydrocracker Unit
    "NHT",    # Naphtha Hydrotreater
    "ISOM",   # Isomerization Unit
    "SRU",    # Sulfur Recovery Unit
    "DHDT",   # Diesel Hydro-desulfurization Unit
    "DHT",    # Diesel Hydrotreater
    "DCU",    # Delayed Coker Unit
    "PPU",    # Polypropylene Unit
    "HGU",    # Hydrogen Generation Unit
    "SWS",    # Sour Water Stripper
    "ARU",    # Amine Recovery Unit
    "ETP",    # Effluent Treatment Plant
]

REFINERY_TERMS = [
    "refinery",
    "refining",
    "throughput",
    "crude charge",
    "crude distillation",
    "gross refining margin",
    "GRM",
    "P&ID",
    "operating pressure",
    "operating temperature",
    "flow rate",
    "naphtha",
    "diesel",
    "LPG",
    "MS",     # Motor Spirit / Gasoline
    "HSD",    # High Speed Diesel
    "ATF",    # Aviation Turbine Fuel
    "VGO",    # Vacuum Gas Oil
    "furnace",
    "boiler",
    "heat exchanger",
    "column",
    "reactor",
    "pump",
    "compressor",
    "relief valve",
    "flare",
    "storage tank",
]

ALL_TECHNICAL_ACRONYMS = list(dict.fromkeys(
    PROCESS_UNITS + [
        "MRPL", "OMPL", "ONGC", "P&ID", "PID", "HSE",
        "LPG", "MS", "HSD", "ATF", "FO", "LOBS", "VGO", "GRM", "EBITDA",
    ]
))

# ── Deterministic Phonetic Corrections ────────────────────────────────────────

# Common Whisper phonetic acoustic confusions for refinery terminology
# Mappings: (regex_pattern, replacement_callable_or_str)
DETERMINISTIC_PHONETIC_RULES: list[tuple[re.Pattern, Any]] = [
    # 1. Company name mis-transcriptions
    (re.compile(r"\b(?:mrpr|mr\s+pear|mr\s+pel|m\.r\.p\.l\.|m\s+r\s+p\s+l|m\s+r\s+p\s+r|marpl)\b", re.IGNORECASE), "MRPL"),

    # 2. P&ID variations
    (re.compile(r"\b(?:p\s*and\s*id|p\s*&\s*id|p\s*n\s*id|p\.n\.i\.d\.|p\s+and\s+i\s+d)\b", re.IGNORECASE), "P&ID"),

    # 3. Acoustic "water the" -> "what are the" preceding refinery concepts
    (
        re.compile(r"\b([Ww]ater|[Ww]atter)\s+the\s+(major|units|refinery|main|operating|pumps|columns|reactors|facilities|products)\b", re.IGNORECASE),
        lambda m: f"{'What' if m.group(1)[0].isupper() else 'what'} are the {m.group(2)}"
    ),
    (
        re.compile(r"\b([Ww]ater|[Ww]atter)\s+(are|is)\s+the\b", re.IGNORECASE),
        lambda m: f"{'What' if m.group(1)[0].isupper() else 'what'} {m.group(2)} the"
    ),

    # 4. "vat is" / "wat is" / "wat are" acoustic errors
    (
        re.compile(r"\b([Vv]at|[Ww]at)\s+is\b", re.IGNORECASE),
        lambda m: f"{'What' if m.group(1)[0].isupper() else 'what'} is"
    ),
    (
        re.compile(r"\b([Vv]at|[Ww]at)\s+are\b", re.IGNORECASE),
        lambda m: f"{'What' if m.group(1)[0].isupper() else 'what'} are"
    ),
    (
        re.compile(r"\b([Ww]er|[Ww]hre)\s+is\b", re.IGNORECASE),
        lambda m: f"{'Where' if m.group(1)[0].isupper() else 'where'} is"
    ),
    (
        re.compile(r"\b([Ww]er|[Ww]hre)\s+are\b", re.IGNORECASE),
        lambda m: f"{'Where' if m.group(1)[0].isupper() else 'where'} are"
    ),

    # 5. Process units acoustic confusions
    (re.compile(r"\b(?:hydro\s*cracker|hydro\s+cracking)\b", re.IGNORECASE), "hydrocracker"),
    (re.compile(r"\b(?:c\s*d\s*u)\b", re.IGNORECASE), "CDU"),
    (re.compile(r"\b(?:v\s*d\s*u)\b", re.IGNORECASE), "VDU"),
    (re.compile(r"\b(?:h\s*c\s*u)\b", re.IGNORECASE), "HCU"),
    (re.compile(r"\b(?:p\s*f\s*c\s*c\s*u|p\s*f\s*c\s*c)\b", re.IGNORECASE), "PFCCU"),
    (re.compile(r"\b(?:d\s*h\s*d\s*t)\b", re.IGNORECASE), "DHDT"),
    (re.compile(r"\b(?:c\s*c\s*r)\b", re.IGNORECASE), "CCR"),
]

TAG_RE = re.compile(r"\b(\d{1,3})[- ]*([A-Za-z]{1,3})[- ]*(\d{1,4})[- ]*([A-Za-z])?\b")


def normalize_asr_transcript(text: str) -> tuple[str, bool]:
    """
    Deterministic ASR normalization using domain vocabulary and phonetic rules.
    Preserves original technical meaning — never uses arbitrary LLM rewrites.
    
    Returns:
        (normalized_text, is_normalized)
    """
    if not text:
        return text, False

    result = text.strip()

    # 1. Apply deterministic phonetic rules
    for pattern, replacement in DETERMINISTIC_PHONETIC_RULES:
        result = pattern.sub(replacement, result)

    # 2. Canonical uppercase for known technical acronyms
    for acronym in ALL_TECHNICAL_ACRONYMS:
        # Match whole word case-insensitively
        pattern = re.compile(rf"\b{re.escape(acronym)}\b", re.IGNORECASE)
        result = pattern.sub(acronym, result)

    # 3. Standardize equipment tag formatting (e.g. "11 p 101 a" -> "11-P-101A")
    def _norm_tag(m: re.Match) -> str:
        suffix = m.group(4).upper() if m.group(4) else ""
        return f"{m.group(1)}-{m.group(2).upper()}-{m.group(3)}{suffix}"

    result = TAG_RE.sub(_norm_tag, result)

    # 4. Clean up any duplicated whitespace
    result = re.sub(r"\s+", " ", result).strip()

    is_normalized = (result != text.strip())
    return result, is_normalized
