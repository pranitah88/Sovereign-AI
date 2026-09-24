"""
MRPL Sovereign AI Workbench — Multilingual Query Normalization & Topic Extraction.

Deterministic, lightweight, zero-external-API normalization layer for
Indian-language (Hindi, Marathi) and English technical queries.

Key Responsibilities:
1. Script & Language Detection (Devanagari vs Latin; Hindi vs Marathi).
2. Domain-Specific Technical Alias Mapping (refinery units, operations, finance, HR).
3. Technical Entity & Embedded English Token Extraction (e.g., MRPL, FY2025, HCU).
4. Deterministic Query-Topic Classification (refinery_unit, financial_metric, recruitment_notice, general).
5. Canonical English Query Construction for High-Precision BM25 & ChromaDB Vector Search.
6. Topic-Aware Category & Document Filtering Directives (+boost / -penalty).
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NormalizedQuery:
    original_query: str
    detected_language: str
    detected_topic: str
    canonical_entities: list[str] = field(default_factory=list)
    normalized_query: str = ""
    primary_metric: str | None = None
    fiscal_year: str | None = None
    boost_categories: list[str] = field(default_factory=list)
    forbidden_categories: list[str] = field(default_factory=list)
    boost_sources: list[str] = field(default_factory=list)
    forbidden_sources: list[str] = field(default_factory=list)
    technical_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "detected_language": self.detected_language,
            "detected_topic": self.detected_topic,
            "canonical_entities": self.canonical_entities,
            "normalized_query": self.normalized_query,
            "primary_metric": self.primary_metric,
            "fiscal_year": self.fiscal_year,
            "boost_categories": self.boost_categories,
            "forbidden_categories": self.forbidden_categories,
            "boost_sources": self.boost_sources,
            "forbidden_sources": self.forbidden_sources,
            "technical_keywords": self.technical_keywords,
        }


# ── Language Detection Markers ────────────────────────────────────────────────

MARATHI_MARKERS = [
    r"\bच्या\b", r"\bमध्ये\b", r"\bआहे\b", r"\bकरा\b", r"\bसोप्या\b",
    r"\bसांगा\b", r"\bकिती\b", r"\bहोता\b", r"\bबद्दल\b", r"\bआणि\b",
    r"\bझाले\b", r"\bनाही\b", r"\bकाही\b", r"\bकसे\b", r"\bमहसूल\b",
    r"\bउलाढाल\b", r"\bनफा\b", r"\bरिफायनरी\b", r"\bरिफायनरीमध्ये\b",
    r"\bरिफायनरीतील\b", r"\bहायड्रोक्रॅकर\b", r"\bयुनिटचा\b", r"\bउद्देश\b",
    r"\bमाहिती\b", r"\bद्या\b", r"\bभरती\b", r"\bघोटाळा\b", r"\bफसवणूक\b",
]

HINDI_MARKERS = [
    r"\bकी\b", r"\bके\b", r"\bमें\b", r"\bहै\b", r"\bबताइए\b",
    r"\bसमझाइए\b", r"\bकितना\b", r"\bथा\b", r"\bबारे\b", r"\bऔर\b",
    r"\bहुआ\b", r"\bनहीं\b", r"\bकुछ\b", r"\bकैसे\b", r"\bराजस्व\b",
    r"\bटर्नओवर\b", r"\bलाभ\b", r"\bमुनाफा\b", r"\bरिफाइनरी\b",
    r"\bहाइड्रोक्रैकर\b", r"\bउद्देश्य\b", r"\bजानकारी\b", r"\bभर्ती\b",
    r"\bघोटाला\b", r"\bफर्जी\b",
]


def _match_devanagari_marker(marker: str, text: str) -> bool:
    clean = marker.replace(r"\b", "").strip()
    if not clean:
        return False
    pattern = rf"(?:^|[^\u0900-\u097fa-zA-Z0-9]){re.escape(clean)}(?:$|[^\u0900-\u097fa-zA-Z0-9])"
    return bool(re.search(pattern, text))


def detect_language(query: str) -> str:
    """
    Detect whether query is Hindi, Marathi, or English.
    Returns: 'english' | 'hindi' | 'marathi'
    """
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", query))
    if not has_devanagari:
        return "english"

    q_lower = query.lower()
    marathi_score = sum(1 for m in MARATHI_MARKERS if _match_devanagari_marker(m, q_lower))
    hindi_score = sum(1 for m in HINDI_MARKERS if _match_devanagari_marker(m, q_lower))

    if marathi_score > hindi_score:
        return "marathi"
    return "hindi"


# ── Domain-Specific Technical Alias Maps ──────────────────────────────────────

# 1. Refinery Units
REFINERY_UNIT_ALIASES: dict[str, list[str]] = {
    "hydrocracker": [
        "हाइड्रोक्रैकर", "हायड्रोक्रॅकर", "हाइड्रो-क्रैकर", "हायड्रो-क्रॅकर",
        "hydrocracker", "hydro-cracker", "hcu", "hydrocracker unit", "hydro cracking",
    ],
    "crude distillation unit": [
        "सीडीयू", "cdu", "crude distillation", "कच्चा तेल आसवन", "क्रूड डिस्टिलेशन",
    ],
    "vacuum distillation unit": [
        "वीडीयू", "vdu", "vacuum distillation", "वैक्यूम डिस्टिलेशन",
    ],
    "diesel hydrotreater": [
        "डीएचडीटी", "dhdt", "diesel hydrotreater", "diesel hydrotreating", "डीजल हाइड्रोसल्फराइजेशन",
    ],
    "fluidized catalytic cracking": [
        "एफसीसीयू", "fccu", "fluid catalytic cracking", "कैटेलिटिक क्रैकिंग",
    ],
    "petrochemical fluidized catalytic cracking": [
        "पीएफसीसीयू", "pfccu", "petrochemical fluidized catalytic cracking",
    ],
    "continuous catalytic reforming": [
        "सीसीआर", "ccr", "continuous catalytic reforming", "catalytic reformer",
    ],
    "hydrogen generation unit": [
        "हाइड्रोजन यूनिट", "हायड्रोजन युनिट", "hgu", "hydrogen unit", "hydrogen generation",
    ],
    "sulfur recovery unit": [
        "सल्फर यूनिट", "सल्फर रिकवरी", "sru", "sulfur recovery", "सल्फर रिकव्हरी",
    ],
    "delayed coker unit": [
        "डीसीयू", "dcu", "delayed coker", "कोकर यूनिट",
    ],
    "polypropylene unit": [
        "पीपीयू", "ppu", "polypropylene", "पॉलीप्रोपाइलीन", "पॉलीप्रॉपिलीन",
    ],
}

# 2. Refinery Operations & Concepts
OPERATION_ALIASES: dict[str, list[str]] = {
    "refinery": [
        "रिफाइनरी", "रिफायनरी", "refinery", "refining", "रिफाइनिंग", "रिफायनिंग",
    ],
    "unit": [
        "यूनिट", "युनिट", "unit", "units", "इकाई", "इकाइयाँ", "इकाइयां", "घटक",
    ],
    "facilities": [
        "सुविधाएं", "सुविधा", "सुविधांचा", "facilities", "facility", "संसाधन",
    ],
    "manufacturing": [
        "निर्माण इकाइयाँ", "निर्माण", "manufacturing", "production units", "उत्पादन घटक",
    ],
    "plant": [
        "संयंत्र", "प्रकल्प", "plant", "plants",
    ],
    "process": [
        "प्रक्रिया", "process", "processing",
    ],
    "capacity": [
        "क्षमता", "capacity", "throughput capacity",
    ],
    "technology": [
        "प्रौद्योगिकी", "तंत्रज्ञान", "technology", "cutting edge",
    ],
    "safety": [
        "सुरक्षा", "सुरक्षितता", "safety", "hse", "पर्यावरण",
    ],
    "production": [
        "उत्पादन", "production",
    ],
    "inspection": [
        "निरीक्षण", "तपासणी", "inspection", "audit",
    ],
}

# 3. Intent & Goal Concepts
INTENT_ALIASES: dict[str, list[str]] = {
    "purpose": [
        "उद्देश्य", "उद्देश", "कार्य", "काम", "role", "function", "purpose",
        "objective", "use", "importance", "कामकाज", "हेतू",
    ],
    "overview": [
        "विवरण", "माहिती", "जानकारी", "overview", "details", "information", "सांगा", "बताइए",
    ],
}

# 4. Recruitment & Job Scam Notice Concepts
RECRUITMENT_ALIASES: dict[str, list[str]] = {
    "job_scam": [
        "जॉब स्कैम", "घोटाला", "घोटाळा", "फर्जी", "फसवणूक", "job scam", "scam",
        "fraud", "fake recruitment", "fake job", "scam notice",
    ],
    "recruitment": [
        "भर्ती", "भरती", "नौकरी", "नोकरी", "रोजगार", "recruitment", "job",
        "vacancy", "vacancies", "careers", "career", "employment",
    ],
    "public_notice": [
        "सार्वजनिक सूचना", "सूचना", "नोटीस", "public notice", "notice",
    ],
}

# 5. Financial Metric Aliases
METRIC_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "राजस्व", "महसूल", "revenue", "gross revenue", "net revenue",
        "turnover", "उलाढाल", "total income", "topline",
    ],
    "exports": [
        "निर्यात", "export", "exports", "overseas sales", "foreign sales",
    ],
    "profit": [
        "लाभ", "नफा", "मुनाफा", "net profit", "pat", "profit after tax",
        "profit before tax", "pbt", "ebitda", "profit",
    ],
    "throughput": [
        "थ्रूपुट", "प्रसंस्करण", "throughput", "crude processed", "refinery throughput",
    ],
    "expenses": [
        "व्यय", "खर्च", "expenses", "expenditure",
    ],
    "debt": [
        "ऋण", "कर्ज", "debt", "borrowings", "liabilities",
    ],
    "dividend": [
        "लाभांश", "dividend", "dividend per share",
    ],
}

# Conversational phrases that dilute BM25 & semantic matching
CONVERSATIONAL_PHRASES = [
    r"सरल\s+शब्दों\s+में\s+समझाइए",
    r"सरल\s+शब्दों\s+में\s+बताइए",
    r"सोप्या\s+भाषेत\s+समजावून\s+सांगा",
    r"सोप्या\s+भाषेत\s+माहिती\s+द्या",
    r"के\s+बारे\s+में\s+बताइए",
    r"बद्दल\s+सोप्या\s+भाषेत\s+माहिती\s+द्या",
    r"बद्दल\s+माहिती\s+द्या",
    r"की\s+जानकारी\s+दीजिए",
    r"की\s+जानकारी\s+क्या\s+है",
    r"ची\s+माहिती\s+द्या",
    r"चा\s+मुख्य\s+उद्देश\s+काय\s+आहे",
    r"का\s+मुख्य\s+उद्देश्य\s+क्या\s+है",
    r"in\s+simple\s+words",
    r"in\s+simple\s+terms",
    r"explain\s+in\s+simple\s+terms",
    r"what\s+is\s+the\s+main\s+purpose\s+of",
    r"what\s+is\s+the\s+purpose\s+of",
    r"tell\s+me\s+about",
    r"give\s+information\s+about",
]


def extract_canonical_aliases(
    text: str,
    alias_dict: dict[str, list[str]],
) -> list[str]:
    """Extract canonical keys from alias dictionary matching text."""
    found: list[str] = []
    text_lower = text.lower()
    for canonical_name, variants in alias_dict.items():
        for var in variants:
            # Word boundary matching for Latin words, substring for Devanagari
            if re.search(r"[\u0900-\u097F]", var):
                if var in text_lower:
                    if canonical_name not in found:
                        found.append(canonical_name)
                    break
            else:
                pat = r"\b" + re.escape(var) + r"\b"
                if re.search(pat, text_lower):
                    if canonical_name not in found:
                        found.append(canonical_name)
                    break
    return found


def extract_fiscal_year(text: str) -> tuple[str | None, list[str]]:
    """Extract standardized fiscal year and search variants."""
    t_lower = text.lower()
    fy_dash = re.search(r"\b(?:fy\s*)?(20\d\d)\s*[-–]\s*(\d\d(?:\d\d)?)\b", t_lower)
    fy_single = re.search(r"\b(?:fy\s*|financial\s+year\s*)(20\d\d|\d\d)\b", t_lower)
    bare_year = re.search(r"\b(20\d\d)\b", t_lower)

    if fy_dash:
        y1 = int(fy_dash.group(1))
        y2 = int(fy_dash.group(2)[-2:])
        return f"FY{y1}-{y2:02d}", [f"{y1}-{y2:02d}", f"FY{y1}-{y2:02d}", str(y1 + 1)]
    elif fy_single:
        val = int(fy_single.group(1))
        full_year = val if val > 2000 else 2000 + val
        return f"FY{full_year}", [f"{full_year - 1}-{full_year % 100:02d}", f"FY{full_year}", str(full_year)]
    elif bare_year:
        full_year = int(bare_year.group(1))
        return str(full_year), [str(full_year), f"FY{full_year}"]
    return None, []


def normalize_query(query: str) -> NormalizedQuery:
    """
    Deterministically normalize Indian-language or English query into
    retrieval concepts, topic classification, and filtering directives.
    """
    clean_q = query.strip()
    lang = detect_language(clean_q)

    # 1. Strip conversational fillers for clean intent identification
    stripped = clean_q
    for pat in CONVERSATIONAL_PHRASES:
        stripped = re.sub(pat, " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()

    # 2. Extract technical entities & canonical terms
    found_units = extract_canonical_aliases(clean_q, REFINERY_UNIT_ALIASES)
    found_ops = extract_canonical_aliases(clean_q, OPERATION_ALIASES)
    found_intents = extract_canonical_aliases(clean_q, INTENT_ALIASES)
    found_recruitment = extract_canonical_aliases(clean_q, RECRUITMENT_ALIASES)
    found_metrics = extract_canonical_aliases(clean_q, METRIC_ALIASES)
    fy_str, fy_variants = extract_fiscal_year(clean_q)

    # 3. Extract embedded Latin / English tokens (e.g. MRPL, HCU, ATF, VGO)
    embedded_english = re.findall(r"\b[A-Za-z0-9\-_]{2,}\b", clean_q)
    # Filter out common stop words if any
    stop_words = {"the", "is", "at", "in", "of", "and", "or", "to", "for", "with", "what", "how", "main"}
    english_tokens = [tok for tok in embedded_english if tok.lower() not in stop_words]

    # Ensure "MRPL" is always present if mentioned or implied
    has_mrpl = "mrpl" in clean_q.lower() or "एमआरपीएल" in clean_q or "mrpl" in [t.lower() for t in english_tokens]

    # 4. Classify Topic
    detected_topic = "general"
    primary_metric = found_metrics[0] if found_metrics else None

    # Priority 1: Job scam / recruitment query
    if ("job_scam" in found_recruitment or "recruitment" in found_recruitment) and not found_units:
        detected_topic = "recruitment_notice"
    # Priority 2: Refinery unit technical query
    elif found_units or (
        ("refinery" in found_ops or has_mrpl)
        and any(op in found_ops for op in ["facilities", "manufacturing", "unit", "plant", "technology", "process"])
    ):
        detected_topic = "refinery_unit"
    # Priority 3: Financial metric query
    elif primary_metric or (fy_str and any(m in clean_q.lower() for m in ["rajswa", "mahsul", "revenue", "turnover", "profit"])):
        detected_topic = "financial_metric"
    # Priority 4: Safety / HSE
    elif "safety" in found_ops:
        detected_topic = "safety_hse"

    # 5. Build Canonical Retrieval Query & Entities
    canonical_entities: list[str] = []
    if has_mrpl:
        canonical_entities.append("MRPL")

    # Add refinery operations
    for op in ["refinery", "facilities", "manufacturing"]:
        if op in found_ops and op not in canonical_entities:
            canonical_entities.append(op)

    # Add units
    for u in found_units:
        if u not in canonical_entities:
            canonical_entities.append(u)

    # Add intent
    for intent in found_intents:
        if intent not in canonical_entities:
            canonical_entities.append(intent)

    # Add recruitment/scam concepts if topic matches
    if detected_topic == "recruitment_notice":
        for r in found_recruitment:
            if r not in canonical_entities:
                canonical_entities.append(r)
        if "notice" not in canonical_entities:
            canonical_entities.append("public notice")

    # Add metric concepts if topic matches
    if detected_topic == "financial_metric" and primary_metric:
        if primary_metric not in canonical_entities:
            canonical_entities.append(primary_metric)
        if fy_str and fy_str not in canonical_entities:
            canonical_entities.append(fy_str)

    # Add specific technical keywords associated with units for enriched search
    technical_keywords: list[str] = []
    if "hydrocracker" in found_units:
        technical_keywords.extend(["hydrocracker", "VGO", "diesel", "ATF", "lighter products", "hydrogen", "cracking"])
    elif "crude distillation unit" in found_units:
        technical_keywords.extend(["crude distillation", "CDU", "atmospheric", "naphtha", "kerosene"])
    elif "continuous catalytic reforming" in found_units:
        technical_keywords.extend(["CCR", "continuous catalytic reforming", "reformate", "hydrogen"])

    # Merge embedded English tokens
    for tok in english_tokens:
        if tok.lower() not in [c.lower() for c in canonical_entities] and tok.lower() not in [k.lower() for k in technical_keywords]:
            if len(tok) >= 3:
                canonical_entities.append(tok)

    # Build normalized search query string
    # For technical units, prioritize: MRPL refinery [unit] purpose function facilities
    query_parts: list[str] = []
    if has_mrpl:
        query_parts.append("MRPL")
    if "refinery" in found_ops or detected_topic == "refinery_unit":
        query_parts.append("refinery")
    for u in found_units:
        query_parts.append(u)
    if "facilities" in found_ops:
        query_parts.append("facilities")
    if "purpose" in found_intents or detected_topic == "refinery_unit":
        query_parts.append("purpose function")
    for r in found_recruitment:
        if detected_topic == "recruitment_notice":
            query_parts.append(r)
    if primary_metric:
        query_parts.append(primary_metric)
    if fy_str:
        query_parts.append(fy_str)

    # If query was English, also retain important technical non-stopwords
    if lang == "english":
        norm_query_str = f"{clean_q} {' '.join(query_parts)}".strip()
        # Deduplicate words while preserving order
        seen_words = set()
        deduped = []
        for w in norm_query_str.split():
            wl = w.lower().strip("?,.!")
            if wl not in seen_words and len(wl) > 1:
                seen_words.add(wl)
                deduped.append(w.strip("?,.!"))
        normalized_query_str = " ".join(deduped)
    else:
        # For Hindi/Marathi, canonical English query ensures accurate vector & BM25 retrieval
        if not query_parts:
            # Fallback if no specific aliases matched: use embedded English tokens or stripped
            query_parts = english_tokens if english_tokens else [stripped]
        normalized_query_str = " ".join(dict.fromkeys(query_parts))

    # 6. Category & Source Directives based on topic
    boost_categories: list[str] = []
    forbidden_categories: list[str] = []
    boost_sources: list[str] = []
    forbidden_sources: list[str] = []

    if detected_topic == "refinery_unit":
        boost_categories = ["01_Refinery_Manufacturing", "06_Website_Content", "manufacturing_refining"]
        boost_sources = [
            "Refining", "Manufacturing Units", "Facilities", "Cutting Edge Technology", "Brochure",
        ]
        # Strictly penalize job scams, recruitment notices, unrelated statutory/CSR for unit queries
        forbidden_sources = [
            "public_notice_on_job_scam", "job_scam", "recruitment", "tender", "vigilance",
        ]
        forbidden_categories = ["04_Company_General"]  # Checked in conjunction with source name

    elif detected_topic == "recruitment_notice":
        boost_sources = ["public_notice_on_job_scam", "recruitment", "careers"]
        boost_categories = ["04_Company_General", "05_Policies_Certifications"]
        # Down-rank refinery engineering documents for recruitment queries
        forbidden_sources = ["refining", "manufacturing units", "crude distillation", "hydrocracker"]

    elif detected_topic == "financial_metric":
        boost_categories = ["07_Finance", "annual_report", "financial_statement"]
        forbidden_sources = ["public_notice_on_job_scam", "recruitment"]

    return NormalizedQuery(
        original_query=clean_q,
        detected_language=lang,
        detected_topic=detected_topic,
        canonical_entities=canonical_entities,
        normalized_query=normalized_query_str,
        primary_metric=primary_metric,
        fiscal_year=fy_str,
        boost_categories=boost_categories,
        forbidden_categories=forbidden_categories,
        boost_sources=boost_sources,
        forbidden_sources=forbidden_sources,
        technical_keywords=technical_keywords,
    )
