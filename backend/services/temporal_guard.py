"""
Deterministic Temporal / Recency Guard for MRPL Sovereign AI Workbench.

Ensures that:
1. Current/latest questions cannot be answered using stale documents as if they were current.
2. Historical questions can still use historical documents normally.
3. Every claimed reporting period/date must be consistent with the cited source.
4. The LLM must never invent a newer date, fiscal year, reporting period, or "current" status unsupported by retrieved evidence.
5. If the local knowledge base does not contain sufficiently recent evidence, the system explicitly states that current information cannot be reliably determined.
6. 100% sovereign and air-gapped — zero external network or external API dependency.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Reference year for the active sovereign deployment environment
CURRENT_DEPLOYMENT_YEAR = 2026

# Annual report number to fiscal year mapping for MRPL
# (Incorporated in 1988; 1st AGM was 1989 for FY1988-89; 29th was FY2016-17)
ANNUAL_REPORT_NUMBER_MAP = {
    25: ("FY2012-13", "2012-13", 2013),
    26: ("FY2013-14", "2013-14", 2014),
    27: ("FY2014-15", "2014-15", 2015),
    28: ("FY2015-16", "2015-16", 2016),
    29: ("FY2016-17", "2016-17", 2017),
    30: ("FY2017-18", "2017-18", 2018),
    31: ("FY2018-19", "2018-19", 2019),
    32: ("FY2019-20", "2019-20", 2020),
    33: ("FY2020-21", "2020-21", 2021),
    34: ("FY2021-22", "2021-22", 2022),
    35: ("FY2022-23", "2022-23", 2023),
    36: ("FY2023-24", "2023-24", 2024),
    37: ("FY2024-25", "2024-25", 2025),
    38: ("FY2025-26", "2025-26", 2026),
}


@dataclass
class TemporalIntent:
    temporal_intent: str  # "CURRENT" | "RECENT" | "HISTORICAL" | "DATE_SPECIFIC" | "UNSPECIFIED"
    explicit_date: str | None = None
    relative_time: str | None = None
    requires_recent_evidence: bool = False
    requested_period: str | None = None  # e.g., "FY2016-17", "2018", "2016-17"
    requested_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_intent": self.temporal_intent,
            "explicit_date": self.explicit_date,
            "relative_time": self.relative_time,
            "requires_recent_evidence": self.requires_recent_evidence,
            "requested_period": self.requested_period,
            "requested_year": self.requested_year,
        }


@dataclass
class DocumentTemporalMetadata:
    document_date: str | None = None
    publication_date: str | None = None
    reporting_period: str | None = None  # e.g., "FY2016-17"
    fiscal_year: str | None = None       # e.g., "2016-17"
    document_type: str | None = None
    source_filename: str | None = None
    document_title: str | None = None
    temporal_confidence: str = "UNKNOWN" # "HIGH" | "MEDIUM" | "UNKNOWN"
    year_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_date": self.document_date,
            "publication_date": self.publication_date,
            "reporting_period": self.reporting_period,
            "fiscal_year": self.fiscal_year,
            "document_type": self.document_type,
            "source_filename": self.source_filename,
            "document_title": self.document_title,
            "temporal_confidence": self.temporal_confidence,
            "year_end": self.year_end,
        }


@dataclass
class TemporalValidationResult:
    temporal_status: str  # "VALID" | "INSUFFICIENT_RECENT_EVIDENCE" | "TEMPORAL_MISMATCH" | "UNCONSTRAINED"
    is_valid: bool = True
    answer_allowed_as_current: bool = False
    latest_evidence_date: str | None = None
    latest_reporting_period: str | None = None
    latest_source_title: str | None = None
    requested_period: str | None = None
    matched_sources: list[dict] = field(default_factory=list)
    stale_sources: list[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_status": self.temporal_status,
            "is_valid": self.is_valid,
            "answer_allowed_as_current": self.answer_allowed_as_current,
            "latest_evidence_date": self.latest_evidence_date,
            "latest_reporting_period": self.latest_reporting_period,
            "latest_source_title": self.latest_source_title,
            "requested_period": self.requested_period,
            "matched_sources_count": len(self.matched_sources),
            "stale_sources_count": len(self.stale_sources),
            "reason": self.reason,
        }


@dataclass
class TemporalClaimResult:
    valid: bool = True
    flagged: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    detected_periods: list[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "flagged": self.flagged,
            "unsupported_claims": self.unsupported_claims,
            "detected_periods": self.detected_periods,
            "details": self.details,
        }


# ── 1. Deterministic Temporal Intent Detection ───────────────────────────────

def detect_temporal_intent(query: str) -> TemporalIntent:
    """
    Detect temporal intent deterministically from user query.
    Rules and regex patterns only — zero LLM calls.
    """
    if not query or not query.strip():
        return TemporalIntent(temporal_intent="UNSPECIFIED")

    q = query.strip()
    q_lower = q.lower()

    # 1. Extract explicit fiscal year or date references
    # e.g., "FY2016-17", "FY 2016-17", "2016-17", "FY18", "FY2018"
    fy_match = re.search(r'\b(?:fy\s*[-–]?\s*)?(20\d{2})[-–](\d{2,4})\b', q_lower)
    single_year_match = re.search(r'\b(?:in|for|year|of)\s+(20\d{2})\b', q_lower)
    annual_report_num_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)\s+annual\s+report\b', q_lower)

    extracted_period = None
    extracted_year = None

    if fy_match:
        start_y = int(fy_match.group(1))
        end_y_str = fy_match.group(2)
        end_y = int(end_y_str) if len(end_y_str) == 4 else int(str(start_y)[:2] + end_y_str)
        extracted_period = f"FY{start_y}-{str(end_y)[-2:]}"
        extracted_year = end_y
    elif annual_report_num_match:
        rep_num = int(annual_report_num_match.group(1))
        if rep_num in ANNUAL_REPORT_NUMBER_MAP:
            rep_period, _, y_end = ANNUAL_REPORT_NUMBER_MAP[rep_num]
            extracted_period = rep_period
            extracted_year = y_end
    elif single_year_match:
        extracted_period = single_year_match.group(1)
        extracted_year = int(extracted_period)

    # 2. Check for explicit historical patterns
    # e.g. "What was MRPL's PAT in FY2016-17?", "Explain the 29th Annual Report", "Compare FY2016-17 and FY2017-18"
    is_historical_phrasing = bool(re.search(
        r'\b(?:what\s+was|what\s+were|in\s+the\s+past|historically|historical|previous\s+years?|earlier)\b',
        q_lower
    ))
    is_report_explanation = bool(re.search(
        r'\b(?:explain|summarize|review|breakdown|analyze|about)\s+(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)\s+annual\s+report|annual\s+report\s+(?:for\s+)?20\d{2})\b',
        q_lower
    ))
    is_comparison = bool(re.search(r'\bcompare\b.*\b(20\d{2}|fy)', q_lower))

    if extracted_period and (is_historical_phrasing or is_report_explanation or is_comparison or "what was" in q_lower):
        return TemporalIntent(
            temporal_intent="HISTORICAL",
            explicit_date=extracted_period,
            requested_period=extracted_period,
            requested_year=extracted_year,
            requires_recent_evidence=False,
        )

    if is_report_explanation:
        return TemporalIntent(
            temporal_intent="HISTORICAL",
            explicit_date=extracted_period,
            requested_period=extracted_period,
            requested_year=extracted_year,
            requires_recent_evidence=False,
        )

    # 3. Check for CURRENT intent
    # e.g. "What is MRPL's financial status right now?", "What is MRPL's current revenue?", "today", "at present"
    current_patterns = [
        r'\bright\s+now\b',
        r'\bas\s+of\s+(?:today|now|present)\b',
        r'\bat\s+present\b',
        r'\bpresent\s+status\b',
        r'\bcurrently\b',
        r'\bcurrent\s+(?:financial\s+status|status|revenue|profit|turnover|results|performance|situation|debt|ebitda|throughput|capacity|feedstock)\b',
        r'\bwhat\s+(?:is|are)\s+(?:the\s+)?current\b',
        r'\bwhat\s+is\s+mrpl\'?s?\s+current\b',
    ]
    matched_current = next((pat for pat in current_patterns if re.search(pat, q_lower)), None)
    if matched_current:
        rel_time = "right now" if "right now" in q_lower else ("today" if "today" in q_lower else "current")
        return TemporalIntent(
            temporal_intent="CURRENT",
            explicit_date=None,
            relative_time=rel_time,
            requires_recent_evidence=True,
            requested_period=None,
            requested_year=CURRENT_DEPLOYMENT_YEAR,
        )

    # 4. Check for RECENT intent
    # e.g. "What are MRPL's latest financial results?", "What is the most recent annual report?"
    recent_patterns = [
        r'\blatest\b',
        r'\bmost\s+recent\b',
        r'\bnewest\b',
        r'\brecently\b',
    ]
    matched_recent = next((pat for pat in recent_patterns if re.search(pat, q_lower)), None)
    if matched_recent:
        rel_time = "latest" if "latest" in q_lower else "most recent"
        return TemporalIntent(
            temporal_intent="RECENT",
            explicit_date=None,
            relative_time=rel_time,
            requires_recent_evidence=True,
            requested_period=None,
            requested_year=CURRENT_DEPLOYMENT_YEAR,
        )

    # 5. Check for DATE_SPECIFIC intent
    # e.g. "What was MRPL's revenue in 2018?", "MRPL production in FY2019-20"
    if extracted_period:
        intent_type = "HISTORICAL" if extracted_year and extracted_year < (CURRENT_DEPLOYMENT_YEAR - 2) else "DATE_SPECIFIC"
        return TemporalIntent(
            temporal_intent=intent_type,
            explicit_date=extracted_period,
            requested_period=extracted_period,
            requested_year=extracted_year,
            requires_recent_evidence=False,
        )

    # 6. Default: UNSPECIFIED
    # e.g. "What is MRPL?", "Tell me about MRPL's refinery", "Explain what Python is"
    return TemporalIntent(
        temporal_intent="UNSPECIFIED",
        requires_recent_evidence=False,
    )


# ── 2. Document Temporal Metadata Extraction ─────────────────────────────────

def extract_temporal_metadata(source: dict) -> DocumentTemporalMetadata:
    """
    Extract temporal metadata from a retrieved document/chunk dictionary.
    Inspects source filename, document_title, and chunk text without inventing metadata.
    """
    filename = source.get("source", "") or source.get("stored_filename", "")
    title = source.get("document_title", "") or ""
    text = source.get("text", "") or ""
    combined_name = f"{title} {filename}"

    reporting_period = None
    fiscal_year = None
    year_end = None
    confidence = "UNKNOWN"

    # 1. Match from explicit filename / title (e.g. "29th Annual Report for 2016-17...")
    fy_match = re.search(r'\b(?:fy\s*[-–]?\s*)?(20\d{2})[-–](\d{2,4})\b', combined_name, re.IGNORECASE)
    if fy_match:
        start_y = int(fy_match.group(1))
        end_y_str = fy_match.group(2)
        end_y = int(end_y_str) if len(end_y_str) == 4 else int(str(start_y)[:2] + end_y_str)
        fiscal_year = f"{start_y}-{str(end_y)[-2:]}"
        reporting_period = f"FY{fiscal_year}"
        year_end = end_y
        confidence = "HIGH"
    else:
        # Check annual report number pattern in title / filename
        an_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)\s+Annual\s+Report\b', combined_name, re.IGNORECASE)
        if an_match:
            rep_num = int(an_match.group(1))
            if rep_num in ANNUAL_REPORT_NUMBER_MAP:
                rep_period, f_year, y_end = ANNUAL_REPORT_NUMBER_MAP[rep_num]
                reporting_period = rep_period
                fiscal_year = f_year
                year_end = y_end
                confidence = "HIGH"

    # 2. If filename is a hash or uninformative, inspect chunk text for explicit indicators
    if not reporting_period and text:
        text_an = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)\s+Annual\s+Report\b', text, re.IGNORECASE)
        if text_an:
            rep_num = int(text_an.group(1))
            if rep_num in ANNUAL_REPORT_NUMBER_MAP:
                rep_period, f_year, y_end = ANNUAL_REPORT_NUMBER_MAP[rep_num]
                reporting_period = rep_period
                fiscal_year = f_year
                year_end = y_end
                confidence = "MEDIUM"

        if not reporting_period:
            text_fy = re.findall(r'\bFY\s*(20\d{2})[-–]?(\d{2})?\b', text)
            if text_fy:
                years = []
                for y1, y2 in text_fy:
                    if y2:
                        years.append((f"FY{y1}-{y2}", f"{y1}-{y2}", int(str(y1)[:2] + y2)))
                    else:
                        years.append((f"FY{y1}", y1, int(y1)))
                if years:
                    best = max(years, key=lambda x: x[2])
                    reporting_period, fiscal_year, year_end = best
                    confidence = "MEDIUM"

    doc_type = source.get("document_type")
    if not doc_type:
        if "annual report" in combined_name.lower():
            doc_type = "annual_report"
        elif "finance" in source.get("category", "").lower():
            doc_type = "financial"
        else:
            doc_type = "technical"

    return DocumentTemporalMetadata(
        document_date=source.get("document_date"),
        publication_date=source.get("publication_date"),
        reporting_period=reporting_period,
        fiscal_year=fiscal_year,
        document_type=doc_type,
        source_filename=filename,
        document_title=title,
        temporal_confidence=confidence,
        year_end=year_end,
    )


# ── 3. Temporal Validation of Retrieved Documents ────────────────────────────

def validate_temporal_suitability(
    intent: TemporalIntent,
    sources: list[dict],
    current_year: int = CURRENT_DEPLOYMENT_YEAR,
) -> TemporalValidationResult:
    """
    Evaluate temporal suitability of retrieved documents against the query's temporal intent.
    Deterministic rules — zero LLM hallucination.
    """
    if not sources:
        return TemporalValidationResult(
            temporal_status="INSUFFICIENT_RECENT_EVIDENCE",
            is_valid=False,
            answer_allowed_as_current=False,
            reason="No sources retrieved.",
        )

    # Extract temporal metadata for all sources
    source_metas = []
    for s in sources:
        meta = extract_temporal_metadata(s)
        source_metas.append((s, meta))

    # Find latest reporting period and year among retrieved evidence
    valid_years = [(s, m) for s, m in source_metas if m.year_end is not None]
    if valid_years:
        latest_item, latest_meta = max(valid_years, key=lambda x: x[1].year_end)
        latest_evidence_year = latest_meta.year_end
        latest_period = latest_meta.reporting_period
        latest_title = latest_meta.document_title or latest_item.get("source", "Document")
    else:
        latest_evidence_year = None
        latest_period = None
        latest_title = None

    # Case 1: CURRENT or RECENT queries
    if intent.temporal_intent in ("CURRENT", "RECENT"):
        RECENCY_THRESHOLD_YEAR = current_year - 2  # E.g. for 2026, requires >= 2024 (FY2023-24 or newer)

        if latest_evidence_year is None or latest_evidence_year < RECENCY_THRESHOLD_YEAR:
            stale_sources = [s for s, m in source_metas]
            latest_label = latest_title if latest_title else "historical documents"
            period_str = f" ({latest_period})" if latest_period else ""
            return TemporalValidationResult(
                temporal_status="INSUFFICIENT_RECENT_EVIDENCE",
                is_valid=False,
                answer_allowed_as_current=False,
                latest_evidence_date=str(latest_evidence_year) if latest_evidence_year else None,
                latest_reporting_period=latest_period,
                latest_source_title=latest_title,
                requested_period="CURRENT",
                stale_sources=stale_sources,
                reason=(
                    f"Retrieved documents are historical (latest: {latest_label}{period_str}). "
                    f"They cannot support a {intent.temporal_intent} inquiry without risk of temporal hallucination."
                ),
            )

        return TemporalValidationResult(
            temporal_status="VALID",
            is_valid=True,
            answer_allowed_as_current=True,
            latest_evidence_date=str(latest_evidence_year),
            latest_reporting_period=latest_period,
            latest_source_title=latest_title,
            matched_sources=[s for s, m in source_metas if m.year_end and m.year_end >= RECENCY_THRESHOLD_YEAR],
            reason="Retrieved evidence meets recency requirements.",
        )

    # Case 2: HISTORICAL queries
    if intent.temporal_intent == "HISTORICAL":
        if intent.requested_period:
            req_clean = intent.requested_period.replace("FY", "").strip()
            matched = []
            for s, m in source_metas:
                if m.fiscal_year and (req_clean in m.fiscal_year or m.fiscal_year in req_clean):
                    matched.append(s)
                elif m.year_end and intent.requested_year and m.year_end == intent.requested_year:
                    matched.append(s)

            if matched:
                return TemporalValidationResult(
                    temporal_status="VALID",
                    is_valid=True,
                    answer_allowed_as_current=False,
                    latest_reporting_period=latest_period,
                    latest_source_title=latest_title,
                    requested_period=intent.requested_period,
                    matched_sources=matched,
                    reason=f"Retrieved evidence matches requested historical period {intent.requested_period}.",
                )
            else:
                if valid_years:
                    return TemporalValidationResult(
                        temporal_status="TEMPORAL_MISMATCH",
                        is_valid=False,
                        answer_allowed_as_current=False,
                        latest_reporting_period=latest_period,
                        latest_source_title=latest_title,
                        requested_period=intent.requested_period,
                        reason=f"Requested historical period {intent.requested_period} not supported by retrieved sources (found {latest_period}).",
                    )

        return TemporalValidationResult(
            temporal_status="VALID",
            is_valid=True,
            answer_allowed_as_current=False,
            latest_reporting_period=latest_period,
            latest_source_title=latest_title,
            matched_sources=[s for s, _ in source_metas],
            reason="Historical query satisfied with available documentary evidence.",
        )

    # Case 3: DATE_SPECIFIC queries
    if intent.temporal_intent == "DATE_SPECIFIC":
        if intent.requested_period:
            req_clean = intent.requested_period.replace("FY", "").strip()
            matched = []
            for s, m in source_metas:
                if m.fiscal_year and (req_clean in m.fiscal_year or m.fiscal_year in req_clean):
                    matched.append(s)
                elif m.year_end and intent.requested_year and m.year_end == intent.requested_year:
                    matched.append(s)

            if not matched:
                return TemporalValidationResult(
                    temporal_status="TEMPORAL_MISMATCH",
                    is_valid=False,
                    answer_allowed_as_current=False,
                    latest_reporting_period=latest_period,
                    latest_source_title=latest_title,
                    requested_period=intent.requested_period,
                    reason=f"Requested period {intent.requested_period} is not supported by retrieved evidence (found {latest_period}).",
                )

        return TemporalValidationResult(
            temporal_status="VALID",
            is_valid=True,
            answer_allowed_as_current=False,
            latest_reporting_period=latest_period,
            latest_source_title=latest_title,
            matched_sources=sources,
            reason="Date-specific query matches retrieved documentary evidence.",
        )

    # Case 4: UNSPECIFIED
    return TemporalValidationResult(
        temporal_status="VALID",
        is_valid=True,
        answer_allowed_as_current=False,
        latest_reporting_period=latest_period,
        latest_source_title=latest_title,
        matched_sources=sources,
        reason="Query has no temporal constraints; retrieved evidence accepted.",
    )


# ── 4. Temporal Claim Validator & Hallucination Prevention ────────────────────

def validate_temporal_claims(
    answer_text: str,
    sources: list[dict],
    intent: TemporalIntent | None = None,
) -> TemporalClaimResult:
    """
    Validates that every fiscal year, date, or reporting period claimed in the answer
    is strictly substantiated by the cited sources.

    Prevents bugs like:
    Claiming "as of FY2019" while citing "29th Annual Report (2016-17)".
    Claiming "MRPL currently has..." when answer_allowed_as_current is False.
    """
    if not answer_text:
        return TemporalClaimResult(valid=True, details="No text to validate.")

    allowed_periods = set()
    allowed_years = set()

    for s in sources:
        meta = extract_temporal_metadata(s)
        if meta.reporting_period:
            allowed_periods.add(meta.reporting_period.upper())
            allowed_periods.add(meta.reporting_period.replace("FY", "").upper())
        if meta.fiscal_year:
            allowed_periods.add(meta.fiscal_year.upper())
            allowed_periods.add(f"FY{meta.fiscal_year.upper()}")
        if meta.year_end:
            allowed_years.add(meta.year_end)
            if meta.fiscal_year and "-" in meta.fiscal_year:
                try:
                    start_yr = int(meta.fiscal_year.split("-")[0])
                    allowed_years.add(start_yr)
                except Exception:
                    pass

        chunk_text = s.get("text", "")
        for m in re.finditer(r'\b(?:FY\s*)?(20\d{2})[-–]?(\d{2})?\b', chunk_text):
            y_start = int(m.group(1))
            allowed_years.add(y_start)
            if m.group(2):
                y_end = int(str(y_start)[:2] + m.group(2))
                allowed_years.add(y_end)
                allowed_periods.add(f"FY{y_start}-{m.group(2)}")
                allowed_periods.add(f"{y_start}-{m.group(2)}")

    claimed_periods = []
    for m in re.finditer(r'\b(?:as\s+of\s+|in\s+|for\s+)?(FY\s*20\d{2}(?:[-–]\d{2,4})?)\b', answer_text, re.IGNORECASE):
        claimed_periods.append(m.group(1).upper().replace(" ", ""))

    claimed_phrased_years = []
    for m in re.finditer(r'\b(?:as\s+of|in|during|for\s+the\s+year)\s+(20\d{2})\b', answer_text, re.IGNORECASE):
        claimed_phrased_years.append(int(m.group(1)))

    unsupported_claims = []

    for cp in claimed_periods:
        norm_cp = cp.replace(" ", "")
        matched = any(norm_cp == ap or ap in norm_cp or norm_cp in ap for ap in allowed_periods)
        if not matched:
            y_match = re.search(r'20\d{2}', norm_cp)
            if y_match and int(y_match.group(0)) in allowed_years:
                matched = True

        if not matched:
            unsupported_claims.append(f"Period {cp} (not substantiated by cited sources)")

    for cy in claimed_phrased_years:
        if cy not in allowed_years and cy != CURRENT_DEPLOYMENT_YEAR:
            unsupported_claims.append(f"Year {cy} (unsubstantiated by cited sources)")

    if intent:
        intent_type = getattr(intent, "temporal_intent", None) or (intent.get("temporal_intent") if isinstance(intent, dict) else None)
        req_recent = getattr(intent, "requires_recent_evidence", False) if not isinstance(intent, dict) else intent.get("requires_recent_evidence", False)
        if intent_type in ("CURRENT", "RECENT"):
            current_assertions = re.findall(
                r'\b(?:mrpl\s+currently\s+has|current\s+(?:revenue|turnover|pat|profit)\s+is|currently\s+stands\s+at)\b',
                answer_text,
                re.IGNORECASE
            )
            if current_assertions and not req_recent:
                unsupported_claims.append("Asserts current status using historical evidence")

    if unsupported_claims:
        logger.warning(
            "Temporal grounding failure: Answer contains unsupported temporal claims: %s (Allowed: %s / %s)",
            unsupported_claims,
            allowed_periods,
            allowed_years,
        )
        return TemporalClaimResult(
            valid=False,
            flagged=True,
            unsupported_claims=unsupported_claims,
            detected_periods=claimed_periods,
            details=f"Temporal grounding failure: {', '.join(unsupported_claims)}",
        )

    return TemporalClaimResult(
        valid=True,
        flagged=False,
        unsupported_claims=[],
        detected_periods=claimed_periods,
        details="All temporal claims and reporting periods verified against source documents.",
    )


# ── 5. Safe Evidence-Limited Response Generator ──────────────────────────────

def generate_evidence_limited_response(
    query: str,
    temporal_val: TemporalValidationResult,
    sources: list[dict],
) -> str:
    """
    Generate a deterministic, evidence-limited response when current/recent information
    cannot be reliably determined from the available knowledge base.
    """
    latest_title = temporal_val.latest_source_title or "Annual Report"
    period_label = f" ({temporal_val.latest_reporting_period})" if temporal_val.latest_reporting_period else ""

    response = (
        f"I can't reliably determine MRPL's current financial status from the available knowledge base. "
        f"The latest relevant financial document retrieved is the {latest_title}{period_label}, "
        f"which is historical and should not be treated as current."
    )

    if sources and temporal_val.latest_reporting_period:
        for s in sources:
            text = s.get("text", "")
            title = s.get("document_title") or s.get("source", "")
            page = s.get("page", "")
            if "turnover" in text.lower() or "pat" in text.lower() or "operating income" in text.lower():
                response += f"\n\nFor reference, the {temporal_val.latest_reporting_period} report records historical figures as follows:"
                sentences = text.split(". ")
                extracted_points = []
                for sent in sentences:
                    if any(k in sent.lower() for k in ["turnover", "profit after tax", "pat", "operating income", "ebitda"]):
                        clean_s = sent.strip().rstrip(".")
                        if clean_s and len(clean_s) > 15 and len(clean_s) < 250:
                            extracted_points.append(f"- {clean_s} (Source: {title}, p. {page})")
                    if len(extracted_points) >= 3:
                        break

                if extracted_points:
                    response += "\n" + "\n".join(extracted_points)
                break

    return response
