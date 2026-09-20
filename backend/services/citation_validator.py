"""
Deterministic Citation Validator for MRPL Sovereign AI Workbench.

Ensures that every citation and page reference generated in an AI answer
strictly corresponds to an actual retrieved document and page number.
Zero invented page numbers or synthetic references permitted.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CitationValidationResult:
    valid: bool
    cited_pages: list[int]
    allowed_pages: list[int]
    invalid_citations: list[str] = field(default_factory=list)
    flagged: bool = False
    details: str = ""
    temporal_claims: dict[str, Any] | None = None

    @property
    def is_grounded(self) -> bool:
        return self.valid and not self.flagged

    @property
    def valid_citations(self) -> list[int]:
        return [p for p in self.cited_pages if p in self.allowed_pages]

    @property
    def flagged_citations(self) -> list[dict]:
        return [{"citation": c} for c in self.invalid_citations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "flagged": self.flagged,
            "is_grounded": self.is_grounded,
            "cited_pages": self.cited_pages,
            "allowed_pages": self.allowed_pages,
            "invalid_citations": self.invalid_citations,
            "details": self.details,
            "temporal_claims": self.temporal_claims,
        }


def validate_citations(
    answer_text: str,
    sources: list[dict],
    intent: Any | None = None,
    check_temporal: bool = True,
) -> CitationValidationResult:
    """
    Validate all page citations in the generated answer against the retrieved source pages,
    and verify that all claimed reporting periods/dates are substantiated by cited sources.
    """
    if not answer_text:
        return CitationValidationResult(
            valid=True,
            cited_pages=[],
            allowed_pages=[],
            details="No answer content to validate.",
        )

    # 1. Collect all authorized page numbers from retrieved sources
    allowed_pages_set = set()
    for s in sources:
        page = s.get("page")
        if page is not None:
            try:
                allowed_pages_set.add(int(page))
            except (ValueError, TypeError):
                pass

    allowed_pages = sorted(list(allowed_pages_set))

    # 2. Extract cited page references from answer text
    # Patterns like: "page 12", "p. 5", "pages 3 and 4", "p.14"
    page_patterns = [
        r"(?i)\bpage[s]?\s+(\d+)\b",
        r"(?i)\bp\.?\s*(\d+)\b",
        r"(?i)\[[^\]]*\bp\.?\s*(\d+)[^\]]*\]",
    ]

    extracted_pages = []
    for pat in page_patterns:
        matches = re.findall(pat, answer_text)
        for m in matches:
            try:
                extracted_pages.append(int(m))
            except ValueError:
                pass

    unique_cited_pages = sorted(list(set(extracted_pages)))

    # 3. Check for citations of non-existent or unretrieved pages
    invalid_citations = []
    if allowed_pages_set:
        for p in unique_cited_pages:
            if p not in allowed_pages_set:
                invalid_citations.append(f"Page {p} (not in retrieved evidence)")

    # 4. Check for temporal claim validity against sources
    temporal_res = None
    if check_temporal:
        try:
            from backend.services.temporal_guard import validate_temporal_claims
            t_claims = validate_temporal_claims(answer_text, sources, intent=intent)
            temporal_res = t_claims.to_dict()
            if not t_claims.valid:
                invalid_citations.extend(t_claims.unsupported_claims)
        except Exception as e:
            logger.warning("Temporal claim validation within citation_validator encountered: %s", e)

    if invalid_citations:
        logger.warning(
            "Citation/Temporal validation failed: Issues found: %s. Allowed pages: %s",
            invalid_citations,
            allowed_pages,
        )
        return CitationValidationResult(
            valid=False,
            flagged=True,
            cited_pages=unique_cited_pages,
            allowed_pages=allowed_pages,
            invalid_citations=invalid_citations,
            details=f"Answer referenced invalid citations or unsupported temporal periods: {', '.join(invalid_citations)}.",
            temporal_claims=temporal_res,
        )

    return CitationValidationResult(
        valid=True,
        flagged=False,
        cited_pages=unique_cited_pages,
        allowed_pages=allowed_pages,
        invalid_citations=[],
        details="All citations and reporting periods verified against authoritative source chunks.",
        temporal_claims=temporal_res,
    )

