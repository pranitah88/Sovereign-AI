"""
Deterministic Query Metric Analyzer for Financial and Quantitative Queries.

Extracts:
- primary_metric: e.g. "revenue", "profit", "exports", "throughput", "expenses", "debt", "dividend", "margin", "ratio"
- metric_terms: specific terms/synonyms associated with the metric
- year: standardized fiscal year representation (e.g. "FY2025", "FY2024-25")
- year_variants: alternative textual representations in annual reports (e.g. "2024-25", "March 31, 2025")
- company: e.g. "MRPL"
- is_exact_value_query: bool (e.g. "what was", "how much", "what is")
- is_percentage_query: bool (e.g. "what percentage", "contribution of exports as a percentage")
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class QueryAnalysis:
    query: str
    primary_metric: Optional[str] = None
    metric_terms: List[str] = field(default_factory=list)
    exact_phrases: List[str] = field(default_factory=list)
    competing_metrics: List[str] = field(default_factory=list)
    year: Optional[str] = None
    year_variants: List[str] = field(default_factory=list)
    company: Optional[str] = "MRPL"
    is_exact_value_query: bool = False
    is_percentage_query: bool = False

    @property
    def is_metric_query(self) -> bool:
        return self.primary_metric is not None

    @property
    def requires_exact_value(self) -> bool:
        return self.is_exact_value_query

    @property
    def requires_percentage(self) -> bool:
        return self.is_percentage_query

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "is_metric_query": self.is_metric_query,
            "primary_metric": self.primary_metric,
            "metric_terms": self.metric_terms,
            "exact_phrases": self.exact_phrases,
            "competing_metrics": self.competing_metrics,
            "year": self.year,
            "year_variants": self.year_variants,
            "company": self.company,
            "is_exact_value_query": self.is_exact_value_query,
            "is_percentage_query": self.is_percentage_query,
        }


# Metric definitions with priority ordering (more specific metrics like "exports percentage" or "profit after tax" match first)
METRIC_DEFINITIONS = {
    "exports": {
        "patterns": [
            r"\b(exports?|export\s+turnover|export\s+revenue|export\s+sales|overseas\s+sales)\b",
            r"\b(contribution\s+of\s+exports|exports?\s+percentage|exports?\s+as\s+a\s+percentage)\b",
            r"(निर्यात)",
        ],
        "primary_terms": [
            "exports", "export", "export turnover", "export revenue",
            "contribution of exports", "percentage of the total turnover", "export customers",
        ],
        "competing": ["revenue", "profit", "throughput"],
    },
    "profit": {
        "patterns": [
            r"\b(profit\s+after\s+tax|pat|profit\s+before\s+tax|pbt|net\s+profit|operating\s+profit|gross\s+profit|ebitda|ebit)\b",
            r"\b(profit|earnings|net\s+income)\b",
            r"(लाभ|नफा|मुनाफा)",
        ],
        "primary_terms": [
            "profit after tax", "pat", "profit before tax", "pbt", "net profit",
            "profit for the year", "profit / (loss)", "ebitda", "profit",
        ],
        "competing": ["revenue", "exports", "throughput"],
    },
    "throughput": {
        "patterns": [
            r"\b(throughput|refinery\s+throughput|crude\s+throughput|crude\s+processed|production\s+volume|refining\s+capacity)\b",
            r"\b(production|crude\s+distillation)\b",
            r"(थ्रूपुट|प्रसंस्करण)",
        ],
        "primary_terms": [
            "throughput", "refinery throughput", "crude throughput",
            "thruput", "crude processed", "million tonnes", "mmt", "production",
        ],
        "competing": ["revenue", "exports", "profit"],
    },
    "revenue": {
        "patterns": [
            r"\b(revenue\s+from\s+operations|gross\s+revenue|net\s+revenue|total\s+revenue|turnover|total\s+income|income\s+from\s+operations|operating\s+revenue)\b",
            r"\b(revenue|sales|topline)\b",
            r"(राजस्व|महसूल|टर्नओवर|उलाढाल)",
        ],
        "primary_terms": [
            "revenue from operations", "revenue (gross)", "gross revenue", "net revenue",
            "total turnover", "turnover", "total income", "income from operations",
            "consolidated turnover", "standalone turnover", "revenue",
        ],
        "competing": ["exports", "profit", "throughput", "csr", "employee benefits"],
    },
    "expenses": {
        "patterns": [
            r"\b(expenses?|total\s+expenses?|expenditure|operating\s+cost|cost\s+of\s+materials)\b",
            r"(खर्च|व्यय)",
        ],
        "primary_terms": ["expenses", "total expenses", "expenditure", "cost of materials consumed"],
        "competing": ["revenue", "profit"],
    },
    "debt": {
        "patterns": [
            r"\b(debt|borrowings|total\s+debt|debt\s+equity\s+ratio|liabilities)\b",
            r"(ऋण|कर्ज)",
        ],
        "primary_terms": ["total debt", "borrowings", "debt equity ratio", "non-current borrowings"],
        "competing": ["revenue", "profit"],
    },
    "dividend": {
        "patterns": [
            r"\b(dividend|dividend\s+per\s+share|dps)\b",
            r"(लाभांश)",
        ],
        "primary_terms": ["dividend", "dividend per share", "final dividend", "interim dividend"],
        "competing": ["revenue", "exports"],
    },
    "margin": {
        "patterns": [
            r"\b(gross\s+refining\s+margin|grm|operating\s+margin|net\s+profit\s+margin|margin)\b",
        ],
        "primary_terms": ["gross refining margin", "grm", "$/bbl", "operating margin"],
        "competing": ["exports", "throughput"],
    },
    "ratio": {
        "patterns": [
            r"\b(inventory\s+turnover\s+ratio|trade\s+receivables\s+turnover|net\s+profit\s+ratio|return\s+on\s+capital\s+employed|roce|roe|ratio)\b",
        ],
        "primary_terms": ["turnover ratio", "net profit ratio", "return on capital employed", "roce"],
        "competing": ["revenue"],
    },
}


def analyze_query(query: str) -> QueryAnalysis:
    """
    Deterministically analyze the user query for quantitative metrics,
    target fiscal years, companies, and answer requirements.
    """
    q_lower = query.strip().lower()

    # 1. Company Detection
    company = "MRPL"
    if "shell" in q_lower or "smafsl" in q_lower:
        company = "SMAFSL"
    elif "ompl" in q_lower:
        company = "OMPL"
    elif "ongc" in q_lower and "mrpl" not in q_lower:
        company = "ONGC"

    # 2. Percentage and Exact Value Intent
    is_percentage = bool(re.search(
        r"\b(percentage|percent|%|proportion|share\s+of|contribution\s+of)\b",
        q_lower,
    ))
    is_exact_value = bool(re.search(
        r"^(what\s+was|what\s+is|how\s+much|how\s+many|what\s+percentage|state|give\s+me|tell\s+me)\b",
        q_lower,
    )) or bool(re.search(r"\b(what\s+was|what\s+is|how\s+much|calculate|report)\b", q_lower)) or bool(re.search(r"(कितना|किती|काय|क्या)", q_lower))

    # 3. Metric Detection
    detected_metric = None
    detected_terms = []
    competing_metrics = []

    # Priority rule: If user asks for "percentage of ... from exports" or "contribution of exports",
    # the primary metric is exports (not turnover/revenue).
    if bool(re.search(r"\b(exports?|contribution\s+of\s+exports)\b", q_lower)) and is_percentage:
        detected_metric = "exports"
    elif "profit after tax" in q_lower or r"\bpat\b" in q_lower:
        detected_metric = "profit"
    elif "refinery throughput" in q_lower or "throughput" in q_lower:
        detected_metric = "throughput"
    else:
        # Check definitions in order
        for metric, defn in METRIC_DEFINITIONS.items():
            matched = False
            for pat in defn["patterns"]:
                if re.search(pat, q_lower):
                    matched = True
                    break
            if matched:
                detected_metric = metric
                break

    exact_phrases = []
    if detected_metric:
        detected_terms = list(METRIC_DEFINITIONS[detected_metric]["primary_terms"])
        competing_metrics = list(METRIC_DEFINITIONS[detected_metric]["competing"])
        exact_phrases = [t for t in detected_terms if len(t.split()) > 1][:4]

    # 4. Fiscal Year Detection
    year = None
    year_variants = []

    # Check for specific FY patterns
    # Matches: FY2025, FY 2025, FY25, FY 2024-25, 2024-25, FY2025-26, etc.
    fy_dash_match = re.search(r"\b(?:fy\s*)?(20\d\d)\s*[-–]\s*(\d\d(?:\d\d)?)\b", q_lower)
    fy_single_match = re.search(r"\b(?:fy\s*|financial\s+year\s*)(20\d\d|\d\d)\b", q_lower)
    bare_year_match = re.search(r"\b(20\d\d)\b", q_lower)

    if fy_dash_match:
        y1 = int(fy_dash_match.group(1))
        y2_str = fy_dash_match.group(2)
        y2 = int(y2_str[-2:])  # 2-digit end year
        year = f"FY{y1}-{y2:02d}"
        year_variants = [
            f"{y1}-{y2:02d}",
            f"{y1}–{y2:02d}",
            f"FY {y1}-{y2:02d}",
            f"FY{y1}-{y2:02d}",
            f"March 31, {y1 + 1}",
            f"31st March, {y1 + 1}",
            str(y1 + 1),
        ]
    elif fy_single_match:
        val = fy_single_match.group(1)
        full_year = int(val) if len(val) == 4 else (2000 + int(val))
        # In Indian reporting, FY2025 refers to 2024-25 (ending March 31, 2025)
        # However, users also might mean calendar year or period ending 2025.
        y_prev = full_year - 1
        y_end = full_year % 100
        year = f"FY{full_year}"
        year_variants = [
            f"{y_prev}-{y_end:02d}",
            f"{y_prev}–{y_end:02d}",
            f"FY {y_prev}-{y_end:02d}",
            f"FY{y_prev}-{y_end:02d}",
            f"March 31, {full_year}",
            f"31st March, {full_year}",
            f"ended March 31, {full_year}",
            str(full_year),
            f"FY{full_year}",
            f"FY {full_year}",
        ]
    elif bare_year_match:
        full_year = int(bare_year_match.group(1))
        y_prev = full_year - 1
        y_end = full_year % 100
        year = str(full_year)
        year_variants = [
            str(full_year),
            f"{y_prev}-{y_end:02d}",
            f"{y_prev}–{y_end:02d}",
            f"March 31, {full_year}",
            f"FY{full_year}",
        ]

    return QueryAnalysis(
        query=query,
        primary_metric=detected_metric,
        metric_terms=detected_terms,
        exact_phrases=exact_phrases,
        competing_metrics=competing_metrics,
        year=year,
        year_variants=year_variants,
        company=company,
        is_exact_value_query=is_exact_value,
        is_percentage_query=is_percentage,
    )
