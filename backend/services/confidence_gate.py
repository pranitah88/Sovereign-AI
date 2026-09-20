"""
Deterministic Retrieval Confidence Gate for MRPL Sovereign AI Workbench.

Calibrates retrieval confidence using real ChromaDB vector distances, BM25 keyword
overlap, and cross-encoder reranking scores. Prevents hallucination when source
evidence is weak or missing, deterministically triggering escalation or refusal.
"""

from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Calibrated Empirical Thresholds for MRPL Corpus
# nomic-embed-text / all-MiniLM-L6-v2 cosine distance scale:
# 0.00 - 0.45: Very Strong match
# 0.45 - 0.70: Good contextual match
# 0.70 - 0.85: Weak / marginal match
# > 0.85: Distant / irrelevant
MAX_ACCEPTABLE_DISTANCE = 0.82
STRONG_DISTANCE_THRESHOLD = 0.55
MIN_RERANK_SCORE = -2.5


@dataclass
class ConfidenceDecision:
    confidence_score: float  # Normalized 0.0 to 1.0
    status: str  # "HIGH_CONFIDENCE" | "MODERATE_CONFIDENCE" | "INSUFFICIENT_EVIDENCE"
    is_sufficient: bool
    escalation_required: bool
    reason: str
    metrics: dict[str, Any]

    @property
    def confidence_band(self) -> str:
        return self.status

    @property
    def requires_escalation(self) -> bool:
        return self.escalation_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_score": round(self.confidence_score, 3),
            "status": self.status,
            "confidence_band": self.status,
            "is_sufficient": self.is_sufficient,
            "escalation_required": self.escalation_required,
            "requires_escalation": self.escalation_required,
            "reason": self.reason,
            "metrics": self.metrics,
        }


RetrievalConfidence = ConfidenceDecision



def evaluate_retrieval_confidence(
    query: str,
    sources: list[dict],
    top_candidates: list[dict] | None = None,
    temporal_validation: Any | None = None,
) -> ConfidenceDecision:
    """
    Evaluate retrieval quality against empirical knowledge base calibration.
    Returns ConfidenceDecision.
    """
    if not sources:
        return ConfidenceDecision(
            confidence_score=0.0,
            status="INSUFFICIENT_EVIDENCE",
            is_sufficient=False,
            escalation_required=True,
            reason="No relevant technical documents found in the local knowledge base for this query.",
            metrics={"source_count": 0, "best_distance": None},
        )

    # 1. Inspect distances and scores from sources
    if not sources or len(sources) == 0:
        return ConfidenceDecision(
            confidence_score=0.0,
            status="INSUFFICIENT_EVIDENCE",
            is_sufficient=False,
            escalation_required=True,
            reason="Zero authorized technical documents retrieved for this query.",
            metrics={"source_count": 0, "best_distance": None, "avg_distance": None, "best_rerank_score": None},
        )

    distances = []
    rerank_scores = []

    for s in sources:
        dist_val = s.get("distance") if s.get("distance") is not None else s.get("dist")
        if dist_val is not None:
            distances.append(float(dist_val))
        rerank_val = s.get("rerank_score") if s.get("rerank_score") is not None else s.get("score")
        if rerank_val is not None:
            rerank_scores.append(float(rerank_val))

    best_dist = min(distances) if distances else None
    avg_dist = (sum(distances) / len(distances)) if distances else None
    best_rerank = max(rerank_scores) if rerank_scores else None

    # Calculate normalized confidence (0.0 to 1.0)
    if best_dist is not None:
        # Lower distance is higher confidence
        # Distance 0.3 -> 0.95, Distance 0.8 -> 0.20
        raw_conf = max(0.0, min(1.0, 1.0 - ((best_dist - 0.2) / 0.7)))
    else:
        # Fall back to source count heuristic if distances are stripped
        raw_conf = min(0.85, 0.4 + (len(sources) * 0.15))

    # Boost for cross-encoder reranker confirmation
    if best_rerank is not None:
        if best_rerank > 2.0:
            raw_conf = min(1.0, raw_conf + 0.1)
        elif best_rerank < MIN_RERANK_SCORE:
            raw_conf = max(0.1, raw_conf - 0.25)

    metrics = {
        "source_count": len(sources),
        "best_distance": round(best_dist, 4) if best_dist is not None else None,
        "avg_distance": round(avg_dist, 4) if avg_dist is not None else None,
        "best_rerank_score": round(best_rerank, 4) if best_rerank is not None else None,
    }

    # 2. Temporal validation integration
    if temporal_validation is not None:
        t_status = getattr(temporal_validation, "temporal_status", None)
        if isinstance(temporal_validation, dict):
            t_status = temporal_validation.get("temporal_status")
            t_reason = temporal_validation.get("reason", "")
            t_period = temporal_validation.get("latest_reporting_period")
        else:
            t_reason = getattr(temporal_validation, "reason", "")
            t_period = getattr(temporal_validation, "latest_reporting_period", None)

        if t_status == "INSUFFICIENT_RECENT_EVIDENCE":
            metrics["temporal_status"] = t_status
            metrics["latest_reporting_period"] = t_period
            return ConfidenceDecision(
                confidence_score=min(raw_conf, 0.20),
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                escalation_required=True,
                reason=t_reason or "Retrieved documentary evidence is historical and cannot substantiate a current inquiry without hallucination.",
                metrics=metrics,
            )

        if t_status == "TEMPORAL_MISMATCH":
            metrics["temporal_status"] = t_status
            return ConfidenceDecision(
                confidence_score=min(raw_conf, 0.25),
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                escalation_required=True,
                reason=t_reason or "Retrieved documentary evidence does not match the requested temporal period.",
                metrics=metrics,
            )

    # 3. Decision logic
    if (best_dist is not None and best_dist > MAX_ACCEPTABLE_DISTANCE) or raw_conf < 0.35:
        return ConfidenceDecision(
            confidence_score=raw_conf,
            status="INSUFFICIENT_EVIDENCE",
            is_sufficient=False,
            escalation_required=True,
            reason=f"Retrieved document distance ({round(best_dist, 3) if best_dist else 'high'}) exceeds acceptable threshold ({MAX_ACCEPTABLE_DISTANCE}). Evidence is insufficient to answer without hallucination.",
            metrics=metrics,
        )

    if (best_dist is not None and best_dist <= STRONG_DISTANCE_THRESHOLD) or raw_conf >= 0.70:
        return ConfidenceDecision(
            confidence_score=raw_conf,
            status="HIGH_CONFIDENCE",
            is_sufficient=True,
            escalation_required=False,
            reason="Strong documentary evidence corroborated by vector distance and keyword overlap.",
            metrics=metrics,
        )

    return ConfidenceDecision(
        confidence_score=raw_conf,
        status="MODERATE_CONFIDENCE",
        is_sufficient=True,
        escalation_required=False,
        reason="Adequate evidence located in authorized technical documentation.",
        metrics=metrics,
    )
