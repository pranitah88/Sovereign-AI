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

    # 1. Extract and normalize signals from retrieved sources
    distances = []
    rerank_scores = []
    bm25_scores = []

    for s in sources:
        dist_val = s.get("cosine_distance") if s.get("cosine_distance") is not None else (
            s.get("distance") if s.get("distance") is not None else s.get("dist")
        )
        if dist_val is not None:
            try:
                d = float(dist_val)
                # If distance > 1.0, it is a raw squared L2 distance from Chroma (||u - v||^2 in [0, 2]).
                # Convert to normalized cosine distance in [0, 1]:
                if d > 1.0:
                    d = min(1.0, max(0.0, d / 2.0))
                distances.append(d)
            except (ValueError, TypeError):
                pass

        rerank_val = s.get("rerank_score") if s.get("rerank_score") is not None else s.get("score")
        if rerank_val is not None:
            try:
                rerank_scores.append(float(rerank_val))
            except (ValueError, TypeError):
                pass

        bm25_val = s.get("bm25_score")
        if bm25_val is not None:
            try:
                bm25_scores.append(float(bm25_val))
            except (ValueError, TypeError):
                pass

    best_dist = min(distances) if distances else None
    avg_dist = (sum(distances) / len(distances)) if distances else None
    distance_available = best_dist is not None

    best_rerank = max(rerank_scores) if rerank_scores else None
    rerank_available = best_rerank is not None

    best_bm25 = max(bm25_scores) if bm25_scores else None
    bm25_available = best_bm25 is not None

    # Deterministic signal normalization (heuristic normalization for fusion, NOT probability calibration)
    import math

    def _sigmoid(x: float) -> float:
        # Logistic curve mapping unbounded CrossEncoder logits to [0, 1] for signal fusion
        return 1.0 / (1.0 + math.exp(-0.5 * x))

    def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, val))

    rerank_signal = _clamp(_sigmoid(best_rerank)) if rerank_available else None
    dense_signal = _clamp(1.0 - ((best_dist - 0.15) / 0.70)) if distance_available else None
    bm25_signal = _clamp(best_bm25 / 15.0) if bm25_available else None

    # Multi-Signal Evidence Fusion (do not penalize a candidate because another signal is absent)
    if rerank_signal is not None and dense_signal is not None:
        confidence = 0.60 * rerank_signal + 0.40 * dense_signal
    elif rerank_signal is not None and bm25_signal is not None:
        confidence = 0.60 * rerank_signal + 0.40 * bm25_signal
    elif rerank_signal is not None:
        confidence = rerank_signal
    elif dense_signal is not None:
        confidence = dense_signal
    elif bm25_signal is not None:
        confidence = bm25_signal
    else:
        # Fallback heuristic if all numerical signals are absent
        confidence = min(0.85, 0.40 + (len(sources) * 0.15))

    metrics = {
        "source_count": len(sources),
        "distance_available": distance_available,
        "best_distance": round(best_dist, 4) if best_dist is not None else None,
        "avg_distance": round(avg_dist, 4) if avg_dist is not None else None,
        "rerank_available": rerank_available,
        "best_rerank_score": round(best_rerank, 4) if best_rerank is not None else None,
        "bm25_available": bm25_available,
        "best_bm25_score": round(best_bm25, 4) if best_bm25 is not None else None,
        "dense_signal": round(dense_signal, 4) if dense_signal is not None else None,
        "rerank_signal": round(rerank_signal, 4) if rerank_signal is not None else None,
        "bm25_signal": round(bm25_signal, 4) if bm25_signal is not None else None,
        "fused_confidence": round(confidence, 4),
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
                confidence_score=min(confidence, 0.20),
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                escalation_required=True,
                reason=t_reason or "Retrieved documentary evidence is historical and cannot substantiate a current inquiry without hallucination.",
                metrics=metrics,
            )

        if t_status == "TEMPORAL_MISMATCH":
            metrics["temporal_status"] = t_status
            return ConfidenceDecision(
                confidence_score=min(confidence, 0.25),
                status="INSUFFICIENT_EVIDENCE",
                is_sufficient=False,
                escalation_required=True,
                reason=t_reason or "Retrieved documentary evidence does not match the requested temporal period.",
                metrics=metrics,
            )

    # 3. Decision logic
    # Distance failure ONLY if distance is available and strictly exceeds MAX_ACCEPTABLE_DISTANCE
    if best_dist is not None and best_dist > MAX_ACCEPTABLE_DISTANCE:
        return ConfidenceDecision(
            confidence_score=confidence,
            status="INSUFFICIENT_EVIDENCE",
            is_sufficient=False,
            escalation_required=True,
            reason=f"Retrieved document distance ({round(best_dist, 3)}) exceeds acceptable threshold ({MAX_ACCEPTABLE_DISTANCE}). Evidence is insufficient to answer without hallucination.",
            metrics=metrics,
        )

    # Low relevance/confidence failure
    if confidence < 0.35:
        return ConfidenceDecision(
            confidence_score=confidence,
            status="INSUFFICIENT_EVIDENCE",
            is_sufficient=False,
            escalation_required=True,
            reason=f"Evidence relevance/confidence ({round(confidence, 3)}) was below the minimum sufficiency threshold (0.35).",
            metrics=metrics,
        )

    if confidence >= 0.70 or (best_dist is not None and best_dist <= STRONG_DISTANCE_THRESHOLD and confidence >= 0.50):
        return ConfidenceDecision(
            confidence_score=confidence,
            status="HIGH_CONFIDENCE",
            is_sufficient=True,
            escalation_required=False,
            reason="Strong documentary evidence corroborated by available retrieval signals.",
            metrics=metrics,
        )

    return ConfidenceDecision(
        confidence_score=confidence,
        status="MODERATE_CONFIDENCE",
        is_sufficient=True,
        escalation_required=False,
        reason="Adequate evidence located in authorized technical documentation.",
        metrics=metrics,
    )
