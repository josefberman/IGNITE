"""Uncertainty estimation from source reliability, analogue distance, and intervals."""

from __future__ import annotations

from typing import Dict, Optional

from backend.app.domain.models import PredictionSource, PropertyEstimate


SOURCE_BASE_CONFIDENCE = {
    PredictionSource.DATASET: 0.95,
    PredictionSource.LITERATURE: 0.85,
    PredictionSource.ANALOGUE: 0.70,
    PredictionSource.XGBOOST: 0.60,
    PredictionSource.DELTA_MODEL: 0.55,
    PredictionSource.UNKNOWN: 0.40,
}


def calibrate_estimate(
    estimate: PropertyEstimate,
    analogue_tanimoto: Optional[float] = None,
    ood_distance: Optional[float] = None,
) -> PropertyEstimate:
    conf = SOURCE_BASE_CONFIDENCE.get(estimate.source, 0.4)
    if estimate.source == PredictionSource.ANALOGUE and analogue_tanimoto is not None:
        conf = 0.4 + 0.5 * max(0.0, min(1.0, analogue_tanimoto))
    if ood_distance is not None:
        # ood_distance in [0,1], higher = more OOD
        conf *= max(0.2, 1.0 - 0.6 * ood_distance)

    conf = max(0.05, min(0.99, conf))
    # Symmetric relative interval width grows as confidence falls
    rel = (1.0 - conf) * 0.25
    lower = estimate.value * (1.0 - rel)
    upper = estimate.value * (1.0 + rel)
    return PropertyEstimate(
        value=estimate.value,
        source=estimate.source,
        confidence=conf,
        lower=lower,
        upper=upper,
        citation=estimate.citation,
    )


def apply_uncertainty(
    props: Dict[str, PropertyEstimate],
    analogue_tanimoto: Optional[float] = None,
    ood_distance: Optional[float] = None,
) -> Dict[str, PropertyEstimate]:
    return {
        k: calibrate_estimate(v, analogue_tanimoto=analogue_tanimoto, ood_distance=ood_distance)
        for k, v in props.items()
    }


def ensemble_spread_confidence(values: list[float]) -> float:
    if len(values) < 2:
        return 0.5
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    rel = std / (abs(mean) + 1e-8)
    return max(0.05, min(0.95, 1.0 / (1.0 + 5.0 * rel)))
