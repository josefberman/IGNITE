"""Property estimate helpers (intervals are not client-facing)."""

from __future__ import annotations

from typing import Dict, Optional

from backend.app.domain.models import PredictionSource, PropertyEstimate


def calibrate_estimate(
    estimate: PropertyEstimate,
    ood_distance: Optional[float] = None,
) -> PropertyEstimate:
    """Attach source only — no confidence intervals for reporting."""
    _ = ood_distance  # retained for call-site compatibility
    return PropertyEstimate(
        value=estimate.value,
        source=estimate.source,
        confidence=0.5,
        lower=None,
        upper=None,
        citation=estimate.citation,
    )


def apply_uncertainty(
    props: Dict[str, PropertyEstimate],
    ood_distance: Optional[float] = None,
) -> Dict[str, PropertyEstimate]:
    return {k: calibrate_estimate(v, ood_distance=ood_distance) for k, v in props.items()}


def ensemble_spread_confidence(values: list[float]) -> float:
    if len(values) < 2:
        return 0.5
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    rel = std / (abs(mean) + 1e-8)
    return max(0.05, min(0.95, 1.0 / (1.0 + 5.0 * rel)))
