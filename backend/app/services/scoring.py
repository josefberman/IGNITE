"""MAPE and multi-objective ranking."""

from __future__ import annotations

from typing import Dict, Optional

from backend.app.domain.models import (
    PROPERTY_KEYS,
    CandidateMolecule,
    PredictionSource,
    ScoreBreakdown,
)


# Final ranking uses MAPE and synthetic accessibility only.
DEFAULT_WEIGHTS = {
    "mape": 0.65,
    "sascore": 0.25,
}


def calculate_mape(
    predicted: Dict[str, float],
    target: Dict[str, float],
    property_weights: Optional[Dict[str, float]] = None,
) -> float:
    weights = property_weights or {k: 0.25 for k in PROPERTY_KEYS}
    mape = 0.0
    total_w = 0.0
    for key in PROPERTY_KEYS:
        if key not in predicted or key not in target:
            continue
        pred, tgt = predicted[key], target[key]
        if abs(tgt) > 1e-10:
            err = abs(pred - tgt) / abs(tgt) * 100.0
        else:
            err = abs(pred - tgt) * 100.0
        w = weights.get(key, 0.25)
        mape += w * err
        total_w += w
    return mape / total_w if total_w > 0 else 100.0


def score_candidate(
    candidate: CandidateMolecule,
    target: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    max_edit: int = 10,
    *,
    explore: bool = False,
) -> ScoreBreakdown:
    """
    Ranking score (lower is better): MAPE + SAScore.

    Hard safety rejection and optional novelty gating happen upstream in beam search.
    Safety / uncertainty / novelty are not reported to clients.
    ``max_edit`` is unused; edit size is recorded on the modification, not in the score.
    """
    del max_edit
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    w.pop("minimal_edit", None)
    if explore:
        w = {**w, "mape": 0.85, "sascore": 0.10}

    props = candidate.property_values()
    mape = calculate_mape(props, target)
    mape_norm = min(mape / 100.0, 1.0)

    sas = candidate.safety.score if candidate.safety else 0.5

    # Dataset seeds should not freeze the search while exploring.
    seed_stagnation = 0.0
    if explore and candidate.generation == 0:
        sources = {est.source for est in candidate.properties.values()}
        if PredictionSource.DATASET in sources:
            seed_stagnation = 0.04

    total = (
        w["mape"] * mape_norm
        + w["sascore"] * sas
        + seed_stagnation
    )
    return ScoreBreakdown(
        mape=mape,
        mape_normalized=mape_norm,
        sascore_penalty=sas,
        minimal_edit_penalty=0.0,
        total=total,
        weights=w,
    )
