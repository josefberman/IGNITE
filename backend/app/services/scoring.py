"""MAPE and multi-objective ranking."""

from __future__ import annotations

from typing import Dict, Optional

from backend.app.domain.models import (
    PROPERTY_KEYS,
    CandidateMolecule,
    NoveltyStatus,
    ScoreBreakdown,
)


DEFAULT_WEIGHTS = {
    "mape": 0.45,
    "sascore": 0.15,
    "uncertainty": 0.20,
    "safety": 0.15,
    "minimal_edit": 0.05,
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


def mean_uncertainty(candidate: CandidateMolecule) -> float:
    if not candidate.properties:
        return 1.0
    vals = []
    for est in candidate.properties.values():
        # confidence 1 → uncertainty 0
        vals.append(1.0 - float(est.confidence))
        if est.lower is not None and est.upper is not None and abs(est.value) > 1e-8:
            width = (est.upper - est.lower) / (2.0 * abs(est.value))
            vals.append(min(1.0, max(0.0, width)))
    return sum(vals) / len(vals) if vals else 1.0


def score_candidate(
    candidate: CandidateMolecule,
    target: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    max_edit: int = 10,
) -> ScoreBreakdown:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    props = candidate.property_values()
    mape = calculate_mape(props, target)
    mape_norm = min(mape / 100.0, 1.0)

    sas = candidate.safety.score if candidate.safety else 0.5
    unc = mean_uncertainty(candidate)
    safety = candidate.safety.score if candidate.safety else 0.5
    edit = 0.0
    if candidate.modification:
        edit = min(1.0, candidate.modification.edit_distance / max(1, max_edit))

    novelty_bonus = 0.0
    if candidate.novelty:
        if candidate.novelty.status == NoveltyStatus.NO_MATCH_FOUND:
            novelty_bonus = 0.05 * candidate.novelty.score
        elif candidate.novelty.status == NoveltyStatus.KNOWN:
            novelty_bonus = -0.02

    total = (
        w["mape"] * mape_norm
        + w["sascore"] * sas
        + w["uncertainty"] * unc
        + w["safety"] * safety
        + w["minimal_edit"] * edit
        - novelty_bonus
    )
    return ScoreBreakdown(
        mape=mape,
        mape_normalized=mape_norm,
        sascore_penalty=sas,
        uncertainty_penalty=unc,
        safety_penalty=safety,
        minimal_edit_penalty=edit,
        novelty_bonus=novelty_bonus,
        total=total,
        weights=w,
    )
