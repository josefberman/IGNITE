"""Shared property-gap encoding for ModificationPlanner (train + inference)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

from backend.app.domain.models import PROPERTY_KEYS

# Match train_planner delta scaling / typical property magnitudes
PROPERTY_SCALES = {
    "density": 1.0,
    "det_velocity": 1000.0,
    "det_pressure": 10.0,
    "hf_solid": 100.0,
}

# Relative threshold for "no change" on each axis (same spirit as strategy pool ±0.01 absolute)
RELATIVE_THRESHOLDS = {
    "density": 0.01,
    "det_velocity": 0.01,
    "det_pressure": 0.01,
    "hf_solid": 0.01,
}


def gap_dict_from_values(
    parent: Dict[str, float],
    target: Dict[str, float],
) -> Dict[str, float]:
    return {k: float(target[k] - parent.get(k, target[k])) for k in PROPERTY_KEYS}


def gap_vector_from_values(
    parent: Dict[str, float],
    target: Dict[str, float],
) -> List[float]:
    return [float(target[k] - parent.get(k, target[k])) for k in PROPERTY_KEYS]


def quantize_direction(gap: Sequence[float], *, absolute_threshold: float = 0.01) -> List[float]:
    """Per-axis sign in {-1, 0, +1} using absolute threshold (legacy strategy-pool style)."""
    out: List[float] = []
    for g in gap:
        if g > absolute_threshold:
            out.append(1.0)
        elif g < -absolute_threshold:
            out.append(-1.0)
        else:
            out.append(0.0)
    return out


def gap_to_planner_input(gap: Sequence[float]) -> List[float]:
    """
    4-d continuous planner input in roughly [-1, 1]:
    signed normalized gap with per-property scale + deadband.
    Used at inference (user target − parent) and during training (same encoding).
    """
    out: List[float] = []
    for key, g in zip(PROPERTY_KEYS, gap):
        scale = PROPERTY_SCALES[key]
        rel = RELATIVE_THRESHOLDS[key]
        deadband = max(abs(scale * rel), 1e-6)
        if abs(g) <= deadband:
            out.append(0.0)
        else:
            out.append(max(-1.0, min(1.0, float(g) / scale)))
    return out


def planner_input_from_parent_target(
    parent: Dict[str, float],
    target: Dict[str, float],
) -> List[float]:
    return gap_to_planner_input(gap_vector_from_values(parent, target))


def training_planner_input(
    parent: Dict[str, float],
    child: Dict[str, float],
    *,
    rng=None,
    jitter_prob: float = 0.35,
) -> List[float]:
    """
    Simulate inference-time user targets from weak-supervision pairs.

    Usually the user target is unknown; we approximate it from child properties with
    optional jitter so the planner does not memorize exact outcome deltas.
    """
    import random

    r = rng or random.Random()
    if r.random() < jitter_prob:
        target = {}
        for k in PROPERTY_KEYS:
            delta = child[k] - parent[k]
            alpha = r.uniform(0.55, 1.45)
            noise = r.gauss(0.0, 0.05 * (abs(parent[k]) + abs(delta) + 1e-6))
            target[k] = parent[k] + alpha * delta + noise
    else:
        target = {k: child[k] for k in PROPERTY_KEYS}
    return planner_input_from_parent_target(parent, target)


def row_parent_child_props(row) -> tuple[Dict[str, float], Dict[str, float]]:
    parent = {k: float(row[f"parent_{k}"]) for k in PROPERTY_KEYS}
    child = {k: float(row[f"child_{k}"]) for k in PROPERTY_KEYS}
    return parent, child
