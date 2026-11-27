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


def child_is_closer(
    parent: Dict[str, float],
    child: Dict[str, float],
    target: Dict[str, float],
) -> bool:
    """True when the child has a lower MAPE to ``target`` than the parent."""
    from backend.app.services.scoring import calculate_mape

    return calculate_mape(child, target) < calculate_mape(parent, target)


def sample_gap_reducing_target(
    parent: Dict[str, float],
    child: Dict[str, float],
    rng=None,
    *,
    jitter_prob: float = 0.35,
    attempts: int = 12,
) -> Dict[str, float]:
    """A target the recorded child is closer to than the parent.

    The exact child is always accepted: the recorded reaction closes that gap.
    A jittered target is kept only when the same reaction still decreases MAPE.
    """
    import random

    r = rng or random.Random()
    exact = {k: float(child[k]) for k in PROPERTY_KEYS}
    if r.random() >= jitter_prob:
        return exact
    for _ in range(attempts):
        target: Dict[str, float] = {}
        for k in PROPERTY_KEYS:
            delta = float(child[k]) - float(parent[k])
            alpha = r.uniform(0.55, 1.45)
            noise = r.gauss(0.0, 0.05 * (abs(float(parent[k])) + abs(delta) + 1e-6))
            target[k] = float(parent[k]) + alpha * delta + noise
        if child_is_closer(parent, child, target):
            return target
    return exact


def training_planner_input(
    parent: Dict[str, float],
    child: Dict[str, float],
    *,
    rng=None,
    jitter_prob: float = 0.35,
) -> List[float]:
    """Inference-style gap for a target the recorded reaction moves closer.

    The encoded vector is target − parent, the same encoding search uses.
    The target is the recorded child, or a nearby point the child reaches
    with a lower MAPE than the parent.
    """
    target = sample_gap_reducing_target(parent, child, rng, jitter_prob=jitter_prob)
    return planner_input_from_parent_target(parent, target)


def mape_rows(props, target):
    """Row-wise MAPE, matching ``calculate_mape`` with equal property weights."""
    import numpy as np

    props = np.asarray(props, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = np.abs(props - target)
    small = np.abs(target) <= 1e-10
    denom = np.maximum(np.abs(target), 1e-10)
    err = np.where(small, diff * 100.0, diff / denom * 100.0)
    return err.mean(axis=1)


def sample_gap_reducing_targets(parent, child, rng, *, jitter_prob: float = 0.5, attempts: int = 8):
    """``(N, 4)`` targets. Each row is closer to ``child`` than to ``parent`` in MAPE.

    Rows that draw no jitter, and jittered rows that never decrease the gap,
    use the recorded child. A reaction that does not change the properties
    keeps that child, which leaves a zero gap.
    """
    import numpy as np

    parent = np.asarray(parent, dtype=np.float64)
    child = np.asarray(child, dtype=np.float64)
    target = child.copy()
    jitter = rng.random(len(parent)) < jitter_prob
    delta = child - parent
    scale = 0.05 * (np.abs(parent) + np.abs(delta) + 1e-6)
    for _ in range(attempts):
        if not np.any(jitter):
            break
        alpha = rng.uniform(0.55, 1.45, size=(int(jitter.sum()), parent.shape[1]))
        noise = rng.normal(0.0, scale[jitter])
        trial = parent[jitter] + alpha * delta[jitter] + noise
        ok = mape_rows(child[jitter], trial) < mape_rows(parent[jitter], trial)
        rows = np.flatnonzero(jitter)
        target[rows[ok]] = trial[ok]
        jitter[rows[ok]] = False
    return target


def encode_property_gaps(gaps):
    """Vectorized ``gap_to_planner_input`` for an ``(N, 4)`` gap."""
    import numpy as np

    gaps = np.asarray(gaps, dtype=np.float64)
    out = np.empty_like(gaps, dtype=np.float32)
    for i, key in enumerate(PROPERTY_KEYS):
        scale = PROPERTY_SCALES[key]
        deadband = max(abs(scale * RELATIVE_THRESHOLDS[key]), 1e-6)
        g = gaps[:, i]
        v = np.clip(g / scale, -1.0, 1.0).astype(np.float32)
        v[np.abs(g) <= deadband] = 0.0
        out[:, i] = v
    return out


def row_parent_child_props(row) -> tuple[Dict[str, float], Dict[str, float]]:
    parent = {k: float(row[f"parent_{k}"]) for k in PROPERTY_KEYS}
    child = {k: float(row[f"child_{k}"]) for k in PROPERTY_KEYS}
    return parent, child
