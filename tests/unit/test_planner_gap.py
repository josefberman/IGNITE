"""Unit tests for planner gap encoding and training helpers."""

import numpy as np

from ml.planner_gap import (
    child_is_closer,
    encode_property_gaps,
    gap_to_planner_input,
    mape_rows,
    planner_input_from_parent_target,
    sample_gap_reducing_target,
    sample_gap_reducing_targets,
    training_planner_input,
)


def test_gap_to_planner_input_deadband():
    assert gap_to_planner_input([0.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 0.0]
    out = gap_to_planner_input([0.5, 500.0, 5.0, 50.0])
    assert all(-1.0 <= x <= 1.0 for x in out)
    assert out[0] > 0


def test_training_planner_input_jitter():
    import random

    parent = {"density": 1.5, "det_velocity": 7000.0, "det_pressure": 20.0, "hf_solid": 50.0}
    child = {"density": 1.8, "det_velocity": 8500.0, "det_pressure": 30.0, "hf_solid": 100.0}
    exact = planner_input_from_parent_target(parent, child)
    no_jitter = training_planner_input(parent, child, rng=random.Random(1), jitter_prob=0.0)
    assert no_jitter == exact
    jittered = [
        training_planner_input(parent, child, rng=random.Random(i), jitter_prob=1.0) for i in range(10)
    ]
    assert any(j != exact for j in jittered)


def test_sampled_target_is_closer_than_the_parent():
    import random

    parent = {"density": 1.5, "det_velocity": 7000.0, "det_pressure": 20.0, "hf_solid": 50.0}
    child = {"density": 1.8, "det_velocity": 8500.0, "det_pressure": 30.0, "hf_solid": 100.0}
    for i in range(30):
        target = sample_gap_reducing_target(parent, child, random.Random(i), jitter_prob=1.0)
        assert child_is_closer(parent, child, target)


def test_batch_targets_match_mape_and_encoding():
    parent = np.array([[1.5, 7000.0, 20.0, 50.0]] * 16)
    child = np.array([[1.8, 8500.0, 30.0, 100.0]] * 16)
    rng = np.random.default_rng(0)
    target = sample_gap_reducing_targets(parent, child, rng, jitter_prob=1.0)
    assert np.all(mape_rows(child, target) < mape_rows(parent, target))
    encoded = encode_property_gaps(target - parent)
    for row, gap in enumerate(target - parent):
        assert np.allclose(encoded[row], gap_to_planner_input(gap))
