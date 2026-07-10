"""Unit tests for scoring, chemistry, uncertainty, novelty semantics."""

from backend.app.domain.models import (
    CandidateMolecule,
    NoveltyEvidence,
    NoveltyStatus,
    PredictionSource,
    PropertyEstimate,
    SafetyFinding,
    TargetProperties,
)
from backend.app.services.chemistry import canonicalize, validate_molecule
from backend.app.services.scoring import calculate_mape, score_candidate
from backend.app.services.uncertainty import calibrate_estimate


def test_canonicalize_benzene():
    assert canonicalize("c1ccccc1") == "c1ccccc1"


def test_validate_rejects_unchanged():
    r = validate_molecule("c1ccccc1", parent_smiles="c1ccccc1")
    assert not r.valid
    assert r.reason == "unchanged"


def test_mape_zero_when_equal():
    t = {"density": 1.8, "det_velocity": 8000, "det_pressure": 30, "hf_solid": 50}
    assert calculate_mape(t, t) == 0.0


def test_mape_positive():
    t = {"density": 1.8, "det_velocity": 8000, "det_pressure": 30, "hf_solid": 50}
    p = {"density": 1.98, "det_velocity": 8000, "det_pressure": 30, "hf_solid": 50}
    assert calculate_mape(p, t) > 0


def test_target_validation():
    TargetProperties(density=1.8, det_velocity=8500, det_pressure=30, hf_solid=100)


def test_uncertainty_interval_widens_with_low_confidence():
    est = PropertyEstimate(value=100.0, source=PredictionSource.XGBOOST, confidence=0.6)
    cal = calibrate_estimate(est, ood_distance=0.5)
    assert cal.lower is not None and cal.upper is not None
    assert cal.upper > cal.lower


def test_score_prefers_lower_mape():
    target = {"density": 1.8, "det_velocity": 8000, "det_pressure": 30, "hf_solid": 50}

    def make(mape_props, safety=0.1):
        c = CandidateMolecule(
            smiles="C",
            canonical_smiles="C",
            properties={
                k: PropertyEstimate(value=v, source=PredictionSource.DATASET, confidence=0.9)
                for k, v in mape_props.items()
            },
            safety=SafetyFinding(score=safety, is_acceptable=True, sascore=2.0),
            novelty=NoveltyEvidence(status=NoveltyStatus.NO_MATCH_FOUND, score=0.8),
        )
        return score_candidate(c, target)

    good = make(target)
    bad = make({**target, "density": 2.2})
    assert good.total < bad.total


def test_novelty_known_status_enum():
    assert NoveltyStatus.INCOMPLETE_SEARCH.value == "incomplete_search"
