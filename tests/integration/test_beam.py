"""Integration tests for seed selection and bounded beam search."""

from pathlib import Path

import pytest

from backend.app.domain.models import BeamParameters, TargetProperties
from backend.app.services.beam_search import BeamSearchController
from backend.app.services.modification import ModificationMechanism
from backend.app.services.novelty import NoveltyService
from backend.app.services.prediction import PropertyPredictor
from backend.app.services.property_estimator import LiteratureRetriever, PropertyEstimator
from backend.app.services.seed_selector import SeedLibrary

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def seeds():
    return SeedLibrary(ROOT / "sample_start_molecules.csv")


@pytest.fixture(scope="module")
def controller(seeds):
    predictor = PropertyPredictor(ROOT / "models")
    estimator = PropertyEstimator(predictor, LiteratureRetriever(enabled=False))
    # Disable external providers for deterministic offline tests
    novelty = NoveltyService(
        seed_smiles=seeds.all_canonical_smiles()[:500],
        pubchem_enabled=False,
        openalex_enabled=False,
    )
    return BeamSearchController(seeds, estimator, novelty, ModificationMechanism())


def test_seed_selection_returns_candidates(seeds):
    targets = TargetProperties(density=1.8, det_velocity=8500, det_pressure=32, hf_solid=50)
    selected = seeds.select_seeds(targets, mape_threshold=50.0, max_seeds=3)
    assert 1 <= len(selected) <= 3
    assert selected[0].mape <= selected[-1].mape


def test_beam_search_bounded(controller):
    targets = TargetProperties(density=1.7, det_velocity=7500, det_pressure=25, hf_solid=20)
    beam = BeamParameters(depth=1, width=2, top_k=2, mape_threshold=1.0, random_seed=0,
                          enable_literature=False, enable_patent_search=False)
    best, top, progress = controller.run("test-job", targets, beam)
    assert best.canonical_smiles
    assert best.score is not None
    assert len(top) >= 1
    assert progress.status.value in ("completed", "cancelled")
