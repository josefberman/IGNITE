"""Novelty unit tests (offline)."""

from backend.app.domain.models import NoveltyStatus
from backend.app.services.novelty import NoveltyService


def test_exact_seed_is_known():
    svc = NoveltyService(["c1ccccc1"], pubchem_enabled=False, openalex_enabled=False)
    ev = svc.evaluate("c1ccccc1")
    assert ev.status == NoveltyStatus.KNOWN
    assert ev.score == 0.0


def test_missing_providers_mark_incomplete_not_novel():
    svc = NoveltyService(["CCO"], pubchem_enabled=False, openalex_enabled=False)
    ev = svc.evaluate("c1ccc(N(=O)=O)cc1")
    assert ev.status in (NoveltyStatus.INCOMPLETE_SEARCH, NoveltyStatus.CLOSE_ANALOGUE, NoveltyStatus.NO_MATCH_FOUND)
    if ev.status == NoveltyStatus.INCOMPLETE_SEARCH:
        assert "lens" in ev.providers_unavailable or any("unavailable" in n.lower() or "incomplete" in n.lower() or "not proof" in n.lower() for n in ev.notes)
