"""Rule-based modifier: reagent-level list, gap-direction ranking, seeded sites."""

from __future__ import annotations

import inspect
import random
import re
from pathlib import Path

from backend.app.domain.models import ModificationMode, PROPERTY_KEYS
from backend.app.services import modification as modification_module
from backend.app.services import reagents as reagents_module
from backend.app.services.modification import ModificationMechanism
from backend.app.services.reaction_primitives import list_primitives
from backend.app.services.reagents import (
    EFFECT_SIGN,
    REAGENTS,
    REAGENTS_BY_ID,
    apply_reagent_random_sites,
    desired_property_signs,
    rank_reagents_for_gap,
)

ROOT = Path(__file__).resolve().parents[2]
_TARGET = {"density": 1.8, "det_velocity": 8500.0, "det_pressure": 30.0, "hf_solid": 100.0}


def _parse_reagents_doc() -> dict[str, tuple[str, ...]]:
    text = (ROOT / "docs" / "reagents.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `(\w+)` \| .+? \| (raise|lower|neutral) \| (raise|lower|neutral) \| (raise|lower|neutral) \| (raise|lower|neutral) \|$",
        text,
        re.M,
    )
    return {r[0]: tuple(r[1:]) for r in rows}


def _parse_family_doc() -> dict[str, tuple[int, ...]]:
    text = (ROOT / "docs" / "reaction_families.md").read_text(encoding="utf-8").replace("−", "-")
    rows = re.findall(r"^\| `(\w+)` \| .+ \| ([-+0-9, ]+) \|\s*$", text, re.M)
    return {r: tuple(int(x) for x in v.split(",")) for r, v in rows}


def test_reagents_have_all_four_effects():
    assert REAGENTS
    for r in REAGENTS:
        assert tuple(k for k, _ in r.effects) == PROPERTY_KEYS
        assert all(v in EFFECT_SIGN for _, v in r.effects)


def test_reagent_list_matches_docs_and_primitives():
    prims = {p["family_id"]: tuple(p["delta"]) for p in list_primitives() if p["family_id"] != "identity"}
    assert set(REAGENTS_BY_ID) == set(prims)
    family_doc = _parse_family_doc()
    reagents_doc = _parse_reagents_doc()
    assert set(family_doc) == set(REAGENTS_BY_ID) == set(reagents_doc)
    for rid, r in REAGENTS_BY_ID.items():
        assert r.signs() == family_doc[rid] == prims[rid]
        assert tuple(r.effect(k) for k in PROPERTY_KEYS) == reagents_doc[rid]


def test_desired_signs_follow_gap():
    s = desired_property_signs({**_TARGET, "density": 1.2, "hf_solid": 400.0}, _TARGET)
    assert s == {"density": 1, "det_velocity": 0, "det_pressure": 0, "hf_solid": -1}


def test_rank_prefers_raise_when_below_target():
    ranked = rank_reagents_for_gap({**_TARGET, "density": 1.2}, _TARGET)
    raise_ = [i for i, r in enumerate(ranked) if REAGENTS_BY_ID[r].effect("density") == "raise"]
    lower = [i for i, r in enumerate(ranked) if REAGENTS_BY_ID[r].effect("density") == "lower"]
    assert max(raise_) < min(lower)


def test_rank_prefers_lower_when_above_target():
    ranked = rank_reagents_for_gap({**_TARGET, "det_pressure": 60.0}, _TARGET)
    assert REAGENTS_BY_ID[ranked[0]].effect("det_pressure") == "lower"
    lower = [i for i, r in enumerate(ranked) if REAGENTS_BY_ID[r].effect("det_pressure") == "lower"]
    raise_ = [i for i, r in enumerate(ranked) if REAGENTS_BY_ID[r].effect("det_pressure") == "raise"]
    assert max(lower) < min(raise_)


def test_rank_skip_and_candidates():
    ranked = rank_reagents_for_gap(
        {**_TARGET, "density": 1.2},
        _TARGET,
        skip={"nitrate_arom_CH"},
        candidates=["nitrate_arom_CH", "remove_nitro_arom", "iodo_arom_CH"],
    )
    assert ranked == ["iodo_arom_CH", "remove_nitro_arom"]


def test_seeded_site_selection_is_deterministic():
    parent = "Cc1ccccc1"
    a = apply_reagent_random_sites(parent, "nitrate_arom_CH", random.Random(7))
    b = apply_reagent_random_sites(parent, "nitrate_arom_CH", random.Random(7))
    assert len(a) >= 3  # several distinct aromatic sites
    assert a == b
    orders = {tuple(apply_reagent_random_sites(parent, "nitrate_arom_CH", random.Random(s))) for s in range(10)}
    assert len(orders) > 1  # site choice actually varies with the seed


def test_expand_rule_based_is_seeded_and_gap_directed():
    props = {**_TARGET, "density": 1.2}
    runs = []
    for _ in range(2):
        mech = ModificationMechanism(checkpoint_dir=None, rng=random.Random(123))
        runs.append(mech.expand("Cc1ccccc1", props, _TARGET, n=4, mode=ModificationMode.RULE_BASED))
    assert [s for s, _ in runs[0]] == [s for s, _ in runs[1]]
    assert runs[0]
    for _, rec in runs[0]:
        assert rec.planner == "rule_based"
        assert rec.reaction_family == rec.reagent  # reagent id, not a family
        assert REAGENTS_BY_ID[rec.reagent].effect("density") == "raise"


def test_no_family_logic_in_rule_based_path(monkeypatch):
    src = inspect.getsource(reagents_module)
    for word in ("_REACTION_GROUP", "group", "consensus", "unknown"):
        assert word not in src.replace('"""', "").split("from __future__")[1], word

    def boom(*a, **k):
        raise AssertionError("family lookup used in rule-based path")

    for name in ("matching_group_ids", "reagent_id_for_smarts"):
        monkeypatch.setattr(modification_module, name, boom)
    monkeypatch.setattr(ModificationMechanism, "_apply_family", boom)
    monkeypatch.setattr(ModificationMechanism, "_ordered_variants", boom)
    mech = ModificationMechanism(checkpoint_dir=None, rng=random.Random(1))
    mech._prims = {}  # no family table available
    out = mech.expand("Cc1ccccc1", {**_TARGET, "density": 1.2}, _TARGET, n=3, mode=ModificationMode.RULE_BASED)
    assert out


def _seed_smiles() -> dict[str, str]:
    import csv
    path = ROOT / "data" / "seeds" / "energetic_seed_library.csv"
    with path.open(encoding="utf-8") as fh:
        return {row["name"]: row["SMILES"] for row in csv.DictReader(fh)}


def test_rule_based_yields_candidates_for_cages_and_saturated_nitramines():
    seeds = _seed_smiles()
    scaffolds = ["RDX", "HMX", "CL-20", "TEX", "Octanitrocubane"]
    for name in scaffolds:
        smiles = seeds[name]
        found = False
        for reagent in REAGENTS:
            products = apply_reagent_random_sites(smiles, reagent.reagent_id, random.Random(0))
            if products:
                found = True
                break
        assert found, f"rule-based modifier produced no candidate for {name}"


def test_new_cage_reagents_match_expected_scaffolds():
    seeds = _seed_smiles()
    expected = {
        "nitramine_to_nitrosamine": ["RDX", "HMX", "CL-20", "TEX"],
        "aliphatic_nitro_to_amino": ["Octanitrocubane"],
    }
    for reagent_id, names in expected.items():
        assert reagent_id in REAGENTS_BY_ID
        for name in names:
            products = apply_reagent_random_sites(seeds[name], reagent_id, random.Random(0))
            assert products, f"{reagent_id} produced no candidate for {name}"
