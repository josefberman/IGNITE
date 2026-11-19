"""Multitask planner: architecture, legacy equivalence, batched trainer, split, backend loading."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from backend.app.services.modification import LearnedPlannerBundle, ModificationMechanism
from backend.app.domain.models import ModificationMode
from backend.app.services.reaction_primitives import list_planner_vocab
from ml.models.architectures import (
    ModificationPlanner,
    MultitaskPlanner,
    build_planner,
    legacy_to_multitask_state_dict,
    mol_to_graph,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "modification_pairs_tiny.csv"
SMILES = "Cc1ccc(-c2noc3nc([N+](=O)[O-])cn23)cc1"


def _vocab():
    families, reagent_rows = list_planner_vocab()
    return families, [r["reagent_id"] for r in reagent_rows]


def _inputs():
    x, edge = mol_to_graph(SMILES)
    return x.unsqueeze(0), torch.tensor([[0.1, -0.3, 0.2, 0.0]]), edge


@pytest.mark.parametrize("site_reagent", [True, False])
def test_baseline_config_reproduces_legacy_planner(site_reagent):
    torch.manual_seed(0)
    nf, nr = 11, 65
    legacy = ModificationPlanner(nf, nr, layers=2, site_reagent=site_reagent).eval()
    mt = MultitaskPlanner(nf, nr, shared_mp=2, site_reagent=site_reagent).eval()
    mt.load_state_dict(legacy_to_multitask_state_dict(legacy.state_dict()))
    x, gap, edge = _inputs()
    z1, _, a1 = legacy.encode(x, gap, edge)
    z2, _, a2 = mt.encode(x, gap, edge)
    fam, reag = torch.tensor([3]), torch.tensor([7])
    pairs = [
        (legacy.forward_family(z1, gap), mt.forward_family(z2, gap)),
        (legacy.forward_reagent(z1, family_idx=fam, gap=gap, atom_emb=a1),
         mt.forward_reagent(z2, family_idx=fam, gap=gap, atom_emb=a2)),
        (legacy.forward_reagent(z1, gap=gap, atom_emb=a1), mt.forward_reagent(z2, gap=gap, atom_emb=a2)),
        (legacy.forward_sites(a1, z1, gap, reagent_idx=reag), mt.forward_sites(a2, z2, gap, reagent_idx=reag)),
        (legacy.forward_sites(a1, z1, gap), mt.forward_sites(a2, z2, gap)),
    ]
    for a, b in pairs:
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(shared_mp=3),
        dict(shared_mp=2, shared_fc=2, family_tower_layers=2, reagent_tower_layers=2, site_tower_layers=2),
        dict(shared_mp=1, branch_mp=2, tower_hidden=64),
        dict(shared_mp=0, branch_mp=1, family_tower_layers=0),
    ],
)
def test_fork_configs_shapes_and_meta_roundtrip(cfg):
    nf, nr = 11, 65
    model = MultitaskPlanner(nf, nr, **cfg).eval()
    x, gap, edge = _inputs()
    z, _, atom_emb = model.encode(x, gap, edge)
    n = x.size(1)
    assert z.shape == (1, 3 * model.hidden)
    assert atom_emb.shape == (1, n, 6 * model.hidden)
    assert model.forward_family(z, gap).shape == (1, nf)
    assert model.forward_reagent(z, family_idx=torch.tensor([0]), gap=gap, atom_emb=atom_emb).shape == (1, nr)
    assert model.forward_sites(atom_emb, z, gap, reagent_idx=torch.tensor([1])).shape == (n,)
    rebuilt = build_planner(model.arch_meta(), nf, nr)
    assert isinstance(rebuilt, MultitaskPlanner)
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    z_r, _, _ = rebuilt.encode(x, gap, edge)
    assert torch.allclose(rebuilt.forward_family(z_r, gap), model.forward_family(z, gap))


def test_branches_have_private_parameters():
    model = MultitaskPlanner(11, 65, shared_mp=1, branch_mp=1)
    x, gap, edge = _inputs()
    z, _, _ = model.encode(x, gap, edge)
    model.forward_family(z, gap).sum().backward()
    def touched(p):
        return p.grad is not None and bool(p.grad.abs().sum() > 0)

    assert touched(model.trunk_mp[0].self_lin.weight)
    assert touched(model.branch_layers["family"][0].self_lin.weight)
    assert not touched(model.branch_layers["site"][0].self_lin.weight)
    assert not touched(model.branch_layers["reagent"][0].self_lin.weight)


def test_invalid_fork_configs_raise():
    with pytest.raises(ValueError):
        MultitaskPlanner(11, 65, shared_mp=1, branch_mp=1, shared_fc=1)
    with pytest.raises(ValueError):
        MultitaskPlanner(11, 65, shared_mp=0, branch_mp=0)


def test_build_planner_legacy_defaults():
    model = build_planner({}, 11, 65)
    assert isinstance(model, ModificationPlanner)
    assert model.family_layers == 0 and model.bond_types == 0 and model.site_reagent is False


def _write_mt_checkpoint(path: Path, **cfg) -> None:
    path.mkdir(parents=True, exist_ok=True)
    families, reagents = _vocab()
    planner = MultitaskPlanner(len(families), len(reagents), **cfg)
    torch.save(
        {"planner": planner.state_dict(), "meta": {"families": families, "reagents": reagents, **planner.arch_meta()}},
        path / "planner.pt",
    )


def test_backend_loads_multitask_checkpoint(tmp_path: Path):
    _write_mt_checkpoint(tmp_path, shared_mp=1, branch_mp=1)
    bundle = LearnedPlannerBundle(tmp_path)
    assert bundle.available, bundle.unavailable_reason
    assert isinstance(bundle._model, MultitaskPlanner)
    assert bundle.ranks_reagent_sites
    gap = [0.1, 100.0, 1.0, 10.0]
    fams = bundle.propose_families("c1ccccc1N(=O)=O", gap, k=3)
    assert len(fams) == 3
    families, sites, per_fam = bundle.plan("c1ccccc1N(=O)=O", gap, k_families=3, k_reagents=2)
    assert families and sites and all(f in per_fam for f in families)
    mech = ModificationMechanism(checkpoint_dir=tmp_path)
    props = {"density": 1.5, "det_velocity": 7000.0, "det_pressure": 20.0, "hf_solid": 50.0}
    targets = {"density": 1.8, "det_velocity": 8500.0, "det_pressure": 30.0, "hf_solid": 100.0}
    out = mech.expand("c1ccccc1N(=O)=O", props, targets, n=3, mode=ModificationMode.PLANNER)
    assert len(out) >= 1


def test_random_group_split_fractions_and_no_parent_leak():
    from ml.training.pair_cache import random_group_split

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(2000), rng.integers(1, 51, 2000))
    codes = random_group_split(groups, (0.7, 0.1, 0.2), seed=42)
    frac = np.bincount(codes, minlength=3) / len(codes)
    assert np.allclose(frac, [0.7, 0.1, 0.2], atol=0.01)
    for g in np.unique(groups)[:200]:
        assert len(set(codes[groups == g].tolist())) == 1
    assert np.array_equal(codes, random_group_split(groups, (0.7, 0.1, 0.2), seed=42))
    assert not np.array_equal(codes, random_group_split(groups, (0.7, 0.1, 0.2), seed=1))


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    from ml.training.pair_cache import PairCache, build_pair_cache

    path = tmp_path_factory.mktemp("cache") / "tiny.npz"
    build_pair_cache(FIXTURE, path, workers=1)
    return PairCache(path)


def test_batched_losses_match_legacy_per_pair_loop(tiny_cache):
    """Batched family / reagent / site losses equal the legacy per-pair losses for the same weights."""
    import pandas as pd
    from rdkit import RDLogger

    from backend.app.services.reaction_primitives import matching_group_ids, matching_reaction_ids
    from ml.training.train_multitask import _forward
    from ml.training.train_planner import (
        PairDataset,
        _candidate_site_loss,
        _mask_class_logits,
        _site_loss,
    )

    RDLogger.DisableLog("rdApp.*")
    families, reagent_rows = list_planner_vocab()
    reagents = [r["reagent_id"] for r in reagent_rows]
    f2i = {f: i for i, f in enumerate(families)}
    r2i = {r: i for i, r in enumerate(reagents)}
    f2r = {f: [] for f in families}
    for r in reagent_rows:
        f2r[r["family_id"]].append(r2i[r["reagent_id"]])
    df = pd.read_csv(FIXTURE).assign(split="train")
    ds = PairDataset(df, f2i, r2i, "train", rng=random.Random(0), jitter_prob=0.0)

    def allowed_fam(s, g):
        a = [f2i[f] for f in matching_group_ids(s) if f in f2i]
        return a + ([g] if a and g not in a else [])

    def allowed_reag(s, f, g):
        pool = f2r[families[f]]
        m = {r2i[r] for r in matching_reaction_ids(s) if r in r2i}
        a = [i for i in pool if i in m] if m else list(pool)
        a = a or list(pool)
        return a + ([g] if g not in a else [])

    torch.manual_seed(0)
    legacy = ModificationPlanner(len(families), len(reagents), layers=2).eval()
    mt = MultitaskPlanner(len(families), len(reagents), shared_mp=2).eval()
    mt.load_state_dict(legacy_to_multitask_state_dict(legacy.state_dict()))
    idx = np.arange(tiny_cache.n_pairs)
    with torch.no_grad():
        out = _forward(mt, tiny_cache.batch(idx), torch.from_numpy(tiny_cache.fam_pool), eval_mode=True)
        for j, i in enumerate(idx):
            px, pe, _, _, fam, reag, _, gap, st, ps, cands, pos = ds[int(tiny_cache.pair_row[i])]
            g = gap.unsqueeze(0)
            z, _, a = legacy.encode(px.unsqueeze(0), g, pe)
            fl = _mask_class_logits(legacy.forward_family(z, g), allowed_fam(ps, fam))
            rl = _mask_class_logits(
                legacy.forward_reagent(z, family_idx=torch.tensor([fam]), gap=g, atom_emb=a),
                allowed_reag(ps, fam, reag),
            )
            sl = legacy.forward_sites(a, z, g, reagent_idx=torch.tensor([reag]))
            ls = _candidate_site_loss(sl, cands, pos)
            if ls is None:
                ls = _site_loss(sl, st)
            assert abs(float(F.cross_entropy(fl, torch.tensor([fam]))) - float(out["l_fam"][j])) < 1e-4
            assert abs(float(F.cross_entropy(rl, torch.tensor([reag]))) - float(out["l_reag"][j])) < 1e-4
            assert abs(float(ls) - float(out["l_site"][j])) < 1e-4


def test_tie_rank_matches_first_index_argmax():
    from ml.training.train_multitask import _tie_rank

    scores = torch.tensor([1.0, 2.0, 2.0, 0.5, 3.0, 3.0])
    pos = torch.tensor([False, False, True, True, True, False])
    seg = torch.tensor([0, 0, 0, 1, 1, 1])
    # segment 0: argmax picks index 1 (negative) -> rank 1; segment 1: index 4 positive -> rank 0
    assert _tie_rank(scores, pos, seg, 2).tolist() == [1, 0]


def test_train_multitask_smoke_writes_loadable_checkpoint(tmp_path: Path, tiny_cache):
    from ml.training.train_multitask import train_multitask

    out = tmp_path / "ckpt"
    card = train_multitask(
        FIXTURE, out, epochs=2, patience=2, batch_size=8, hidden=32, dropout=0.1, bond_types=5,
        site_reagent=True, shared_mp=1, branch_mp=1, shared_fc=0, tower_hidden=32,
        family_tower_layers=1, reagent_tower_layers=1, site_tower_layers=1,
        split="random_pair", eval_test=True, cache_path=tmp_path / "c.npz", cache_workers=1, threads=1,
    )
    assert card["architecture"] == "multitask" and card["mt_branch_mp"] == 1
    assert card["test"] is not None and 0.0 <= card["test"]["family_top1"] <= 1.0
    blob = torch.load(out / "planner.pt", map_location="cpu", weights_only=False)
    assert blob["meta"]["split"]["mode"] == "random_pair"
    bundle = LearnedPlannerBundle(out)
    assert bundle.available and isinstance(bundle._model, MultitaskPlanner)


def test_unconditioned_heads_ignore_family_and_reagent():
    model = MultitaskPlanner(11, 65, shared_mp=1, conditioning="none").eval()
    assert model.site_reagent is False and model.site_reagent_emb is None
    x, gap, edge = _inputs()
    z, _, atom_emb = model.encode(x, gap, edge)
    r0 = model.forward_reagent(z, family_idx=torch.tensor([0]), gap=gap, atom_emb=atom_emb)
    r1 = model.forward_reagent(z, family_idx=torch.tensor([4]), gap=gap, atom_emb=atom_emb)
    r2 = model.forward_reagent(z, gap=gap, atom_emb=atom_emb)
    assert torch.equal(r0, r1) and torch.equal(r0, r2)
    s0 = model.forward_sites(atom_emb, z, gap, reagent_idx=torch.tensor([1]))
    s1 = model.forward_sites(atom_emb, z, gap, reagent_idx=torch.tensor([9]))
    assert torch.equal(s0, s1)
    rebuilt = build_planner(model.arch_meta(), 11, 65)
    assert rebuilt.conditioning == "none"
    rebuilt.load_state_dict(model.state_dict())
    with pytest.raises(ValueError):
        MultitaskPlanner(11, 65, conditioning="bogus")


def test_backend_loads_unconditioned_checkpoint(tmp_path: Path):
    _write_mt_checkpoint(tmp_path, shared_mp=1, conditioning="none")
    bundle = LearnedPlannerBundle(tmp_path)
    assert bundle.available, bundle.unavailable_reason
    assert not bundle.ranks_reagent_sites  # reagent-blind site head -> backend uses site proposals
    gap = [0.1, 100.0, 1.0, 10.0]
    families, sites, per_fam = bundle.plan("c1ccccc1N(=O)=O", gap, k_families=3, k_reagents=2)
    assert families and sites and all(f in per_fam for f in families)
    for fam, reags in per_fam.items():
        assert all(r in bundle.family_to_reagent_ids[fam] for r in reags)
