"""Unit tests for learned modification planner wiring."""

from __future__ import annotations

from pathlib import Path

import torch

from backend.app.services.modification import LearnedPlannerBundle, ModificationMechanism
from backend.app.services.reaction_primitives import list_planner_vocab, list_primitives, list_reagents
from ml.models.architectures import DeltaPropertyPredictor, ModificationPlanner, mol_to_graph


def _write_dummy_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    families, reagent_rows = list_planner_vocab()
    reagents = [r["reagent_id"] for r in reagent_rows]
    planner = ModificationPlanner(n_families=len(families), n_reagents=len(reagents))
    delta = DeltaPropertyPredictor()
    torch.save(
        {
            "planner": planner.state_dict(),
            "delta": delta.state_dict(),
            "meta": {"families": families, "reagents": reagents},
        },
        path / "planner.pt",
    )


def test_mol_to_graph_feature_dim():
    from ml.models.architectures import ATOM_FEAT_DIM

    g = mol_to_graph("c1ccccc1")
    assert g is not None
    x, _ = g
    assert x.shape[1] == ATOM_FEAT_DIM


def test_list_reagents_stable_ids():
    reagents = list_reagents()
    assert len(reagents) > len([p for p in list_primitives() if p["family_id"] != "identity"])
    ids = [r["reagent_id"] for r in reagents]
    assert len(ids) == len(set(ids))
    assert all("__v" in rid for rid in ids)


def test_propose_families_and_reagents(tmp_path: Path):
    _write_dummy_checkpoint(tmp_path)
    bundle = LearnedPlannerBundle(tmp_path)
    assert bundle.available
    gap = [0.1, 100.0, 1.0, 10.0]
    fams = bundle.propose_families("c1ccccc1N(=O)=O", gap, k=3)
    assert len(fams) == 3
    reags = bundle.propose_reagents("c1ccccc1N(=O)=O", gap, family_id=fams[0], k=2)
    assert len(reags) >= 1
    assert all(r in bundle.family_to_reagent_ids[fams[0]] for r in reags)
    families, sites, per_fam = bundle.plan("c1ccccc1N(=O)=O", gap, k_families=3, k_reagents=2)
    assert families
    assert sites
    assert all(f in per_fam for f in families)


def test_reacting_atom_indices_nonzero():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ml.data_gen.generate_pairs import reacting_atom_indices

    mol = Chem.MolFromSmiles("c1ccccc1")
    rxn = AllChem.ReactionFromSmarts("[cH:1]>>[c:1]N")
    assert rxn is not None
    sites = reacting_atom_indices(mol, rxn, 0)
    assert len(sites) >= 1
    assert all(0 <= i < mol.GetNumAtoms() for i in sites)


def test_site_loss_trains_site_head():
    from ml.training.train_planner import _site_loss

    families = [p["family_id"] for p in list_primitives() if p["family_id"] != "identity"]
    reagents = [r["reagent_id"] for r in list_reagents()]
    planner = ModificationPlanner(n_families=len(families), n_reagents=len(reagents))
    g = mol_to_graph("c1ccccc1N(=O)=O")
    assert g is not None
    x, edge = g
    target = torch.zeros(x.size(0))
    target[0] = 1.0
    opt = torch.optim.Adam(planner.parameters(), lr=1e-2)
    before = planner.site_head.weight.detach().clone()
    z, _, atom_emb = planner.encode(x.unsqueeze(0), torch.zeros(1, 4), edge)
    logits = planner.forward_sites(atom_emb, z)
    loss = _site_loss(logits, target)
    loss.backward()
    opt.step()
    assert not torch.allclose(before, planner.site_head.weight.detach())


def test_expand_records_reagent_id(tmp_path: Path):
    _write_dummy_checkpoint(tmp_path)
    mech = ModificationMechanism(checkpoint_dir=tmp_path)
    assert mech.learned.available
    parent = "c1ccccc1N(=O)=O"
    props = {
        "density": 1.5,
        "det_velocity": 7000.0,
        "det_pressure": 20.0,
        "hf_solid": 50.0,
    }
    targets = {
        "density": 1.8,
        "det_velocity": 8500.0,
        "det_pressure": 30.0,
        "hf_solid": 100.0,
    }
    out = mech.expand(parent, props, targets, n=3)
    assert len(out) >= 1
    for _, rec in out:
        assert rec.reagent
        known = {p["family_id"] for p in list_primitives() if p["family_id"] != "identity"}
        assert rec.reagent in known
        if rec.planner == "learned":
            assert rec.reaction_family in {p["group"] for p in list_primitives()}


def test_eval_inference_metrics_oracle_vs_predicted_family():
    from ml.training.train_planner import _eval_inference_metrics

    families = ["fam_a", "fam_b"]
    family_to_reagent_idxs = {
        "fam_a": [0, 1],
        "fam_b": [2, 3],
    }
    fam_logits = torch.tensor([[0.1, 2.0]])  # predict fam_b
    site_target = torch.zeros(4)
    site_target[1] = 1.0
    site_logits = torch.tensor([0.0, 3.0, 0.0, 0.0])

    def reag_logits_fn(family_idx: torch.Tensor) -> torch.Tensor:
        # family 0 -> prefers reagent 0; family 1 -> prefers reagent 2
        if int(family_idx.item()) == 0:
            return torch.tensor([[5.0, 0.0, -5.0, -5.0]])
        return torch.tensor([[-5.0, -5.0, 5.0, 0.0]])

    # true fam_a + reag 0: oracle ok, inference reag wrong (predicted fam_b -> reag 2)
    m = _eval_inference_metrics(
        fam_logits,
        reag_logits_fn,
        site_logits,
        true_fam=0,
        true_reag=0,
        site_target=site_target,
        families=families,
        family_to_reagent_idxs=family_to_reagent_idxs,
    )
    assert m["fam"] is False
    assert m["reag_oracle"] is True
    assert m["reag"] is False
    assert m["plan"] is False
    assert m["site"] is True
