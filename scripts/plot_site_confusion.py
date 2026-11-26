#!/usr/bin/env python
"""Site confusion matrices on the evaluation split.

A site is a distinct product of the labeled reagent on the parent. Atom sets
that give the same product (for example symmetric ring positions) are one site,
so picking either counts as correct. Sites are numbered from 1 by their lowest
atom index; the same number is the same product only within one parent and
reagent.

The row is the site whose product is the pair's child. The planner column is
the site with the highest mean site-head score on its reacting atoms, with the
head conditioned on the labeled reagent when the checkpoint supports it. The
rule-based column is the first product in a seeded shuffle of that reagent's
products. Both choices use the labeled reagent, so this is the site decision
rather than the reagent decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROP_COLS = ("density", "det_velocity", "det_pressure", "hf_solid")
RULE_SEED = 0


def plot_confusion(true_idx, pred_idx, labels: list[str], out_dir: Path, stem: str, title: str) -> None:
    """Rows are the true site, columns the selected site. Color is the row fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_classes = len(labels)
    counts = np.zeros((n_classes, n_classes), dtype=float)
    for truth, pred in zip(true_idx, pred_idx):
        counts[int(truth), int(pred)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    side = max(6.0, n_classes * 0.36)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(side + 1.6, side))
    image = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues", origin="upper", aspect="equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true site")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="both", length=2, pad=1)
    ax.set_xlabel("Selected site")
    ax.set_ylabel("True site")
    ax.set_title(title)
    fig.tight_layout()
    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(out_dir / f"{stem}.{ext}", **kw)
    plt.close(fig)
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_site", "selected_site", "count", "row_fraction"])
        for i in range(n_classes):
            for j in range(n_classes):
                if counts[i, j] == 0:
                    continue
                w.writerow([labels[i], labels[j], int(counts[i, j]), f"{fraction[i, j]:.6g}"])


def rule_choice(cands) -> str | None:
    """Product of the first site after the seeded shuffle used by the rule-based modifier."""
    children = sorted({c.child for c in cands})
    if not children:
        return None
    random.Random(RULE_SEED).shuffle(children)
    return children[0]


def encode_parent(model, atom_x, edge):
    """Gap-independent encoder state. Message passing does not see the gap."""
    import torch

    with torch.inference_mode():
        return model.parent_enc(atom_x, edge, return_atom_emb=True)


def site_scores(model, encoded, gap, reagent_idx):
    """Match ModificationPlanner.encode, then the site head, for one cached parent."""
    import torch

    parent_z, atom_ctx, atom_local = encoded
    with torch.inference_mode():
        gamma, beta = model.film(gap).chunk(2, dim=-1)
        parent_cond = parent_z * (1.0 + gamma) + beta
        n_atoms = atom_local.size(1)
        gap_exp = gap.unsqueeze(1).expand(-1, n_atoms, -1)
        attn = model.atom_score(torch.cat([atom_local, gap_exp], dim=-1)).squeeze(-1)
        alpha = torch.softmax(attn, dim=-1)
        attended = torch.bmm(alpha.unsqueeze(1), atom_local).squeeze(1)
        z = model.fuse(torch.cat([parent_cond, attended, gap], dim=-1))
        atom_emb = torch.cat([atom_local, atom_ctx], dim=-1)
        return model.forward_sites(atom_emb, z, gap, reagent_idx=reagent_idx)


def planner_atom_logits(model, atom_x, edge, gap, reagent_idx, encoded_cache, smiles):
    """Site logit per atom. Multitask checkpoints encode with the gap; the legacy encoder does not."""
    import torch

    if not hasattr(model, "parent_enc"):
        with torch.inference_mode():
            z, _, atom_emb = model.encode(atom_x, gap, edge)
            return model.forward_sites(atom_emb, z, gap, reagent_idx=reagent_idx)
    if smiles not in encoded_cache:
        encoded_cache[smiles] = encode_parent(model, atom_x, edge)
    encoded = encoded_cache[smiles]
    if encoded is None:
        return None
    return site_scores(model, encoded, gap, reagent_idx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "modification_pairs.csv")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "ml" / "checkpoints")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    args = ap.parse_args(argv)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    import pandas as pd
    import torch

    from backend.app.services.modification import LearnedPlannerBundle
    from backend.app.services.reaction_primitives import list_primitives
    from ml.models.architectures import mol_to_graph
    from ml.planner_gap import planner_input_from_parent_target
    from ml.site_candidates import candidate_logits, positive_mask, product_groups, reaction_sites

    if not args.data.exists():
        print(f"missing {args.data}")
        return 1
    bundle = LearnedPlannerBundle(args.checkpoint_dir)
    if not bundle.available or bundle._model is None:
        print(bundle.unavailable_reason)
        return 1
    print(f"site head conditioned on reagent: {bundle.ranks_reagent_sites}")

    smarts_by_reagent = {
        p["family_id"]: list(p.get("smarts_variants") or [])
        for p in list_primitives()
        if p["family_id"] != "identity"
    }
    usecols = ["split", "parent_smiles", "child_smiles", "reagent_id", "site_atom_indices"]
    usecols += [f"parent_{k}" for k in PROP_COLS] + [f"child_{k}" for k in PROP_COLS]
    df = pd.read_csv(args.data, usecols=usecols)
    df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print(f"no rows in split {args.split}")
        return 1

    model = bundle._model
    rxn_cache: dict = {}
    site_cache: dict = {}
    encoded_cache: dict = {}
    rows: list[tuple[int, int, int]] = []
    n = len(df)
    n_no_site = n_unmatched = 0

    for i, rec in enumerate(df.itertuples(index=False), start=1):
        try:
            label = tuple(int(a) for a in json.loads(rec.site_atom_indices))
        except (TypeError, json.JSONDecodeError):
            label = tuple()
        smiles = str(rec.parent_smiles)
        reagent = str(rec.reagent_id)
        key = (smiles, reagent)
        packed = site_cache.get(key)
        if packed is None:
            mol = Chem.MolFromSmiles(smiles)
            cands = []
            if mol is not None:
                cands = reaction_sites(mol, smiles, smarts_by_reagent.get(reagent, []), rxn_cache)
            group_of = {}
            for g, members in enumerate(product_groups(cands)):
                for m in members:
                    group_of[m] = g
            child_group = {cands[m].child: g for m, g in group_of.items()}
            rule = rule_choice(cands)
            packed = (cands, group_of, child_group.get(rule) if rule else None)
            site_cache[key] = packed
        cands, group_of, rule_group = packed
        if not cands or rule_group is None:
            n_no_site += 1
            continue
        pos = positive_mask(cands, str(rec.child_smiles), label)
        if not any(pos):
            n_unmatched += 1
            continue
        true_group = min(group_of[m] for m, p in enumerate(pos) if p)

        parent = {k: float(getattr(rec, f"parent_{k}")) for k in PROP_COLS}
        target = {k: float(getattr(rec, f"child_{k}")) for k in PROP_COLS}
        gap = torch.tensor([planner_input_from_parent_target(parent, target)], dtype=torch.float32)
        reag_idx = None
        if reagent in bundle.reagent_names:
            reag_idx = torch.tensor([bundle.reagent_names.index(reagent)])
        if smiles not in encoded_cache:
            graph = mol_to_graph(smiles)
            if graph is None:
                encoded_cache[smiles] = None
            elif hasattr(model, "parent_enc"):
                x, edge = graph
                encoded_cache[smiles] = encode_parent(model, x.unsqueeze(0), edge)
            else:
                encoded_cache[smiles] = graph
        cached = encoded_cache[smiles]
        if cached is None:
            n_no_site += 1
            continue
        if hasattr(model, "parent_enc"):
            atom_logits = site_scores(model, cached, gap, reag_idx)
        else:
            x, edge = cached
            atom_logits = planner_atom_logits(model, x.unsqueeze(0), edge, gap, reag_idx, encoded_cache, smiles)
        planner_group = group_of[int(candidate_logits(atom_logits, cands).argmax().item())]
        rows.append((true_group, planner_group, rule_group))
        if i % 2000 == 0 or i == n:
            print(f"  {i}/{n} kept={len(rows)}", flush=True)

    if not rows:
        print("no comparable rows")
        return 1

    n_classes = max(max(t, p, r) for t, p, r in rows) + 1
    labels = [f"site {i + 1}" for i in range(n_classes)]
    true_idx = [t for t, _p, _r in rows]
    planner_idx = [p for _t, p, _r in rows]
    rule_idx = [r for _t, _p, r in rows]
    plot_confusion(
        true_idx, planner_idx, labels, args.out_dir,
        "site_confusion_planner", "Planner site confusion (evaluation set)",
    )
    plot_confusion(
        true_idx, rule_idx, labels, args.out_dir,
        "site_confusion_rule_based", "Rule-based site confusion (evaluation set)",
    )
    kept = len(rows)
    planner_hit = sum(t == p for t, p, _r in rows) / kept
    rule_hit = sum(t == r for t, _p, r in rows) / kept
    print(
        f"n={kept} no_site={n_no_site} unmatched_label={n_unmatched} "
        f"planner_site={planner_hit:.3f} rule_site={rule_hit:.3f}"
    )
    for name in ("site_confusion_planner", "site_confusion_rule_based"):
        print(f"wrote {args.out_dir / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
