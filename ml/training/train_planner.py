"""Train planner / delta predictor on generated weak-supervision pairs."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.app.services.reaction_primitives import (
    list_planner_vocab,
    matching_group_ids,
    matching_reaction_ids,
    planner_labels,
)
from ml.models.architectures import (
    DEFAULT_DROPOUT,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    ModificationPlanner,
    mol_to_graph,
)
from ml.planner_gap import training_planner_input


def _parse_sites(raw) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    s = str(raw).strip()
    if not s:
        return []
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        val = ast.literal_eval(s)
    return [int(x) for x in val]


class PairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        family_to_idx: dict,
        reagent_to_idx: dict,
        split: str,
        *,
        rng: random.Random | None = None,
        jitter_prob: float = 0.35,
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.family_to_idx = family_to_idx
        self.reagent_to_idx = reagent_to_idx
        self.rng = rng or random.Random(42)
        self.jitter_prob = jitter_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parent = mol_to_graph(row["parent_smiles"])
        child = mol_to_graph(row["child_smiles"])
        if parent is None or child is None:
            return self[max(0, idx - 1)]
        px, p_edge = parent
        cx, c_edge = child
        raw_reag = row["reagent_id"] if "reagent_id" in row.index and pd.notna(row["reagent_id"]) else None
        group, specific = planner_labels(str(row["reaction_family"]), None if raw_reag is None else str(raw_reag))
        fam = self.family_to_idx.get(group, 0)
        reag = self.reagent_to_idx.get(specific, 0)

        from ml.planner_gap import row_parent_child_props

        parent_props, child_props = row_parent_child_props(row)
        planner_gap = training_planner_input(
            parent_props,
            child_props,
            rng=self.rng,
            jitter_prob=self.jitter_prob,
        )

        delta = torch.tensor(
            [
                row["delta_density"],
                row["delta_det_velocity"],
                row["delta_det_pressure"],
                row["delta_hf_solid"],
            ],
            dtype=torch.float32,
        )
        sites = _parse_sites(row["site_atom_indices"] if "site_atom_indices" in row.index else None)
        site_target = torch.zeros(px.size(0), dtype=torch.float32)
        for a in sites:
            if 0 <= a < site_target.numel():
                site_target[a] = 1.0
        return (
            px,
            p_edge,
            cx,
            c_edge,
            fam,
            reag,
            delta,
            torch.tensor(planner_gap, dtype=torch.float32),
            site_target,
            str(row["parent_smiles"]),
        )


def collate(batch):
    return batch


def _site_loss(site_logits: torch.Tensor, site_target: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over atoms.

    BCE-with-logits does not push the reacting atom to be the argmax, which is
    the decision used at inference. With about one positive site in ~25 atoms,
    that left top-1 barely above chance.
    """
    if site_target.sum() <= 0:
        return 0.01 * (site_logits ** 2).mean()
    log_prob = F.log_softmax(site_logits, dim=-1)
    return -(log_prob * site_target).sum() / site_target.sum().clamp_min(1.0)


def _family_class_weights(labels: pd.Series, families: list[str]) -> torch.Tensor:
    counts = labels.value_counts()
    weights = []
    for fam in families:
        c = max(1, int(counts.get(fam, 1)))
        weights.append(1.0 / (c ** 0.5))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.mean()


def _mask_class_logits(
    logits: torch.Tensor,
    allowed_idxs: list[int],
) -> torch.Tensor:
    """Set disallowed class logits to a large negative. Empty allow-list = no mask."""
    if not allowed_idxs:
        return logits
    masked = logits.clone()
    ban = torch.ones(masked.shape[-1], dtype=torch.bool)
    for idx in allowed_idxs:
        if 0 <= int(idx) < ban.numel():
            ban[int(idx)] = False
    if not bool((~ban).any()):
        return logits
    masked[..., ban] = -1e9
    return masked


def _mask_reagent_logits(
    reag_logits: torch.Tensor,
    allowed_idxs: list[int],
) -> torch.Tensor:
    return _mask_class_logits(reag_logits, allowed_idxs)


def _eval_inference_metrics(
    fam_logits: torch.Tensor,
    reag_logits_fn,
    site_logits: torch.Tensor,
    *,
    true_fam: int,
    true_reag: int,
    site_target: torch.Tensor,
    families: list[str],
    family_to_reagent_idxs: dict[str, list[int]],
    allowed_family_idxs: list[int] | None = None,
) -> dict[str, bool]:
    """Mirror LearnedPlannerBundle: predict family → family-conditioned reagent → site."""
    fam_for_pred = _mask_class_logits(fam_logits, allowed_family_idxs or [])
    pred_fam = int(fam_for_pred.argmax().item())
    pred_fam_t = torch.tensor([pred_fam])
    k3 = min(3, int(fam_for_pred.shape[-1]))
    top3 = {int(i) for i in torch.topk(fam_for_pred, k=k3).indices.view(-1).tolist()}

    reag_logits = reag_logits_fn(pred_fam_t)
    pred_pool = family_to_reagent_idxs.get(families[pred_fam], [])
    pred_reag = int(_mask_reagent_logits(reag_logits, pred_pool).argmax().item())

    oracle_pool = family_to_reagent_idxs.get(families[true_fam], [])
    oracle_reag = int(_mask_reagent_logits(reag_logits_fn(torch.tensor([true_fam])), oracle_pool).argmax().item())

    fam_ok = pred_fam == true_fam
    reag_ok = pred_reag == true_reag
    reag_oracle_ok = oracle_reag == true_reag
    site_ok = False
    if site_target.sum() > 0:
        site_ok = bool(site_target[int(site_logits.argmax().item())].item() > 0.5)

    return {
        "fam": fam_ok,
        "reag": reag_ok,
        "reag_oracle": reag_oracle_ok,
        "site": site_ok,
        "site_labeled": site_target.sum() > 0,
        "plan": fam_ok and reag_ok,
        "plan_all": fam_ok and reag_ok and site_ok,
        "fam_top3": true_fam in top3,
    }


def train(
    data_csv: Path,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 2e-4,
    *,
    patience: int = 7,
    jitter_prob: float = 0.0,
    accum_steps: int = 16,
    seed: int = 42,
    hidden: int = DEFAULT_HIDDEN,
    layers: int = DEFAULT_LAYERS,
    heads: int = DEFAULT_HEADS,
    dropout: float = DEFAULT_DROPOUT,
    label_smoothing: float = 0.0,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    loss_reag_weight: float = 1.0,
    loss_site_weight: float = 1.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    df = pd.read_csv(data_csv)
    if "site_atom_indices" not in df.columns:
        raise SystemExit(
            "Dataset missing site_atom_indices; re-run ml.data_gen.generate_pairs before training"
        )
    families, reagent_rows = list_planner_vocab()
    reagents = [r["reagent_id"] for r in reagent_rows]
    family_to_idx = {f: i for i, f in enumerate(families)}
    reagent_to_idx = {r: i for i, r in enumerate(reagents)}

    family_to_reagent_idxs = {f: [] for f in families}
    for r in reagent_rows:
        family_to_reagent_idxs[r["family_id"]].append(reagent_to_idx[r["reagent_id"]])

    df_train = df[df["split"] == "train"]
    broad = df_train.apply(
        lambda r: planner_labels(
            str(r["reaction_family"]),
            None if "reagent_id" not in r.index or pd.isna(r["reagent_id"]) else str(r["reagent_id"]),
        )[0],
        axis=1,
    )
    fam_weights = _family_class_weights(broad, families)

    family_allow: dict[str, list[int]] = {}
    reagent_allow: dict[str, list[int]] = {}
    parents = df["parent_smiles"].astype(str).unique()
    print(f"Matching reaction templates on {len(parents)} parents...", flush=True)
    for i, smiles in enumerate(parents, start=1):
        fams = matching_group_ids(smiles)
        family_allow[smiles] = [family_to_idx[f] for f in fams if f in family_to_idx]
        reagent_allow[smiles] = [
            reagent_to_idx[r] for r in matching_reaction_ids(smiles) if r in reagent_to_idx
        ]
        if i % 50 == 0 or i == len(parents):
            print(f"  templates {i}/{len(parents)}", flush=True)

    train_ds = PairDataset(
        df, family_to_idx, reagent_to_idx, "train", rng=rng, jitter_prob=jitter_prob
    )
    val_ds = PairDataset(
        df, family_to_idx, reagent_to_idx, "val", rng=random.Random(0), jitter_prob=0.0
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout)
    planner = ModificationPlanner(
        n_families=len(families), n_reagents=len(reagents), **enc_kw
    )
    opt = torch.optim.AdamW(planner.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)

    def _allowed_families(smiles: str, gold: int) -> list[int]:
        allowed = list(family_allow.get(smiles) or [])
        if not allowed:
            return []
        if gold not in allowed:
            allowed.append(gold)
        return allowed

    def _allowed_reagents(smiles: str, fam_idx: int, gold: int | None = None) -> list[int]:
        fam_pool = family_to_reagent_idxs.get(families[fam_idx], [])
        matched = set(reagent_allow.get(smiles) or [])
        allowed = [i for i in fam_pool if i in matched] if matched else list(fam_pool)
        if not allowed:
            allowed = list(fam_pool)
        if gold is not None and gold not in allowed:
            allowed.append(gold)
        return allowed

    best_score = -1.0
    best_fam_acc = -1.0
    best_state = None
    stale = 0
    history = []
    accum_steps = max(1, int(accum_steps))

    n_train = len(train_ds)
    log_every = 500
    for epoch in range(epochs):
        planner.train()
        total_loss = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        micro = 0
        print(f"epoch {epoch+1}/{epochs} train ({n_train} pairs)", flush=True)
        for batch in train_loader:
            for px, p_edge, _cx, _c_edge, fam, reag, _delta, planner_gap, site_target, parent_smi in batch:
                px_b = px.unsqueeze(0)
                gap_b = planner_gap.unsqueeze(0)
                fam_t = torch.tensor([fam])
                z, _, atom_emb = planner.encode(px_b, gap_b, p_edge)
                fam_logits = _mask_class_logits(
                    planner.forward_family(z),
                    _allowed_families(parent_smi, fam),
                )
                reag_logits = _mask_reagent_logits(
                    planner.forward_reagent(z, family_idx=fam_t),
                    _allowed_reagents(parent_smi, fam, reag),
                )
                site_logits = planner.forward_sites(atom_emb)
                # Applicability masks use -1e9; label smoothing would assign
                # mass to those classes and blow up the loss.
                loss_fam = F.cross_entropy(fam_logits, fam_t, weight=fam_weights)
                loss_reag = F.cross_entropy(reag_logits, torch.tensor([reag]))
                loss_site = _site_loss(site_logits, site_target)
                loss = loss_fam + loss_reag_weight * loss_reag + loss_site_weight * loss_site
                (loss / accum_steps).backward()
                micro += 1
                total_loss += float(loss.item())
                n += 1
                if n % log_every == 0 or n == n_train:
                    print(f"  train {n}/{n_train} loss={total_loss / n:.4f}", flush=True)
                if micro % accum_steps == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(planner.parameters(), grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        if micro % accum_steps:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()
        avg = total_loss / max(1, n)

        planner.eval()
        print(f"epoch {epoch+1}/{epochs} val ({len(val_ds)} pairs)", flush=True)
        val_loss = 0.0
        counts = {
            "fam": 0,
            "fam_top3": 0,
            "reag": 0,
            "reag_oracle": 0,
            "site": 0,
            "site_labeled": 0,
            "plan": 0,
            "plan_all": 0,
        }
        totals = {k: 0 for k in counts}
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                for (
                    px,
                    p_edge,
                    _cx,
                    _c_edge,
                    fam,
                    reag,
                    _delta,
                    planner_gap,
                    site_target,
                    parent_smi,
                ) in batch:
                    px_b = px.unsqueeze(0)
                    gap_b = planner_gap.unsqueeze(0)
                    fam_t = torch.tensor([fam])
                    allowed_fam = _allowed_families(parent_smi, fam)
                    z, _, atom_emb = planner.encode(px_b, gap_b, p_edge)
                    fam_logits = planner.forward_family(z)
                    fam_logits_m = _mask_class_logits(fam_logits, allowed_fam)
                    site_logits = planner.forward_sites(atom_emb)

                    def reag_logits_fn(family_idx: torch.Tensor) -> torch.Tensor:
                        idx = int(family_idx.item())
                        # Include the gold reagent only for the true family.
                        # Adding it under a wrong family leaks the label into top-1.
                        gold = reag if idx == fam else None
                        pooled = _allowed_reagents(parent_smi, idx, gold)
                        return _mask_reagent_logits(
                            planner.forward_reagent(z, family_idx=family_idx),
                            pooled,
                        )

                    oracle_reag = reag_logits_fn(fam_t)
                    loss = (
                        F.cross_entropy(fam_logits_m, fam_t, weight=fam_weights)
                        + loss_reag_weight * F.cross_entropy(oracle_reag, torch.tensor([reag]))
                        + loss_site_weight * _site_loss(site_logits, site_target)
                    )
                    val_loss += float(loss.item())
                    vn += 1
                    if vn % log_every == 0 or vn == len(val_ds):
                        print(f"  val {vn}/{len(val_ds)}", flush=True)

                    m = _eval_inference_metrics(
                        fam_logits,
                        reag_logits_fn,
                        site_logits,
                        true_fam=fam,
                        true_reag=reag,
                        site_target=site_target,
                        families=families,
                        family_to_reagent_idxs=family_to_reagent_idxs,
                        allowed_family_idxs=allowed_fam,
                    )
                    totals["fam"] += 1
                    totals["fam_top3"] += 1
                    totals["reag"] += 1
                    totals["reag_oracle"] += 1
                    totals["plan"] += 1
                    for key in ("fam", "fam_top3", "reag", "reag_oracle", "plan"):
                        counts[key] += int(m[key])
                    if m["site_labeled"]:
                        totals["site"] += 1
                        totals["site_labeled"] += 1
                        totals["plan_all"] += 1
                        counts["site"] += int(m["site"])
                        counts["plan_all"] += int(m["plan_all"])

        vavg = val_loss / max(1, vn)
        fam_acc = counts["fam"] / max(1, totals["fam"])
        fam_top3 = counts["fam_top3"] / max(1, totals["fam_top3"])
        reag_acc = counts["reag"] / max(1, totals["reag"])
        reag_oracle_acc = counts["reag_oracle"] / max(1, totals["reag_oracle"])
        site_acc = counts["site"] / max(1, totals["site"])
        plan_acc = counts["plan"] / max(1, totals["plan"])
        plan_all_acc = counts["plan_all"] / max(1, totals["plan_all"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg,
                "val_loss": vavg,
                "lr": scheduler.get_last_lr()[0],
                "val_family_top1": fam_acc,
                "val_family_top3": fam_top3,
                "val_reagent_top1": reag_acc,
                "val_reagent_oracle_top1": reag_oracle_acc,
                "val_site_top1": site_acc,
                "val_plan_top1": plan_acc,
                "val_plan_all_top1": plan_all_acc,
            }
        )
        print(
            f"epoch {epoch+1}: train={avg:.4f} val={vavg:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
            f"fam={fam_acc:.3f} fam3={fam_top3:.3f} reag={reag_acc:.3f} "
            f"reag_oracle={reag_oracle_acc:.3f} site={site_acc:.3f} plan={plan_acc:.3f}",
            flush=True,
        )

        # Family alone peaked at epoch 1 while site/reagent were still random.
        score = fam_acc + 0.5 * site_acc + 0.25 * reag_oracle_acc
        if score > best_score + 1e-4:
            best_score = score
            best_fam_acc = fam_acc
            stale = 0
            best_state = {
                "planner": copy.deepcopy(planner.state_dict()),
                "epoch": epoch + 1,
            }
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"Early stop at epoch {epoch+1} (best score={best_score:.3f}, fam={best_fam_acc:.3f})",
                    flush=True,
                )
                break

    if best_state is None:
        best_state = {
            "planner": planner.state_dict(),
            "epoch": epochs,
        }

    arch_meta = {
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "dropout": dropout,
        "encoder": "residual_mp_mean_max",
        "gap_fuse": "film",
        "site_head": "local_context",
        "reagent_local": "family_only",
        "vocab": "broad_family_specific_reagent",
        "applicability_mask": True,
        "family_conditioned_reagent": True,
    }
    bundle = {
        "families": families,
        "reagents": reagents,
        "family_to_reagent_idxs": family_to_reagent_idxs,
        "history": history,
        "site_supervision": True,
        "reagent_supervision": True,
        **arch_meta,
        "planner_gap": "signed_normalized",
        "train_hparams": {
            "lr": lr,
            "label_smoothing": label_smoothing,
            "loss_reag_weight": loss_reag_weight,
            "loss_site_weight": loss_site_weight,
            "jitter_prob": jitter_prob,
            "accum_steps": accum_steps,
        },
        "best_epoch": best_state["epoch"],
        "best_val_family_top1": best_fam_acc,
    }
    torch.save(
        {"planner": best_state["planner"], "meta": bundle},
        out_dir / "planner.pt",
    )
    (out_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "ignite_modification_planner",
                "version": "0.10.0",
                "label_type": "weak_xgb_delta+reacting_atom_sites+reagent_variant",
                **arch_meta,
                "n_families": len(families),
                "n_reagents": len(reagents),
                "best_val_family_top1": best_fam_acc,
                "best_epoch": best_state["epoch"],
                "history": history,
                "disclaimer": "Trained on synthetic weak labels; not experimental ground truth.",
            },
            indent=2,
        )
    )
    print(f"Saved checkpoint to {out_dir / 'planner.pt'} (best epoch {best_state['epoch']})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/modification_pairs.csv"))
    ap.add_argument("--out", type=Path, default=Path("ml/checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--jitter-prob", type=float, default=0.0)
    ap.add_argument("--accum-steps", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    ap.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--loss-reag-weight", type=float, default=1.0)
    ap.add_argument("--loss-site-weight", type=float, default=1.0)
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Missing dataset {args.data}; run ml.data_gen.generate_pairs.py first")
    train(
        args.data,
        args.out,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        jitter_prob=args.jitter_prob,
        accum_steps=args.accum_steps,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        loss_reag_weight=args.loss_reag_weight,
        loss_site_weight=args.loss_site_weight,
    )
