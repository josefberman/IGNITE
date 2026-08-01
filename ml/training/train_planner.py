"""Train planner / delta predictor on generated weak-supervision pairs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.app.services.reaction_primitives import list_primitives, list_reagents
from ml.models.architectures import (
    DEFAULT_HIDDEN,
    DeltaPropertyPredictor,
    ModificationPlanner,
    mol_to_graph,
)


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
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.family_to_idx = family_to_idx
        self.reagent_to_idx = reagent_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parent = mol_to_graph(row["parent_smiles"])
        child = mol_to_graph(row["child_smiles"])
        if parent is None or child is None:
            return self[max(0, idx - 1)]
        px, _ = parent
        cx, _ = child
        fam = self.family_to_idx.get(row["reaction_family"], 0)
        reagent_id = row["reagent_id"] if "reagent_id" in row.index and pd.notna(row["reagent_id"]) else None
        if reagent_id is None:
            # Legacy CSV: synthesize from family + smarts position when possible
            reagent_id = f"{row['reaction_family']}__v0"
        reag = self.reagent_to_idx.get(str(reagent_id), 0)
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
        return px, cx, fam, reag, delta, delta.clone(), site_target


def collate(batch):
    return batch


def _site_loss(site_logits: torch.Tensor, site_target: torch.Tensor) -> torch.Tensor:
    if site_target.sum() <= 0:
        return 0.01 * (site_logits ** 2).mean()
    pos = float(site_target.sum().item())
    neg = float(site_target.numel() - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(site_logits, site_target, pos_weight=pos_weight)


def train(data_csv: Path, out_dir: Path, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv)
    if "site_atom_indices" not in df.columns:
        raise SystemExit(
            "Dataset missing site_atom_indices; re-run ml.data_gen.generate_pairs before training"
        )
    families = [p["family_id"] for p in list_primitives() if p["family_id"] != "identity"]
    reagents = [r["reagent_id"] for r in list_reagents()]
    family_to_idx = {f: i for i, f in enumerate(families)}
    reagent_to_idx = {r: i for i, r in enumerate(reagents)}

    # Family → allowed reagent indices (chemistry mask for val metrics / logging)
    family_to_reagent_idxs = {f: [] for f in families}
    for r in list_reagents():
        family_to_reagent_idxs[r["family_id"]].append(reagent_to_idx[r["reagent_id"]])

    train_ds = PairDataset(df, family_to_idx, reagent_to_idx, "train")
    val_ds = PairDataset(df, family_to_idx, reagent_to_idx, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    planner = ModificationPlanner(
        n_families=len(families), n_reagents=len(reagents), hidden=DEFAULT_HIDDEN
    )
    delta_model = DeltaPropertyPredictor(hidden=DEFAULT_HIDDEN)
    opt = torch.optim.Adam(list(planner.parameters()) + list(delta_model.parameters()), lr=lr)
    scale = torch.tensor([1.0, 1000.0, 10.0, 100.0], dtype=torch.float32)

    history = []
    for epoch in range(epochs):
        planner.train()
        delta_model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            for px, cx, fam, reag, delta, target_delta, site_target in batch:
                opt.zero_grad()
                px_b = px.unsqueeze(0)
                cx_b = cx.unsqueeze(0)
                z, _ = planner.encode(px_b, target_delta.unsqueeze(0))
                fam_logits = planner.forward_family(z)
                reag_logits = planner.forward_reagent(z)
                site_logits = planner.forward_sites(px_b, z)
                loss_fam = F.cross_entropy(fam_logits, torch.tensor([fam]))
                loss_reag = F.cross_entropy(reag_logits, torch.tensor([reag]))
                loss_site = _site_loss(site_logits, site_target)
                pred_delta = delta_model(px_b, cx_b)
                loss_delta = F.mse_loss(pred_delta.squeeze(0) / scale, delta / scale)
                loss = loss_fam + loss_reag + 2.0 * loss_site + loss_delta
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n += 1
        avg = total_loss / max(1, n)

        planner.eval()
        delta_model.eval()
        val_loss = 0.0
        site_hits = site_total = 0
        reag_hits = reag_total = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                for px, cx, fam, reag, delta, target_delta, site_target in batch:
                    px_b = px.unsqueeze(0)
                    cx_b = cx.unsqueeze(0)
                    z, _ = planner.encode(px_b, target_delta.unsqueeze(0))
                    fam_logits = planner.forward_family(z)
                    reag_logits = planner.forward_reagent(z)
                    site_logits = planner.forward_sites(px_b, z)
                    pred_delta = delta_model(px_b, cx_b)
                    loss = (
                        F.cross_entropy(fam_logits, torch.tensor([fam]))
                        + F.cross_entropy(reag_logits, torch.tensor([reag]))
                        + F.mse_loss(pred_delta.squeeze(0) / scale, delta / scale)
                        + 2.0 * _site_loss(site_logits, site_target)
                    )
                    val_loss += float(loss.item())
                    vn += 1
                    if site_target.sum() > 0:
                        site_total += 1
                        if site_target[int(site_logits.argmax().item())] > 0.5:
                            site_hits += 1
                    # Family-masked reagent top-1
                    allowed = family_to_reagent_idxs.get(families[fam], [])
                    if allowed:
                        reag_total += 1
                        masked = reag_logits.squeeze(0).clone()
                        ban = torch.ones_like(masked, dtype=torch.bool)
                        ban[allowed] = False
                        masked[ban] = -1e9
                        if int(masked.argmax().item()) == reag:
                            reag_hits += 1
        vavg = val_loss / max(1, vn)
        site_acc = site_hits / max(1, site_total)
        reag_acc = reag_hits / max(1, reag_total)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg,
                "val_loss": vavg,
                "val_site_top1_acc": site_acc,
                "val_reagent_masked_top1_acc": reag_acc,
            }
        )
        print(
            f"epoch {epoch+1}: train={avg:.4f} val={vavg:.4f} "
            f"site_top1={site_acc:.3f} reag_top1={reag_acc:.3f}"
        )

    bundle = {
        "families": families,
        "reagents": reagents,
        "family_to_reagent_idxs": family_to_reagent_idxs,
        "history": history,
        "site_supervision": True,
        "reagent_supervision": True,
        "hidden": DEFAULT_HIDDEN,
        "delta_scale": [1.0, 1000.0, 10.0, 100.0],
    }
    torch.save(
        {"planner": planner.state_dict(), "delta": delta_model.state_dict(), "meta": bundle},
        out_dir / "planner.pt",
    )
    (out_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "ignite_modification_planner",
                "version": "0.3.1",
                "label_type": "weak_xgb_delta+reacting_atom_sites+reagent_variant",
                "hidden": DEFAULT_HIDDEN,
                "n_families": len(families),
                "n_reagents": len(reagents),
                "families": families,
                "history": history,
                "disclaimer": "Trained on synthetic weak labels; not experimental ground truth.",
            },
            indent=2,
        )
    )
    print(f"Saved checkpoint to {out_dir / 'planner.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/modification_pairs.csv"))
    ap.add_argument("--out", type=Path, default=Path("ml/checkpoints"))
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Missing dataset {args.data}; run ml/data_gen/generate_pairs.py first")
    train(args.data, args.out, epochs=args.epochs)
