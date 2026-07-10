"""Train planner / delta predictor on generated weak-supervision pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.app.services.reaction_primitives import list_primitives
from ml.models.architectures import (
    DeltaPropertyPredictor,
    ModificationPlanner,
    mol_to_graph,
)


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, family_to_idx: dict, split: str):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.family_to_idx = family_to_idx

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
        delta = torch.tensor(
            [
                row["delta_density"],
                row["delta_det_velocity"],
                row["delta_det_pressure"],
                row["delta_hf_solid"],
            ],
            dtype=torch.float32,
        )
        # target delta for planner: child-parent absolute gap proxy from deltas
        target_delta = delta.clone()
        return px, cx, fam, delta, target_delta


def collate(batch):
    # variable atom counts — process one-by-one in training loop for simplicity
    return batch


def train(data_csv: Path, out_dir: Path, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv)
    families = [p["family_id"] for p in list_primitives() if p["family_id"] != "identity"]
    family_to_idx = {f: i for i, f in enumerate(families)}
    reagents = families  # reagent vocab aligned to family for bootstrap

    train_ds = PairDataset(df, family_to_idx, "train")
    val_ds = PairDataset(df, family_to_idx, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    planner = ModificationPlanner(n_families=len(families), n_reagents=len(reagents))
    delta_model = DeltaPropertyPredictor()
    opt = torch.optim.Adam(list(planner.parameters()) + list(delta_model.parameters()), lr=lr)

    history = []
    for epoch in range(epochs):
        planner.train()
        delta_model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            for px, cx, fam, delta, target_delta in batch:
                opt.zero_grad()
                px_b = px.unsqueeze(0)
                cx_b = cx.unsqueeze(0)
                z, _ = planner.encode(px_b, target_delta.unsqueeze(0))
                fam_logits = planner.forward_family(z)
                reag_logits = planner.forward_reagent(z)
                loss_fam = F.cross_entropy(fam_logits, torch.tensor([fam]))
                loss_reag = F.cross_entropy(reag_logits, torch.tensor([fam]))
                pred_delta = delta_model(px_b, cx_b)
                loss_delta = F.mse_loss(pred_delta.squeeze(0), delta)
                loss = loss_fam + loss_reag + loss_delta
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n += 1
        avg = total_loss / max(1, n)

        # quick val
        planner.eval()
        delta_model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                for px, cx, fam, delta, target_delta in batch:
                    px_b = px.unsqueeze(0)
                    cx_b = cx.unsqueeze(0)
                    z, _ = planner.encode(px_b, target_delta.unsqueeze(0))
                    fam_logits = planner.forward_family(z)
                    pred_delta = delta_model(px_b, cx_b)
                    loss = F.cross_entropy(fam_logits, torch.tensor([fam])) + F.mse_loss(
                        pred_delta.squeeze(0), delta
                    )
                    val_loss += float(loss.item())
                    vn += 1
        vavg = val_loss / max(1, vn)
        history.append({"epoch": epoch + 1, "train_loss": avg, "val_loss": vavg})
        print(f"epoch {epoch+1}: train={avg:.4f} val={vavg:.4f}")

    bundle = {
        "families": families,
        "reagents": reagents,
        "history": history,
    }
    torch.save({"planner": planner.state_dict(), "delta": delta_model.state_dict(), "meta": bundle}, out_dir / "planner.pt")
    (out_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "ignite_modification_planner",
                "version": "0.1.0",
                "label_type": "weak_xgb_delta",
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
