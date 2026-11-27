#!/usr/bin/env python
"""Grid-search planner head depth on a small pair subset.

Each trial is one combination of message-passing depth, hidden width, and the
hidden-layer counts of the family, reagent, and site heads. ``0`` hidden
layers is a single linear map to the logits. The shipped heads are family 0,
reagent 1, and site 1. Defaults vary message-passing depth over 1–4, width
over 32, 64, and 128, and each head by one extra hidden layer.

Every trial trains on the same subsample of the train split and is scored on
the same subsample of the val split. The score is the one ``train_planner``
uses for early stopping:

    0.25 * family top-1 + site top-1 + reagent top-1 given the true family

Higher is better. Checkpoints go under ``--out`` and are not copied over
``ml/checkpoints/planner.pt``.
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

DEFAULT_LAYERS = (1, 2, 3, 4)
DEFAULT_HIDDEN = (32, 64, 128)
DEFAULT_FAMILY_LAYERS = (0, 1)
DEFAULT_REAGENT_LAYERS = (1, 2)
DEFAULT_SITE_LAYERS = (1, 2)


def parse_ints(text: str, minimum: int = 1) -> tuple[int, ...]:
    vals = tuple(int(part) for part in text.split(",") if part.strip())
    if not vals or any(v < minimum for v in vals):
        raise argparse.ArgumentTypeError(f"expected integers >= {minimum}, got {text!r}")
    return vals


def parse_nonneg_ints(text: str) -> tuple[int, ...]:
    return parse_ints(text, 0)


def selection_score(row: dict) -> float:
    return (
        0.25 * float(row["val_family_top1"])
        + float(row["val_site_top1"])
        + float(row["val_reagent_oracle_top1"])
    )


def subset_pairs(df, max_train: int, max_val: int, seed: int):
    import pandas as pd

    rng = random.Random(seed)
    parts = []
    for split, cap in (("train", max_train), ("val", max_val)):
        block = df[df["split"] == split]
        if cap >= 0 and len(block) > cap:
            order = list(block.index)
            rng.shuffle(order)
            block = block.loc[order[:cap]]
        parts.append(block)
    return pd.concat(parts)


def best_epoch(history: list[dict]) -> dict:
    return max(history, key=selection_score)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "unified_pairs.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "checkpoints" / "arch_search")
    ap.add_argument("--layers", type=parse_ints, default=DEFAULT_LAYERS, help="comma-separated message-passing depths")
    ap.add_argument("--hidden", type=parse_ints, default=DEFAULT_HIDDEN, help="comma-separated layer widths")
    ap.add_argument("--family-layers", type=parse_nonneg_ints, default=DEFAULT_FAMILY_LAYERS, help="comma-separated family-head hidden layers; 0 is linear")
    ap.add_argument("--reagent-layers", type=parse_ints, default=DEFAULT_REAGENT_LAYERS, help="comma-separated reagent-head hidden layers")
    ap.add_argument("--site-layers", type=parse_ints, default=DEFAULT_SITE_LAYERS, help="comma-separated site-head hidden layers")
    ap.add_argument("--max-train", type=int, default=8000, help="train rows per trial; -1 keeps the split")
    ap.add_argument("--max-val", type=int, default=2000, help="val rows per trial; -1 keeps the split")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.data.exists():
        print(f"Missing dataset {args.data}; run ml.data_gen.generate_pairs first")
        return 1

    import pandas as pd

    from ml.training.train_planner import train

    df = pd.read_csv(args.data)
    small = subset_pairs(df, args.max_train, args.max_val, args.seed)
    n_train = int((small["split"] == "train").sum())
    n_val = int((small["split"] == "val").sum())
    if n_train == 0 or n_val == 0:
        print(f"subset is empty (train={n_train}, val={n_val})")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    subset_path = args.out / "subset.csv"
    small.to_csv(subset_path, index=False)
    n_trials = (
        len(args.layers) * len(args.hidden)
        * len(args.family_layers) * len(args.reagent_layers) * len(args.site_layers)
    )
    print(
        f"subset train={n_train} val={n_val} grid={n_trials} trials -> {args.out}",
        flush=True,
    )

    rows = []
    for layers in args.layers:
        for hidden in args.hidden:
            for family_layers in args.family_layers:
                for reagent_layers in args.reagent_layers:
                    for site_layers in args.site_layers:
                        trial = (
                            f"layers{layers}_hidden{hidden}"
                            f"_fam{family_layers}_reag{reagent_layers}_site{site_layers}"
                        )
                        trial_dir = args.out / trial
                        print(f"\n=== {trial} ===", flush=True)
                        row = {
                            "layers": layers,
                            "hidden": hidden,
                            "family_layers": family_layers,
                            "reagent_layers": reagent_layers,
                            "site_layers": site_layers,
                            "n_train": n_train,
                            "n_val": n_val,
                            "checkpoint": str(trial_dir / "planner.pt"),
                            "error": "",
                        }
                        try:
                            train(
                                subset_path,
                                trial_dir,
                                epochs=args.epochs,
                                patience=args.patience,
                                seed=args.seed,
                                hidden=hidden,
                                layers=layers,
                                family_layers=family_layers,
                                reagent_layers=reagent_layers,
                                site_layers=site_layers,
                            )
                            card = json.loads((trial_dir / "model_card.json").read_text(encoding="utf-8"))
                            chosen = best_epoch(card["history"])
                            row.update(
                                {
                                    "score": f"{selection_score(chosen):.6g}",
                                    "best_epoch": int(chosen["epoch"]),
                                    "val_family_top1": f"{chosen['val_family_top1']:.6g}",
                                    "val_site_top1": f"{chosen['val_site_top1']:.6g}",
                                    "val_reagent_oracle_top1": f"{chosen['val_reagent_oracle_top1']:.6g}",
                                    "val_plan_top1": f"{chosen['val_plan_top1']:.6g}",
                                }
                            )
                        except (Exception, SystemExit) as exc:
                            row["score"] = ""
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            print(f"FAIL {trial}: {row['error']}", flush=True)
                        rows.append(row)

    def rank_key(row: dict) -> float:
        try:
            return float(row["score"])
        except (TypeError, ValueError):
            return -1.0

    rows.sort(key=rank_key, reverse=True)
    summary = args.out / "arch_search.csv"
    fields = [
        "layers", "hidden", "family_layers", "reagent_layers", "site_layers",
        "score", "best_epoch",
        "val_family_top1", "val_site_top1", "val_reagent_oracle_top1", "val_plan_top1",
        "n_train", "n_val", "checkpoint", "error",
    ]
    with summary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print(f"\nwrote {summary}", flush=True)
    if best.get("score"):
        print(
            f"best layers={best['layers']} hidden={best['hidden']} "
            f"family={best['family_layers']} reagent={best['reagent_layers']} "
            f"site={best['site_layers']} score={best['score']} checkpoint={best['checkpoint']}",
            flush=True,
        )
        return 0
    print("every trial failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
