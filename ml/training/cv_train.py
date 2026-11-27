"""Train one planner on a seeded 80% / 10% / 10% parent-grouped split.

Used by ``scripts/train_arch1_parallel.py``, ``train_arch2_cascade_fc.py``,
and ``train_arch3_cascade.py``. ``--seed`` sets both the training RNG and the
split so a rerun with the same seed writes the same train / val / test parents.
The test split is scored once, on the best validation epoch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.models.architectures import (
    DEFAULT_BOND_TYPES,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN,
    DEFAULT_MT_SHARED_MP,
    DEFAULT_MT_TOWER_HIDDEN,
    DEFAULT_MT_TOWER_LAYERS,
)
from ml.training.pair_cache import default_cache_path
from ml.training.train_multitask import train_multitask

ROOT = Path(__file__).resolve().parents[2]
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)


def main(
    argv=None,
    *,
    shared_fc: int,
    conditioning: str,
    out: Path,
    site_reagent: bool = False,
    branch_mp: int = 0,
    shared_mp: int | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "unified_pairs.csv")
    ap.add_argument("--out", type=Path, default=out)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    ap.add_argument("--bond-types", type=int, default=DEFAULT_BOND_TYPES)
    ap.add_argument(
        "--shared-mp", type=int, default=DEFAULT_MT_SHARED_MP if shared_mp is None else shared_mp
    )
    ap.add_argument("--tower-layers", type=int, default=DEFAULT_MT_TOWER_LAYERS)
    ap.add_argument("--tower-hidden", type=int, default=DEFAULT_MT_TOWER_HIDDEN)
    ap.add_argument("--seed", type=int, default=42, help="training RNG and parent-grouped split")
    ap.add_argument("--cache-workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)
    cache = args.cache or default_cache_path(args.data)
    tl = args.tower_layers
    train_multitask(
        args.data,
        args.out,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden=args.hidden,
        dropout=args.dropout,
        bond_types=args.bond_types,
        site_reagent=site_reagent,
        shared_mp=args.shared_mp,
        branch_mp=branch_mp,
        shared_fc=shared_fc,
        tower_hidden=args.tower_hidden,
        family_tower_layers=tl,
        reagent_tower_layers=tl,
        site_tower_layers=max(1, tl),
        split="random_parent",
        split_fractions=SPLIT_FRACTIONS,
        split_seed=args.seed,
        eval_test=True,
        cache_path=cache,
        cache_workers=args.cache_workers,
        threads=args.threads,
        conditioning=conditioning,
    )
    return 0
