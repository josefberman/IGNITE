#!/usr/bin/env python
"""Validation search over the multitask planner's fork point, depth, widths, and loss weights.

Every trial trains ``train_planner --arch multitask`` on the same seeded
random parent-grouped split (70% train / 10% val / 20% test), on a fixed
subsample of the train split (``--max-train``) and is scored on the whole
validation split. The test split is never touched here.

Score (higher is better), the one ``train_planner`` early-stops on:

    0.25 * family top-1 + site top-1 + reagent top-1 given the true family

Stage 1 varies where the trunk forks and how deep the shared and per-task
parts are. Stage 2 takes the best stage-1 architecture and varies width and
loss weighting, and repeats the top configs and the baseline with a second
seed. Stage 3 takes the best config so far and tries the other fork points at
the same message-passing depth, a shared dense layer, deeper / wider towers,
and dropout 0.05 / 0.3. Stage 4 retrains the strongest fork configs with
``--mt-conditioning none``: parallel family / reagent / site heads with no
family input to the reagent tower and no reagent input to the site tower
(pure multitask), against the hierarchical towers of stages 1-3. Results are appended to ``docs/multitask_search.csv`` and
``docs/multitask_search.md`` is regenerated. Checkpoints and logs go under
``--out`` (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (shared_mp, branch_mp, shared_fc, tower_layers, tower_hidden, hidden)
BASELINE = dict(shared_mp=3, branch_mp=0, shared_fc=0, tower=1, tower_hidden=128, hidden=128)
STAGE1 = [
    BASELINE,                                   # = current ModificationPlanner defaults
    dict(BASELINE, shared_mp=1),                # = shipped checkpoint depth
    dict(BASELINE, shared_mp=2),
    dict(BASELINE, shared_mp=4),
    dict(BASELINE, shared_fc=1),
    dict(BASELINE, shared_fc=2),
    dict(BASELINE, shared_mp=2, branch_mp=1),
    dict(BASELINE, shared_mp=1, branch_mp=2),
    dict(BASELINE, shared_mp=0, branch_mp=3),
    dict(BASELINE, tower=2),
    dict(BASELINE, shared_fc=1, tower=2),
    dict(BASELINE, tower_hidden=256),
    dict(BASELINE, tower=2, tower_hidden=256),
    dict(BASELINE, shared_mp=2, branch_mp=1, tower=2),
]

FIELDS = [
    "trial", "stage", "seed", "conditioning", "hidden", "dropout", "shared_mp", "branch_mp", "shared_fc", "tower", "tower_hidden",
    "loss_reag_weight", "loss_site_weight", "loss_weighting", "n_params", "best_epoch", "epochs_run",
    "val_score", "val_family_top1", "val_family_top3", "val_reagent_top1", "val_reagent_oracle_top1",
    "val_site_top1", "val_site_top3", "val_plan_top1", "val_plan_all_top1", "val_loss",
    "n_train", "n_val", "minutes", "checkpoint", "error",
]


def trial_name(c: dict) -> str:
    name = (
        f"h{c['hidden']}_s{c['shared_mp']}_b{c['branch_mp']}_fc{c['shared_fc']}"
        f"_t{c['tower']}_w{c['tower_hidden']}"
    )
    if c.get("loss_weighting", "fixed") == "uncertainty":
        name += "_unc"
    elif (c.get("lr_w", 1.0), c.get("ls_w", 1.0)) != (1.0, 1.0):
        name += f"_lr{c.get('lr_w', 1.0)}_ls{c.get('ls_w', 1.0)}"
    cond = c.get("cond", "hierarchical")
    if cond == "none":
        name += "_cnone"
    elif cond == "parallel":
        name += "_cparallel"
    elif cond == "cascade":
        name += "_ccascade"
    if float(c.get("dropout", 0.15)) != 0.15:
        name += f"_d{c['dropout']}"
    if c.get("seed", 42) != 42:
        name += f"_seed{c['seed']}"
    return name


def command(c: dict, args, out_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "ml.training.train_planner", "--arch", "multitask",
        "--data", str(args.data), "--out", str(out_dir),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--max-train", str(args.max_train), "--max-val", str(args.max_val),
        "--threads", str(args.threads), "--seed", str(c.get("seed", 42)),
        "--split", "random_parent", "--split-seed", str(args.split_seed),
        "--hidden", str(c["hidden"]), "--dropout", str(c.get("dropout", 0.15)),
        "--mt-shared-mp", str(c["shared_mp"]), "--mt-branch-mp", str(c["branch_mp"]),
        "--mt-shared-fc", str(c["shared_fc"]), "--mt-tower-layers", str(c["tower"]),
        "--mt-tower-hidden", str(c["tower_hidden"]),
        "--loss-reag-weight", str(c.get("lr_w", 1.0)), "--loss-site-weight", str(c.get("ls_w", 1.0)),
        "--mt-loss-weighting", c.get("loss_weighting", "fixed"),
        "--mt-conditioning", c.get("cond", "hierarchical"),
    ]


def row_from_card(c: dict, stage: str, out_dir: Path, minutes: float, err: str) -> dict:
    row = {
        "trial": trial_name(c), "stage": stage, "seed": c.get("seed", 42), "hidden": c["hidden"],
        "dropout": c.get("dropout", 0.15), "conditioning": c.get("cond", "hierarchical"),
        "shared_mp": c["shared_mp"], "branch_mp": c["branch_mp"], "shared_fc": c["shared_fc"],
        "tower": c["tower"], "tower_hidden": c["tower_hidden"],
        "loss_reag_weight": c.get("lr_w", 1.0), "loss_site_weight": c.get("ls_w", 1.0),
        "loss_weighting": c.get("loss_weighting", "fixed"),
        "minutes": f"{minutes:.1f}", "checkpoint": str(out_dir / "planner.pt"), "error": err,
    }
    card_p = out_dir / "model_card.json"
    if not err and card_p.exists():
        card = json.loads(card_p.read_text())
        v = card["best_val"]
        row.update(
            n_params=card["n_params"], best_epoch=card["best_epoch"], epochs_run=len(card["history"]),
            val_score=f"{v['score']:.4f}", val_family_top1=f"{v['family_top1']:.4f}",
            val_family_top3=f"{v['family_top3']:.4f}", val_reagent_top1=f"{v['reagent_top1']:.4f}",
            val_reagent_oracle_top1=f"{v['reagent_oracle_top1']:.4f}", val_site_top1=f"{v['site_top1']:.4f}",
            val_site_top3=f"{v['site_top3']:.4f}", val_plan_top1=f"{v['plan_top1']:.4f}",
            val_plan_all_top1=f"{v['plan_all_top1']:.4f}", val_loss=f"{v['loss']:.4f}",
            n_train=card["split"]["n_train_used"], n_val=card["split"]["n_val_used"],
        )
    return row


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def score_of(r: dict) -> float:
    try:
        return float(r["val_score"])
    except (KeyError, TypeError, ValueError):
        return -1.0


def config_of(r: dict) -> dict:
    c = {k: int(r[k]) for k in ("hidden", "shared_mp", "branch_mp", "shared_fc", "tower", "tower_hidden")}
    c.update(lr_w=float(r["loss_reag_weight"]), ls_w=float(r["loss_site_weight"]),
             loss_weighting=r["loss_weighting"], seed=int(r["seed"]),
             dropout=float(r.get("dropout") or 0.15), cond=r.get("conditioning") or "hierarchical")
    return c


def stage2_configs(rows: list[dict]) -> list[dict]:
    s1 = sorted([r for r in rows if r["stage"] == "1"], key=score_of, reverse=True)
    best = config_of(s1[0])
    out = []
    for h in (96, 192):
        out.append(dict(best, hidden=h, tower_hidden=h if best["tower_hidden"] == best["hidden"] else best["tower_hidden"]))
    for lr_w, ls_w in ((1.0, 2.0), (0.5, 1.0), (2.0, 1.0)):
        out.append(dict(best, lr_w=lr_w, ls_w=ls_w))
    out.append(dict(best, loss_weighting="uncertainty"))
    # second seed for the two best architectures and the baseline (seed noise)
    for r in s1[:2]:
        out.append(dict(config_of(r), seed=7))
    base = dict(BASELINE, lr_w=1.0, ls_w=1.0, loss_weighting="fixed", seed=7)
    if all(trial_name(base) != trial_name(c) for c in out):
        out.append(base)
    return out


def stage3_configs(rows: list[dict]) -> list[dict]:
    ranked = sorted([r for r in rows if not r.get("error")], key=score_of, reverse=True)
    best = dict(config_of(ranked[0]), seed=42)
    depth = best["shared_mp"] + best["branch_mp"]
    out = []
    for shared in range(depth):  # every fork point inside the encoder at this depth
        out.append(dict(best, shared_mp=shared, branch_mp=depth - shared, shared_fc=0))
    if best["branch_mp"] == 0:
        out.append(dict(best, shared_fc=best["shared_fc"] + 1))
    out.append(dict(best, tower=best["tower"] + 1))
    out.append(dict(best, tower_hidden=2 * best["tower_hidden"]))
    out.append(dict(best, dropout=0.05))
    out.append(dict(best, dropout=0.3))
    return out


def stage4_configs(rows: list[dict]) -> list[dict]:
    """Pure multitask: the strongest fork configs with parallel, unconditioned heads."""
    ok = [r for r in rows if not r.get("error") and (r.get("conditioning") or "hierarchical") == "hierarchical"]
    seen, out = set(), []

    def add(c):
        c = dict(c, cond="none")
        if trial_name(c) not in seen:
            seen.add(trial_name(c))
            out.append(c)

    for seed in (42, 7):
        add(dict(BASELINE, shared_mp=1, seed=seed))   # stage-1/2 winner
        add(dict(BASELINE, seed=seed))                # current depth, fully shared
    add(dict(BASELINE, shared_mp=0, branch_mp=3))     # current depth, fork at the input
    add(dict(BASELINE, shared_mp=0, branch_mp=1))     # depth 1, fork at the input
    for r in sorted(ok, key=score_of, reverse=True)[:3]:
        add(dict(config_of(r), seed=42))
    return out


def run_trials(configs: list[dict], stage: str, args) -> None:
    csv_path = args.results
    done = {r["trial"] for r in load_rows(csv_path) if not r.get("error")}
    pending = [c for c in configs if trial_name(c) not in done]
    print(f"stage {stage}: {len(pending)} trials to run ({len(configs) - len(pending)} already done)", flush=True)
    running: list = []
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    while pending or running:
        while pending and len(running) < args.parallel:
            c = pending.pop(0)
            out_dir = args.out / trial_name(c)
            out_dir.mkdir(parents=True, exist_ok=True)
            log = (out_dir / "train.log").open("w")
            p = subprocess.Popen(command(c, args, out_dir), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, c, out_dir, time.time(), log))
            print(f"  start {trial_name(c)}", flush=True)
        time.sleep(10)
        for item in list(running):
            p, c, out_dir, t0, log = item
            if p.poll() is None:
                continue
            log.close()
            running.remove(item)
            err = "" if p.returncode == 0 else f"exit {p.returncode}; see {out_dir / 'train.log'}"
            row = row_from_card(c, stage, out_dir, (time.time() - t0) / 60, err)
            rows = [r for r in load_rows(csv_path) if r["trial"] != row["trial"]] + [row]
            write_rows(csv_path, rows)
            write_markdown(rows, args)
            print(f"  done {row['trial']} score={row.get('val_score', '')} {err}", flush=True)


def write_markdown(rows: list[dict], args) -> None:
    rows = sorted(rows, key=score_of, reverse=True)
    lines = [
        "# Multitask planner architecture search",
        "",
        "Generated by `scripts/optimize_multitask_arch.py`; raw rows in `docs/multitask_search.csv`.",
        "",
        f"Split: seeded random split grouped by parent molecule (seed {args.split_seed}), "
        "70% train / 10% val / 20% test of the pairs. Each trial trains on a fixed "
        f"{args.max_train}-pair subsample of the train split for up to {args.epochs} epochs "
        f"(patience {args.patience}) and is scored on the whole validation split at its best epoch. "
        "The test split is not used here.",
        "",
        "Score = 0.25 family top-1 + site top-1 + reagent top-1 given the true family. "
        "`s` shared message-passing layers, `b` private message-passing layers per task "
        "(fork inside the encoder), `fc` shared dense layers after the graph vector, "
        "`t` hidden layers per task tower, `w` tower width, `h` trunk width. "
        "`cnone` parallel heads without hierarchical conditioning (pure multitask; otherwise the "
        "reagent tower sees the family and the site tower the reagent), `d` dropout when not 0.15, `lr`/`ls` reagent/site loss weights when not 1, `unc` learned "
        "uncertainty task weights, `seed` training seed when not 42. "
        "`h128_s3_b0_fc0_t1_w128` is the current `ModificationPlanner` architecture.",
        "",
        "| trial | stage | cond | params | epoch | score | family | family top-3 | reagent | reagent (true fam) | site | site top-3 | plan | val loss | min |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['trial']} | {r['stage']} | | | | failed | | | | | | | | | {r['error']} |")
            continue
        lines.append(
            f"| {r['trial']} | {r['stage']} | {r.get('conditioning') or 'hierarchical'} | {r['n_params']} | {r['best_epoch']}/{r['epochs_run']} | **{r['val_score']}** | "
            f"{r['val_family_top1']} | {r['val_family_top3']} | {r['val_reagent_top1']} | {r['val_reagent_oracle_top1']} | "
            f"{r['val_site_top1']} | {r['val_site_top3']} | {r['val_plan_top1']} | {r['val_loss']} | {r['minutes']} |"
        )
    extra = ROOT / "docs" / "multitask_search_notes.md"
    if extra.exists():
        lines += ["", extra.read_text().strip()]
    args.results.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "unified_pairs.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "multitask_search")
    ap.add_argument("--results", type=Path, default=ROOT / "docs" / "multitask_search.csv")
    ap.add_argument("--stage", choices=("1", "2", "3", "4", "none", "all"), default="all")
    ap.add_argument(
        "--extra", type=str, default="",
        help='JSON list of extra trial configs to run as stage "x", e.g. confirmation seeds',
    )
    ap.add_argument("--max-train", type=int, default=150000)
    ap.add_argument("--max-val", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.stage in ("1", "all"):
        run_trials(STAGE1, "1", args)
    if args.stage in ("2", "all"):
        run_trials(stage2_configs(load_rows(args.results)), "2", args)
    if args.stage in ("3", "all"):
        run_trials(stage3_configs(load_rows(args.results)), "3", args)
    if args.stage in ("4", "all"):
        run_trials(stage4_configs(load_rows(args.results)), "4", args)
    if args.extra:
        extra = [dict(BASELINE, **c) for c in json.loads(args.extra)]
        run_trials(extra, "x", args)
    rows = sorted(load_rows(args.results), key=score_of, reverse=True)
    if rows:
        print(f"best: {rows[0]['trial']} score={rows[0]['val_score']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
