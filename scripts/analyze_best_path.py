#!/usr/bin/env python
"""Plot mean best_path ``overall.score`` and ``fit.mape_normalized`` vs generation.

``--run`` executes the benchmark first at a fixed beam of width=2, top_k=1,
depth=20 (logs under ``logs/benchmark_w2_k1_d20``), then plots those logs.

Score direction: ``overall.score`` is ``ScoreBreakdown.total`` (MAPE + SAScore
penalty) and the beam search ranks ascending, so
LOWER IS BETTER. ``fit.mape_normalized`` is the logged MAPE term (MAPE/100,
capped at 1) and is also lower-is-better. The y-axes are labelled accordingly.

Variable-length policy (carry-forward): best paths have different lengths
(a run may finish with a best molecule found at generation 2 of 8). For each
run, the score at generation g is the score of the last best_path step with
generation <= g; after the path ends its final score is carried forward up to
the maximum generation observed in that mode (the best molecule is unchanged,
so the achieved score is unchanged). Mean and +/-1 SD (population, ddof=0)
are computed per generation over all runs of a mode; ``n`` per generation is
written to the summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OKABE_ITO = {"planner": "#0072B2", "rule_based": "#E69F00"}
LABELS = {"planner": "Learned planner", "rule_based": "Rule-based modifier"}
# Tight beam: one survivor, two proposals per expansion, long enough that
# proposal order is not washed out by a wide search. Separate log dir so
# logs from the cycled benchmark beam are not reused.
RUN_BEAM = {"width": 2, "top_k": 1, "depth": 20}
RUN_LOG_DIR = ROOT / "logs" / "benchmark_w2_k1_d20"


def _step_value(step: dict, *keys: str):
    node = step
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def load_runs(log_dir: Path, *keys: str) -> dict[str, list[list[tuple[int, float]]]]:
    """Best-path (generation, value) series per mode. ``keys`` is the logged path, e.g. ``fit``, ``mape_normalized``."""
    runs: dict[str, list] = {"planner": [], "rule_based": []}
    for p in sorted(Path(log_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        mode = rec.get("mode")
        steps = []
        for s in rec.get("best_path") or []:
            value = _step_value(s, *keys)
            if value is None:
                continue
            steps.append((int(s["generation"]), float(value)))
        if mode in runs and steps:
            runs[mode].append(sorted(steps))
    return runs


def carry_forward(steps: list[tuple[int, float]], max_gen: int) -> np.ndarray:
    out = np.full(max_gen + 1, np.nan)
    cur = np.nan
    lookup = dict(steps)
    for g in range(max_gen + 1):
        if g in lookup:
            cur = lookup[g]
        out[g] = cur
    return out


def summarize(runs: list[list[tuple[int, float]]]):
    if not runs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    max_gen = max(g for r in runs for g, _ in r)
    mat = np.vstack([carry_forward(r, max_gen) for r in runs])
    n = np.sum(~np.isnan(mat), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
    return np.arange(max_gen + 1), mean, sd, n


def plot(
    summaries: dict,
    out_dir: Path,
    stem: str,
    *,
    ylabel: str,
    title: str,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = []
    for mode, (gen, mean, sd, n) in summaries.items():
        if len(gen) == 0:
            continue
        plotted.append(gen)
        c = OKABE_ITO[mode]
        ax.plot(gen, mean, color=c, marker="o", lw=2, label=f"{LABELS[mode]} (n={int(n[0])} runs)")
        ax.fill_between(gen, mean - sd, mean + sd, color=c, alpha=0.2, label=f"{LABELS[mode]} ±1 SD")
    if plotted:
        lo = int(min(g.min() for g in plotted))
        hi = int(max(g.max() for g in plotted))
        ax.set_xlim(lo-1, hi+1)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_xlabel("Generation (beam-search expansion round)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{stem}.svg"
    fig.savefig(p)
    paths = [p]
    plt.close(fig)
    return paths


def write_csv(path: Path, summaries: dict, value_name: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "generation", f"mean_{value_name}", f"sd_{value_name}", "n_runs"])
        for mode, (gen, mean, sd, n) in summaries.items():
            for row in zip(gen, mean, sd, n):
                w.writerow([mode, int(row[0]), f"{row[1]:.6g}", f"{row[2]:.6g}", int(row[3])])


METRICS = (
    {
        "keys": ("overall", "score"),
        "stem": "best_path_score_vs_generation",
        "value_name": "score",
        "ylabel": "Best-path overall score (lower is better)",
        "title": "Best-path score vs generation (carry-forward after path end)",
    },
    {
        "keys": ("fit", "mape_normalized"),
        "stem": "best_path_mape_normalized_vs_generation",
        "value_name": "mape_normalized",
        "ylabel": "Best-path normalized MAPE (lower is better)",
        "title": "Best-path normalized MAPE vs generation (carry-forward after path end)",
    },
)


def _run_benchmark(argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(
        "ignite_run_benchmark", ROOT / "scripts" / "run_benchmark.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(argv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run", action="store_true",
        help="Run the benchmark at width=2, top_k=1, depth=20, then plot",
    )
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--suite", choices=("50", "500"), default="50")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.run:
        log_dir = args.log_dir or RUN_LOG_DIR
        bench_argv = [
            "--suite", args.suite,
            "--log-dir", str(log_dir),
            "--depth-override", str(RUN_BEAM["depth"]),
            "--width-override", str(RUN_BEAM["width"]),
            "--top-k-override", str(RUN_BEAM["top_k"]),
        ]
        if args.limit:
            bench_argv += ["--limit", str(args.limit)]
        rc = _run_benchmark(bench_argv)
        if rc != 0:
            return rc
    else:
        log_dir = args.log_dir or Path(os.environ.get("IGNITE_RUN_LOG_DIR") or ROOT / "logs" / "benchmark")
    wrote = []
    counted = False
    for spec in METRICS:
        runs = load_runs(log_dir, *spec["keys"])
        if not any(runs.values()):
            continue
        if not counted:
            for mode, series in runs.items():
                print(f"{mode}: {len(series)} runs")
            counted = True
        summaries = {m: summarize(r) for m, r in runs.items()}
        wrote.extend(plot(
            summaries, args.out_dir, spec["stem"],
            ylabel=spec["ylabel"], title=spec["title"],
        ))
        csv_path = args.out_dir / f"{spec['stem']}.csv"
        write_csv(csv_path, summaries, spec["value_name"])
        wrote.append(csv_path)
    if not wrote:
        print(f"no valid run logs in {log_dir}")
        return 1
    for p in wrote:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
