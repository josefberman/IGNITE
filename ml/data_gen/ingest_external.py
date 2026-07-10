"""Adapters for licensed external reaction / property / hazard datasets.

Place downloaded files under data/raw/ and run this script to normalize them.
Without external files, writes an empty curated schema so the pipeline stays reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")


SCHEMA_HAZARD = ["smiles", "impact_sensitivity_J", "decomposition_temp_C", "source", "license"]
SCHEMA_REACTION = ["parent_smiles", "child_smiles", "reaction_family", "source", "license", "label_type"]


def ingest_hazard(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(raw_dir.glob("hazard_*.csv")):
        df = pd.read_csv(path)
        missing = [c for c in SCHEMA_HAZARD if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        frames.append(df[SCHEMA_HAZARD])
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=SCHEMA_HAZARD)
    dest = out_dir / "hazard_labels.csv"
    out.to_csv(dest, index=False)
    return {"path": str(dest), "n": len(out), "note": "Empty until licensed hazard CSVs are provided."}


def ingest_measured_reactions(raw_dir: Path, out_dir: Path) -> dict:
    frames = []
    for path in sorted(raw_dir.glob("reactions_*.csv")):
        df = pd.read_csv(path)
        missing = [c for c in SCHEMA_REACTION if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        df = df[SCHEMA_REACTION]
        df["label_type"] = df["label_type"].fillna("measured")
        frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=SCHEMA_REACTION)
    dest = out_dir / "measured_reactions.csv"
    out.to_csv(dest, index=False)
    return {"path": str(dest), "n": len(out), "note": "Measured labels preferred over weak_xgb_delta."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    report = {
        "hazard": ingest_hazard(args.raw, args.out),
        "reactions": ingest_measured_reactions(args.raw, args.out),
    }
    (args.out / "external_ingest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
