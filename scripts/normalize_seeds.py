"""Normalize and index seed molecules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.app.services.chemistry import canonicalize, inchikey_of, murcko_scaffold
from backend.app.domain.models import PROPERTY_KEYS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = ROOT / "sample_start_molecules.csv"
    out_dir = ROOT / "data" / "processed"
    idx_dir = ROOT / "data" / "indexes"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    rows = []
    for i, row in df.iterrows():
        can = canonicalize(str(row["SMILES"]))
        if not can:
            continue
        incomplete = any(pd.isna(row.get(k)) for k in PROPERTY_KEYS)
        rec = {
            "source_row": int(i),
            "smiles": str(row["SMILES"]),
            "canonical_smiles": can,
            "inchikey": inchikey_of(can),
            "scaffold": murcko_scaffold(can),
            "incomplete": incomplete,
        }
        for k in PROPERTY_KEYS:
            rec[k] = None if pd.isna(row.get(k)) else float(row[k])
        rows.append(rec)

    out = pd.DataFrame(rows)
    # Dedup complete molecules by canonical SMILES (keep first)
    complete = out[~out["incomplete"]].drop_duplicates("canonical_smiles")
    complete.to_csv(out_dir / "seeds_canonical.csv", index=False)
    out.to_csv(out_dir / "seeds_all_with_lineage.csv", index=False)

    index = {
        "n_raw": len(df),
        "n_parseable": len(out),
        "n_complete_unique": len(complete),
        "inchikey_to_smiles": {
            r["inchikey"]: r["canonical_smiles"]
            for _, r in complete.iterrows()
            if r["inchikey"]
        },
    }
    (idx_dir / "seed_index.json").write_text(json.dumps({
        "n_raw": index["n_raw"],
        "n_parseable": index["n_parseable"],
        "n_complete_unique": index["n_complete_unique"],
    }, indent=2))
    print(json.dumps({k: index[k] for k in ("n_raw", "n_parseable", "n_complete_unique")}, indent=2))


if __name__ == "__main__":
    main()
