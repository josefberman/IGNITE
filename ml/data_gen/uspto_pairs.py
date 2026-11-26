"""Build a pairs table from recorded USPTO reactions.

Source: Daniel Lowe, "Chemical reactions from US patents (1976-Sep2016)",
figshare, https://doi.org/10.6084/m9.figshare.5104873 (CC0). The file used
here is ``1976_Sep2016_USPTOgrants_smiles.rsmi`` (patent grants only).

A row is kept when one reactant component, run through one IGNITE reagent
SMARTS, sanitizes to one recorded product component. The site is the reacting
atoms of that match. Property columns are XGBoost predictions of the parent
and that child, not measurements from the patent. A small stratified sample
of parents also gets a ``hold`` row: the child is the parent, every property
delta is 0, and the site is the first carbon. That is the no-gap identity
example, kept small so the class does not dominate the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from backend.app.domain.models import PROPERTY_KEYS
from backend.app.services.chemistry import edit_distance_proxy, sanitize_reaction_product
from backend.app.services.reaction_primitives import list_primitives
from backend.app.services.safety import estimate_safety
from ml.data_gen.generate_pairs import scaffold_of
from ml.site_candidates import reacting_atom_indices

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "uspto"
ARCHIVE_NAME = "1976_Sep2016_USPTOgrants_smiles.7z"
RSMI_NAME = "1976_Sep2016_USPTOgrants_smiles.rsmi"
FIGSHARE_URL = "https://ndownloader.figshare.com/files/8664379"
SOURCE_DOI = "10.6084/m9.figshare.5104873"
SOURCE_NAME = "uspto_grants_1976_2016"
MIN_HEAVY = 6
MAX_HEAVY = 80
HOLD_REAGENT = "hold"
HOLD_FAMILY = "identity"
HOLD_SMARTS = "[#6:1]>>[#6:1]"
# About the median USPTO reagent count, so identity is a normal small class.
HOLD_SAMPLES = 64
HOLD_SEED = 42

_TEMPLATES: list[dict] = []


def compile_templates() -> list[dict]:
    """One entry per SMARTS variant. ``reagent_id`` is the specific reaction."""
    out = []
    for prim in list_primitives():
        if prim["family_id"] == "identity":
            continue
        for smarts in prim["smarts_variants"]:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None or rxn.GetNumReactantTemplates() < 1 or rxn.GetNumProductTemplates() < 1:
                continue
            reactant = rxn.GetReactantTemplate(0)
            product = rxn.GetProductTemplate(0)
            if reactant is None or product is None:
                continue
            out.append(
                {
                    "reagent_id": prim["family_id"],
                    "reaction_family": prim["group"],
                    "smarts": smarts,
                    "rxn": rxn,
                    "reactant_q": Chem.Mol(reactant),
                    "product_q": Chem.Mol(product),
                }
            )
    return out


def _canon(mol: Chem.Mol) -> str | None:
    try:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(0)
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None
    if not smi or "." in smi:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < MIN_HEAVY or n_heavy > MAX_HEAVY:
        return None
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return None
    return smi


def _components(smiles: str) -> list[tuple[str, Chem.Mol]]:
    found = []
    for part in smiles.split("."):
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        smi = _canon(mol)
        if smi is None:
            continue
        found.append((smi, mol))
    return found


def match_reaction(reaction_smiles: str, templates: list[dict] | None = None) -> list[dict]:
    """Recorded reaction → hits whose SMARTS reproduces a recorded product."""
    bank = _TEMPLATES if templates is None else templates
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        reactant_s, product_s = parts[0], parts[2]
    elif len(parts) == 2:
        reactant_s, product_s = parts[0], parts[1]
    else:
        return []
    products = _components(product_s)
    reactants = _components(reactant_s)
    if not products or not reactants:
        return []
    product_mols = [mol for _, mol in products]
    product_smiles = {smi for smi, _ in products}
    candidates = []
    for tmpl in bank:
        query = tmpl["product_q"]
        try:
            if any(mol.HasSubstructMatch(query) for mol in product_mols):
                candidates.append(tmpl)
        except Exception:
            continue
    if not candidates:
        return []
    hits = []
    seen: set[tuple[str, str]] = set()
    for parent_smi, parent in reactants:
        for tmpl in candidates:
            try:
                if not parent.HasSubstructMatch(tmpl["reactant_q"]):
                    continue
                outcomes = tmpl["rxn"].RunReactants((parent,))
            except Exception:
                continue
            for pi, outcome in enumerate(outcomes):
                matched = None
                for prod in outcome:
                    prod = sanitize_reaction_product(prod)
                    if prod is None:
                        continue
                    try:
                        child = Chem.MolToSmiles(prod, canonical=True, isomericSmiles=False)
                    except Exception:
                        continue
                    if child in product_smiles and child != parent_smi:
                        matched = child
                        break
                if matched is None:
                    continue
                key = (parent_smi, tmpl["reagent_id"])
                if key in seen:
                    break
                seen.add(key)
                sites = reacting_atom_indices(parent, tmpl["rxn"], pi)
                hits.append(
                    {
                        "parent_smiles": parent_smi,
                        "child_smiles": matched,
                        "reaction_family": tmpl["reaction_family"],
                        "reagent_id": tmpl["reagent_id"],
                        "smarts": tmpl["smarts"],
                        "site_atom_indices": json.dumps(sites),
                    }
                )
                break
    return hits


def _init_worker() -> None:
    global _TEMPLATES
    RDLogger.DisableLog("rdApp.*")
    _TEMPLATES = compile_templates()


def _match_batch(lines: list[str]) -> tuple[int, list[dict]]:
    hits = []
    n = 0
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if not cols or ">" not in cols[0]:
            continue
        n += 1
        for hit in match_reaction(cols[0]):
            hit["patent"] = cols[1] if len(cols) > 1 else ""
            hit["year"] = cols[3] if len(cols) > 3 else ""
            hits.append(hit)
    return n, hits


def _batches(path: Path, size: int):
    batch: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ">" not in line:
                continue
            batch.append(line)
            if len(batch) >= size:
                yield batch
                batch = []
    if batch:
        yield batch


def ensure_rsmi(rsmi: Path, download: bool) -> Path:
    if rsmi.exists():
        return rsmi
    archive = rsmi.parent / ARCHIVE_NAME
    if not archive.exists():
        if not download:
            raise SystemExit(f"Missing {rsmi}. Re-run with --download.")
        import urllib.request

        rsmi.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {FIGSHARE_URL}", flush=True)
        urllib.request.urlretrieve(FIGSHARE_URL, archive)
    print(f"extracting {archive.name}", flush=True)
    import subprocess

    subprocess.check_call(["7z", "e", "-y", f"-o{rsmi.parent}", str(archive)])
    if not rsmi.exists():
        raise SystemExit(f"7z did not write {rsmi}")
    return rsmi


def match_file(rsmi: Path, hits_path: Path, workers: int, batch_size: int, limit: int) -> dict:
    import multiprocessing as mp

    hits_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    n_read = 0
    n_raw = 0
    n_unique = 0
    ctx = mp.get_context("fork")
    pending = 0
    max_pending = max(workers * 2, 1)
    with hits_path.open("w", encoding="utf-8") as out, ctx.Pool(workers, initializer=_init_worker) as pool:
        jobs = []
        for batch in _batches(rsmi, batch_size):
            if limit and n_read >= limit:
                break
            if limit and n_read + len(batch) > limit:
                batch = batch[: limit - n_read]
            jobs.append(pool.apply_async(_match_batch, (batch,)))
            pending += 1
            n_read += len(batch)
            if pending >= max_pending:
                n_raw, n_unique = _drain(jobs, out, seen, n_raw, n_unique, wait_one=True)
                pending = len(jobs)
                print(f"reactions {n_read} hits {n_unique}", flush=True)
        n_raw, n_unique = _drain(jobs, out, seen, n_raw, n_unique, wait_one=False)
    stats = {"n_reactions_read": n_read, "n_template_hits": n_raw, "n_unique_hits": n_unique}
    print(f"match done {json.dumps(stats)}", flush=True)
    return stats


def _drain(jobs, out, seen, n_raw, n_unique, wait_one: bool):
    if not jobs:
        return n_raw, n_unique
    ready = [jobs.pop(0)] if wait_one else jobs[:]
    if not wait_one:
        jobs.clear()
    for job in ready:
        n, hits = job.get()
        n_raw += len(hits)
        for hit in hits:
            key = (hit["parent_smiles"], hit["child_smiles"], hit["reagent_id"])
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(hit) + "\n")
            n_unique += 1
    return n_raw, n_unique


def _init_predictor(models_dir: str) -> None:
    global _PRED
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    from backend.app.services.prediction import PropertyPredictor

    _PRED = PropertyPredictor(models_dir)


def _predict_one(smiles: str):
    est = _PRED.predict(smiles)
    if not est:
        return smiles, None
    return smiles, {k: float(est[k].value) for k in PROPERTY_KEYS}


def add_gaps(hits_path: Path, out_csv: Path, models_dir: Path, workers: int, seed: int) -> dict:
    import multiprocessing as mp

    rows = [json.loads(line) for line in hits_path.read_text(encoding="utf-8").splitlines() if line]
    smiles = sorted({r["parent_smiles"] for r in rows} | {r["child_smiles"] for r in rows})
    cache_path = hits_path.with_name("uspto_prop_cache.jsonl")
    predicted: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            rec = json.loads(line)
            if rec["props"] is not None:
                predicted[rec["smiles"]] = rec["props"]
    todo = [smi for smi in smiles if smi not in predicted]
    print(f"predicting properties for {len(todo)} of {len(smiles)} molecules", flush=True)
    if todo:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(workers, initializer=_init_predictor, initargs=(str(models_dir),))
        try:
            with cache_path.open("a", encoding="utf-8") as cache:
                done = len(smiles) - len(todo)
                for smi, props in pool.imap_unordered(_predict_one, todo, chunksize=32):
                    cache.write(json.dumps({"smiles": smi, "props": props}) + "\n")
                    if props is not None:
                        predicted[smi] = props
                    done += 1
                    if done % 5000 == 0 or done == len(smiles):
                        print(f"  properties {done}/{len(smiles)}", flush=True)
        finally:
            pool.close()
            pool.join()
    kept = []
    dropped = 0
    safety_cache: dict[str, tuple[float, float]] = {}
    for row in rows:
        parent = predicted.get(row["parent_smiles"])
        child = predicted.get(row["child_smiles"])
        if parent is None or child is None:
            dropped += 1
            continue
        safety = safety_cache.get(row["child_smiles"])
        if safety is None:
            report = estimate_safety(row["child_smiles"])
            safety = (report.score, report.sascore)
            safety_cache[row["child_smiles"]] = safety
        out = {
            "parent_smiles": row["parent_smiles"],
            "child_smiles": row["child_smiles"],
            "scaffold": scaffold_of(row["parent_smiles"]),
            "reaction_family": row["reaction_family"],
            "reagent_id": row["reagent_id"],
            "smarts": row["smarts"],
            "site_atom_indices": row["site_atom_indices"],
            "edit_distance": edit_distance_proxy(row["parent_smiles"], row["child_smiles"]),
            "label_type": "uspto_recorded+xgb_delta",
            "safety_score": safety[0],
            "sascore": safety[1],
            "patent": row.get("patent", ""),
            "year": row.get("year", ""),
            "source": SOURCE_NAME,
        }
        for key in PROPERTY_KEYS:
            out[f"parent_{key}"] = parent[key]
            out[f"child_{key}"] = child[key]
            out[f"delta_{key}"] = child[key] - parent[key]
        kept.append(out)
    frame = pd.DataFrame(kept)
    if frame.empty:
        raise SystemExit("No pairs left after property prediction")
    scaffolds = sorted(frame["scaffold"].fillna("").unique())
    import random

    rng = random.Random(seed)
    rng.shuffle(scaffolds)
    n = len(scaffolds)
    train_sc = set(scaffolds[: int(0.7 * n)])
    val_sc = set(scaffolds[int(0.7 * n) : int(0.85 * n)])

    def split_name(sc: str) -> str:
        if sc in train_sc:
            return "train"
        if sc in val_sc:
            return "val"
        return "test"

    frame["split"] = frame["scaffold"].map(split_name)
    frame, n_hold = append_hold_samples(frame)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return {
        "n_pairs": int(len(frame)),
        "n_hold": int(n_hold),
        "n_dropped_no_property": int(dropped),
        "n_scaffolds": int(n),
    }


def _first_carbon(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            return [atom.GetIdx()]
    return []


def _sample_by_split(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Keep ``n`` rows, split in proportion to how many candidates each split has."""
    if len(rows) <= n:
        return rows
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(str(row.get("split", "")), []).append(row)
    raw = {split: n * len(group) / len(rows) for split, group in by.items()}
    quota = {split: int(v) for split, v in raw.items()}
    left = n - sum(quota.values())
    for split, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if left <= 0:
            break
        quota[split] += 1
        left -= 1
    rng = random.Random(seed)
    chosen = []
    for split, group in by.items():
        k = min(quota[split], len(group))
        chosen.extend(rng.sample(group, k))
    return chosen


def append_hold_samples(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A few identity rows: same molecule, zero property gap, reagent ``hold``."""
    if frame.empty or "parent_smiles" not in frame.columns:
        return frame, 0
    already = set()
    if "reagent_id" in frame.columns:
        already = set(frame.loc[frame["reagent_id"].astype(str) == HOLD_REAGENT, "parent_smiles"].astype(str))
    safety_by: dict[str, tuple[float, float]] = {}
    if "safety_score" in frame.columns:
        for smi, score, sa in zip(frame["child_smiles"], frame["safety_score"], frame["sascore"]):
            safety_by.setdefault(str(smi), (float(score), float(sa)))
    rows = []
    seen: set[str] = set()
    for rec in frame.itertuples(index=False):
        smi = str(rec.parent_smiles)
        if smi in seen or smi in already:
            seen.add(smi)
            continue
        seen.add(smi)
        site = _first_carbon(smi)
        if not site:
            continue
        safety = safety_by.get(smi)
        if safety is None:
            report = estimate_safety(smi)
            safety = (float(report.score), float(report.sascore))
            safety_by[smi] = safety
        row = rec._asdict()
        row["child_smiles"] = smi
        row["reaction_family"] = HOLD_FAMILY
        row["reagent_id"] = HOLD_REAGENT
        row["smarts"] = HOLD_SMARTS
        row["site_atom_indices"] = json.dumps(site)
        row["edit_distance"] = 0
        row["label_type"] = "hold_identity"
        row["safety_score"] = safety[0]
        row["sascore"] = safety[1]
        for key in PROPERTY_KEYS:
            row[f"child_{key}"] = row[f"parent_{key}"]
            row[f"delta_{key}"] = 0.0
        rows.append(row)
    rows = _sample_by_split(rows, HOLD_SAMPLES, HOLD_SEED)
    if not rows:
        return frame, 0
    extra = pd.DataFrame(rows, columns=list(frame.columns))
    return pd.concat([frame, extra], ignore_index=True), len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rsmi", type=Path, default=RAW_DIR / RSMI_NAME)
    ap.add_argument("--download", action="store_true", help="download and extract the figshare archive if the RSMI is missing")
    ap.add_argument("--hits", type=Path, default=ROOT / "data" / "processed" / "uspto_pair_hits.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "uspto_pairs.csv")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0, help="read at most this many reaction lines (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-match", action="store_true", help="reuse --hits and only predict gaps")
    args = ap.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    stats = {}
    if not args.skip_match:
        rsmi = ensure_rsmi(args.rsmi, args.download)
        stats.update(match_file(rsmi, args.hits, args.workers, args.batch_size, args.limit))
    elif not args.hits.exists():
        raise SystemExit(f"Missing hits file {args.hits}")
    gap_stats = add_gaps(args.hits, args.out, args.models_dir, args.workers, args.seed)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta = {
        "source": "Daniel Lowe, Chemical reactions from US patents (1976-Sep2016)",
        "source_doi": SOURCE_DOI,
        "source_url": "https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873",
        "source_file": RSMI_NAME,
        "figshare_file_url": FIGSHARE_URL,
        "license": "CC0-1.0",
        "method": (
            "Each patent-grant reaction is kept when one reactant component, "
            "transformed by one IGNITE reagent SMARTS, sanitizes to one recorded "
            "product component. The site atoms are the reacting atoms of that match. "
            f"Components outside {MIN_HEAVY}-{MAX_HEAVY} heavy atoms, or with no carbon, "
            "are ignored. Stereochemistry is dropped so a template match is not "
            "rejected for a stereo mark the SMARTS does not write. "
            "Duplicate parent/child/reagent rows collapse to one. "
            f"A stratified sample of {HOLD_SAMPLES} parents also gets one hold row: "
            "child equals parent, every delta is 0, and the reagent is hold. "
            "parent_*, child_*, and delta_* are XGBoost predictions from models/, "
            "not properties reported in the patent. The split is a 70/15/15 shuffle "
            "of Bemis-Murcko scaffolds."
        ),
        "label_type": "uspto_recorded+xgb_delta",
        "sha256": digest,
        "seed": args.seed,
        **stats,
        **gap_stats,
    }
    meta_path = args.out.with_name("uspto_pairs_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
