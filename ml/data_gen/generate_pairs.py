"""Generate parent→child weak-supervision dataset from SMARTS primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from backend.app.services.chemistry import canonicalize, edit_distance_proxy, validate_molecule
from backend.app.services.prediction import PropertyPredictor
from backend.app.services.reaction_primitives import list_primitives, reagent_id_for_smarts
from backend.app.services.safety import estimate_safety
from backend.app.domain.models import PROPERTY_KEYS


def scaffold_of(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def reacting_atom_indices(mol: Chem.Mol, rxn, product_index: int) -> list[int]:
    """Map RDKit reacting template atoms onto parent atom indices for one RunReactants hit."""
    try:
        tmpl = rxn.GetReactantTemplate(0)
        if tmpl is None:
            return []
        matches = mol.GetSubstructMatches(tmpl)
        if product_index >= len(matches):
            return []
        match = matches[product_index]
        reacting = []
        try:
            reacting_sets = rxn.GetReactingAtoms()
            if reacting_sets:
                reacting = list(reacting_sets[0])
        except Exception:
            reacting = []
        if reacting:
            sites = sorted({int(match[j]) for j in reacting if j < len(match)})
            if sites:
                return sites
        # Weak fallback: all atoms in the matched reactant pattern
        return sorted({int(a) for a in match})
    except Exception:
        return []


def generate(seed_csv: Path, models_dir: Path, out_dir: Path, max_parents: int = 500, max_per_parent: int = 5, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    df = pd.read_csv(seed_csv)
    smiles_list = [canonicalize(s) for s in df["SMILES"].astype(str)]
    smiles_list = [s for s in smiles_list if s]
    smiles_list = list(dict.fromkeys(smiles_list))
    rng.shuffle(smiles_list)
    smiles_list = smiles_list[:max_parents]

    predictor = PropertyPredictor(models_dir)
    primitives = [p for p in list_primitives() if p["family_id"] != "identity"]
    rows = []
    failures = []
    n_with_sites = 0

    for parent in smiles_list:
        parent_props = predictor.predict(parent)
        if not parent_props:
            continue
        mol = Chem.MolFromSmiles(parent)
        if mol is None:
            continue
        rng.shuffle(primitives)
        produced = 0
        for prim in primitives:
            if produced >= max_per_parent:
                break
            for smarts in prim["smarts_variants"]:
                if produced >= max_per_parent:
                    break
                try:
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    if rxn is None:
                        continue
                    product_sets = rxn.RunReactants((mol,))
                    for pi, product_tuple in enumerate(product_sets):
                        sites = reacting_atom_indices(mol, rxn, pi)
                        for product in product_tuple:
                            try:
                                Chem.SanitizeMol(product)
                                child = Chem.MolToSmiles(product, canonical=True)
                            except Exception as e:
                                failures.append({"parent": parent, "smarts": smarts, "error": str(e)})
                                continue
                            vr = validate_molecule(child, parent_smiles=parent)
                            if not vr.valid or not vr.canonical_smiles:
                                failures.append(
                                    {"parent": parent, "smarts": smarts, "error": vr.reason}
                                )
                                continue
                            child = vr.canonical_smiles
                            child_props = predictor.predict(child)
                            if not child_props:
                                continue
                            safety = estimate_safety(child)
                            if sites:
                                n_with_sites += 1
                            reagent_id = reagent_id_for_smarts(prim["family_id"], smarts) or (
                                f"{prim['family_id']}__v0"
                            )
                            row = {
                                "parent_smiles": parent,
                                "child_smiles": child,
                                "scaffold": scaffold_of(parent),
                                "reaction_family": prim["family_id"],
                                "reagent_id": reagent_id,
                                "smarts": smarts,
                                "site_atom_indices": json.dumps(sites),
                                "edit_distance": edit_distance_proxy(parent, child),
                                "label_type": "weak_xgb_delta",
                                "safety_score": safety.score,
                                "sascore": safety.sascore,
                            }
                            for k in PROPERTY_KEYS:
                                row[f"parent_{k}"] = parent_props[k].value
                                row[f"child_{k}"] = child_props[k].value
                                row[f"delta_{k}"] = child_props[k].value - parent_props[k].value
                            rows.append(row)
                            produced += 1
                            if produced >= max_per_parent:
                                break
                        if produced >= max_per_parent:
                            break
                except Exception as e:
                    failures.append({"parent": parent, "smarts": smarts, "error": str(e)})

    out = pd.DataFrame(rows)
    # Scaffold split
    scaffolds = sorted(out["scaffold"].unique())
    rng.shuffle(scaffolds)
    n = len(scaffolds)
    train_sc = set(scaffolds[: int(0.7 * n)])
    val_sc = set(scaffolds[int(0.7 * n) : int(0.85 * n)])
    test_sc = set(scaffolds[int(0.85 * n) :])

    def split_name(sc):
        if sc in train_sc:
            return "train"
        if sc in val_sc:
            return "val"
        return "test"

    out["split"] = out["scaffold"].map(split_name)
    csv_path = out_dir / "modification_pairs.csv"
    out.to_csv(csv_path, index=False)

    meta = {
        "n_pairs": len(out),
        "n_with_site_labels": int(n_with_sites),
        "n_failures": len(failures),
        "max_parents": max_parents,
        "seed": seed,
        "label_types": ["weak_xgb_delta", "reacting_atom_sites", "reagent_variant"],
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "license_note": "Synthetic weak labels from local SMARTS + XGBoost; not experimental ground truth.",
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "failures.json").write_text(json.dumps(failures[:1000], indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-csv", type=Path, default=Path("sample_start_molecules.csv"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--max-parents", type=int, default=200)
    ap.add_argument("--max-per-parent", type=int, default=3)
    args = ap.parse_args()
    generate(args.seed_csv, args.models_dir, args.out_dir, args.max_parents, args.max_per_parent)
