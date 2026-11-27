"""Featurized pairs cache and the seeded random split used by ``--arch multitask``.

Featurizing a pair (parent graph, applicable reaction sites, which site gives
the labeled child) costs far more than a forward pass, and the result never
changes between epochs or trials. ``build_pair_cache`` does it once, in
parallel, and stores flat numpy arrays:

* per parent: atom features, typed edges, applicable-family and
  applicable-reagent masks;
* per pair: family / reagent index, the exact child−parent gap, parent and child
  properties, labeled site atoms, the reagent's candidate sites and which of
  them give the labeled child. Training batches replace the exact gap with a
  target the recorded child is closer to.

The labels, masks, and candidates are the ones ``PairDataset`` builds on the
fly; pairs whose parent or child SMILES does not parse are dropped (the legacy
loader silently substitutes the previous row).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
CACHE_VERSION = 2


def random_group_split(groups: np.ndarray, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> np.ndarray:
    """Split codes (0 train, 1 val, 2 test) for rows, keeping each group in one split.

    Groups are shuffled with ``seed`` and filled into train until it holds
    ``fractions[0]`` of the rows, then val until ``fractions[0] + fractions[1]``;
    the rest is test. With one row per group this is a plain random row split.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    cum = np.cumsum(counts[order]) / len(groups)
    code_of_group = np.full(len(uniq), 2, dtype=np.int8)
    b1, b2 = fractions[0], fractions[0] + fractions[1]
    # A group goes to the split its first row falls in.
    start = cum - counts[order] / len(groups)
    code_sorted = np.where(start < b1 - 1e-12, 0, np.where(start < b2 - 1e-12, 1, 2)).astype(np.int8)
    code_of_group[order] = code_sorted
    return code_of_group[inv]


def holdout_and_kfold(
    groups: np.ndarray, *, test_frac: float = 0.2, n_folds: int = 10, seed: int = 42
) -> tuple:
    """Parent-grouped 20% holdout plus k folds on the rest.

    Returns ``(test_mask, fold_id)``. ``fold_id`` is ``-1`` on holdout rows and
    ``0 .. n_folds-1`` on the rest. Each group is in exactly one piece.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    groups = np.asarray(groups)
    test_mask = random_group_split(groups, (1.0 - test_frac, 0.0, test_frac), seed) == 2
    fold_id = np.full(len(groups), -1, dtype=np.int8)
    rest = ~test_mask
    if not bool(rest.any()):
        return test_mask, fold_id
    rest_groups = groups[rest]
    uniq, inv, counts = np.unique(rest_groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed + 1).permutation(len(uniq))
    fold_size = np.zeros(n_folds, dtype=np.int64)
    fold_of = np.zeros(len(uniq), dtype=np.int8)
    for i in order:
        f = int(np.argmin(fold_size))
        fold_of[i] = f
        fold_size[f] += counts[i]
    fold_id[rest] = fold_of[inv]
    return test_mask, fold_id


def assign_split(df: pd.DataFrame, mode: str, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> np.ndarray:
    """Split codes per row. ``csv`` keeps the CSV's ``split`` column (scaffold split)."""
    if mode == "csv":
        return df["split"].map(SPLIT_CODES).fillna(2).to_numpy(dtype=np.int8)
    if mode == "random_parent":
        groups = pd.factorize(df["parent_smiles"].astype(str))[0]
    elif mode == "random_pair":
        groups = np.arange(len(df))
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    return random_group_split(groups, fractions, seed)


def _planner_gap(df: pd.DataFrame) -> np.ndarray:
    """``training_planner_input`` with jitter 0, vectorized."""
    from backend.app.domain.models import PROPERTY_KEYS
    from ml.planner_gap import PROPERTY_SCALES, RELATIVE_THRESHOLDS

    cols = []
    for k in PROPERTY_KEYS:
        g = df[f"child_{k}"].to_numpy(dtype=np.float64) - df[f"parent_{k}"].to_numpy(dtype=np.float64)
        scale = PROPERTY_SCALES[k]
        deadband = max(abs(scale * RELATIVE_THRESHOLDS[k]), 1e-6)
        v = np.clip(g / scale, -1.0, 1.0)
        v[np.abs(g) <= deadband] = 0.0
        cols.append(v)
    return np.stack(cols, axis=1).astype(np.float32)


_W: dict = {}


def _worker_init(smarts, fam_to_idx, reag_to_idx):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _W["smarts"] = smarts
    _W["fam"] = fam_to_idx
    _W["reag"] = reag_to_idx


def _featurize_parent(task):
    """One parent and its pairs -> graph, masks, candidates, per-pair positives."""
    from rdkit import Chem

    from backend.app.services.reaction_primitives import matching_group_ids, matching_reaction_ids
    from ml.models.architectures import mol_to_graph
    from ml.site_candidates import positive_mask, reaction_sites
    from ml.training.train_planner import _parse_sites

    parent, rows = task  # rows: list of (row_idx, child, raw_reagent, site_json)
    g = mol_to_graph(parent)
    mol = Chem.MolFromSmiles(parent)
    if g is None or mol is None or g[0].size(0) == 0:
        return None
    x, edge = g
    fam_allow = [_W["fam"][f] for f in matching_group_ids(parent) if f in _W["fam"]]
    reag_match = [_W["reag"][r] for r in matching_reaction_ids(parent) if r in _W["reag"]]
    rxn_cache: dict = {}
    keys: dict = {}
    key_cands: list = []
    pairs = []
    for row_idx, child, raw_reag, site_json in rows:
        if Chem.MolFromSmiles(child) is None:
            continue
        if raw_reag not in keys:
            cands = reaction_sites(mol, parent, _W["smarts"].get(raw_reag, []), rxn_cache)
            keys[raw_reag] = (len(key_cands), cands)
            key_cands.append([list(c.atoms) for c in cands])
        kid, cands = keys[raw_reag]
        sites = _parse_sites(site_json)
        pos = positive_mask(cands, child, sites)
        pairs.append((row_idx, kid, sites, [i for i, p in enumerate(pos) if p]))
    return parent, x.numpy(), edge.numpy(), fam_allow, reag_match, key_cands, pairs


def _csr(lists, dtype=np.int32):
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(v) for v in lists])
    flat = np.fromiter((a for v in lists for a in v), dtype=dtype, count=int(ptr[-1]))
    return ptr, flat


def build_pair_cache(data_csv: Path, cache_path: Path, workers: int = 8) -> Path:
    from backend.app.services.reaction_primitives import list_planner_vocab, list_primitives, planner_labels

    df = pd.read_csv(data_csv)
    if "site_atom_indices" not in df.columns:
        raise SystemExit("Dataset missing site_atom_indices; re-run ml.data_gen.generate_pairs")
    families, reagent_rows = list_planner_vocab()
    reagents = [r["reagent_id"] for r in reagent_rows]
    fam_to_idx = {f: i for i, f in enumerate(families)}
    reag_to_idx = {r: i for i, r in enumerate(reagents)}
    smarts = {
        p["family_id"]: list(p.get("smarts_variants") or [])
        for p in list_primitives()
        if p["family_id"] != "identity"
    }

    raw_reag = df["reagent_id"].where(df["reagent_id"].notna(), None)
    raw_reag = [None if v is None else str(v) for v in raw_reag]
    rf = df["reaction_family"].astype(str).tolist()
    lab_cache: dict = {}
    fam_idx = np.zeros(len(df), dtype=np.int16)
    reag_idx = np.zeros(len(df), dtype=np.int16)
    for i, (f, r) in enumerate(zip(rf, raw_reag)):
        key = (f, r)
        if key not in lab_cache:
            group, specific = planner_labels(f, r)
            lab_cache[key] = (fam_to_idx.get(group, 0), reag_to_idx.get(specific, 0))
        fam_idx[i], reag_idx[i] = lab_cache[key]
    gap = _planner_gap(df)
    from backend.app.domain.models import PROPERTY_KEYS

    parent_props = np.stack(
        [df[f"parent_{k}"].to_numpy(dtype=np.float64) for k in PROPERTY_KEYS], axis=1
    ).astype(np.float32)
    child_props = np.stack(
        [df[f"child_{k}"].to_numpy(dtype=np.float64) for k in PROPERTY_KEYS], axis=1
    ).astype(np.float32)

    parent_codes, parent_uniques = pd.factorize(df["parent_smiles"].astype(str))
    by_parent: dict = {}
    child = df["child_smiles"].astype(str).tolist()
    sites_col = df["site_atom_indices"].tolist()
    for i, pc in enumerate(parent_codes):
        by_parent.setdefault(pc, []).append((i, child[i], raw_reag[i] or "", sites_col[i]))
    tasks = [(parent_uniques[pc], rows) for pc, rows in by_parent.items()]
    print(f"featurizing {len(df)} pairs over {len(tasks)} parents with {workers} workers", flush=True)

    P_x, P_e, P_fam, P_reag = [], [], [], []
    pair_rows, pair_parent, pair_key, pair_sites, pair_pos = [], [], [], [], []
    key_cand_lists: list = []  # per key: list of candidate atom lists
    with mp.get_context("fork").Pool(
        workers, initializer=_worker_init, initargs=(smarts, fam_to_idx, reag_to_idx)
    ) as pool:
        for n, res in enumerate(pool.imap(_featurize_parent, tasks, chunksize=16), start=1):
            if n % 1000 == 0 or n == len(tasks):
                print(f"  parents {n}/{len(tasks)}", flush=True)
            if res is None:
                continue
            _parent, x, edge, fam_allow, reag_match, key_cands, pairs = res
            pid = len(P_x)
            P_x.append(x)
            P_e.append(edge)
            P_fam.append(fam_allow)
            P_reag.append(reag_match)
            key_base = len(key_cand_lists)
            key_cand_lists.extend(key_cands)
            for row_idx, kid, sites, pos in pairs:
                pair_rows.append(row_idx)
                pair_parent.append(pid)
                pair_key.append(key_base + kid)
                pair_sites.append(sites)
                pair_pos.append(pos)

    n_par = len(P_x)
    x_ptr = np.zeros(n_par + 1, dtype=np.int64)
    x_ptr[1:] = np.cumsum([a.shape[0] for a in P_x])
    e_ptr = np.zeros(n_par + 1, dtype=np.int64)
    e_ptr[1:] = np.cumsum([a.shape[1] for a in P_e])
    fam_allow = np.zeros((n_par, len(families)), dtype=bool)
    reag_match = np.zeros((n_par, len(reagents)), dtype=bool)
    for i, (fa, rm) in enumerate(zip(P_fam, P_reag)):
        fam_allow[i, fa] = True
        reag_match[i, rm] = True
    key_ptr = np.zeros(len(key_cand_lists) + 1, dtype=np.int64)
    key_ptr[1:] = np.cumsum([len(k) for k in key_cand_lists])
    cand_ptr, cand_atoms = _csr([c for k in key_cand_lists for c in k])
    site_ptr, site_atoms = _csr(pair_sites)
    pos_ptr, pos_idx = _csr(pair_pos)
    rows = np.asarray(pair_rows, dtype=np.int64)
    fam_pool = np.zeros((len(families), len(reagents)), dtype=bool)
    for r in reagent_rows:
        fam_pool[fam_to_idx[r["family_id"]], reag_to_idx[r["reagent_id"]]] = True

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        x=np.concatenate(P_x).astype(np.float32),
        x_ptr=x_ptr,
        edge=np.concatenate(P_e, axis=1).astype(np.int64),
        e_ptr=e_ptr,
        fam_allow=fam_allow,
        reag_match=reag_match,
        fam_pool=fam_pool,
        pair_row=rows,
        pair_parent=np.asarray(pair_parent, dtype=np.int32),
        pair_key=np.asarray(pair_key, dtype=np.int64),
        pair_fam=fam_idx[rows],
        pair_reag=reag_idx[rows],
        pair_gap=gap[rows],
        pair_parent_props=parent_props[rows],
        pair_child_props=child_props[rows],
        csv_split=df["split"].map(SPLIT_CODES).fillna(2).to_numpy(dtype=np.int8)[rows],
        row_parent_code=parent_codes.astype(np.int64),
        site_ptr=site_ptr,
        site_atoms=site_atoms,
        pos_ptr=pos_ptr,
        pos_idx=pos_idx,
        key_ptr=key_ptr,
        cand_ptr=cand_ptr,
        cand_atoms=cand_atoms,
        n_rows=np.int64(len(df)),
    )
    meta = {
        "version": CACHE_VERSION,
        "data_csv": str(data_csv),
        "data_size": data_csv.stat().st_size,
        "data_mtime": data_csv.stat().st_mtime,
        "families": families,
        "reagents": reagents,
        "n_rows": int(len(df)),
        "n_pairs": int(len(rows)),
        "n_parents": int(n_par),
        "n_dropped": int(len(df) - len(rows)),
    }
    Path(str(cache_path) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {cache_path}: {json.dumps({k: v for k, v in meta.items() if k not in ('families', 'reagents')})}", flush=True)
    return cache_path


def default_cache_path(data_csv: Path) -> Path:
    tag = hashlib.sha1(f"{data_csv.resolve()}:{data_csv.stat().st_size}:{CACHE_VERSION}".encode()).hexdigest()[:10]
    return Path("output") / "pair_cache" / f"{data_csv.stem}_{tag}.npz"


def ensure_pair_cache(data_csv: Path, cache_path: Path | None = None, workers: int = 8) -> Path:
    cache_path = cache_path or default_cache_path(data_csv)
    meta_p = Path(str(cache_path) + ".json")
    if cache_path.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if (
            meta.get("version") == CACHE_VERSION
            and meta.get("data_size") == data_csv.stat().st_size
            and abs(meta.get("data_mtime", 0) - data_csv.stat().st_mtime) < 1e-3
        ):
            return cache_path
    return build_pair_cache(data_csv, cache_path, workers)


class PairCache:
    """Loaded cache plus batch assembly into disjoint-union graph tensors."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.__dict__.update({k: d[k] for k in d.files})
        meta = json.loads(Path(str(path) + ".json").read_text())
        self.families = meta["families"]
        self.reagents = meta["reagents"]
        self.n_pairs = len(self.pair_row)
        self.n_atoms = np.diff(self.x_ptr)

    def split_codes(self, mode: str, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> np.ndarray:
        """Split code per cached pair; computed over all CSV rows so it matches ``assign_split``."""
        if mode == "csv":
            return self.csv_split
        n_rows = int(self.n_rows)
        if mode == "random_parent":
            groups = self.row_parent_code
        elif mode == "random_pair":
            groups = np.arange(n_rows)
        else:
            raise ValueError(f"unknown split mode {mode!r}")
        return random_group_split(groups, fractions, seed)[self.pair_row]

    def batch(self, idx: np.ndarray, *, train: bool = False, rng: np.random.Generator | None = None):
        import torch

        from ml.planner_gap import encode_property_gaps, sample_gap_reducing_targets

        xs, es, batch_vec = [], [], []
        site_t_idx, cand_seg, cand_atom_idx, cand_atom_cand, cand_pos = [], [], [], [], []
        off = 0
        n_cand = 0
        for b, i in enumerate(idx):
            p = int(self.pair_parent[i])
            x0, x1 = self.x_ptr[p], self.x_ptr[p + 1]
            n = int(x1 - x0)
            xs.append(self.x[x0:x1])
            e = self.edge[:, self.e_ptr[p] : self.e_ptr[p + 1]].copy()
            e[:2] += off
            es.append(e)
            batch_vec.append(np.full(n, b, dtype=np.int64))
            s = self.site_atoms[self.site_ptr[i] : self.site_ptr[i + 1]]
            s = s[(s >= 0) & (s < n)]
            site_t_idx.append(np.unique(s) + off)
            k = int(self.pair_key[i])
            c0, c1 = self.key_ptr[k], self.key_ptr[k + 1]
            pos = set(self.pos_idx[self.pos_ptr[i] : self.pos_ptr[i + 1]].tolist())
            for local, c in enumerate(range(c0, c1)):
                atoms = self.cand_atoms[self.cand_ptr[c] : self.cand_ptr[c + 1]]
                atoms = atoms[(atoms >= 0) & (atoms < n)]
                cand_seg.append(b)
                cand_pos.append(local in pos)
                cand_atom_idx.append(atoms + off)
                cand_atom_cand.append(np.full(len(atoms), n_cand, dtype=np.int64))
                n_cand += 1
            off += n
        par = self.pair_parent[idx]
        cat = lambda v, dt: torch.from_numpy(np.concatenate(v).astype(dt)) if v else torch.zeros(0, dtype=torch.long)
        if train and getattr(self, "pair_parent_props", None) is not None:
            gen = rng if rng is not None else np.random.default_rng()
            target = sample_gap_reducing_targets(
                self.pair_parent_props[idx], self.pair_child_props[idx], gen
            )
            gap = encode_property_gaps(target - self.pair_parent_props[idx])
        else:
            gap = self.pair_gap[idx]
        return {
            "x": torch.from_numpy(np.concatenate(xs)),
            "edge": torch.from_numpy(np.concatenate(es, axis=1)),
            "batch": torch.from_numpy(np.concatenate(batch_vec)),
            "n": len(idx),
            "gap": torch.from_numpy(np.ascontiguousarray(gap)),
            "fam": torch.from_numpy(self.pair_fam[idx].astype(np.int64)),
            "reag": torch.from_numpy(self.pair_reag[idx].astype(np.int64)),
            "fam_allow": torch.from_numpy(self.fam_allow[par]),
            "reag_match": torch.from_numpy(self.reag_match[par]),
            "site_idx": cat(site_t_idx, np.int64),
            "cand_seg": torch.tensor(cand_seg, dtype=torch.long),
            "cand_pos": torch.tensor(cand_pos, dtype=torch.bool),
            "cand_atom_idx": cat(cand_atom_idx, np.int64),
            "cand_atom_cand": cat(cand_atom_cand, np.int64),
            "n_cand": n_cand,
        }
