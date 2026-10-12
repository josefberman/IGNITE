"""Modification planner: learned graph MP heads with rule-based fallback."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.app.domain.models import ModificationRecord, PROPERTY_KEYS
from backend.app.services.chemistry import edit_distance_proxy, validate_molecule
from backend.app.services.reaction_primitives import (
    StrategyPoolModifier,
    list_planner_vocab,
    list_primitives,
    list_reagents,
    matching_group_ids,
    matching_reaction_ids,
    planner_labels,
    reagent_id_for_smarts,
)
from ml.planner_gap import planner_input_from_parent_target

logger = logging.getLogger(__name__)

_GAP_DISPLAY = {
    "density": "Density",
    "det_velocity": "Det Velocity",
    "det_pressure": "Det Pressure",
    "hf_solid": "Hf solid",
}


def _gap_display(parent_props: Dict[str, float], targets: Dict[str, float]) -> Dict[str, float]:
    return {_GAP_DISPLAY[k]: targets[k] - parent_props.get(k, targets[k]) for k in PROPERTY_KEYS}


def _gap_vector(parent_props: Dict[str, float], targets: Dict[str, float]) -> List[float]:
    """Planner input: signed normalized (target − parent), aligned with training."""
    return planner_input_from_parent_target(parent_props, targets)


def _primitives_by_family() -> Dict[str, dict]:
    return {p["family_id"]: p for p in list_primitives() if p["family_id"] != "identity"}


def _run_smarts_on_parent(
    parent_smiles: str,
    mol: Chem.Mol,
    smarts: str,
    *,
    preferred_atoms: Optional[Sequence[int]] = None,
) -> List[str]:
    """Apply one SMARTS; prefer products from reactant matches overlapping preferred sites."""
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            return []
    except Exception:
        return []

    preferred = set(int(i) for i in (preferred_atoms or []))
    match_scores: List[int] = []
    try:
        tmpl = rxn.GetReactantTemplate(0)
        matches = mol.GetSubstructMatches(tmpl) if tmpl is not None else ()
        for match in matches:
            match_scores.append(sum(1 for a in match if a in preferred) if preferred else 0)
    except Exception:
        match_scores = []

    scored: List[Tuple[int, str]] = []
    try:
        products = rxn.RunReactants((mol,))
        for pi, product_tuple in enumerate(products):
            mscore = match_scores[pi] if pi < len(match_scores) else 0
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product, canonical=True)
                except Exception:
                    continue
                if not smi or "." in smi or smi == parent_smiles:
                    continue
                scored.append((mscore, smi))
    except Exception:
        return []

    scored.sort(key=lambda t: -t[0])
    out: List[str] = []
    seen: set[str] = set()
    for _, smi in scored:
        if smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out


class LearnedPlannerBundle:
    """Torch ModificationPlanner (+ optional delta predictor) for family / site / reagent proposals."""

    def __init__(self, checkpoint_dir: Path | str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.available = False
        groups, reagent_rows = list_planner_vocab()
        self.family_names = groups
        self.reagent_names = [r["reagent_id"] for r in reagent_rows]
        self.reagent_to_smarts = {r["reagent_id"]: r["smarts"] for r in list_reagents()}
        self.family_to_reagent_ids = {
            f: [r["reagent_id"] for r in reagent_rows if r["family_id"] == f] for f in self.family_names
        }
        self._model = None
        self._delta_model = None
        if self.checkpoint_dir and (self.checkpoint_dir / "planner.pt").exists():
            try:
                import torch

                from ml.models.architectures import (
                    DEFAULT_DROPOUT,
                    DEFAULT_HEADS,
                    DEFAULT_HIDDEN,
                    DEFAULT_LAYERS,
                    DeltaPropertyPredictor,
                    ModificationPlanner,
                )

                blob = torch.load(self.checkpoint_dir / "planner.pt", map_location="cpu", weights_only=False)
                meta = blob.get("meta", {})
                families = meta.get("families") or self.family_names
                reagents = meta.get("reagents") or self.reagent_names
                hidden = int(meta.get("hidden", DEFAULT_HIDDEN))
                layers = int(meta.get("layers", DEFAULT_LAYERS))
                heads = int(meta.get("heads", DEFAULT_HEADS))
                dropout = float(meta.get("dropout", DEFAULT_DROPOUT))
                enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout)
                model = ModificationPlanner(
                    n_families=len(families), n_reagents=len(reagents), **enc_kw
                )
                model.load_state_dict(blob["planner"])
                model.eval()
                self._model = model
                self.family_names = families
                self.reagent_names = reagents
                # Keep SMARTS map aligned with checkpoint reagent order when IDs match
                live = {r["reagent_id"]: r["smarts"] for r in list_reagents()}
                self.reagent_to_smarts = {rid: live[rid] for rid in reagents if rid in live}
                group_of = {
                    p["family_id"]: p["group"]
                    for p in list_primitives()
                    if p["family_id"] != "identity"
                }
                self.family_to_reagent_ids = {f: [] for f in families}
                for rid in reagents:
                    group = group_of.get(rid)
                    if group in self.family_to_reagent_ids:
                        self.family_to_reagent_ids[group].append(rid)
                if "delta" in blob:
                    delta_model = DeltaPropertyPredictor(**enc_kw)
                    delta_model.load_state_dict(blob["delta"])
                    delta_model.eval()
                    self._delta_model = delta_model
                self.available = True
                logger.info(
                    "Loaded learned planner from %s (%d families, %d reagents)",
                    self.checkpoint_dir,
                    len(families),
                    len(reagents),
                )
            except Exception as e:
                logger.warning("Could not load learned planner: %s", e)
                self.available = False
                self._model = None
                self._delta_model = None

    def _encode(self, parent_smiles: str, gap_vector: Sequence[float]):
        if not self.available or self._model is None:
            return None, None, None, None
        import torch

        from ml.models.architectures import mol_to_graph

        graph = mol_to_graph(parent_smiles)
        if graph is None:
            return None, None, None, None
        x, edge = graph
        if x.numel() == 0:
            return None, None, None, None
        delta = torch.tensor([list(gap_vector)], dtype=torch.float32)
        with torch.no_grad():
            z, _, atom_emb = self._model.encode(x.unsqueeze(0), delta, edge)
        return x, z, atom_emb, edge

    def propose_families(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        k: int = 3,
    ) -> List[str]:
        x, z, _, _ = self._encode(parent_smiles, gap_vector)
        if z is None or self._model is None:
            return []
        import torch

        with torch.no_grad():
            gap_t = torch.tensor([list(gap_vector)], dtype=torch.float32)
            logits = self._model.forward_family(z, gap_t).squeeze(0).clone()
            applicable = set(matching_group_ids(parent_smiles))
            if applicable:
                for i, name in enumerate(self.family_names):
                    if name not in applicable:
                        logits[i] = -1e9
                if not bool((logits > -1e8).any()):
                    logits = self._model.forward_family(z, gap_t).squeeze(0)
            k_eff = max(1, min(k, int((logits > -1e8).sum().item()), len(self.family_names)))
            if k_eff <= 0:
                return []
            idx = torch.topk(logits, k=k_eff).indices.tolist()
        return [self.family_names[i] for i in idx if i < len(self.family_names)]

    def propose_reagents(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        *,
        family_id: Optional[str] = None,
        k: int = 3,
        z=None,
        atom_emb=None,
    ) -> List[str]:
        """Top-k reagent_ids; optionally masked to a single reaction family."""
        if z is None or atom_emb is None:
            _, z, atom_emb, _ = self._encode(parent_smiles, gap_vector)
        if z is None or self._model is None:
            return []
        import torch

        fam_idx = None
        if family_id and family_id in self.family_names:
            fam_idx = torch.tensor([self.family_names.index(family_id)])

        with torch.no_grad():
            gap_t = torch.tensor([list(gap_vector)], dtype=torch.float32)
            logits = self._model.forward_reagent(
                z, family_idx=fam_idx, gap=gap_t, atom_emb=atom_emb
            ).squeeze(0).clone()
            matched = set(matching_reaction_ids(parent_smiles))
            if family_id:
                allowed = set(self.family_to_reagent_ids.get(family_id, []))
                for i, rid in enumerate(self.reagent_names):
                    if rid not in allowed or (matched and rid not in matched):
                        logits[i] = -1e9
                if matched and not bool((logits > -1e8).any()):
                    logits = self._model.forward_reagent(
                        z, family_idx=fam_idx, gap=gap_t, atom_emb=atom_emb
                    ).squeeze(0).clone()
                    for i, rid in enumerate(self.reagent_names):
                        if rid not in allowed:
                            logits[i] = -1e9
            k_eff = max(1, min(k, int((logits > -1e8).sum().item()), len(self.reagent_names)))
            if k_eff <= 0:
                return list(self.family_to_reagent_ids.get(family_id or "", []))[:k]
            idx = torch.topk(logits, k=k_eff).indices.tolist()
        return [self.reagent_names[i] for i in idx if i < len(self.reagent_names)]

    def propose_sites(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        top_m: int = 8,
        z=None,
        atom_emb=None,
    ) -> List[int]:
        if z is None or atom_emb is None:
            _, z, atom_emb, _ = self._encode(parent_smiles, gap_vector)
        if z is None or atom_emb is None or self._model is None:
            return []
        import torch

        with torch.no_grad():
            gap_t = torch.tensor([list(gap_vector)], dtype=torch.float32)
            scores = self._model.forward_sites(atom_emb, z, gap_t)
            m = max(1, min(top_m, scores.numel()))
            return torch.topk(scores, k=m).indices.tolist()

    def plan(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        *,
        k_families: int = 5,
        k_reagents: int = 3,
        top_m_sites: int = 8,
    ) -> Tuple[List[str], List[int], Dict[str, List[str]]]:
        """Encode once → families, sites, and per-family ranked reagents."""
        _, z, atom_emb, _ = self._encode(parent_smiles, gap_vector)
        if z is None:
            return [], [], {}
        families = self.propose_families(parent_smiles, gap_vector, k=k_families)
        sites = self.propose_sites(
            parent_smiles, gap_vector, top_m=top_m_sites, z=z, atom_emb=atom_emb
        )
        per_family: Dict[str, List[str]] = {}
        for fam in families:
            per_family[fam] = self.propose_reagents(
                parent_smiles, gap_vector, family_id=fam, k=k_reagents, z=z, atom_emb=atom_emb
            )
        return families, sites, per_family

    def rank_children_by_delta(
        self,
        parent_smiles: str,
        children: Sequence[str],
        gap_vector: Sequence[float],
    ) -> List[str]:
        if not self._delta_model or not children:
            return list(children)
        import torch

        from ml.models.architectures import mol_to_graph

        from ml.planner_gap import PROPERTY_SCALES

        parent_g = mol_to_graph(parent_smiles)
        if parent_g is None:
            return list(children)
        px, p_edge = parent_g
        gap = torch.tensor(list(gap_vector), dtype=torch.float32)
        # Predictor emits raw property deltas; the planner gap is already
        # scaled into roughly [-1, 1]. Compare in that same space.
        delta_scale = torch.tensor(
            [PROPERTY_SCALES[k] for k in PROPERTY_KEYS],
            dtype=torch.float32,
        )
        scored: List[Tuple[float, str]] = []
        with torch.no_grad():
            for smi in children:
                cg = mol_to_graph(smi)
                if cg is None:
                    scored.append((-1e9, smi))
                    continue
                cx, c_edge = cg
                pred = self._delta_model(
                    px.unsqueeze(0), cx.unsqueeze(0), p_edge, c_edge
                ).squeeze(0)
                pred_n = pred / delta_scale
                if float(gap.norm()) < 1e-8 or float(pred_n.norm()) < 1e-8:
                    score = -float(pred_n.norm())
                else:
                    score = float(torch.nn.functional.cosine_similarity(pred_n, gap, dim=0))
                scored.append((score, smi))
        scored.sort(key=lambda t: -t[0])
        return [s for _, s in scored]


class ModificationMechanism:
    """Diagram-aligned modification: plan → execute → validity filter."""

    def __init__(self, checkpoint_dir: Path | str | None = None, rng: Optional[random.Random] = None):
        self.rule = StrategyPoolModifier()
        self.learned = LearnedPlannerBundle(checkpoint_dir)
        self.rng = rng or random.Random(42)
        self._prims = _primitives_by_family()

    def _ordered_variants(
        self,
        family_id: str,
        preferred_reagent_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, str]]:
        """Return (reagent_id, smarts) ordered by learned reagent preference."""
        prim = self._prims.get(family_id)
        if not prim:
            return []
        pairs = list(zip(prim.get("reagent_ids") or [], prim.get("smarts_variants") or []))
        if not pairs:
            return []
        smarts_by_id = {r: s for r, s in pairs}
        if preferred_reagent_ids:
            pref = [rid for rid in preferred_reagent_ids if rid in smarts_by_id]
            rest = [p for p in pairs if p[0] not in pref]
            self.rng.shuffle(rest)
            return [(rid, smarts_by_id[rid]) for rid in pref] + rest
        self.rng.shuffle(pairs)
        return pairs

    def _apply_family(
        self,
        parent_smiles: str,
        mol: Chem.Mol,
        family_id: str,
        *,
        preferred_atoms: Optional[Sequence[int]] = None,
        preferred_reagent_ids: Optional[Sequence[str]] = None,
        max_products: int = 8,
    ) -> List[Tuple[str, str, str]]:
        """Return (child_smiles, reagent_id, smarts) for a reaction family."""
        out: List[Tuple[str, str, str]] = []
        seen: set[str] = set()
        for reagent_id, smarts in self._ordered_variants(family_id, preferred_reagent_ids):
            for smi in _run_smarts_on_parent(
                parent_smiles, mol, smarts, preferred_atoms=preferred_atoms
            ):
                if smi in seen:
                    continue
                seen.add(smi)
                out.append((smi, reagent_id, smarts))
                if len(out) >= max_products:
                    return out
        return out

    def _accept(
        self,
        parent_smiles: str,
        child_smi: str,
        *,
        family: str,
        reagent: Optional[str],
        smarts: Optional[str],
        planner: str,
        seen: set[str],
        products: List[Tuple[str, ModificationRecord]],
        n: int,
    ) -> bool:
        if len(products) >= n:
            return True
        vr = validate_molecule(child_smi, parent_smiles=parent_smiles)
        if not vr.valid or not vr.canonical_smiles or vr.canonical_smiles in seen:
            return False
        seen.add(vr.canonical_smiles)
        if reagent is None and smarts:
            reagent = reagent_id_for_smarts(family, smarts) or family
        products.append(
            (
                vr.canonical_smiles,
                ModificationRecord(
                    reaction_family=family,
                    reagent=reagent or family,
                    smarts=smarts,
                    edit_distance=edit_distance_proxy(parent_smiles, vr.canonical_smiles),
                    planner=planner,
                ),
            )
        )
        return len(products) >= n

    def expand(
        self,
        parent_smiles: str,
        parent_props: Dict[str, float],
        targets: Dict[str, float],
        n: int,
        *,
        forbidden: Optional[set[str]] = None,
        skip_reagents: Optional[set[str]] = None,
    ) -> List[Tuple[str, ModificationRecord]]:
        gap_disp = _gap_display(parent_props, targets)
        gap_vec = _gap_vector(parent_props, targets)
        products: List[Tuple[str, ModificationRecord]] = []
        seen: set[str] = set(forbidden or ())
        # Caller keeps this set. Reactions tried here are not proposed again, so
        # the next call on the same molecule uses the next family and reagent.
        tried = skip_reagents if skip_reagents is not None else set()
        mol = Chem.MolFromSmiles(parent_smiles)
        rule_target = max(n * 10, n + len(seen))
        family_cap = max(8, n * 3)

        # --- Learned path: family → reagent → site-biased SMARTS ---
        if self.learned.available and mol is not None:
            k_fam = max(3, min(8, n * 2))
            families, sites, reag_by_fam = self.learned.plan(
                parent_smiles,
                gap_vec,
                k_families=k_fam,
                k_reagents=max(2, min(5, n + 1)),
                top_m_sites=8,
            )
            raw_children: List[Tuple[str, str, str, str]] = []  # smi, family, reagent, smarts
            for group in families:
                for reaction_id in reag_by_fam.get(group, []):
                    if reaction_id in tried:
                        continue
                    tried.add(reaction_id)
                    for smi, _variant, smarts in self._apply_family(
                        parent_smiles,
                        mol,
                        reaction_id,
                        preferred_atoms=sites,
                        max_products=family_cap,
                    ):
                        raw_children.append((smi, group, reaction_id, smarts))
            order = self.learned.rank_children_by_delta(
                parent_smiles, [c[0] for c in raw_children], gap_vec
            )
            by_smi = {c[0]: c for c in raw_children}
            for smi in order:
                fam, reagent_id, smarts = by_smi[smi][1], by_smi[smi][2], by_smi[smi][3]
                if self._accept(
                    parent_smiles,
                    smi,
                    family=fam,
                    reagent=reagent_id,
                    smarts=smarts,
                    planner="learned",
                    seen=seen,
                    products=products,
                    n=n,
                ):
                    break

        # --- Rule-based strategy pool fallback / fill ---
        if len(products) < n:
            strategy = self.rule.get_strategy(gap_disp)
            specific = strategy.get("primitive", "strategy_pool")
            if specific in tried:
                specific = ""
            else:
                tried.add(specific)
            group, _ = planner_labels(specific, specific) if specific else ("", "")
            smarts = (strategy.get("variants") or [{}])[0].get("smarts")
            for child in ([] if not specific else self.rule.apply_strategies(
                parent_smiles, gap_disp, target_count=rule_target
            )):
                if self._accept(
                    parent_smiles,
                    child,
                    family=group,
                    reagent=specific,
                    smarts=smarts,
                    planner="rule_based",
                    seen=seen,
                    products=products,
                    n=n,
                ):
                    break

        # --- Exhaust remaining primitives if still short ---
        if len(products) < n and mol is not None:
            prims = list(self._prims.values())
            self.rng.shuffle(prims)
            for prim in prims:
                if len(products) >= n:
                    break
                specific = prim["family_id"]
                if specific in tried:
                    continue
                tried.add(specific)
                group = prim.get("group", specific)
                for smi, _variant, smarts in self._apply_family(
                    parent_smiles, mol, specific, max_products=family_cap
                ):
                    if self._accept(
                        parent_smiles,
                        smi,
                        family=group,
                        reagent=specific,
                        smarts=smarts,
                        planner="rule_based",
                        seen=seen,
                        products=products,
                        n=n,
                    ):
                        if len(products) >= n:
                            break
        return products[:n]
