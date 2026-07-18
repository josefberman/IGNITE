"""Modification planner: learned Graph Transformer heads with rule-based fallback."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.app.domain.models import ModificationRecord, PROPERTY_KEYS
from backend.app.services.chemistry import edit_distance_proxy, validate_molecule
from backend.app.services.reaction_primitives import StrategyPoolModifier, list_primitives

logger = logging.getLogger(__name__)

# Display keys used by StrategyPoolModifier.get_strategy_key
_GAP_DISPLAY = {
    "density": "Density",
    "det_velocity": "Det Velocity",
    "det_pressure": "Det Pressure",
    "hf_solid": "Hf solid",
}


def _gap_display(parent_props: Dict[str, float], targets: Dict[str, float]) -> Dict[str, float]:
    return {_GAP_DISPLAY[k]: targets[k] - parent_props.get(k, targets[k]) for k in PROPERTY_KEYS}


def _gap_vector(parent_props: Dict[str, float], targets: Dict[str, float]) -> List[float]:
    """Target − parent in PROPERTY_KEYS order (matches train_planner deltas)."""
    return [float(targets[k] - parent_props.get(k, targets[k])) for k in PROPERTY_KEYS]


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
            # RunReactants enumerates matches in order; align score when possible
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
    """Torch ModificationPlanner (+ optional delta predictor) for family / site proposals."""

    def __init__(self, checkpoint_dir: Path | str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.available = False
        self.family_names = [p["family_id"] for p in list_primitives() if p["family_id"] != "identity"]
        self.reagent_names = list(self.family_names)
        self._model = None
        self._delta_model = None
        if self.checkpoint_dir and (self.checkpoint_dir / "planner.pt").exists():
            try:
                import torch

                from ml.models.architectures import DeltaPropertyPredictor, ModificationPlanner

                blob = torch.load(self.checkpoint_dir / "planner.pt", map_location="cpu", weights_only=False)
                meta = blob.get("meta", {})
                families = meta.get("families") or self.family_names
                reagents = meta.get("reagents") or families
                model = ModificationPlanner(n_families=len(families), n_reagents=len(reagents))
                model.load_state_dict(blob["planner"])
                model.eval()
                self._model = model
                self.family_names = families
                self.reagent_names = reagents
                if "delta" in blob:
                    delta_model = DeltaPropertyPredictor()
                    delta_model.load_state_dict(blob["delta"])
                    delta_model.eval()
                    self._delta_model = delta_model
                self.available = True
                logger.info("Loaded learned planner from %s (%d families)", self.checkpoint_dir, len(families))
            except Exception as e:
                logger.warning("Could not load learned planner: %s", e)
                self.available = False
                self._model = None
                self._delta_model = None

    def propose_families(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        k: int = 3,
    ) -> List[str]:
        if not self.available or self._model is None:
            return []
        import torch

        from ml.models.architectures import mol_to_graph

        graph = mol_to_graph(parent_smiles)
        if graph is None:
            return []
        x, _ = graph
        if x.numel() == 0:
            return []
        delta = torch.tensor([list(gap_vector)], dtype=torch.float32)
        with torch.no_grad():
            z, _ = self._model.encode(x.unsqueeze(0), delta)
            logits = self._model.forward_family(z).squeeze(0)
            k_eff = max(1, min(k, logits.numel(), len(self.family_names)))
            idx = torch.topk(logits, k=k_eff).indices.tolist()
        return [self.family_names[i] for i in idx if i < len(self.family_names)]

    def propose_sites(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        top_m: int = 8,
    ) -> List[int]:
        """Return top-m atom indices by site head score (empty if unavailable)."""
        if not self.available or self._model is None:
            return []
        import torch

        from ml.models.architectures import mol_to_graph

        graph = mol_to_graph(parent_smiles)
        if graph is None:
            return []
        x, _ = graph
        if x.numel() == 0:
            return []
        delta = torch.tensor([list(gap_vector)], dtype=torch.float32)
        with torch.no_grad():
            z, _ = self._model.encode(x.unsqueeze(0), delta)
            scores = self._model.forward_sites(x.unsqueeze(0), z)
            m = max(1, min(top_m, scores.numel()))
            return torch.topk(scores, k=m).indices.tolist()

    def rank_children_by_delta(
        self,
        parent_smiles: str,
        children: Sequence[str],
        gap_vector: Sequence[float],
    ) -> List[str]:
        """Reorder children by predicted Δy alignment with the target gap (higher better)."""
        if not self._delta_model or not children:
            return list(children)
        import torch

        from ml.models.architectures import mol_to_graph

        parent_g = mol_to_graph(parent_smiles)
        if parent_g is None:
            return list(children)
        px, _ = parent_g
        gap = torch.tensor(list(gap_vector), dtype=torch.float32)
        scored: List[Tuple[float, str]] = []
        with torch.no_grad():
            for smi in children:
                cg = mol_to_graph(smi)
                if cg is None:
                    scored.append((-1e9, smi))
                    continue
                cx, _ = cg
                pred = self._delta_model(px.unsqueeze(0), cx.unsqueeze(0)).squeeze(0)
                # Cosine similarity toward desired gap direction
                if float(gap.norm()) < 1e-8 or float(pred.norm()) < 1e-8:
                    score = -float(pred.norm())
                else:
                    score = float(torch.nn.functional.cosine_similarity(pred, gap, dim=0))
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

    def _apply_family(
        self,
        parent_smiles: str,
        mol: Chem.Mol,
        family_id: str,
        *,
        preferred_atoms: Optional[Sequence[int]] = None,
        max_products: int = 8,
    ) -> List[Tuple[str, str]]:
        """Return (child_smiles, smarts) for a reaction family."""
        prim = self._prims.get(family_id)
        if not prim:
            return []
        out: List[Tuple[str, str]] = []
        seen: set[str] = set()
        variants = list(prim.get("smarts_variants") or [])
        self.rng.shuffle(variants)
        for smarts in variants:
            for smi in _run_smarts_on_parent(
                parent_smiles, mol, smarts, preferred_atoms=preferred_atoms
            ):
                if smi in seen:
                    continue
                seen.add(smi)
                out.append((smi, smarts))
                if len(out) >= max_products:
                    return out
        return out

    def _accept(
        self,
        parent_smiles: str,
        child_smi: str,
        *,
        family: str,
        smarts: Optional[str],
        planner: str,
        seen: set[str],
        products: List[Tuple[str, ModificationRecord]],
        n: int,
    ) -> bool:
        """Validate and append; return True if `products` is full."""
        if len(products) >= n:
            return True
        vr = validate_molecule(child_smi, parent_smiles=parent_smiles)
        if not vr.valid or not vr.canonical_smiles or vr.canonical_smiles in seen:
            return False
        seen.add(vr.canonical_smiles)
        products.append(
            (
                vr.canonical_smiles,
                ModificationRecord(
                    reaction_family=family,
                    reagent=family,
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
    ) -> List[Tuple[str, ModificationRecord]]:
        gap_disp = _gap_display(parent_props, targets)
        gap_vec = _gap_vector(parent_props, targets)
        products: List[Tuple[str, ModificationRecord]] = []
        seen: set[str] = set()
        mol = Chem.MolFromSmiles(parent_smiles)

        # --- Learned path: top-k families (+ site prior) ---
        if self.learned.available and mol is not None:
            k_fam = max(3, min(8, n * 2))
            families = self.learned.propose_families(parent_smiles, gap_vec, k=k_fam)
            sites = self.learned.propose_sites(parent_smiles, gap_vec, top_m=8)
            raw_children: List[Tuple[str, str, str]] = []  # smi, family, smarts
            for fam in families:
                for smi, smarts in self._apply_family(
                    parent_smiles, mol, fam, preferred_atoms=sites, max_products=max(4, n)
                ):
                    raw_children.append((smi, fam, smarts))
            # Rank by delta predictor when available
            order = self.learned.rank_children_by_delta(
                parent_smiles, [c[0] for c in raw_children], gap_vec
            )
            by_smi = {c[0]: c for c in raw_children}
            for smi in order:
                fam, smarts = by_smi[smi][1], by_smi[smi][2]
                if self._accept(
                    parent_smiles,
                    smi,
                    family=fam,
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
            for child in self.rule.apply_strategies(parent_smiles, gap_disp, target_count=max(n * 3, n)):
                if self._accept(
                    parent_smiles,
                    child,
                    family=strategy.get("primitive", "strategy_pool"),
                    smarts=(strategy.get("variants") or [{}])[0].get("smarts"),
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
                fam = prim["family_id"]
                for smi, smarts in self._apply_family(parent_smiles, mol, fam, max_products=4):
                    if self._accept(
                        parent_smiles,
                        smi,
                        family=fam,
                        smarts=smarts,
                        planner="rule_based",
                        seen=seen,
                        products=products,
                        n=n,
                    ):
                        break
        return products[:n]
