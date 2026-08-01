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
from backend.app.services.reaction_primitives import (
    StrategyPoolModifier,
    list_primitives,
    list_reagents,
    reagent_id_for_smarts,
)

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
        self.family_names = [p["family_id"] for p in list_primitives() if p["family_id"] != "identity"]
        self.reagent_names = [r["reagent_id"] for r in list_reagents()]
        self.reagent_to_smarts = {r["reagent_id"]: r["smarts"] for r in list_reagents()}
        self.family_to_reagent_ids = {
            f: [r["reagent_id"] for r in list_reagents() if r["family_id"] == f] for f in self.family_names
        }
        self._model = None
        self._delta_model = None
        if self.checkpoint_dir and (self.checkpoint_dir / "planner.pt").exists():
            try:
                import torch

                from ml.models.architectures import (
                    DEFAULT_HIDDEN,
                    DeltaPropertyPredictor,
                    ModificationPlanner,
                )

                blob = torch.load(self.checkpoint_dir / "planner.pt", map_location="cpu", weights_only=False)
                meta = blob.get("meta", {})
                families = meta.get("families") or self.family_names
                reagents = meta.get("reagents") or self.reagent_names
                hidden = int(meta.get("hidden", DEFAULT_HIDDEN))
                model = ModificationPlanner(
                    n_families=len(families), n_reagents=len(reagents), hidden=hidden
                )
                model.load_state_dict(blob["planner"])
                model.eval()
                self._model = model
                self.family_names = families
                self.reagent_names = reagents
                # Keep SMARTS map aligned with checkpoint reagent order when IDs match
                live = {r["reagent_id"]: r["smarts"] for r in list_reagents()}
                self.reagent_to_smarts = {rid: live[rid] for rid in reagents if rid in live}
                self.family_to_reagent_ids = {
                    f: [rid for rid in reagents if rid.startswith(f + "__v")] for f in families
                }
                if "delta" in blob:
                    delta_model = DeltaPropertyPredictor(hidden=hidden)
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
            return None, None
        import torch

        from ml.models.architectures import mol_to_graph

        graph = mol_to_graph(parent_smiles)
        if graph is None:
            return None, None
        x, _ = graph
        if x.numel() == 0:
            return None, None
        delta = torch.tensor([list(gap_vector)], dtype=torch.float32)
        with torch.no_grad():
            z, _ = self._model.encode(x.unsqueeze(0), delta)
        return x, z

    def propose_families(
        self,
        parent_smiles: str,
        gap_vector: Sequence[float],
        k: int = 3,
    ) -> List[str]:
        x, z = self._encode(parent_smiles, gap_vector)
        if z is None or self._model is None:
            return []
        import torch

        with torch.no_grad():
            logits = self._model.forward_family(z).squeeze(0)
            k_eff = max(1, min(k, logits.numel(), len(self.family_names)))
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
    ) -> List[str]:
        """Top-k reagent_ids; optionally masked to a single reaction family."""
        if z is None:
            _, z = self._encode(parent_smiles, gap_vector)
        if z is None or self._model is None:
            return []
        import torch

        with torch.no_grad():
            logits = self._model.forward_reagent(z).squeeze(0).clone()
            if family_id:
                allowed = set(self.family_to_reagent_ids.get(family_id, []))
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
        x=None,
    ) -> List[int]:
        if z is None or x is None:
            x, z = self._encode(parent_smiles, gap_vector)
        if z is None or x is None or self._model is None:
            return []
        import torch

        with torch.no_grad():
            scores = self._model.forward_sites(x.unsqueeze(0), z)
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
        x, z = self._encode(parent_smiles, gap_vector)
        if z is None:
            return [], [], {}
        families = self.propose_families(parent_smiles, gap_vector, k=k_families)
        # Re-use z for sites/reagents (propose_families re-encodes; recompute once here)
        import torch

        from ml.models.architectures import mol_to_graph

        graph = mol_to_graph(parent_smiles)
        if graph is None or self._model is None:
            return families, [], {}
        x, _ = graph
        delta = torch.tensor([list(gap_vector)], dtype=torch.float32)
        with torch.no_grad():
            z, _ = self._model.encode(x.unsqueeze(0), delta)
        sites = self.propose_sites(parent_smiles, gap_vector, top_m=top_m_sites, z=z, x=x)
        per_family: Dict[str, List[str]] = {}
        for fam in families:
            per_family[fam] = self.propose_reagents(
                parent_smiles, gap_vector, family_id=fam, k=k_reagents, z=z
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
    ) -> List[Tuple[str, ModificationRecord]]:
        gap_disp = _gap_display(parent_props, targets)
        gap_vec = _gap_vector(parent_props, targets)
        products: List[Tuple[str, ModificationRecord]] = []
        seen: set[str] = set()
        mol = Chem.MolFromSmiles(parent_smiles)

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
            for fam in families:
                for smi, reagent_id, smarts in self._apply_family(
                    parent_smiles,
                    mol,
                    fam,
                    preferred_atoms=sites,
                    preferred_reagent_ids=reag_by_fam.get(fam),
                    max_products=max(4, n),
                ):
                    raw_children.append((smi, fam, reagent_id, smarts))
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
            fam = strategy.get("primitive", "strategy_pool")
            smarts = (strategy.get("variants") or [{}])[0].get("smarts")
            for child in self.rule.apply_strategies(parent_smiles, gap_disp, target_count=max(n * 3, n)):
                if self._accept(
                    parent_smiles,
                    child,
                    family=fam,
                    reagent=reagent_id_for_smarts(fam, smarts) if smarts else fam,
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
                fam = prim["family_id"]
                for smi, reagent_id, smarts in self._apply_family(parent_smiles, mol, fam, max_products=4):
                    if self._accept(
                        parent_smiles,
                        smi,
                        family=fam,
                        reagent=reagent_id,
                        smarts=smarts,
                        planner="rule_based",
                        seen=seen,
                        products=products,
                        n=n,
                    ):
                        break
        return products[:n]
