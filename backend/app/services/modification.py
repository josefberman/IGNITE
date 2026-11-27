"""Modification planner: learned graph MP heads or documented rule-based modifier."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.app.domain.models import PROPERTY_KEYS, ModificationMode, ModificationRecord
from backend.app.services.scoring import calculate_mape
from backend.app.services.chemistry import (
    edit_distance_proxy,
    sanitize_reaction_product,
    validate_molecule,
)
from backend.app.services.reagents import apply_reagent_random_sites, rank_reagents_for_gap
from backend.app.services.reaction_primitives import (
    StrategyPoolModifier,
    list_planner_vocab,
    list_primitives,
    list_reagents,
    matching_group_ids,
    matching_reaction_ids,
    reagent_id_for_smarts,
)
from ml.planner_gap import planner_input_from_parent_target

logger = logging.getLogger(__name__)


class PlannerUnavailableError(RuntimeError):
    """Raised when the learned planner is requested but torch/checkpoint cannot run it."""


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
                product = sanitize_reaction_product(product)
                if product is None:
                    continue
                try:
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
    """Torch ModificationPlanner for family, site, and reagent proposals."""

    def __init__(self, checkpoint_dir: Path | str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.available = False
        self.unavailable_reason = self._missing_checkpoint_reason()
        groups, reagent_rows = list_planner_vocab()
        self.family_names = groups
        self.reagent_names = [r["reagent_id"] for r in reagent_rows]
        self.reagent_to_smarts = {r["reagent_id"]: r["smarts"] for r in list_reagents()}
        self.family_to_reagent_ids = {
            f: [r["reagent_id"] for r in reagent_rows if r["family_id"] == f] for f in self.family_names
        }
        self._model = None
        ckpt = self.checkpoint_dir / "planner.pt" if self.checkpoint_dir else None
        if ckpt is not None and ckpt.exists():
            try:
                import torch
            except ImportError as e:
                self.unavailable_reason = (
                    "Learned planner is unavailable: torch is not installed. "
                    "Install with: pip install -e '.[torch]'"
                )
                logger.warning("%s (%s)", self.unavailable_reason, e)
                return
            try:
                from ml.models.architectures import build_planner

                blob = torch.load(self.checkpoint_dir / "planner.pt", map_location="cpu", weights_only=False)
                meta = blob.get("meta", {})
                families = meta.get("families") or self.family_names
                reagents = meta.get("reagents") or self.reagent_names
                # Legacy or multitask, from the architecture keys saved in the
                # checkpoint. Older checkpoints without those keys rebuild the
                # layout they were written with.
                model = build_planner(meta, len(families), len(reagents))
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
                self.available = True
                self.unavailable_reason = ""
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
                self.unavailable_reason = (
                    f"Learned planner is unavailable: failed to load checkpoint ({e})."
                )

    def _missing_checkpoint_reason(self) -> str:
        if not self.checkpoint_dir:
            return (
                "Learned planner is unavailable: no checkpoint directory configured "
                "(IGNITE_LEARNED_CHECKPOINT_DIR)."
            )
        ckpt = self.checkpoint_dir / "planner.pt"
        return f"Learned planner is unavailable: checkpoint not found at {ckpt}."

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

    @property
    def ranks_reagent_sites(self) -> bool:
        """True when the site head is conditioned on the reagent it places."""
        return bool(self._model is not None and getattr(self._model, "site_reagent", False))

    def encode(self, parent_smiles: str, gap_vector: Sequence[float]):
        """(z, atom_emb) for one parent and gap, or (None, None)."""
        _, z, atom_emb, _ = self._encode(parent_smiles, gap_vector)
        return z, atom_emb

    def rank_reagent_sites(
        self,
        parent_smiles: str,
        mol: Chem.Mol,
        gap_vector: Sequence[float],
        reagent_id: str,
        *,
        z=None,
        atom_emb=None,
    ) -> List[Tuple[str, str]]:
        """(child_smiles, smarts) for each distinct product of one reagent, best site first.

        A product's score is its best site's mean site-head logit; equivalent
        sites give the same product and appear once.
        """
        from backend.app.services.reagents import reagent_smarts
        from ml.site_candidates import candidate_logits, product_groups, reaction_sites

        cands = reaction_sites(mol, parent_smiles, reagent_smarts(reagent_id))
        if not cands:
            return []
        if z is None or atom_emb is None:
            z, atom_emb = self.encode(parent_smiles, gap_vector)
        if z is None or atom_emb is None or self._model is None:
            return [(c.child, c.smarts) for c in cands]
        import torch

        reag_idx = None
        if reagent_id in self.reagent_names:
            reag_idx = torch.tensor([self.reagent_names.index(reagent_id)])
        with torch.no_grad():
            gap_t = torch.tensor([list(gap_vector)], dtype=torch.float32)
            atom_logits = self._model.forward_sites(atom_emb, z, gap_t, reagent_idx=reag_idx)
            scores = candidate_logits(atom_logits, cands).tolist()
        ranked = []
        for group in product_groups(cands):
            best = max(group, key=lambda i: scores[i])
            ranked.append((scores[best], cands[best]))
        ranked.sort(key=lambda t: -t[0])
        return [(c.child, c.smarts) for _, c in ranked]

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


def planner_capability(checkpoint_dir: Path | str | None = None) -> dict:
    """Whether the learned planner can run, with a GUI-safe reason if not."""
    from backend.app.core.config import get_settings

    path = Path(checkpoint_dir) if checkpoint_dir else get_settings().learned_checkpoint_dir
    bundle = LearnedPlannerBundle(path)
    return {
        "planner_available": bundle.available,
        "planner_unavailable_reason": bundle.unavailable_reason or None,
        "rule_based_available": True,
        "checkpoint_dir": str(path),
    }


class ModificationMechanism:
    """Diagram-aligned modification: plan → execute → validity filter."""

    def __init__(
        self,
        checkpoint_dir: Path | str | None = None,
        rng: Optional[random.Random] = None,
        property_predictor=None,
    ):
        self.rule = StrategyPoolModifier()
        self.learned = LearnedPlannerBundle(checkpoint_dir)
        self.property_predictor = property_predictor
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

    def _predicted_mape(self, smiles: str, targets: Dict[str, float]) -> Optional[float]:
        """MAPE of one molecule against the targets, or None if it cannot be predicted."""
        if self.property_predictor is None:
            return None
        predicted = self.property_predictor.predict(smiles)
        if not predicted:
            return None
        values: Dict[str, float] = {}
        for key in PROPERTY_KEYS:
            est = predicted.get(key)
            if est is None:
                continue
            values[key] = float(est.value if hasattr(est, "value") else est)
        if len(values) != len(PROPERTY_KEYS):
            return None
        return calculate_mape(values, targets)

    def expand(
        self,
        parent_smiles: str,
        parent_props: Dict[str, float],
        targets: Dict[str, float],
        n: int,
        *,
        forbidden: Optional[set[str]] = None,
        skip_reagents: Optional[set[str]] = None,
        mode: ModificationMode = ModificationMode.RULE_BASED,
    ) -> List[Tuple[str, ModificationRecord]]:
        gap_vec = _gap_vector(parent_props, targets)
        products: List[Tuple[str, ModificationRecord]] = []
        seen: set[str] = set(forbidden or ())
        # Caller keeps this set. Reactions tried here are not proposed again, so
        # the next call on the same molecule uses the next family and reagent.
        tried = skip_reagents if skip_reagents is not None else set()
        mol = Chem.MolFromSmiles(parent_smiles)

        if mode == ModificationMode.PLANNER:
            if not self.learned.available:
                raise PlannerUnavailableError(
                    self.learned.unavailable_reason
                    or "Learned planner is unavailable: torch or the planner checkpoint is missing."
                )
            if mol is None:
                return products[:n]
            k_fam = max(3, min(8, n * 2))
            families, _sites, reag_by_fam = self.learned.plan(
                parent_smiles,
                gap_vec,
                k_families=k_fam,
                k_reagents=max(2, min(5, n + 1)),
            )
            # The three heads propose the reactions. With a property model,
            # return the products whose predicted properties are closer to the
            # target than the parent, closest first. Without one, keep the
            # heads' own order.
            z_enc, atom_emb = self.learned.encode(parent_smiles, gap_vec)
            proposals: List[Tuple[str, str, str, str]] = []
            for group in families:
                for reaction_id in reag_by_fam.get(group, []):
                    if reaction_id in tried:
                        continue
                    tried.add(reaction_id)
                    for smi, smarts in self.learned.rank_reagent_sites(
                        parent_smiles, mol, gap_vec, reaction_id, z=z_enc, atom_emb=atom_emb
                    ):
                        proposals.append((smi, group, reaction_id, smarts))
            if self.property_predictor is None:
                for smi, group, reaction_id, smarts in proposals:
                    if self._accept(
                        parent_smiles,
                        smi,
                        family=group,
                        reagent=reaction_id,
                        smarts=smarts,
                        planner="learned",
                        seen=seen,
                        products=products,
                        n=n,
                    ):
                        break
                return products[:n]
            parent_mape = calculate_mape(parent_props, targets)
            ranked: List[Tuple[float, str, str, str, str]] = []
            proposed: set[str] = set()
            for smi, group, reaction_id, smarts in proposals:
                vr = validate_molecule(smi, parent_smiles=parent_smiles)
                if not vr.valid or not vr.canonical_smiles or vr.canonical_smiles in seen:
                    continue
                if vr.canonical_smiles in proposed:
                    continue
                child_mape = self._predicted_mape(vr.canonical_smiles, targets)
                if child_mape is None or not child_mape < parent_mape:
                    continue
                proposed.add(vr.canonical_smiles)
                ranked.append((child_mape, vr.canonical_smiles, group, reaction_id, smarts))
            ranked.sort(key=lambda item: item[0])
            for _, smi, group, reaction_id, smarts in ranked:
                if self._accept(
                    parent_smiles,
                    smi,
                    family=group,
                    reagent=reaction_id,
                    smarts=smarts,
                    planner="learned",
                    seen=seen,
                    products=products,
                    n=n,
                ):
                    break
            return products[:n]

        # Rule-based: reagent level only. Rank reagents by whether their documented
        # effects close the gap, then try each reagent's valid sites in seeded
        # random order. No reaction family is consulted.
        if mol is None:
            return products[:n]
        for reagent_id in rank_reagents_for_gap(parent_props, targets, skip=tried):
            if len(products) >= n:
                break
            tried.add(reagent_id)
            for smi, smarts in apply_reagent_random_sites(parent_smiles, reagent_id, self.rng):
                if self._accept(
                    parent_smiles,
                    smi,
                    family=reagent_id,
                    reagent=reagent_id,
                    smarts=smarts,
                    planner="rule_based",
                    seen=seen,
                    products=products,
                    n=n,
                ):
                    break
                # One site per reagent per expansion; move to the next reagent.
                if products and products[-1][1].reagent == reagent_id:
                    break
        return products[:n]
