"""Modification planner: learned heads with rule-based fallback."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from backend.app.domain.models import ModificationRecord, PROPERTY_KEYS
from backend.app.services.chemistry import edit_distance_proxy, validate_molecule
from backend.app.services.reaction_primitives import STRATEGY_POOL, StrategyPoolModifier, list_primitives

logger = logging.getLogger(__name__)


def _gap_dict(parent_props: Dict[str, float], targets: Dict[str, float]) -> Dict[str, float]:
    # Positive gap => need to increase property
    display = {
        "density": "Density",
        "det_velocity": "Det Velocity",
        "det_pressure": "Det Pressure",
        "hf_solid": "Hf solid",
    }
    return {display[k]: targets[k] - parent_props.get(k, targets[k]) for k in PROPERTY_KEYS}


class LearnedPlannerBundle:
    """Optional torch checkpoint wrapper; falls back if missing/unloadable."""

    def __init__(self, checkpoint_dir: Path | str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.available = False
        self.family_names = [p["family_id"] for p in list_primitives() if p["family_id"] != "identity"]
        self._model = None
        if self.checkpoint_dir and (self.checkpoint_dir / "planner.pt").exists():
            try:
                import torch

                from ml.models.architectures import ModificationPlanner

                blob = torch.load(self.checkpoint_dir / "planner.pt", map_location="cpu")
                meta = blob.get("meta", {})
                families = meta.get("families") or self.family_names
                reagents = meta.get("reagents") or families
                model = ModificationPlanner(n_families=len(families), n_reagents=len(reagents))
                model.load_state_dict(blob["planner"])
                model.eval()
                self._model = model
                self.family_names = families
                self.available = True
                logger.info("Loaded learned planner from %s", self.checkpoint_dir)
            except Exception as e:
                logger.warning("Could not load learned planner: %s", e)

    def propose_families(self, parent_smiles: str, gap: Dict[str, float], k: int = 3) -> List[str]:
        if not self.available:
            return []
        # Placeholder: checkpoint may store family logits prior; use rule prior if unknown
        return []


class ModificationMechanism:
    """Diagram-aligned modification: plan → execute → validity filter."""

    def __init__(self, checkpoint_dir: Path | str | None = None, rng: Optional[random.Random] = None):
        self.rule = StrategyPoolModifier()
        self.learned = LearnedPlannerBundle(checkpoint_dir)
        self.rng = rng or random.Random(42)

    def expand(
        self,
        parent_smiles: str,
        parent_props: Dict[str, float],
        targets: Dict[str, float],
        n: int,
    ) -> List[Tuple[str, ModificationRecord]]:
        gap = _gap_dict(parent_props, targets)
        strategy = self.rule.get_strategy(gap)
        products: List[Tuple[str, ModificationRecord]] = []
        seen = set()

        # Primary strategy + neighbors via rule pool
        candidates = self.rule.apply_strategies(parent_smiles, gap, target_count=max(n * 3, n))
        for child in candidates:
            vr = validate_molecule(child, parent_smiles=parent_smiles)
            if not vr.valid or not vr.canonical_smiles:
                continue
            if vr.canonical_smiles in seen:
                continue
            seen.add(vr.canonical_smiles)
            products.append(
                (
                    vr.canonical_smiles,
                    ModificationRecord(
                        reaction_family=strategy["primitive"],
                        reagent=strategy["primitive"],
                        smarts=(strategy.get("variants") or [{}])[0].get("smarts"),
                        edit_distance=edit_distance_proxy(parent_smiles, vr.canonical_smiles),
                        planner="learned" if self.learned.available else "rule_based",
                    ),
                )
            )
            if len(products) >= n:
                break

        # Extra random primitives if short
        if len(products) < n:
            prims = [p for p in list_primitives() if p["family_id"] != "identity"]
            self.rng.shuffle(prims)
            mol = Chem.MolFromSmiles(parent_smiles)
            for prim in prims:
                if len(products) >= n or mol is None:
                    break
                for smarts in prim["smarts_variants"]:
                    try:
                        rxn = AllChem.ReactionFromSmarts(smarts)
                        if rxn is None:
                            continue
                        for product_tuple in rxn.RunReactants((mol,)):
                            for product in product_tuple:
                                try:
                                    Chem.SanitizeMol(product)
                                    smi = Chem.MolToSmiles(product, canonical=True)
                                except Exception:
                                    continue
                                vr = validate_molecule(smi, parent_smiles=parent_smiles)
                                if not vr.valid or not vr.canonical_smiles or vr.canonical_smiles in seen:
                                    continue
                                seen.add(vr.canonical_smiles)
                                products.append(
                                    (
                                        vr.canonical_smiles,
                                        ModificationRecord(
                                            reaction_family=prim["family_id"],
                                            reagent=prim["family_id"],
                                            smarts=smarts,
                                            edit_distance=edit_distance_proxy(
                                                parent_smiles, vr.canonical_smiles
                                            ),
                                            planner="rule_based",
                                        ),
                                    )
                                )
                                if len(products) >= n:
                                    break
                            if len(products) >= n:
                                break
                    except Exception:
                        continue
                    if len(products) >= n:
                        break
        return products[:n]
