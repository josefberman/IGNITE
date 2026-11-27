"""Rule-based modifier: a flat list of reagents and their property effects.

Each reagent's four effects (density, det_velocity, det_pressure, hf_solid)
match the primitive delta in ``reaction_primitives.py`` for the same reagent id.
Every reagent lists all four effects as ``raise``, ``lower``, or ``neutral``.
The rule-based path does not consult reaction families.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.app.domain.models import PROPERTY_KEYS
from backend.app.services.chemistry import sanitize_reaction_product
from ml.planner_gap import PROPERTY_SCALES, RELATIVE_THRESHOLDS

RAISE = "raise"
LOWER = "lower"
NEUTRAL = "neutral"
EFFECT_SIGN: Dict[str, int] = {RAISE: 1, LOWER: -1, NEUTRAL: 0}


@dataclass(frozen=True)
class Reagent:
    reagent_id: str
    effects: Tuple[Tuple[str, str], ...]  # ((property, raise|lower|neutral), ...)

    def effect(self, key: str) -> str:
        return dict(self.effects)[key]

    def signs(self) -> Tuple[int, int, int, int]:
        return tuple(EFFECT_SIGN[self.effect(k)] for k in PROPERTY_KEYS)  # type: ignore[return-value]


def _r(reagent_id: str, density: str, det_velocity: str, det_pressure: str, hf_solid: str) -> Reagent:
    return Reagent(
        reagent_id,
        (
            ("density", density),
            ("det_velocity", det_velocity),
            ("det_pressure", det_pressure),
            ("hf_solid", hf_solid),
        ),
    )


# (density, det_velocity, det_pressure, hf_solid)
REAGENTS: List[Reagent] = [
    _r("nitrate_arom_CH", RAISE, RAISE, RAISE, NEUTRAL),
    _r("nitrate_alkyl_CH", RAISE, RAISE, RAISE, NEUTRAL),
    _r("nitramine_NH", RAISE, RAISE, RAISE, RAISE),
    _r("methyl_to_nitromethyl", NEUTRAL, RAISE, RAISE, NEUTRAL),
    _r("halogen_Br_to_nitro", LOWER, RAISE, RAISE, RAISE),
    _r("nitrate_ester_OH", RAISE, RAISE, NEUTRAL, NEUTRAL),
    _r("trinitromethyl", RAISE, RAISE, NEUTRAL, RAISE),
    _r("nitroso_arom", NEUTRAL, LOWER, RAISE, RAISE),
    _r("azide_arom", NEUTRAL, NEUTRAL, LOWER, RAISE),
    _r("azide_methyl", NEUTRAL, RAISE, LOWER, RAISE),
    _r("amino_to_azide", LOWER, NEUTRAL, NEUTRAL, RAISE),
    _r("tetrazole_CH", RAISE, NEUTRAL, RAISE, RAISE),
    _r("triazole_CH", NEUTRAL, NEUTRAL, RAISE, RAISE),
    _r("pyrazole_CH", NEUTRAL, NEUTRAL, NEUTRAL, RAISE),
    _r("nitrile_to_tetrazole", RAISE, RAISE, RAISE, NEUTRAL),
    _r("n_oxide", NEUTRAL, RAISE, RAISE, RAISE),
    _r("hydroxyl_CH", RAISE, LOWER, RAISE, LOWER),
    _r("methoxy_CH", RAISE, NEUTRAL, LOWER, NEUTRAL),
    _r("carboxyl_CH", NEUTRAL, NEUTRAL, RAISE, LOWER),
    _r("methylester_CH", RAISE, NEUTRAL, LOWER, LOWER),
    _r("amino_to_nitro", NEUTRAL, RAISE, RAISE, LOWER),
    _r("hydroperoxy_CH", RAISE, NEUTRAL, NEUTRAL, RAISE),
    _r("epoxidize_CC", RAISE, NEUTRAL, RAISE, NEUTRAL),
    _r("formyl_CH", NEUTRAL, LOWER, NEUTRAL, LOWER),
    _r("cyano_CH", RAISE, NEUTRAL, RAISE, RAISE),
    _r("hydrazino_CH", RAISE, NEUTRAL, LOWER, RAISE),
    _r("amino_CH", NEUTRAL, LOWER, LOWER, RAISE),
    _r("amide_CH", NEUTRAL, LOWER, RAISE, LOWER),
    _r("isocyanate_CH", NEUTRAL, RAISE, LOWER, NEUTRAL),
    _r("fluoro_arom_CH", RAISE, RAISE, RAISE, LOWER),
    _r("trifluoromethyl_CH", RAISE, RAISE, NEUTRAL, LOWER),
    _r("chloro_arom_CH", RAISE, RAISE, LOWER, LOWER),
    _r("bromo_arom_CH", RAISE, RAISE, NEUTRAL, NEUTRAL),
    _r("iodo_arom_CH", RAISE, NEUTRAL, NEUTRAL, NEUTRAL),
    _r("fluoro_alkyl", RAISE, NEUTRAL, RAISE, LOWER),
    _r("dehydrogenate_CC", NEUTRAL, RAISE, LOWER, RAISE),
    _r("alkyne_CC", NEUTRAL, RAISE, NEUTRAL, RAISE),
    _r("add_phenyl_CH", NEUTRAL, NEUTRAL, RAISE, NEUTRAL),
    _r("benzyl_CH", NEUTRAL, LOWER, RAISE, NEUTRAL),
    _r("remove_nitro_arom", LOWER, LOWER, LOWER, NEUTRAL),
    _r("remove_bromo_arom", LOWER, NEUTRAL, NEUTRAL, NEUTRAL),
    _r("remove_azide_arom", NEUTRAL, NEUTRAL, NEUTRAL, LOWER),
    _r("nitro_to_methyl_arom", LOWER, LOWER, LOWER, LOWER),
    _r("nitramine_to_amine", NEUTRAL, NEUTRAL, LOWER, LOWER),
    _r("methyl_to_H", LOWER, NEUTRAL, LOWER, RAISE),
    _r("hydrogenate_CC", NEUTRAL, LOWER, RAISE, LOWER),
    _r("nitro_to_amino", NEUTRAL, LOWER, LOWER, RAISE),
    _r("azide_to_amino", LOWER, NEUTRAL, RAISE, LOWER),
    _r("methyl_CH", LOWER, LOWER, NEUTRAL, NEUTRAL),
    _r("ethyl_CH", LOWER, NEUTRAL, LOWER, NEUTRAL),
    _r("hydroxymethyl_CH", NEUTRAL, LOWER, NEUTRAL, LOWER),
    _r("aromatize_diene", NEUTRAL, RAISE, NEUTRAL, LOWER),
    _r("nitramine_to_nitrosamine", LOWER, LOWER, LOWER, RAISE),
    _r("aliphatic_nitro_to_amino", NEUTRAL, LOWER, LOWER, RAISE),
    _r("hold", NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL),
    _r("remove_chloro_arom", LOWER, LOWER, NEUTRAL, LOWER),
    _r("remove_fluoro_arom", LOWER, LOWER, NEUTRAL, RAISE),
    _r("remove_iodo_arom", LOWER, LOWER, RAISE, LOWER),
    _r("remove_trifluoromethyl", LOWER, LOWER, RAISE, NEUTRAL),
    _r("chloro_to_amino", LOWER, LOWER, RAISE, RAISE),
    _r("remove_hydroxy_arom", LOWER, NEUTRAL, LOWER, LOWER),
    _r("remove_methoxy_arom", LOWER, NEUTRAL, NEUTRAL, LOWER),
    _r("remove_cyano_arom", LOWER, NEUTRAL, RAISE, NEUTRAL),
    _r("bromo_to_amino", LOWER, NEUTRAL, RAISE, RAISE),
    _r("iodo_to_cyano", LOWER, RAISE, LOWER, LOWER),
    _r("bromo_to_hydroxy", LOWER, RAISE, LOWER, NEUTRAL),
    _r("chloro_to_azide", LOWER, RAISE, LOWER, RAISE),
    _r("iodo_to_fluoro", LOWER, RAISE, NEUTRAL, LOWER),
    _r("bromo_to_chloro", LOWER, RAISE, NEUTRAL, NEUTRAL),
    _r("bromo_to_fluoro", LOWER, RAISE, NEUTRAL, RAISE),
    _r("iodo_to_nitro", LOWER, RAISE, RAISE, LOWER),
    _r("chloro_to_nitro", LOWER, RAISE, RAISE, NEUTRAL),
    _r("acetyl_CH", NEUTRAL, LOWER, LOWER, LOWER),
    _r("tertbutyl_CH", NEUTRAL, LOWER, LOWER, NEUTRAL),
    _r("isopropyl_CH", NEUTRAL, LOWER, NEUTRAL, NEUTRAL),
    _r("vinyl_CH", NEUTRAL, LOWER, NEUTRAL, RAISE),
    _r("remove_phenyl", NEUTRAL, NEUTRAL, LOWER, NEUTRAL),
    _r("thiol_CH", NEUTRAL, RAISE, LOWER, LOWER),
    _r("alcohol_to_ketone", NEUTRAL, RAISE, NEUTRAL, NEUTRAL),
    _r("carboxyl_alkyl", RAISE, LOWER, LOWER, LOWER),
    _r("hydroxyl_alkyl", RAISE, LOWER, LOWER, NEUTRAL),
    _r("amino_alkyl", RAISE, LOWER, LOWER, RAISE),
    _r("cyano_alkyl", RAISE, LOWER, NEUTRAL, LOWER),
    _r("chloro_alkyl", RAISE, LOWER, NEUTRAL, NEUTRAL),
    _r("bromo_alkyl", RAISE, LOWER, NEUTRAL, RAISE),
    _r("iodo_alkyl", RAISE, LOWER, RAISE, NEUTRAL),
    _r("azide_alkyl", RAISE, LOWER, RAISE, RAISE),
    _r("formyl_alkyl", RAISE, NEUTRAL, NEUTRAL, LOWER),
    _r("methylsulfonyl_CH", RAISE, RAISE, LOWER, NEUTRAL),
    _r("allyl_CH", RAISE, RAISE, LOWER, RAISE),
]

REAGENTS_BY_ID: Dict[str, Reagent] = {r.reagent_id: r for r in REAGENTS}


def _deadband(key: str) -> float:
    return max(abs(PROPERTY_SCALES[key] * RELATIVE_THRESHOLDS[key]), 1e-6)


def desired_property_signs(parent: Mapping[str, float], target: Mapping[str, float]) -> Dict[str, int]:
    """``+1`` if the parent is below target, ``-1`` if above, ``0`` inside the deadband."""
    out: Dict[str, int] = {}
    for key in PROPERTY_KEYS:
        gap = float(target[key]) - float(parent.get(key, target[key]))
        out[key] = 0 if abs(gap) <= _deadband(key) else (1 if gap > 0 else -1)
    return out


def gap_weights(parent: Mapping[str, float], target: Mapping[str, float]) -> Dict[str, float]:
    """Per-property weight in ``[0, 1]`` from the scaled absolute gap (0 inside the deadband)."""
    out: Dict[str, float] = {}
    for key in PROPERTY_KEYS:
        gap = float(target[key]) - float(parent.get(key, target[key]))
        out[key] = 0.0 if abs(gap) <= _deadband(key) else min(1.0, abs(gap) / PROPERTY_SCALES[key])
    return out


def reagent_gap_score(reagent: Reagent, parent: Mapping[str, float], target: Mapping[str, float]) -> float:
    """Positive when the reagent's effects move properties toward the targets."""
    desired = desired_property_signs(parent, target)
    weights = gap_weights(parent, target)
    return sum(
        EFFECT_SIGN[reagent.effect(k)] * desired[k] * weights[k] for k in PROPERTY_KEYS
    )


def rank_reagents_for_gap(
    parent_props: Mapping[str, float],
    targets: Mapping[str, float],
    *,
    skip: Optional[Iterable[str]] = None,
    candidates: Optional[Sequence[str]] = None,
) -> List[str]:
    """Reagent ids, best gap-closing first. Ties break by reagent id."""
    skip_set = set(skip or ())
    pool = list(candidates) if candidates is not None else [r.reagent_id for r in REAGENTS]
    scored = [
        (reagent_gap_score(REAGENTS_BY_ID[rid], parent_props, targets), rid)
        for rid in pool
        if rid in REAGENTS_BY_ID and rid not in skip_set
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rid for _, rid in scored]


def reagent_smarts(reagent_id: str) -> List[str]:
    """SMARTS spellings for one reagent, from reaction_primitives."""
    from backend.app.services.reaction_primitives import list_primitives

    for prim in list_primitives():
        if prim["family_id"] == reagent_id:
            return list(prim.get("smarts_variants") or [])
    return []


def apply_reagent_random_sites(
    parent_smiles: str,
    reagent_id: str,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """All distinct (child_smiles, smarts) products of one reagent, in seeded random site order.

    Each distinct product is one valid reaction site. Products are first collected
    in a deterministic order, then shuffled with ``rng`` so the same seed gives
    the same site choice.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return []
    seen: Dict[str, str] = {}
    for smarts in reagent_smarts(reagent_id):
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            continue
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            continue
        for tup in products:
            for product in tup:
                product = sanitize_reaction_product(product)
                if product is None:
                    continue
                try:
                    smi = Chem.MolToSmiles(product, canonical=True)
                except Exception:
                    continue
                if smi and "." not in smi and smi != parent_smiles and smi not in seen:
                    seen[smi] = smarts
    sites = sorted(seen.items())
    rng.shuffle(sites)
    return sites
