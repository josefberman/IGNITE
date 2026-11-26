"""Applicable reaction sites for one parent and reagent.

A site is the set of parent atoms that react for one template match. Sites that
give the same product (for example symmetric ring positions) are equivalent:
training treats all of them as correct, and evaluation counts any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.app.services.chemistry import sanitize_reaction_product


@dataclass(frozen=True)
class SiteCandidate:
    atoms: Tuple[int, ...]
    child: str
    smarts: str


def reacting_atoms_for_match(rxn, match: Sequence[int]) -> List[int]:
    """Parent atoms that react in one template match; the whole match if RDKit reports none."""
    reacting: List[int] = []
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
    return sorted({int(a) for a in match})


def reacting_atom_indices(mol: Chem.Mol, rxn, product_index: int) -> List[int]:
    """Map RDKit reacting template atoms onto parent atom indices for one RunReactants hit."""
    try:
        tmpl = rxn.GetReactantTemplate(0)
        if tmpl is None:
            return []
        matches = mol.GetSubstructMatches(tmpl)
        if product_index >= len(matches):
            return []
        return reacting_atoms_for_match(rxn, matches[product_index])
    except Exception:
        return []


def _identity_smarts(smarts: str) -> bool:
    left, sep, right = smarts.partition(">>")
    return bool(sep) and left == right


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def reaction_sites(
    mol: Chem.Mol,
    parent_smiles: str,
    smarts_list: Sequence[str],
    rxn_cache: Optional[Dict[str, object]] = None,
) -> List[SiteCandidate]:
    """Distinct reacting-atom sets with a valid product, in template and match order.

    Atom sets follow the same product-to-match pairing as ``generate_pairs``, so
    a stored ``site_atom_indices`` label is one of the returned ``atoms``.
    """
    cache = rxn_cache if rxn_cache is not None else {}
    out: List[SiteCandidate] = []
    seen: set[Tuple[int, ...]] = set()
    for smarts in smarts_list:
        if smarts not in cache:
            try:
                cache[smarts] = AllChem.ReactionFromSmarts(smarts)
            except Exception:
                cache[smarts] = None
        rxn = cache[smarts]
        if rxn is None:
            continue
        try:
            products = rxn.RunReactants((mol,))
            tmpl = rxn.GetReactantTemplate(0)
            matches = mol.GetSubstructMatches(tmpl) if tmpl is not None else ()
        except Exception:
            continue
        for pi, product_tuple in enumerate(products):
            if pi >= len(matches):
                break
            atoms = tuple(reacting_atoms_for_match(rxn, matches[pi]))
            if not atoms or atoms in seen:
                continue
            for product in product_tuple:
                product = sanitize_reaction_product(product)
                if product is None:
                    continue
                try:
                    child = _canonical(Chem.MolToSmiles(product, canonical=True))
                except Exception:
                    continue
                if not child or "." in child:
                    continue
                # A real reagent that regenerates the parent is not a site.
                # An identity template (same SMARTS on both sides) is the hold reaction.
                if child == parent_smiles and not _identity_smarts(smarts):
                    continue
                seen.add(atoms)
                out.append(SiteCandidate(atoms=atoms, child=child, smarts=smarts))
                break
    return out


def product_groups(candidates: Sequence[SiteCandidate]) -> List[List[int]]:
    """Candidate indices grouped by product, ordered by each group's lowest atom index."""
    by_child: Dict[str, List[int]] = {}
    for i, cand in enumerate(candidates):
        by_child.setdefault(cand.child, []).append(i)
    groups = list(by_child.values())
    groups.sort(key=lambda idx: min(min(candidates[i].atoms) for i in idx))
    return groups


def positive_mask(
    candidates: Sequence[SiteCandidate],
    child_smiles: Optional[str],
    label_atoms: Sequence[int] = (),
) -> List[bool]:
    """Candidates equivalent to the labeled site: same product, else the same atoms."""
    child = _canonical(child_smiles) if child_smiles else None
    mask = [bool(child) and cand.child == child for cand in candidates]
    if not any(mask) and label_atoms:
        label = tuple(sorted(int(a) for a in label_atoms))
        mask = [cand.atoms == label for cand in candidates]
    return mask


def candidate_logits(atom_logits, candidates: Sequence[SiteCandidate]):
    """Mean site-head logit over each candidate's reacting atoms."""
    import torch

    n_atoms = int(atom_logits.shape[-1])
    scores = []
    for cand in candidates:
        idx = [a for a in cand.atoms if 0 <= a < n_atoms]
        if idx:
            scores.append(atom_logits[idx].mean())
        else:
            scores.append(atom_logits.new_tensor(-1e9))
    return torch.stack(scores)
