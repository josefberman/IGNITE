"""Canonicalization, validity filtering, and chemistry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Set, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors

ALLOWED_ATOMIC_NUMS: Set[int] = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


@dataclass
class ValidityResult:
    valid: bool
    canonical_smiles: Optional[str] = None
    inchikey: Optional[str] = None
    reason: Optional[str] = None


def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _product_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    if not smiles or "." in smiles or Chem.MolFromSmiles(smiles) is None:
        return None
    return smiles


def _clear_aromatic_flags(mol: Chem.Mol) -> Chem.Mol:
    """Drop aromatic flags so RDKit can re-perceive them from the bond orders."""
    repaired = Chem.Mol(mol)
    for atom in repaired.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in repaired.GetBonds():
        bond.SetIsAromatic(False)
    return repaired


def _ring_nitrogen_missing_h(mol: Chem.Mol) -> List[int]:
    """Neutral divalent ring nitrogens with no hydrogen (pyrrole-like atoms missing H)."""
    found: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 0 or not atom.IsInRing():
            continue
        if atom.GetDegree() != 2 or atom.GetNumExplicitHs() != 0:
            continue
        try:
            if atom.GetTotalNumHs() != 0:
                continue
        except Exception:
            continue
        found.append(atom.GetIdx())
    return found


def _with_explicit_h(mol: Chem.Mol, atom_idxs: Sequence[int]) -> Chem.Mol:
    repaired = Chem.Mol(mol)
    for idx in atom_idxs:
        repaired.GetAtomWithIdx(idx).SetNumExplicitHs(1)
    return repaired


def _n_oxide_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        for neighbor in atom.GetNeighbors():
            if (
                neighbor.GetAtomicNum() == 8
                and neighbor.GetFormalCharge() == -1
                and neighbor.GetDegree() == 1
            ):
                pairs.append((atom.GetIdx(), neighbor.GetIdx()))
    return pairs


def _as_n_hydroxy(mol: Chem.Mol, nitrogen_idx: int, oxygen_idx: int) -> Chem.Mol:
    """Charge-separated N-oxide to neutral N–OH, which a 5-membered ring can kekulize."""
    repaired = Chem.Mol(mol)
    repaired.GetAtomWithIdx(nitrogen_idx).SetFormalCharge(0)
    oxygen = repaired.GetAtomWithIdx(oxygen_idx)
    oxygen.SetFormalCharge(0)
    oxygen.SetNumExplicitHs(1)
    return repaired


def _valence_atom_index(exc: Exception) -> Optional[int]:
    text = str(exc)
    marker = "atom #"
    if marker not in text:
        return None
    digits = []
    for char in text.split(marker, 1)[1].strip():
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    return int("".join(digits)) if digits else None


def _clear_impossible_chirality(mol: Chem.Mol) -> None:
    """A double bond left over from the reactant cannot keep a tetrahedral tag."""
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        multiple = any(
            bond.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC)
            for bond in atom.GetBonds()
        )
        if multiple or atom.GetDegree() > 4:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)


def _shed_excess_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Remove explicit hydrogens that push an atom over its valence.

    Reaction SMARTS keeps the reactant's hydrogens when it raises a bond order
    (``C–C`` to ``C=C``) or writes ``[CH2]`` onto a carbon that already has three
    neighbors. RDKit then reports valence 5. Stereocenters that are not the
    failing atom are left alone.
    """
    current = Chem.Mol(mol)
    _clear_impossible_chirality(current)
    budget = sum(atom.GetNumExplicitHs() for atom in current.GetAtoms()) + 1
    for _ in range(budget):
        trial = Chem.Mol(current)
        try:
            Chem.SanitizeMol(trial)
            return trial
        except Chem.AtomValenceException as exc:
            idx = _valence_atom_index(exc)
            if idx is None:
                return current
            atom = current.GetAtomWithIdx(idx)
            if atom.GetNumExplicitHs() <= 0:
                return current
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        except Exception:
            return current
    return current


def _drop_extra_n_oxide(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Remove a second ``O-`` from an aromatic nitrogen that is already an N-oxide.

    ``[n+]([O-])[O-]`` is not a kekulizable pyridine. One oxide oxygen is kept.
    """
    editable = Chem.RWMol(mol)
    remove: List[int] = []
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 7 or not atom.GetIsAromatic():
            continue
        oxides = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 8
            and neighbor.GetFormalCharge() == -1
            and neighbor.GetDegree() == 1
        ]
        if len(oxides) >= 2:
            remove.extend(oxides[1:])
    if not remove:
        return None
    for idx in sorted(set(remove), reverse=True):
        editable.RemoveAtom(idx)
    return editable.GetMol()


def _repaired_smiles(mol: Chem.Mol, *, drop_extra_oxide: bool) -> Optional[str]:
    prepared = _shed_excess_hydrogens(mol)
    if drop_extra_oxide:
        dropped = _drop_extra_n_oxide(prepared)
        if dropped is not None:
            smiles = _repaired_smiles(dropped, drop_extra_oxide=False)
            if smiles is not None:
                return smiles
    smiles = _product_smiles(Chem.Mol(prepared))
    if smiles is None:
        smiles = _product_smiles(_clear_aromatic_flags(prepared))
    if smiles is None:
        nitrogens = _ring_nitrogen_missing_h(prepared)[:8]
        for added in (1, 2):
            if len(nitrogens) < added:
                break
            for combo in combinations(nitrogens, added):
                smiles = _product_smiles(_with_explicit_h(prepared, combo))
                if smiles is not None:
                    break
            if smiles is not None:
                break
    if smiles is None:
        flagged = _clear_aromatic_flags(prepared)
        for idx in _ring_nitrogen_missing_h(flagged)[:8]:
            smiles = _product_smiles(_with_explicit_h(flagged, (idx,)))
            if smiles is not None:
                break
    if smiles is None:
        for nitrogen_idx, oxygen_idx in _n_oxide_bonds(prepared)[:6]:
            hydroxy = _as_n_hydroxy(prepared, nitrogen_idx, oxygen_idx)
            smiles = _product_smiles(hydroxy)
            if smiles is not None:
                break
            for idx in _ring_nitrogen_missing_h(hydroxy)[:4]:
                smiles = _product_smiles(_with_explicit_h(hydroxy, (idx,)))
                if smiles is not None:
                    break
            if smiles is not None:
                break
    return smiles


def sanitize_reaction_product(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Sanitize a SMARTS product.

    A product that already sanitizes is returned unchanged. Otherwise the
    repair, in order, is: drop explicit hydrogens that make valence 5 and
    clear a tetrahedral tag on a new double bond; drop a second N-oxide
    oxygen from an aromatic nitrogen; clear aromatic flags; add a hydrogen
    to a neutral ring nitrogen; write a non-kekulizable N-oxide as N–OH.
    """
    if mol is None:
        return None
    with rdBase.BlockLogs():
        smiles = _product_smiles(Chem.Mol(mol))
        if smiles is None:
            smiles = _repaired_smiles(mol, drop_extra_oxide=True)
    if smiles is None:
        return None
    return Chem.MolFromSmiles(smiles)


def inchikey_of(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def morgan_fp_bits(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(smiles_a: str, smiles_b: str) -> Optional[float]:
    from rdkit import DataStructs

    fa, fb = morgan_fp_bits(smiles_a), morgan_fp_bits(smiles_b)
    if fa is None or fb is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def validate_molecule(
    smiles: str,
    parent_smiles: Optional[str] = None,
    require_single_component: bool = True,
) -> ValidityResult:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ValidityResult(False, reason="parse_failed")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ValidityResult(False, reason="sanitize_failed")

    if require_single_component and len(Chem.GetMolFrags(mol)) != 1:
        return ValidityResult(False, reason="multi_component")

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS:
            return ValidityResult(False, reason=f"disallowed_element_{atom.GetSymbol()}")

    can = Chem.MolToSmiles(mol, canonical=True)
    if parent_smiles:
        parent_can = canonicalize(parent_smiles)
        if parent_can and can == parent_can:
            return ValidityResult(False, reason="unchanged")

    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        key = None

    return ValidityResult(True, canonical_smiles=can, inchikey=key)


def murcko_scaffold(smiles: str) -> Optional[str]:
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol else 0


def edit_distance_proxy(parent: str, child: str) -> int:
    """Atom-count delta as a minimal-edit proxy."""
    return abs(atom_count(child) - atom_count(parent))
