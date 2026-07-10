"""Canonicalization, validity filtering, and chemistry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from rdkit import Chem
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
