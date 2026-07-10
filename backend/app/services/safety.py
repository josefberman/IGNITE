"""Safety estimator: SAScore + unstable motif heuristics."""

from __future__ import annotations

import logging
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

from backend.app.domain.models import SafetyFinding

logger = logging.getLogger(__name__)

UNSTABLE_SMARTS = {
    "peroxide": "[O]-[O]",
    "long_n_chain": "[N]-[N]-[N]-[N]",
    "azide": "[N-]=[N+]=[N]",
    "diazo": "[C]=[N+]=[N-]",
    "nitro_peroxide": "[N+](=O)[O-].[O]-[O]",
}


def _normalize_sascore(x: float) -> float:
    t = (x - 1.0) / 9.0
    t = max(0.0, min(1.0, t))
    a, b, c = 0.20460496, 40.24350687, 4.21003837
    if t == 0:
        return 0.0
    if t == 1:
        return 1.0
    return (t ** a) / (t ** a + b * ((1 - t) ** c))


def calculate_sascore(smiles: str) -> float:
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        from BRSAScore import SAScorer  # type: ignore

        score, _ = SAScorer().calculateScore(smiles)
        return float(score)
    except Exception:
        return _simple_sascore(smiles)


def _simple_sascore(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    score = 2.0
    rings = Descriptors.RingCount(mol)
    if rings > 4:
        score += (rings - 4) * 0.4
    score += min(Descriptors.NumRotatableBonds(mol) * 0.15, 1.5)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > 30:
        score += min((n_atoms - 30) * 0.05, 1.5)
    hetero = Descriptors.NumHeteroatoms(mol)
    if hetero > 8:
        score += min((hetero - 8) * 0.1, 1.0)
    for pat in ("O-O", "N-N-N-N", "[N-]=[N+]=N"):
        if pat in smiles:
            score += 0.3
    return max(1.0, min(score, 10.0))


def find_unstable_motifs(smiles: str) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["invalid"]
    hits = []
    for name, smarts in UNSTABLE_SMARTS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            hits.append(name)
    return hits


def estimate_safety(smiles: str) -> SafetyFinding:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return SafetyFinding(score=1.0, is_acceptable=False, sascore=10.0, unstable_motifs=["invalid"])

    sas = calculate_sascore(smiles)
    motifs = find_unstable_motifs(smiles)
    base = _normalize_sascore(sas)
    motif_penalty = min(0.4, 0.1 * len(motifs))
    score = min(1.0, base + motif_penalty)
    notes: List[str] = []
    if motifs:
        notes.append(f"Unstable motifs detected: {', '.join(motifs)}")
    if sas > 7:
        notes.append("High synthetic accessibility score")
    return SafetyFinding(
        score=score,
        is_acceptable=score < 0.85 and "peroxide" not in motifs,
        sascore=sas,
        unstable_motifs=motifs,
        notes=notes,
    )
