"""2D molecular structure depiction via RDKit."""

from __future__ import annotations

import io
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw


def smiles_to_png(smiles: str, size: int = 320) -> Optional[bytes]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    img = Draw.MolToImage(mol, size=(size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
