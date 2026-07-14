"""Resolve SMILES to IUPAC and commercial/common names (exact molecule only)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import quote

import httpx

from backend.app.services.chemistry import canonicalize, inchikey_of

logger = logging.getLogger(__name__)


@dataclass
class MoleculeIdentity:
    smiles: str
    canonical_smiles: str
    inchikey: Optional[str] = None
    iupac_name: Optional[str] = None
    commercial_names: List[str] = field(default_factory=list)
    cid: Optional[int] = None
    query_terms: List[str] = field(default_factory=list)


class MoleculeIdentityResolver:
    """PubChem PUG REST exact-structure name lookup + local cache."""

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self._cache: Dict[str, MoleculeIdentity] = {}

    def resolve(self, smiles: str) -> MoleculeIdentity:
        can = canonicalize(smiles) or smiles
        if can in self._cache:
            return self._cache[can]

        key = inchikey_of(can)
        identity = MoleculeIdentity(
            smiles=smiles,
            canonical_smiles=can,
            inchikey=key,
        )
        terms = [can]
        if key:
            terms.append(key)

        try:
            cid = self._cid_from_smiles(can)
            if cid is not None:
                identity.cid = cid
                props = self._compound_props(cid)
                identity.iupac_name = props.get("iupac")
                identity.commercial_names = props.get("synonyms", [])
                if identity.iupac_name:
                    terms.append(identity.iupac_name)
                terms.extend(identity.commercial_names[:8])
        except Exception as e:
            logger.debug("Identity resolve failed for %s: %s", can[:40], e)

        # Deduplicate terms while preserving order
        seen = set()
        unique_terms = []
        for t in terms:
            t_norm = (t or "").strip()
            if not t_norm:
                continue
            low = t_norm.lower()
            if low in seen:
                continue
            seen.add(low)
            unique_terms.append(t_norm)
        identity.query_terms = unique_terms
        self._cache[can] = identity
        return identity

    def _cid_from_smiles(self, smiles: str) -> Optional[int]:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles, safe='')}/cids/JSON"
        r = httpx.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            return None
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        return int(cids[0]) if cids else None

    def _compound_props(self, cid: int) -> dict:
        out: dict = {"iupac": None, "synonyms": []}
        # IUPAC
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
        r = httpx.get(url, timeout=self.timeout)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                out["iupac"] = props[0].get("IUPACName")

        # Synonyms / commercial names
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        r = httpx.get(url, timeout=self.timeout)
        if r.status_code == 200:
            info = r.json().get("InformationList", {}).get("Information", [])
            syns = []
            if info:
                syns = info[0].get("Synonym", []) or []
            # Prefer short trade/common names over long systematic strings
            filtered = []
            for s in syns:
                if not isinstance(s, str):
                    continue
                if len(s) > 80:
                    continue
                if re.search(r"\d{2,}[-,]", s) and len(s) > 40:
                    continue
                filtered.append(s)
            out["synonyms"] = filtered[:12]
        return out
