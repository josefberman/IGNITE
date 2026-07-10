"""Novelty screening: local fingerprints + optional literature/patent providers."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import httpx

from backend.app.domain.models import NoveltyEvidence, NoveltyStatus
from backend.app.services.chemistry import canonicalize, inchikey_of, morgan_fp_bits, tanimoto

logger = logging.getLogger(__name__)


class NoveltyService:
    def __init__(
        self,
        seed_smiles: Sequence[str],
        pubchem_enabled: bool = True,
        openalex_enabled: bool = True,
        epo_ops_key: str = "",
        lens_api_key: str = "",
        close_analogue_threshold: float = 0.85,
    ):
        self.seed_smiles = list(seed_smiles)
        self.seed_fps = []
        for s in self.seed_smiles:
            fp = morgan_fp_bits(s)
            if fp is not None:
                self.seed_fps.append((s, fp))
        self.pubchem_enabled = pubchem_enabled
        self.openalex_enabled = openalex_enabled
        self.epo_ops_key = epo_ops_key
        self.lens_api_key = lens_api_key
        self.close_analogue_threshold = close_analogue_threshold
        self._cache: Dict[str, NoveltyEvidence] = {}

    def _local_similarity(self, smiles: str) -> tuple[Optional[float], Optional[str]]:
        from rdkit import DataStructs

        fp = morgan_fp_bits(smiles)
        if fp is None or not self.seed_fps:
            return None, None
        best = -1.0
        nearest = None
        for s, sfp in self.seed_fps:
            sim = float(DataStructs.TanimotoSimilarity(fp, sfp))
            if sim > best:
                best = sim
                nearest = s
        return best, nearest

    def _query_pubchem(self, inchikey: str) -> tuple[List[dict], bool]:
        if not self.pubchem_enabled:
            return [], False
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
        try:
            r = httpx.get(url, timeout=8.0)
            if r.status_code == 404:
                return [], True
            if r.status_code != 200:
                return [], False
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            return [{"provider": "pubchem", "cid": c} for c in cids[:5]], True
        except Exception as e:
            logger.debug("PubChem unavailable: %s", e)
            return [], False

    def _query_openalex(self, smiles: str) -> tuple[List[dict], bool]:
        if not self.openalex_enabled:
            return [], False
        # Conservative metadata search by InChIKey substring if available
        key = inchikey_of(smiles) or smiles[:40]
        url = "https://api.openalex.org/works"
        try:
            r = httpx.get(url, params={"search": key, "per_page": 3}, timeout=8.0)
            if r.status_code != 200:
                return [], False
            results = []
            for w in r.json().get("results", [])[:3]:
                results.append(
                    {
                        "provider": "openalex",
                        "id": w.get("id"),
                        "title": w.get("display_name"),
                        "doi": (w.get("doi") or ""),
                    }
                )
            return results, True
        except Exception as e:
            logger.debug("OpenAlex unavailable: %s", e)
            return [], False

    def _query_patents(self, smiles: str) -> tuple[List[dict], List[str], List[str]]:
        hits: List[dict] = []
        checked: List[str] = []
        unavailable: List[str] = []
        if self.lens_api_key:
            checked.append("lens")
            # Credential present but full Lens query requires paid schema; mark incomplete.
            unavailable.append("lens_query_not_configured")
        else:
            unavailable.append("lens")
        if self.epo_ops_key:
            checked.append("epo_ops")
            unavailable.append("epo_ops_query_not_configured")
        else:
            unavailable.append("epo_ops")
        return hits, checked, unavailable

    def evaluate(self, smiles: str) -> NoveltyEvidence:
        can = canonicalize(smiles) or smiles
        if can in self._cache:
            return self._cache[can]

        notes: List[str] = []
        providers_checked: List[str] = ["local_seed_library"]
        providers_unavailable: List[str] = []

        # Exact match against seeds
        seed_set = set(self.seed_smiles)
        if can in seed_set:
            ev = NoveltyEvidence(
                status=NoveltyStatus.KNOWN,
                score=0.0,
                max_tanimoto_to_seeds=1.0,
                nearest_seed_smiles=can,
                providers_checked=providers_checked,
                notes=["Exact match in seed library"],
            )
            self._cache[can] = ev
            return ev

        max_sim, nearest = self._local_similarity(can)
        key = inchikey_of(can)

        lit_hits: List[dict] = []
        if key:
            pub_hits, pub_ok = self._query_pubchem(key)
            if pub_ok:
                providers_checked.append("pubchem")
                lit_hits.extend(pub_hits)
            else:
                providers_unavailable.append("pubchem")
        else:
            providers_unavailable.append("pubchem")

        oa_hits, oa_ok = self._query_openalex(can)
        if oa_ok:
            providers_checked.append("openalex")
            lit_hits.extend(oa_hits)
        else:
            providers_unavailable.append("openalex")

        patent_hits, p_checked, p_unavail = self._query_patents(can)
        providers_checked.extend(p_checked)
        providers_unavailable.extend(p_unavail)

        incomplete = bool(providers_unavailable) or ("lens" in providers_unavailable)

        if max_sim is not None and max_sim >= self.close_analogue_threshold:
            status = NoveltyStatus.CLOSE_ANALOGUE
            score = max(0.0, 1.0 - max_sim)
            notes.append(f"Close analogue (Tanimoto={max_sim:.3f})")
        elif lit_hits or patent_hits:
            status = NoveltyStatus.KNOWN
            score = 0.1
            notes.append("External database hit")
        elif incomplete:
            status = NoveltyStatus.INCOMPLETE_SEARCH
            score = max(0.0, 1.0 - (max_sim or 0.0)) * 0.5
            notes.append(
                "Some novelty providers unavailable — absence of hits is not proof of novelty."
            )
        else:
            status = NoveltyStatus.NO_MATCH_FOUND
            score = max(0.0, 1.0 - (max_sim or 0.0))
            notes.append(
                "No match found in checked sources; not a legal novelty or patentability opinion."
            )

        ev = NoveltyEvidence(
            status=status,
            score=score,
            max_tanimoto_to_seeds=max_sim,
            nearest_seed_smiles=nearest,
            literature_hits=lit_hits,
            patent_hits=patent_hits,
            providers_checked=providers_checked,
            providers_unavailable=providers_unavailable,
            notes=notes,
        )
        self._cache[can] = ev
        return ev
