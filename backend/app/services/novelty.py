"""Novelty screening via exact identifiers: SMILES, IUPAC, commercial names."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote

import httpx

from backend.app.domain.models import NoveltyEvidence, NoveltyStatus
from backend.app.services.chemistry import canonicalize
from backend.app.services.molecule_identity import MoleculeIdentity, MoleculeIdentityResolver

logger = logging.getLogger(__name__)


class NoveltyService:
    def __init__(
        self,
        seed_smiles: Sequence[str],
        pubchem_enabled: bool = True,
        openalex_enabled: bool = True,
        epo_ops_key: str = "",
        lens_api_key: str = "",
        identity_resolver: Optional[MoleculeIdentityResolver] = None,
    ):
        self.seed_smiles = set(canonicalize(s) or s for s in seed_smiles)
        self.pubchem_enabled = pubchem_enabled
        self.openalex_enabled = openalex_enabled
        self.epo_ops_key = epo_ops_key
        self.lens_api_key = lens_api_key
        self.identity = identity_resolver or MoleculeIdentityResolver()
        self._cache: Dict[str, NoveltyEvidence] = {}

    def _query_pubchem_exact(self, identity: MoleculeIdentity) -> tuple[List[dict], bool]:
        if not self.pubchem_enabled:
            return [], False
        hits: List[dict] = []
        try:
            if identity.cid is not None:
                hits.append(
                    {
                        "provider": "pubchem",
                        "cid": identity.cid,
                        "iupac": identity.iupac_name,
                        "commercial_names": identity.commercial_names[:5],
                        "query": "exact SMILES",
                        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{identity.cid}",
                        "title": identity.iupac_name
                        or (identity.commercial_names[0] if identity.commercial_names else f"CID {identity.cid}"),
                    }
                )
                return hits, True
            # No CID ⇒ exact structure not in PubChem
            return [], True
        except Exception as e:
            logger.debug("PubChem unavailable: %s", e)
            return [], False

    def _query_openalex(self, identity: MoleculeIdentity) -> tuple[List[dict], bool]:
        if not self.openalex_enabled:
            return [], False
        results: List[dict] = []
        checked_ok = False
        # Prefer human names; also try SMILES / InChIKey for exact-molecule mentions
        terms = []
        if identity.iupac_name:
            terms.append(identity.iupac_name)
        terms.extend(identity.commercial_names[:5])
        if identity.inchikey:
            terms.append(identity.inchikey)
        terms.append(identity.canonical_smiles)

        for term in terms[:6]:
            try:
                r = httpx.get(
                    "https://api.openalex.org/works",
                    params={"search": term, "per_page": 3},
                    timeout=8.0,
                    headers={"User-Agent": "IGNITE/0.1 (research; mailto:ignite@local)"},
                )
                if r.status_code != 200:
                    continue
                checked_ok = True
                for w in r.json().get("results", [])[:3]:
                    openalex_id = w.get("id") or ""
                    doi = (w.get("doi") or "").strip()
                    doi_url = doi if doi.startswith("http") else (f"https://doi.org/{doi.replace('https://doi.org/', '')}" if doi else "")
                    results.append(
                        {
                            "provider": "openalex",
                            "id": openalex_id,
                            "title": w.get("display_name"),
                            "doi": doi,
                            "query": term,
                            "url": doi_url or openalex_id,
                        }
                    )
                if results:
                    break
            except Exception as e:
                logger.debug("OpenAlex unavailable for %s: %s", term[:40], e)
        return results, checked_ok

    def _query_patents(self, identity: MoleculeIdentity) -> tuple[List[dict], List[str], List[str]]:
        hits: List[dict] = []
        checked: List[str] = []
        unavailable: List[str] = []

        # Build exact-molecule patent query strings
        queries = []
        if identity.iupac_name:
            queries.append(identity.iupac_name)
        queries.extend(identity.commercial_names[:4])
        if identity.inchikey:
            queries.append(identity.inchikey)
        queries.append(identity.canonical_smiles)

        if self.lens_api_key:
            checked.append("lens")
            # Lens full-text search requires configured schema; record queries attempted.
            unavailable.append("lens_query_not_configured")
            for q in queries[:3]:
                hits.append(
                    {
                        "provider": "lens",
                        "status": "not_queried",
                        "query": q,
                        "note": "API key present but Lens query schema not configured",
                        "url": f"https://www.lens.org/lens/search/patent/list?q={quote(q)}",
                        "title": f"Lens search: {q}",
                    }
                )
        else:
            unavailable.append("lens")

        if self.epo_ops_key:
            checked.append("epo_ops")
            unavailable.append("epo_ops_query_not_configured")
            for q in queries[:3]:
                hits.append(
                    {
                        "provider": "epo_ops",
                        "status": "not_queried",
                        "query": q,
                        "note": "API key present but EPO OPS client not configured",
                        "url": f"https://worldwide.espacenet.com/patent/search?q={quote(q)}",
                        "title": f"Espacenet search: {q}",
                    }
                )
        else:
            unavailable.append("epo_ops")

        # Public Google Patents deep-link evidence (no scrape; user-verifiable)
        if queries:
            checked.append("google_patents_link")
            q = queries[0]
            hits.append(
                {
                    "provider": "google_patents",
                    "status": "link_only",
                    "query": q,
                    "url": f"https://patents.google.com/?q={quote(q)}",
                    "title": f"Google Patents: {q}",
                    "note": "Deep link for exact-name/SMILES patent search — not an automated claim hit",
                }
            )

        return hits, checked, unavailable

    def evaluate(self, smiles: str) -> NoveltyEvidence:
        can = canonicalize(smiles) or smiles
        if can in self._cache:
            return self._cache[can]

        notes: List[str] = []
        providers_checked: List[str] = ["local_seed_library"]
        providers_unavailable: List[str] = []

        if can in self.seed_smiles:
            ev = NoveltyEvidence(
                status=NoveltyStatus.KNOWN,
                score=0.0,
                max_tanimoto_to_seeds=None,
                nearest_seed_smiles=can,
                providers_checked=providers_checked,
                notes=["Exact SMILES match in seed library"],
            )
            self._cache[can] = ev
            return ev

        identity = self.identity.resolve(can)
        if identity.iupac_name:
            notes.append(f"IUPAC: {identity.iupac_name}")
        if identity.commercial_names:
            notes.append(f"Names: {', '.join(identity.commercial_names[:3])}")

        lit_hits: List[dict] = []
        pub_hits, pub_ok = self._query_pubchem_exact(identity)
        if pub_ok:
            providers_checked.append("pubchem")
            lit_hits.extend(pub_hits)
            if pub_hits:
                notes.append("Exact structure found in PubChem")
        else:
            providers_unavailable.append("pubchem")

        oa_hits, oa_ok = self._query_openalex(identity)
        if oa_ok:
            providers_checked.append("openalex")
            lit_hits.extend(oa_hits)
        else:
            providers_unavailable.append("openalex")

        patent_hits, p_checked, p_unavail = self._query_patents(identity)
        providers_checked.extend(p_checked)
        providers_unavailable.extend(p_unavail)

        # Patent link_only does not count as a confirmed hit
        confirmed_patents = [h for h in patent_hits if h.get("status") not in ("link_only", "not_queried")]
        incomplete = bool(providers_unavailable)

        if lit_hits or confirmed_patents:
            status = NoveltyStatus.KNOWN
            score = 0.1
            notes.append("Exact-molecule literature/database hit via SMILES/IUPAC/commercial name")
        elif incomplete:
            status = NoveltyStatus.INCOMPLETE_SEARCH
            score = 0.5
            notes.append(
                "Some novelty providers unavailable — absence of hits is not proof of novelty."
            )
        else:
            status = NoveltyStatus.NO_MATCH_FOUND
            score = 0.9
            notes.append(
                "No exact-molecule match found for SMILES/IUPAC/commercial name in checked sources; "
                "not a legal novelty or patentability opinion."
            )

        ev = NoveltyEvidence(
            status=status,
            score=score,
            max_tanimoto_to_seeds=None,
            nearest_seed_smiles=None,
            literature_hits=lit_hits,
            patent_hits=patent_hits,
            providers_checked=providers_checked,
            providers_unavailable=providers_unavailable,
            notes=notes,
        )
        self._cache[can] = ev
        return ev
