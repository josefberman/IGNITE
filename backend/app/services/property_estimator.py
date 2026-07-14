"""Source-aware property estimation: exact literature match, else XGBoost prediction."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

from backend.app.domain.models import PROPERTY_KEYS, PredictionSource, PropertyEstimate
from backend.app.services.chemistry import canonicalize
from backend.app.services.prediction import PropertyPredictor
from backend.app.services.uncertainty import apply_uncertainty

logger = logging.getLogger(__name__)


class LiteratureRetriever:
    """
    Exact-molecule property lookup only (no analogue/fingerprint transfer).

    Sources:
    1. Exact canonical SMILES match in the seed experimental library
    2. Optional in-memory cache of previously retrieved exact-molecule literature values
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.cache: Dict[str, Dict[str, PropertyEstimate]] = {}
        self._exact: Dict[str, Dict[str, float]] = {}

    def load_seed_library(self, rows: Sequence[dict]) -> None:
        """Index exact experimental seed properties by canonical SMILES."""
        self._exact.clear()
        for row in rows:
            smi = canonicalize(str(row.get("canonical_smiles") or row.get("smiles") or ""))
            if not smi:
                continue
            props = {}
            ok = True
            for k in PROPERTY_KEYS:
                if k not in row or row[k] is None:
                    ok = False
                    break
                props[k] = float(row[k])
            if ok:
                self._exact[smi] = props
        logger.info("LiteratureRetriever indexed %d exact seed molecules", len(self._exact))

    def retrieve(self, smiles: str) -> Optional[Dict[str, PropertyEstimate]]:
        if not self.enabled:
            return None

        can = canonicalize(smiles) or smiles
        if can in self.cache:
            return self.cache[can]

        if can in self._exact:
            props = self._exact[can]
            out = {
                k: PropertyEstimate(
                    value=v,
                    source=PredictionSource.LITERATURE,
                    confidence=0.92,
                    citation=f"exact experimental seed match: {can}",
                )
                for k, v in props.items()
            }
            self.cache[can] = out
            return out

        return None


class PropertyEstimator:
    def __init__(self, predictor: PropertyPredictor, literature: Optional[LiteratureRetriever] = None):
        self.predictor = predictor
        self.literature = literature or LiteratureRetriever(enabled=False)

    def estimate(
        self,
        smiles: str,
        known: Optional[Dict[str, float]] = None,
        prefer_literature: bool = True,
    ) -> Optional[Dict[str, PropertyEstimate]]:
        if known and all(k in known for k in PROPERTY_KEYS):
            props = {
                k: PropertyEstimate(value=float(known[k]), source=PredictionSource.DATASET, confidence=0.9)
                for k in PROPERTY_KEYS
            }
            return apply_uncertainty(props)

        if prefer_literature and self.literature.enabled:
            lit = self.literature.retrieve(smiles)
            if lit and all(k in lit for k in PROPERTY_KEYS):
                return apply_uncertainty(lit)

        pred = self.predictor.predict(smiles)
        if pred is None:
            return None
        return apply_uncertainty(pred, ood_distance=0.15)
