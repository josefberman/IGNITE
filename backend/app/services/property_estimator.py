"""Source-aware property estimation: dataset/literature/analogue/XGBoost."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from backend.app.domain.models import PROPERTY_KEYS, PredictionSource, PropertyEstimate
from backend.app.services.prediction import PropertyPredictor
from backend.app.services.uncertainty import apply_uncertainty

logger = logging.getLogger(__name__)


class LiteratureRetriever:
    """Lightweight literature/analogue stub with optional PubChem-free local cache."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.cache: Dict[str, Dict[str, PropertyEstimate]] = {}

    def retrieve(self, smiles: str) -> Optional[Dict[str, PropertyEstimate]]:
        if not self.enabled:
            return None
        return self.cache.get(smiles)


class PropertyEstimator:
    def __init__(self, predictor: PropertyPredictor, literature: Optional[LiteratureRetriever] = None):
        self.predictor = predictor
        self.literature = literature or LiteratureRetriever(enabled=False)

    def estimate(
        self,
        smiles: str,
        known: Optional[Dict[str, float]] = None,
        analogue_tanimoto: Optional[float] = None,
    ) -> Optional[Dict[str, PropertyEstimate]]:
        if known and all(k in known for k in PROPERTY_KEYS):
            props = {
                k: PropertyEstimate(value=float(known[k]), source=PredictionSource.DATASET, confidence=0.95)
                for k in PROPERTY_KEYS
            }
            return apply_uncertainty(props)

        lit = self.literature.retrieve(smiles)
        if lit and all(k in lit for k in PROPERTY_KEYS):
            return apply_uncertainty(lit, analogue_tanimoto=analogue_tanimoto)

        pred = self.predictor.predict(smiles)
        if pred is None:
            return None
        return apply_uncertainty(pred, analogue_tanimoto=analogue_tanimoto, ood_distance=0.2)
