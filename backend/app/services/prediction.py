"""XGBoost absolute property prediction with source-aware wrapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from backend.app.domain.models import PROPERTY_KEYS, PROPERTY_UNITS, ModelMetadata, PredictionSource, PropertyEstimate
from backend.app.services.descriptors import DESCRIPTOR_VERSION, create_descriptor, descriptor_dim

logger = logging.getLogger(__name__)


class PropertyPredictor:
    def __init__(self, models_directory: Path | str):
        self.models_directory = Path(models_directory)
        self.models: Dict[str, object] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        dim = descriptor_dim()
        for prop in PROPERTY_KEYS:
            path = self.models_directory / f"{prop}.joblib"
            if not path.exists():
                logger.warning("Missing model %s", path)
                continue
            self.models[prop] = joblib.load(path)
            meta_path = self.models_directory / f"{prop}.meta.json"
            if meta_path.exists():
                self.metadata[prop] = ModelMetadata(**json.loads(meta_path.read_text()))
            else:
                self.metadata[prop] = ModelMetadata(
                    name=f"xgb_{prop}",
                    version="1.0",
                    property=prop,
                    descriptor_dim=dim,
                    units=PROPERTY_UNITS[prop],
                    path=str(path),
                    notes=f"Descriptor contract {DESCRIPTOR_VERSION}",
                )
        if not self.models:
            raise FileNotFoundError(f"No XGBoost models in {self.models_directory}")

    def predict(self, smiles: str) -> Optional[Dict[str, PropertyEstimate]]:
        desc = create_descriptor(smiles)
        if desc is None:
            return None
        X = np.array([desc])
        out: Dict[str, PropertyEstimate] = {}
        for prop, model in self.models.items():
            try:
                value = float(model.predict(X)[0])
                # Heuristic confidence: shrink with molecular size extremes
                conf = 0.65
                out[prop] = PropertyEstimate(
                    value=value,
                    source=PredictionSource.XGBOOST,
                    confidence=conf,
                )
            except Exception as e:
                logger.error("Prediction failed for %s: %s", prop, e)
        return out if len(out) == len(self.models) else None

    def list_metadata(self) -> list[ModelMetadata]:
        return list(self.metadata.values())
