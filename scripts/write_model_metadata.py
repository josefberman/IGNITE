"""Write model metadata sidecars for the four XGBoost checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.descriptors import DESCRIPTOR_VERSION, descriptor_dim
from backend.app.domain.models import PROPERTY_UNITS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    dim = descriptor_dim()
    for prop, units in PROPERTY_UNITS.items():
        meta = {
            "name": f"xgb_{prop}",
            "version": "1.0",
            "property": prop,
            "descriptor_dim": dim,
            "units": units,
            "path": str(MODELS / f"{prop}.joblib"),
            "notes": f"Descriptor contract {DESCRIPTOR_VERSION}; absolute-property regressor.",
        }
        path = MODELS / f"{prop}.meta.json"
        path.write_text(json.dumps(meta, indent=2))
        print("wrote", path)


if __name__ == "__main__":
    main()
