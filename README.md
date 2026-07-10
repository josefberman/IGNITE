# IGNITE

Learned energetic-material generation platform with a web GUI.

Given target energetic properties and beam-search parameters, IGNITE selects MAPE-close seeds, expands candidates via a learned (or rule-based fallback) modification planner, estimates properties with literature/XGBoost sources, scores uncertainty/safety/novelty, and returns the best molecule with an audit trail.

**Outputs are computational candidates only — not safety certifications, legal novelty opinions, or patentability advice.**

## Architecture

See [docs/architecture/TRACEABILITY.md](docs/architecture/TRACEABILITY.md) and the supplied diagrams in `docs/architecture/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/write_model_metadata.py
python scripts/normalize_seeds.py
uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

## Training the learned modification stack

```bash
# Generate weak-supervision parent→child pairs (scaffold-split)
python -m ml.data_gen.generate_pairs --max-parents 200 --max-per-parent 3

# Train Graph Transformer planner + delta predictor
python -m ml.training.train_planner --epochs 3
```

Checkpoints land in `ml/checkpoints/planner.pt` and are loaded automatically when present.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Health |
| GET | `/api/v1/models` | Model metadata |
| POST | `/api/v1/search` | Start search |
| GET | `/api/v1/search/{id}/progress` | Progress |
| GET | `/api/v1/search/{id}/events` | SSE progress |
| GET | `/api/v1/search/{id}` | Result |
| POST | `/api/v1/search/{id}/cancel` | Cancel |
| GET | `/api/v1/search/{id}/export.csv` | CSV export |
| GET | `/api/v1/search/{id}/export.sdf` | SDF export |

## Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

## Novelty providers

- Local seed fingerprint screening (always on)
- PubChem InChIKey lookup (optional network)
- OpenAlex literature metadata (optional network)
- Lens / EPO OPS require API keys; without keys status is `incomplete_search`, never “novel”

## License / data notes

Seed CSV and XGBoost models are project assets. Synthetic modification pairs are weak labels derived from SMARTS + model predictions — mark measured experimental labels separately when ingested.
