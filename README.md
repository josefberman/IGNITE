# IGNITE

Learned energetic-material generation platform with a web GUI.

Given target energetic properties and beam-search parameters, IGNITE selects MAPE-close seeds, expands candidates via a learned (or rule-based fallback) modification planner, estimates properties with literature/XGBoost sources, ranks by MAPE + SAScore + minimal-edit, and returns the best molecule with an audit trail. An optional novelty gate can hard-reject exact known molecules.

**Outputs are computational candidates only — not experimental validation or a legal novelty / patentability opinion.** Safety, uncertainty, and novelty evidence are not reported in the UI.

## Architecture

See the full system monograph:

- **[IGNITE System Thesis](docs/IGNITE_System_Thesis.md)** — end-to-end flow, algorithms, scores, gates, and limitations

Also: [docs/architecture/TRACEABILITY.md](docs/architecture/TRACEABILITY.md) and the diagrams in `docs/architecture/`:

- [High-level architecture](docs/architecture/IGNITE_High_Level_Architecture.png)
- [Modification architecture](docs/architecture/IGNITE_Modification_Architecture.png)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/write_model_metadata.py
python scripts/normalize_seeds.py
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8080
```

> **Note:** Use port **8080** by default. Port 8000 is often claimed by the HPLIP printer web UI on Linux, which returns bare `400 - Bad Request` for API calls.

Frontend:

```bash
cd frontend && npm install && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) (Vite proxies `/api` → `http://127.0.0.1:8080`)

## Training the learned modification stack

```bash
# Generate weak-supervision parent→child pairs (scaffold-split)
python -m ml.data_gen.generate_pairs --max-parents 200 --max-per-parent 3

# Train Graph Transformer planner + delta predictor (early-stops on val family accuracy)
python -m ml.training.train_planner --epochs 20 --patience 5
```

Checkpoints land in `ml/checkpoints/planner.pt` and are loaded automatically when present.
Beam expansion then: top-k **families** → family-masked **reagent** (SMARTS variant) ranking → site-biased execution → optional Δy ranking → rule-based fill if short.

## API


| Method | Path                             | Purpose        |
| ------ | -------------------------------- | -------------- |
| GET    | `/api/v1/health`                 | Health         |
| GET    | `/api/v1/models`                 | Model metadata |
| POST   | `/api/v1/search`                 | Start search   |
| GET    | `/api/v1/search/{id}/progress`   | Progress       |
| GET    | `/api/v1/search/{id}/events`     | SSE progress   |
| GET    | `/api/v1/search/{id}`            | Result         |
| POST   | `/api/v1/search/{id}/cancel`     | Cancel         |
| GET    | `/api/v1/search/{id}/export.csv` | CSV export     |
| GET    | `/api/v1/search/{id}/export.sdf` | SDF export     |


## Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

## Novelty providers

- Local seed fingerprint screening (always on)
- PubChem InChIKey lookup (optional network)
- OpenAlex literature metadata (optional network)
- Lens / EPO OPS require API keys; without keys status is `incomplete_search`, never “novel”

## License

Code and project documentation are licensed under the [Apache License 2.0](LICENCE.md) (see also [NOTICE](NOTICE)).

This repository is **research software in progress**. Public release supports open science and reproducibility; it is not a substitute for peer-reviewed journal publication of associated manuscripts. Most chemistry / ML venues accept prior public code and preprints — confirm the target journal’s preprint / prior-disclosure policy before submission.

Data and trained artifacts: seed CSV and XGBoost models are project assets (see [docs/LICENSING.md](docs/LICENSING.md)). Synthetic modification pairs are weak labels derived from SMARTS + model predictions — mark measured experimental labels separately when ingested.