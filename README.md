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

## Workflow

Run commands from the repository root, with the project environment active.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

There is no separate `predict` command. Property estimates come from the XGBoost models in `models/` (density, detonation velocity, detonation pressure, solid heat of formation). Pair generation uses them to label parent→child deltas. The GUI search uses them, plus optional literature values, to score candidates.

```text
seeds + XGBoost models
        │
        ▼
generate_pairs  →  data/processed/modification_pairs.csv
        │
        ▼
train_planner   →  ml/checkpoints/planner.pt
        │
        ▼
API (port 8080) + GUI (port 5173)  →  beam search → ranked molecules
```

If `ml/checkpoints/planner.pt` is missing or does not match the current architecture, search still runs and fills expansions from the rule-based SMARTS pool.

### 1. Prepare seeds and property models

```bash
python scripts/write_model_metadata.py
python scripts/normalize_seeds.py
```

Neither script takes flags.


| Script                            | Reads                                                        | Writes                                                                                                            |
| --------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `scripts/write_model_metadata.py` | `models/{density,det_velocity,det_pressure,hf_solid}.joblib` | a `.meta.json` sidecar next to each model                                                                         |
| `scripts/normalize_seeds.py`      | `sample_start_molecules.csv`                                 | `data/processed/seeds_canonical.csv`, `data/processed/seeds_all_with_lineage.csv`, `data/indexes/seed_index.json` |


`sample_start_molecules.csv` must have a `SMILES` column and, for complete seeds, `density`, `det_velocity`, `det_pressure`, and `hf_solid`.

Optional licensed tables (not required for the learned planner):

```bash
python -m ml.data_gen.ingest_external
```


| Flag    | Default          | Meaning                                                                                            |
| ------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `--raw` | `data/raw`       | Directory of `hazard_*.csv` and `reactions_*.csv`                                                  |
| `--out` | `data/processed` | Where `hazard_labels.csv`, `measured_reactions.csv`, and `external_ingest_report.json` are written |


Column schemas are in [data/raw/README.md](data/raw/README.md). With no input files the script writes empty tables.

### 2. Generate parent→child pairs

Applies SMARTS reaction primitives to seed molecules, estimates properties with XGBoost, and writes a scaffold-split CSV (70% train / 15% val / 15% test). Labels are weak: a broad reaction family (nitration, halogenation, oxidation, reduction, heterocycle, alkylation, aromatization, and so on), the specific reaction inside that family (aromatic nitration, nitrate ester, furazan, hydroperoxide, …), reacting-atom sites, and property deltas. They are not experimental measurements. The current pairs file still trains: older rows store the specific reaction in `reaction_family`, and training maps that onto the broad family.

```bash
python -m ml.data_gen.generate_pairs --max-parents 500 --max-per-parent 24 --max-per-family 2
```


| Flag               | Default                      | Meaning                                                                           |
| ------------------ | ---------------------------- | --------------------------------------------------------------------------------- |
| `--seed-csv`       | `sample_start_molecules.csv` | Parent molecules. Column: `SMILES`                                                |
| `--models-dir`     | `models`                     | Directory of the four XGBoost `.joblib` property models                           |
| `--out-dir`        | `data/processed`             | Output directory                                                                  |
| `--max-parents`    | `500`                        | How many unique canonical parents to sample                                       |
| `--max-per-parent` | `24`                         | Cap on children kept for one parent                                               |
| `--max-per-family` | `2`                          | Cap per reaction family on that parent, so one easy SMARTS cannot fill the budget |
| `--seed`           | `42`                         | Shuffle seed for parents, families, and the scaffold split                        |


Outputs in `--out-dir`:

- `modification_pairs.csv` — training table
- `dataset_meta.json` — pair count and checksum
- `failures.json` — up to 1000 SMARTS / sanitization failures

### 3. Train the planner

Trains `ModificationPlanner` (family, reagent, and site heads) on the pairs CSV. Property changes are scored by the XGBoost regressors, not by a second network. Early stopping uses validation family top-1, site top-1, and reagent-given-true-family accuracy. The best epoch is written to the checkpoint.

```bash
python -m ml.training.train_planner --epochs 30 --patience 7
```


| Flag                 | Default                                 | Meaning                                                                              |
| -------------------- | --------------------------------------- | ------------------------------------------------------------------------------------ |
| `--data`             | `data/processed/modification_pairs.csv` | Pairs CSV. Must contain `site_atom_indices`                                          |
| `--out`              | `ml/checkpoints`                        | Writes `planner.pt` and `model_card.json`                                            |
| `--epochs`           | `30`                                    | Maximum epochs                                                                       |
| `--patience`         | `7`                                     | Stop after this many epochs without a better combined validation score               |
| `--lr`               | `0.0002`                                | AdamW learning rate                                                                  |
| `--jitter-prob`      | `0.0`                                   | Probability of jittering the training property gap. `0` matches inference            |
| `--accum-steps`      | `16`                                    | Examples accumulated before one optimizer step                                       |
| `--hidden`           | `64`                                    | Encoder and head width                                                               |
| `--layers`           | `3`                                     | Residual bond message-passing layers in `GraphEncoder`                               |
| `--heads`            | `4`                                     | Kept for checkpoint metadata. The current encoder does not use attention heads       |
| `--dropout`          | `0.15`                                  | Dropout                                                                              |
| `--label-smoothing`  | `0.0`                                   | Recorded in the model card. Not applied to the loss while applicability masks are on |
| `--loss-reag-weight` | `1.0`                                   | Weight of the reagent cross-entropy                                                  |
| `--loss-site-weight` | `1.0`                                   | Weight of the site softmax cross-entropy                                             |


Family cross-entropy and delta MSE stay at weight 1. The API loads `ml/checkpoints/planner.pt` automatically (`IGNITE_LEARNED_CHECKPOINT_DIR` overrides the directory).

### 4. Run the API and the GUI

Terminal 1, API:

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8080
```

Use port **8080**. Port 8000 is often the HPLIP printer web UI on Linux, which returns `400` for these API calls.


| Uvicorn option         | Value used here  | Meaning                          |
| ---------------------- | ---------------- | -------------------------------- |
| `backend.app.main:app` | —                | FastAPI application              |
| `--reload`             | on for local dev | Restart when Python files change |
| `--host`               | `127.0.0.1`      | Bind address                     |
| `--port`               | `8080`           | Must match the Vite proxy        |


Terminal 2, GUI:

```bash
cd frontend && npm install && npm run dev
```


| npm script        | Meaning                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `npm run dev`     | Vite dev server on [http://localhost:5173](http://localhost:5173). Proxies `/api` to `http://127.0.0.1:8080` |
| `npm run build`   | Typecheck and production build                                                                               |
| `npm run preview` | Serve the production build                                                                                   |


Open [http://localhost:5173](http://localhost:5173). You do not type a SMILES. The search starts from the seed library and edits molecules toward the targets you enter.

#### What to enter

**Target properties** (accepted ranges are enforced by the API):


| Field                     | Unit   | GUI default | Allowed range |
| ------------------------- | ------ | ----------- | ------------- |
| Density                   | g/cm³  | 1.80        | 0.8 – 2.5     |
| Detonation velocity       | m/s    | 8500        | 1000 – 12000  |
| Detonation pressure       | GPa    | 30          | 0.1 – 60      |
| Heat of formation (solid) | kJ/mol | 100         | −1000 – 1200  |


**Search bounds:**


| Field          | GUI default | Meaning                                                                          | API limit                                              |
| -------------- | ----------- | -------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Depth          | 10          | Expansion rounds                                                                 | 1–50, clamped by `IGNITE_MAX_BEAM_DEPTH` (default 20)  |
| Width          | 5           | Children generated from each parent                                              | 1–100, clamped by `IGNITE_MAX_BEAM_WIDTH` (default 50) |
| Top K          | 3           | Molecules kept after each round                                                  | 1–50, clamped by `IGNITE_MAX_TOP_K` (default 25)       |
| MAPE threshold | 1%          | Target mean absolute percentage error across the four properties (equal weights) | 0–100                                                  |


**Toggles:**


| Toggle                 | GUI default | Effect                                                                                                                               |
| ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Literature properties  | on          | Prefer curated literature property values over the XGBoost estimate when a match exists                                              |
| Novelty gate           | off         | When on, drop candidates that match a known seed, PubChem, literature, or patent record                                              |
| Stop at MAPE threshold | off         | When on, stop after at least one expansion once the best MAPE is at or below the threshold. When off, the search runs the full depth |


Then **Start search**. **Cancel** stops the running job. After completion, **Export CSV** downloads `/api/v1/search/{id}/export.csv`. The canvas shows the beam tree, the best molecules, estimated properties, and MAPE.

These are sent with every search but have no GUI control: random seed `42`, and patent lookup on (`enable_patent_search`). Patent providers still need API keys (below) or the patent check is incomplete.

#### Server environment

Optional `.env` variables, prefix `IGNITE_`:


| Variable                        | Default                                       | Meaning                                 |
| ------------------------------- | --------------------------------------------- | --------------------------------------- |
| `IGNITE_DATA_DIR`               | `data`                                        | Data root                               |
| `IGNITE_MODELS_DIR`             | `models`                                      | XGBoost property models                 |
| `IGNITE_SEED_CSV`               | `sample_start_molecules.csv`                  | Seed library for search                 |
| `IGNITE_DATABASE_URL`           | SQLite at `data/ignite.db`                    | Job store                               |
| `IGNITE_CORS_ORIGINS`           | `http://localhost:5173,http://localhost:3000` | Allowed browser origins                 |
| `IGNITE_MAX_BEAM_DEPTH`         | `20`                                          | Clamp on GUI depth                      |
| `IGNITE_MAX_BEAM_WIDTH`         | `50`                                          | Clamp on GUI width                      |
| `IGNITE_MAX_TOP_K`              | `25`                                          | Clamp on GUI top K                      |
| `IGNITE_LEARNED_CHECKPOINT_DIR` | `ml/checkpoints`                              | Where `planner.pt` is loaded from       |
| `IGNITE_PROCESSED_DATA_DIR`     | `data/processed`                              | Processed tables                        |
| `IGNITE_PUBCHEM_ENABLED`        | `true`                                        | PubChem exact-match lookup              |
| `IGNITE_OPENALEX_ENABLED`       | `true`                                        | OpenAlex literature lookup              |
| `IGNITE_EPO_OPS_KEY`            | empty                                         | EPO OPS patent API key                  |
| `IGNITE_LENS_API_KEY`           | empty                                         | Lens patent API key                     |
| `IGNITE_REDIS_URL`              | empty                                         | Unused by the current in-process search |
| `IGNITE_OPENAI_API_KEY`         | empty                                         | Unused by the current search            |


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