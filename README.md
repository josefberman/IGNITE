# IGNITE

Learned energetic-material generation platform with a web GUI.

Given target energetic properties and beam-search parameters, IGNITE selects MAPE-close seeds, expands candidates with either the learned planner or a rule-based modifier (chosen per run in the GUI), estimates properties with literature/XGBoost sources, ranks by MAPE + SAScore + minimal-edit, and returns the best molecule with an audit trail. An optional novelty gate can hard-reject exact known molecules.

**Outputs are computational candidates only — not experimental validation or a legal novelty / patentability opinion.** Safety, uncertainty, and novelty evidence are not reported in the UI.

## Architecture

See the full system monograph:

- **[IGNITE System Thesis](docs/IGNITE_System_Thesis.md)** — end-to-end flow, algorithms, scores, gates, and limitations

Also: [docs/architecture/TRACEABILITY.md](docs/architecture/TRACEABILITY.md) and the diagrams in `docs/architecture/`:

- [High-level architecture](docs/architecture/IGNITE_High_Level_Architecture.png)
- [Modification architecture](docs/architecture/IGNITE_Modification_Architecture.png)

## Workflow

Run commands from the repository root, with the project environment active.

### pip / venv

`pyproject.toml` is the source of truth for Python dependencies. Torch is an optional extra. If you select the learned planner without torch or `ml/checkpoints/planner.pt`, the API returns an error and the GUI shows the reason — it does **not** silently fall back to the rule-based modifier. Choose the rule-based modifier in the GUI when the planner is unavailable.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sascore]"
# optional, for the learned modification planner:
pip install -e ".[dev,sascore,torch]"
```

### Conda

`environment.yml` lists the same third-party packages as `pyproject.toml`. Compiled packages (Python 3.10–3.11, XGBoost, PyTorch, NumPy, SciPy, pandas, scikit-learn, Pillow) come from conda-forge. RDKit is installed as **rdkit-pypi** via pip (not conda-forge) so the `sascore` extra (`BRSAScore`) can install; Python is pinned below 3.12 because rdkit-pypi has no 3.12 wheels. Remaining packages and this project (`pip install -e ".[dev,sascore]"`) also come from pip. PyTorch is included for a full local install; omit it from a custom env if you only need the rule-based modifier.

```bash
conda env create -f environment.yml
conda activate ignite
```

```bash
pytest tests/unit tests/integration
```

Equivalent with micromamba: `micromamba env create -f environment.yml` then `micromamba activate ignite`.

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

`data/processed/` (pair labels, canonical seeds) and `ml/checkpoints/planner.pt` are shipped in the repo, copied from [josefberman/IGNITE](https://github.com/josefberman/IGNITE) commit `4df03fe`. If `planner.pt` is missing or does not match the current architecture, select **Rule-based modifier** in the GUI (or `mode: "rule_based"` on the API). Planner mode will refuse to start rather than swap modifiers.

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


`sample_start_molecules.csv` must have a `SMILES` column and, for complete rows, `density`, `det_velocity`, `det_pressure`, and `hf_solid`.

`sample_start_molecules.csv` (~16k rows) is the XGBoost training dataset only. It is not a seed library: `SeedLibrary` refuses to load it.

### Seed library

Search seeds come from `data/seeds/energetic_seed_library.csv`: 77 published neutral energetic molecules. The original 27 are TNT, RDX, HMX, CL-20, PETN, TATB, HNS, FOX-7, NTO, LLM-105, TNAZ, nitroglycerin, picric acid, DNAN, TNB, tetryl, DATB, nitroguanidine, EGDN, BTF, ETN, MHN, HNB, ONC, TEX, nitromethane and 2,4-DNT. Five expansion batches add 50 more: nitroaromatics (1,3-DNB, picramide, TNA, styphnic acid, picryl chloride, HNDPA, PYX, HNAB, DDNP, ...), nitrate esters and nitroalkanes (methyl/ethyl/isopropyl nitrate, DEGDN, TEGDN, TMETN, BTTN, PGDN, tetranitromethane, nitroform, ...), nitramines and ureas (DINGU, EDNA, nitrourea, keto-RDX, BCHMX, heptanitrocubane), heterocycles (ANPZ, LLM-116, ANTA, DNTF, DAAF, dinitropyrazoles, 2,4-DNI, cyanuric triazide, ...) and peroxides (TATP, HMTD). Structures that have no checked value were matched to PubChem by InChIKey. Salts and ions (TKX-50, ADN, AN) are left out. The columns are `name`, `SMILES` (RDKit canonical), and, for each of `density` (g/cm³), `det_velocity` (m/s), `det_pressure` (GPa), and `hf_solid` (kJ/mol), the value plus `<prop>_source` (citation or URL) and `<prop>_provenance` (`literature` or `predicted`). Literature values were checked against the cited source. Values with no checked source come from the repo's XGBoost models and are marked `predicted`. Predicted seed values are not indexed as literature by the property estimator. To rebuild the file:

```bash
python scripts/build_energetic_seed_library.py
```

This is the only seed library. To use a different CSV with the same schema, set `IGNITE_SEED_CSV`.

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
| `--hidden`           | `128`                                   | Encoder and head width                                                               |
| `--layers`           | `3`                                     | Residual bond message-passing layers in `GraphEncoder`                               |
| `--bond-types`       | `5`                                     | Bond-type embeddings added to messages (single, double, triple, aromatic, other). `0` is untyped |
| `--no-site-reagent`  | off                                     | Use a reagent-blind site head. By default the site head is conditioned on the reagent |
| `--heads`            | `4`                                     | Kept for checkpoint metadata. The current encoder does not use attention heads       |
| `--dropout`          | `0.15`                                  | Dropout                                                                              |
| `--label-smoothing`  | `0.0`                                   | Recorded in the model card. Not applied to the loss while applicability masks are on |
| `--loss-reag-weight` | `1.0`                                   | Weight of the reagent cross-entropy                                                  |
| `--loss-site-weight` | `1.0`                                   | Weight of the site loss: a softmax over the reagent's applicable sites, where every site giving the labeled product counts as correct |


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
| Depth          | 10          | Expansion rounds                                                                 | 1–100, clamped by `IGNITE_MAX_BEAM_DEPTH` (default 100) |
| Width          | 5           | Children generated from each parent                                              | 1–100, clamped by `IGNITE_MAX_BEAM_WIDTH` (default 50) |
| Top K          | 3           | Molecules kept after each round                                                  | 1–50, clamped by `IGNITE_MAX_TOP_K` (default 25)       |
| MAPE threshold | 1%          | Target mean absolute percentage error across the four properties (equal weights) | 0–100                                                  |


**Modifier** (per run; not swapped automatically):


| Mode | GUI default | Effect |
| ---- | ----------- | ------ |
| Learned planner | selected when torch and `planner.pt` are available | Graph model proposes family, reagent, and site. If torch or the checkpoint is missing, the GUI explains why and the search does not start. |
| Rule-based modifier | selected when the planner is unavailable | Works only at the reagent level: a flat list of reagents in [docs/reagents.md](docs/reagents.md), each with a raise/lower/neutral effect on density, detonation velocity, detonation pressure, and Hf. It ranks reagents by whether their effects close the gap between the seed's predicted properties and the targets, and picks the reaction site at random among the reagent's valid sites (seeded by the run's random seed). Reaction families play no part. |


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
| `IGNITE_SEED_CSV`               | `data/seeds/energetic_seed_library.csv`       | Seed library for search                 |
| `IGNITE_DATABASE_URL`           | SQLite at `data/ignite.db`                    | Job store                               |
| `IGNITE_CORS_ORIGINS`           | `http://localhost:5173,http://localhost:3000` | Allowed browser origins                 |
| `IGNITE_MAX_BEAM_DEPTH`         | `100`                                         | Clamp on GUI depth                      |
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
| `IGNITE_RUN_LOG_DIR`            | `logs`                                        | Directory for per-run JSON search logs  |
| `IGNITE_RUN_LOGGING_ENABLED`    | `true`                                        | Write a JSON file after each search     |


Each completed search writes one JSON file under `IGNITE_RUN_LOG_DIR` (default `logs/`, gitignored). The file includes the UTC timestamp, target properties, modifier mode (`planner` or `rule_based`), seed SMILES, and every accepted candidate with predicted properties, MAPE fit, SAScore / synthesis-ease penalty, edit-distance / change size, and overall rank/score.

It also contains `best_path`: the ordered chain from the seed (step 0) to the final best molecule, reconstructed from parent links in the beam search. Each step carries `step`, `generation`, `smiles`, `parent_smiles`, `predicted_properties` (density, det_velocity, det_pressure, hf_solid), `fit.mape`, `synthesis_ease` (SAScore and penalty), `change_size` (edit distance and penalty), `overall.score`, and `edit`: the edit applied to reach that step (`null` for the seed). In `rule_based` mode `edit.reagent` (and `edit.reaction_family`) is the reagent id; in `planner` mode `edit.reaction_family` is the planner's reaction family and `edit.reagent` the reaction applied. Abridged example from a real `rule_based` run:

```json
"best_path": [
  {"step": 0, "smiles": "Cc1ccc([N+](=O)[O-])cc1", "fit": {"mape": 60.57}, "overall": {"score": 0.557}, "edit": null},
  {"step": 1, "smiles": "Cc1ccc([N+](=O)[O-])c(-c2nncnn2)c1", "fit": {"mape": 32.68}, "overall": {"score": 0.318},
   "edit": {"reaction_family": "tetrazine_CH", "reagent": "tetrazine_CH", "planner": "rule_based", "smarts": "[cH:1]>>[c:1]c1nncnn1"}},
  {"step": 2, "smiles": "O=[N+]([O-])Cc1ccc([N+](=O)[O-])c(-c2nncnn2)c1", "fit": {"mape": 23.89}, "overall": {"score": 0.228},
   "edit": {"reaction_family": "nitrate_alkyl_CH", "reagent": "nitrate_alkyl_CH", "planner": "rule_based", "smarts": "[CX4;!H0:1]>>[C:1][N+](=O)[O-]"}}
]
```


## API


| Method | Path                             | Purpose        |
| ------ | -------------------------------- | -------------- |
| GET    | `/api/v1/health`                 | Health         |
| GET    | `/api/v1/capabilities`           | Planner / modifier availability |
| GET    | `/api/v1/models`                 | Model metadata |
| POST   | `/api/v1/search`                 | Start search (`mode`: `planner` or `rule_based`) |
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
## Benchmark: best-path score, planner vs rule-based

`configs/benchmark_runs.json` holds 50 target/beam combos. Targets fall inside the 10th–90th percentile range of the seed library, and velocity and pressure scale with density. Depth is 3–8, width 3–8, top-k 2–5, and each combo has its own random seed. The file explains how the combos were chosen. `scripts/run_benchmark.py` runs every combo in both `planner` and `rule_based` mode, so 100 runs in total. `scripts/analyze_best_path.py` plots the mean best-path `overall.score` against generation, one line per mode with a ±1 SD band.

Copy-paste from a fresh clone (Linux/macOS; on Windows use `.venv\Scripts\python`):

```bash
git clone https://github.com/josefberman/IGNITE.git && cd IGNITE
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/bin/python -e ".[dev,sascore,torch,benchmark]"
# (without uv: python3.11 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -e ".[dev,sascore,torch,benchmark]")

# optional smoke test: 1 combo, depth 2, both modes, separate log dir
.venv/bin/python scripts/run_benchmark.py --limit 1 --depth-override 2 --log-dir logs/benchmark_smoke

# full benchmark (100 runs); re-run the same command to resume after an interruption
IGNITE_RUN_LOG_DIR=logs/benchmark .venv/bin/python scripts/run_benchmark.py

# plot
.venv/bin/python scripts/analyze_best_path.py --log-dir logs/benchmark --out-dir docs/figures
```

**500-combo suite (1000 runs).** `configs/benchmark_runs_500.json` is produced by `scripts/generate_benchmark_config.py` (fixed `random.Random(20261018)`, same ranges and schema, ids `combo_001`–`combo_500`, seeds 2000–2499). Select it with `--suite 500` (or any file with `--config PATH`; the default stays the 50-combo suite). Use a separate log dir:

```bash
git pull
uv pip install -p .venv/bin/python -e ".[dev,sascore,torch,benchmark]"
.venv/bin/python scripts/generate_benchmark_config.py   # optional: regenerates the identical file
.venv/bin/python scripts/run_benchmark.py --suite 500 --log-dir logs/benchmark_500 --dry-run
.venv/bin/python scripts/run_benchmark.py --suite 500 --log-dir logs/benchmark_500   # resumable
.venv/bin/python scripts/analyze_best_path.py --log-dir logs/benchmark_500 --out-dir docs/figures/benchmark_500
```

Outputs:

- Each pair writes `logs/benchmark/<combo_id>__<mode>.json`. These are normal run logs with a `best_path`, plus a `benchmark` block that holds the combo id, the clamped beam, and the elapsed time. `logs/` is gitignored.
- The runner skips a pair if its log already exists, parses, and has a non-empty `best_path`. Each failure is appended to `logs/benchmark/failures.jsonl` with the error and traceback, and the runner moves on to the next pair. It exits with code 1 if any pair failed. Use `--dry-run` to list pending pairs and `--modes rule_based` to run one mode only.
- `docs/figures/best_path_score_vs_generation.{pdf,svg,png}` (the PNG is 300 dpi) and `best_path_score_vs_generation.csv`, which has the mean, SD, and n for each mode and generation.
- PubChem/OpenAlex and patent lookups are off, so runs need no network. Literature values come from the local seed library only.

**Lower is better.** `overall.score` is `ScoreBreakdown.total`: MAPE plus the SAScore penalty plus the minimal-edit penalty. Beam search keeps the lowest totals.

**Variable-length policy (carry-forward).** Best paths differ in length. At generation g, a run's value is the score of the last best-path step with generation ≤ g. After the path ends, its final score carries forward up to the highest generation seen in that mode, because the best molecule no longer changes. Mean and SD (ddof=0) are taken over all runs of a mode at each generation.
