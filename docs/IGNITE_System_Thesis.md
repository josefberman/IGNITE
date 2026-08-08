# IGNITE: A Computational Framework for Target-Driven Energetic Molecular Design

**Intelligent Generation and Navigation through Iterative Transformations of Energetics**

*Technical thesis / system monograph*

---

## Abstract

IGNITE is an end-to-end computational platform for designing candidate energetic molecules that match user-specified energetic property targets. The system combines (i) MAPE-based selection from a curated seed library of approximately sixteen thousand molecules, (ii) iterative beam search over chemical modifications executed with RDKit SMARTS reaction primitives, (iii) property estimation via exact experimental matches when available and otherwise pretrained XGBoost regressors, (iv) an optional novelty gate that rejects exact known molecules, and (v) a transparent multi-objective ranking score. A React web interface exposes targets, beam parameters, live progress, 2D structure depictions, a top-of-page scoring explanation (including ranking weights), and per-candidate score components without repeating those weights.

**Scope and disclaimer.** All outputs are computational design candidates. They are not experimental validations, legal novelty opinions, or patentability advice. Safety heuristics, uncertainty intervals, and novelty/literature/patent evidence are used only internally (hard gates / SAScore) and are not reported in the UI or API responses.

---


## Table of contents

1. [Introduction and motivation](#1-introduction-and-motivation)
2. [Problem formulation](#2-problem-formulation)
3. [System architecture](#3-system-architecture)
4. [Data assets](#4-data-assets)
5. [Molecular representation and descriptors](#5-molecular-representation-and-descriptors)
6. [Seed selection](#6-seed-selection)
7. [Modification mechanism](#7-modification-mechanism)
8. [Chemical validity filtering](#8-chemical-validity-filtering)
9. [Property estimation](#9-property-estimation)
10. [Internal uncertainty (not reported)](#10-internal-uncertainty-not-reported)
11. [Safety estimation and hard gates](#11-safety-estimation-and-hard-gates)
12. [Novelty gate (optional)](#12-novelty-gate-optional)
13. [Candidate scoring and ranking](#13-candidate-scoring-and-ranking)
14. [Beam search algorithm](#14-beam-search-algorithm)
15. [Learned modification stack](#15-learned-modification-stack)
16. [Software stack and API](#16-software-stack-and-api)
17. [User interface](#17-user-interface)
18. [Default parameters and validation ranges](#18-default-parameters-and-validation-ranges)
19. [Limitations and ethical considerations](#19-limitations-and-ethical-considerations)
20. [Glossary](#20-glossary)
21. [References within the repository](#21-references-within-the-repository)

---


## 1. Introduction and motivation

Energetic materials design traditionally relies on expert intuition, group-contribution heuristics, and expensive experimental campaigns. Machine learning can accelerate the search for molecules whose predicted density, detonation velocity, detonation pressure, and solid heat of formation approach a desired profile, while still respecting chemical validity and synthetic accessibility.

IGNITE operationalizes this search as an interactive beam-search pipeline:

- start from real molecules with known or tabulated properties;
- propose chemically motivated edits;
- estimate properties for each child;
- reject invalid or critically hazardous structures;
- rank survivors with an explicit, auditable score;
- optionally hard-reject exact known molecules via the novelty gate.

The platform is deliberately modular: each box in the architecture diagrams maps to a concrete module (see `[docs/architecture/TRACEABILITY.md](architecture/TRACEABILITY.md)`).

---


## 2. Problem formulation


### 2.1 Target vector

The user specifies a four-dimensional target:

```text
t = (t_ρ, t_D, t_P, t_ΔHf)
```

corresponding to:

| Symbol | Internal key | Meaning | Units | Allowed range |
|--------|--------------|---------|-------|---------------|
| t_ρ | `density` | Crystal / bulk density | g/cm³ | [0.8, 2.5] |
| t_D | `det_velocity` | Detonation velocity | m/s | [1000, 12000] |
| t_P | `det_pressure` | Detonation pressure | GPa | [0.1, 60] |
| t_ΔHf | `hf_solid` | Solid heat of formation | kJ/mol | [−1000, 1200] |


### 2.2 Beam parameters


| Parameter                           | Symbol | Meaning                                                    |
| ----------------------------------- | ------ | ---------------------------------------------------------- |
| Depth                               | X      | Maximum number of modification iterations                  |
| Width                               | N      | Maximum valid expansions attempted per beam parent         |
| Top K                               | K      | Number of survivors retained after each iteration          |
| MAPE threshold | τ | Early-stop target for property error (%)                   |
| Random seed                         | —      | Deterministic RNG for modification sampling                |
| Literature properties / novelty gate | flags  | Exact literature properties; optional hard novelty filter |


### 2.3 Optimization objective (informal)

Find a molecule m* that minimizes a weighted combination of property error (MAPE), synthetic-accessibility penalty, and edit complexity, subject to:

- chemical validity (RDKit sanitize, valency, single component, allowed elements);
- hard safety rejection for critical motifs;
- optional novelty gate (reject exact known hits).

Safety, uncertainty, and novelty evidence are **not reported** to the client; only the optional novelty gate is exposed as a user input.

---


## 3. System architecture


### 3.1 High-level pipeline

```text
User targets + beam params + seed CSV
        │
        ▼
 Seed selector (MAPE vs targets)
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │              Beam search loop (≤ X)          │
 │  for each parent in beam:                    │
 │     Modification mechanism → children        │
 │     Validity filter                          │
 │     Property estimation (exact lit / XGB)    │
 │     Uncertainty intervals (report only)      │
 │     Safety estimate + hard gates             │
 │     Novelty / literature / patents (exact)   │
 │     Score + rank → keep top K                │
 └──────────────────────────────────────────────┘
        │
        ▼
 Best molecule + top-K audit trail → GUI / export
```

Diagram: `[docs/architecture/IGNITE_High_Level_Architecture.png](architecture/IGNITE_High_Level_Architecture.png)`

### 3.2 Modification sub-pipeline

```text
Parent molecule ──► Encoder (graph / descriptors)
Target gap/delta ──► Delta encoder
        │
        ▼
 Modification planner
   (reaction family → reactive site → reagent/fragment)
        │
        ▼
 RDKit SMARTS reaction executor
        │
        ▼
 Chemical validity filter
        │
        ▼
 Property / Δ-property estimate → safety → ranker
```

Diagram: `[docs/architecture/IGNITE_Modification_Architecture.png](architecture/IGNITE_Modification_Architecture.png)`

In the current runtime path, planning is primarily **rule-based** via a curated SMARTS strategy pool, with an optional learned Graph Transformer planner loaded when `ml/checkpoints/planner.pt` exists.

---


## 4. Data assets


### 4.1 Seed library

File: `sample_start_molecules.csv`

- Approximately **16,359** rows of SMILES with energetic properties.
- After canonicalization and completeness filtering, on the order of **~14k** unique complete molecules are available for seeding.
- Columns: `density`, `det_velocity`, `det_pressure`, `hf_solid`, `SMILES`.
- Incomplete property rows are excluded from seed selection.
- Duplicate canonical SMILES are collapsed, retaining lineage to the source row where recorded.

Normalized derivatives (optional scripts):

- `data/processed/seeds_canonical.csv`
- `data/indexes/seed_index.json`


### 4.2 Absolute property models

Directory: `models/`


| File                  | Target                  |
| --------------------- | ----------------------- |
| `density.joblib`      | Density                 |
| `det_velocity.joblib` | Detonation velocity     |
| `det_pressure.joblib` | Detonation pressure     |
| `hf_solid.joblib`     | Solid heat of formation |


Each is an `xgboost.sklearn.XGBRegressor` consuming a fixed **119-dimensional** descriptor vector (see §5). Sidecar metadata (`*.meta.json`) records units, descriptor contract version, and path.

### 4.3 Reaction primitive library

Module: `backend/app/services/reaction_primitives.py`

- **34** named primitives (including identity), each with literature-cited SMARTS variants where applicable.
- Direction tuples (Δρ, ΔD, ΔP, ΔHf) ∈ {−1, 0, +1}⁴ define an **81-entry** strategy pool by L1-nearest primitive matching.
- Used both for online beam expansion and for offline weak-supervision pair generation (`ml/data_gen/generate_pairs.py`).

---


## 5. Molecular representation and descriptors


### 5.1 Identifiers


| Identifier                 | Role                                                       |
| -------------------------- | ---------------------------------------------------------- |
| Input SMILES               | User / generator string                                    |
| Canonical SMILES           | Deduplication, exact literature match, novelty exact match |
| InChIKey                   | External lookup key when available                         |
| IUPAC name                 | Literature / patent query term (PubChem)                   |
| Commercial / synonym names | Literature / patent query terms (PubChem synonyms)         |


Canonicalization and InChIKey are computed with RDKit (`backend/app/services/chemistry.py`).

### 5.2 Descriptor contract (XGBoost)

Module: `backend/app/services/descriptors.py`

The feature vector (length **119**) is:

1. Molecular weight
2. Estimated zero-point energy proxy from bond frequency heuristics
3. Volumetric heat capacity stub (currently 0.0; retained for model compatibility)
4. HOMO–LUMO proxy from Gasteiger charge span
5. Counts of a fixed list of SMARTS substructures
6. Number of aromatic rings, H-bond acceptors, H-bond donors

If descriptor generation fails, the molecule cannot be XGBoost-predicted and is rejected from expansion.

---


## 6. Seed selection

Module: `backend/app/services/seed_selector.py`

### 6.1 Algorithm

1. Load and canonicalize the seed CSV; keep rows with all four properties present.
2. For each seed s with property vector y_s, compute MAPE(y_s, t).
3. Prefer seeds with MAPE ≤ τ_select.
4. If none qualify, take the globally closest seeds (lowest MAPE).
5. Return up to `max_seeds` candidates (beam search typically requests at least K).

In the beam controller, the selection threshold used for pooling is max(τ, 25%) so that a very tight user τ (e.g. 1%) still yields a diverse starting set of nearby seeds; the user τ still governs early stopping of the search itself.

### 6.2 MAPE definition

For properties i ∈ {ρ, D, P, ΔHf} with equal weights w_i = 0.25:

```text
MAPE(y, t) = Σ_i  w_i · percentage_error_i

where percentage_error_i =
    |y_i − t_i| / |t_i| × 100    if |t_i| > 1e-10
    |y_i − t_i| × 100            otherwise
```

Missing predicted properties cause that term to be skipped (with renormalization by the total weight of present terms). If no terms are present, MAPE defaults to 100%.

Seed properties are tagged `source = dataset` with confidence 0.9.


## 7. Modification mechanism

Modules: `modification.py`, `reaction_primitives.py`

### 7.1 Property gap → direction tuple

Given parent properties y and targets t, the gap for each property is t_i − y_i. Each gap is quantized:

```text
d_i = +1  if gap > 0.01
d_i = −1  if gap < −0.01
d_i =  0  otherwise
```

yielding a direction key d ∈ {−1, 0, +1}⁴.


### 7.2 Strategy pool lookup

The 81-key strategy pool maps d to a reaction primitive (L1-nearest Δ-vector among curated primitives; ties broken by energetic-chemistry priority). The identity key (0, 0, 0, 0) is a no-op.


### 7.3 Execution

1. Sample a SMARTS variant for the selected primitive.
2. Run `AllChem.ReactionFromSmarts` / `RunReactants` on the parent.
3. Sanitize products; drop multi-component or unchanged structures.
4. If fewer than N children are obtained, explore neighboring direction keys and remaining primitives.

Each accepted child is annotated with a `ModificationRecord` (family, SMARTS, edit-distance proxy, planner mode: `rule_based` or `learned`).

### 7.4 Edit-distance proxy

```text
edit(parent, child) = |n_atoms(child) − n_atoms(parent)|
```

Used later in scoring as a minimal-edit penalty.


---


## 8. Chemical validity filtering

Module: `backend/app/services/chemistry.py` — `validate_molecule`

A candidate fails the hard filter if any of the following hold:


| Condition                                                      | Reason code            |
| -------------------------------------------------------------- | ---------------------- |
| SMILES parse failure                                           | `parse_failed`         |
| RDKit sanitize / valency failure                               | `sanitize_failed`      |
| More than one connected component (when required)              | `multi_component`      |
| Disallowed element (outside H, B, C, N, O, F, P, S, Cl, Br, I) | `disallowed_element_*` |
| Canonical SMILES identical to parent                           | `unchanged`            |


Survivors are stored under their canonical SMILES and optional InChIKey. Global beam deduplication uses a set of seen canonical SMILES.

---


## 9. Property estimation

Module: `backend/app/services/property_estimator.py`

### 9.1 Cascade (exact molecule only)

For a candidate SMILES m:

1. **Known seed properties** (initialization): tag as `dataset`.
2. Else if literature enabled: **exact canonical SMILES** match in the experimental seed index → tag as `literature` (confidence ≈ 0.92).
3. Else: **XGBoost** absolute predictors → tag as `xgboost`.

**There is no analogue / fingerprint property transfer.** Nearby molecules do not donate property values.

### 9.2 Prediction

Module: `prediction.py`


```text
ŷ_i = f_i(φ(m))  for i ∈ {ρ, D, P, ΔHf}
```

where φ(m) is the 119-d descriptor. Failure of any required model prediction causes the candidate to be discarded from that expansion step.

---


## 10. Internal uncertainty (not reported)

Module: `backend/app/services/uncertainty.py`

Property estimates carry a source tag (`dataset`, `literature`, `xgboost`, …). Confidence intervals and progress-level uncertainty summaries are **not** computed for client reporting and do not appear in the GUI, progress payload, or exports.

---


## 11. Safety estimation and hard gates

Module: `backend/app/services/safety.py`

Safety findings are **internal only** (excluded from API serialization). They feed hard rejection and the SAScore ranking term; they are never shown in the UI.

### 11.1 SAScore

Prefer RDKit Contrib `SA_Score` / `BRSAScore` when available; otherwise a heuristic based on rings, rotatable bonds, size, heteroatoms, and motif strings (range ≈ 1–10).

Normalize to [0,1] with a smooth map f(SA).

### 11.2 Motif scan

SMARTS / patterns for: peroxide, long nitrogen chain, azide, diazo, nitro+peroxide.


```text
safety_score = min(1, f(SA) + min(0.4, 0.1 · #motifs))
```


### 11.3 Acceptability vs ranking

- Soft signal `safety.score` contributes only via the SAScore ranking weight (as the accessibility penalty input).
- Hard rejection in beam search for critical motifs: `peroxide`, `invalid`, `long_n_chain`.
- Also reject if `safety.score ≥ 0.95` or peroxide present under the acceptability rule.
- Azides are common in energetic chemistry and are not automatic hard rejects solely for being azides.

---


## 12. Novelty gate (optional)

Modules: `molecule_identity.py`, `novelty.py`

Novelty / literature / patent evidence is **not reported** in results. Providers run only when `novelty_as_gate` is enabled, and only to hard-reject candidates with status `known`. Rejection increments the rejected count; accepted molecules carry no novelty payload.

### 12.1 Exact identity resolution (gate only)

For SMILES m:

1. Canonicalize; compute InChIKey.
2. Query PubChem by SMILES → CID.
3. Fetch IUPAC name and synonyms when available.
4. Check exact seed SMILES, PubChem presence, OpenAlex, and configured patent providers.

### 12.2 Status used by the gate


| Condition                             | Status              | Gate action    |
| ------------------------------------- | ------------------- | -------------- |
| Exact seed or confirmed database hit  | `known`             | Hard-reject    |
| Providers missing / incomplete        | `incomplete_search` | Keep candidate |
| Checked providers return no exact hit | `no_match_found`    | Keep candidate |


**No fingerprint “close analogue” novelty status is used.** Analogue search was intentionally removed.

### 12.3 Novelty gate (user input)

If `novelty_as_gate` is enabled, candidates (and seeds) with status `known` are hard-rejected from the beam. This gate does **not** alter the numerical ranking formula for accepted molecules.

---


## 13. Candidate scoring and ranking

Module: `backend/app/services/scoring.py`

### 13.1 Design principle

The ranking objective prioritizes **property match**, then **synthetic accessibility**, then **edit simplicity**. Safety, uncertainty, and novelty are not reported to clients; only hard safety rejection and the optional novelty gate act as filters.

### 13.2 Components

#### (A) Normalized MAPE — weight 0.65

```text
mape_norm = min(MAPE / 100, 1)
```

#### (B) SAScore / accessibility penalty — weight 0.25

Uses the normalized accessibility signal s ∈ [0, 1] (lower is better for ranking). This comes from the SAScore-derived field on `SafetyFinding.score` (including soft motif penalties in that field), but it is **not** a separate safety ranking term.

#### (C) Minimal-edit penalty — weight 0.10

```text
e = min(1, edit_distance / 10)
```

Seeds without a modification record contribute e = 0.

### 13.3 Total score (final ranking)

```text
total = 0.65 · mape_norm + 0.25 · s + 0.10 · e
```

**Lower is better.** Top-K selection and the final “best molecule” use this total (with MAPE as a tie-break where implemented).

In the web UI, the numeric weights (0.65 / 0.25 / 0.10) are documented only in the top scoring-explanation panel (§17). Candidate cards show the component values (MAPE, SA penalty, edit penalty, total) without repeating the weights.

### 13.4 Explore-mode weights

While the search is still above the MAPE threshold, an explore mode uses:

```text
total_explore = 0.85 · mape_norm + 0.10 · s + 0.05 · e + δ_seed
```

where δ_seed = 0.04 if the candidate is an unedited generation-0 dataset seed. This prevents the beam from freezing on the original seeds and reserves beam slots for generated children (§14). After the run, survivors are **re-scored with the final weights** of §13.3 for display and export.


### 13.5 Explicitly excluded from reporting and from the total


| Signal                             | Role                                              |
| ---------------------------------- | ------------------------------------------------- |
| Uncertainty / confidence intervals | Not reported                                      |
| Safety findings                    | Internal hard gates + SAScore input only          |
| Novelty / literature / patents     | Optional gate only; no status or links in results |


---


## 14. Beam search algorithm

Module: `backend/app/services/beam_search.py`

### 14.1 Inputs

- Job id, `TargetProperties`, `BeamParameters`  
- Shared services: seed library, property estimator, novelty service, modification mechanism


### 14.2 Initialization

1. Select seed pool (§6).
2. Attach internal safety (and novelty only if gate enabled); score with explore weights.
3. Initialize beam = top K seeds; `seen` = their canonical SMILES; track global best.


### 14.3 Iteration g = 1 ldots X

For each iteration:

1. **Stop** if cancellation requested.
2. Set `exploring = (best.MAPE > τ)`.
3. **Early stop** if `best.MAPE ≤ τ` and at least one expansion iteration has completed (iteration index > 1). On the first iteration, even if a seed already meets tau, one modification pass is still attempted to seek improved / alternative candidates.
4. For each parent in the beam, request up to N expansions.
5. For each child:
  - skip if SMILES already in `seen`;  
  - estimate properties (literature flag honored);  
  - compute safety; hard-reject critical motifs / extreme scores;  
  - evaluate novelty; optionally novelty-gate;  
  - score with explore or final weights as appropriate;  
  - add to children set.
6. Form pool = beam ∪ children; re-score under current explore flag.
7. **Survivor selection** `_select_survivors`:
  - if exploring and K > 1, reserve at least lceil K/2 rceil slots for `generation > 0` children when available;  
  - fill remaining slots by ascending total score;  
  - return sorted top K.
8. Update global best (prefer lower total; tie-break by MAPE; while exploring, prefer children that are within ~1 MAPE point of the seed best).
9. Emit progress (iteration, beam size, generated/rejected counts, best MAPE/score, source tallies). Strip safety/novelty/uncertainty before returning results.


### 14.4 Termination

- MAPE threshold reached  
- Depth X exhausted  
- Cancellation  
- (Implicit) empty feasible expansions continue with remaining beam / seed fallbacks as coded


### 14.5 Output

- `best`: lowest final-score molecule among survivors  
- `top_k`: ranked survivors  
- Full lineage of parent SMILES for modified candidates  
- Progress metadata and disclaimer

---


## 15. Learned modification stack

Directory: `ml/`

### 15.1 Weak supervision

`ml/data_gen/generate_pairs.py` applies SMARTS primitives to seed parents, validates children, predicts absolute properties with XGBoost, and stores:

- parent/child SMILES, scaffold, reaction family, SMARTS
- `reagent_id` — stable SMARTS-variant id `{family}__v{i}` for the reagent head
- `site_atom_indices` — parent atom indices that react (from RDKit `GetReactingAtoms` mapped through the reactant match; falls back to the full match)
- Delta y_i = y_i^{child} - y_i^{parent} labeled `weak_xgb_delta`
- scaffold-based train/val/test splits

External measured reaction / hazard CSVs can be ingested via `ml/data_gen/ingest_external.py` when licensed data are available. Sensitivity / decomposition heads are **not** trained from SAScore proxies alone.

### 15.2 Models

`ml/models/architectures.py`:

- `GraphTransformerEncoder` — parent (and child) graph encoding from RDKit atom features  
- `TargetDeltaEncoder` — numeric property-delta MLP  
- `ModificationPlanner` — hierarchical heads for reaction family, site scores, and reagent (SMARTS variant)
- `DeltaPropertyPredictor` — multi-task Δy regression
- Optional `UncertaintyEnsemble` for epistemic spread

Training entrypoint: `ml/training/train_planner.py` → checkpoint `ml/checkpoints/planner.pt` (family CE + reagent CE + reacting-site BCE + Δy MSE). The planner sees a **signed normalized (target − parent)** gap (same encoding at inference), with optional target jitter during training so it does not memorize exact outcome deltas. Early stopping saves the best checkpoint by validation **family top‑1** (unmasked).

### 15.3 Online use

When `ml/checkpoints/planner.pt` loads:

1. Encode parent graph + target−parent Δ vector with `ModificationPlanner`.
2. Take top-k **reaction families** from the family head.
3. For each family, take top-k **reagents** from the reagent head **masked** to that family’s variants (`{family}__v{i}`).
4. Score atoms with the **site head** and prefer SMARTS matches that hit those atoms.
5. Execute preferred SMARTS via RDKit in reagent order; optionally reorder children with `DeltaPropertyPredictor` (cosine alignment of predicted Δy to the target gap).
6. Tag accepted children `planner="learned"` and record `reagent` as the variant id.

If the checkpoint is missing/unloadable or too few valid children are produced, the rule-based strategy pool (and remaining primitives) fill the expansion. Beam search itself is unchanged: it still calls `ModificationMechanism.expand` per parent.

---


## 16. Software stack and API


### 16.1 Components


| Layer        | Technology                                   |
| ------------ | -------------------------------------------- |
| API          | FastAPI + Pydantic                           |
| Jobs         | SQLite-backed job store + background threads |
| Chemistry    | RDKit                                        |
| ML inference | XGBoost + joblib; optional PyTorch planner   |
| Frontend     | React + TypeScript + Vite                    |
| Containers   | Docker Compose under `infra/`                |


### 16.2 Principal endpoints


| Method | Path                                    | Purpose           |
| ------ | --------------------------------------- | ----------------- |
| GET    | `/api/v1/health`                        | Liveness          |
| GET    | `/api/v1/models`                        | Model metadata    |
| POST   | `/api/v1/search`                        | Start beam search |
| GET    | `/api/v1/search/{id}/progress`          | Poll progress     |
| GET    | `/api/v1/search/{id}/events`            | SSE progress      |
| GET    | `/api/v1/search/{id}`                   | Final result      |
| POST   | `/api/v1/search/{id}/cancel`            | Cancel            |
| GET    | `/api/v1/search/{id}/export.csv`        | CSV export        |
| GET    | `/api/v1/search/{id}/export.sdf`        | SDF export        |
| GET    | `/api/v1/molecule/depict?smiles=&size=` | RDKit 2D PNG      |


Default API port: **8080** (avoids conflict with HPLIP on port 8000). Vite proxies `/api` to that port.

---


## 17. User interface

Frontend: `frontend/src/App.tsx`

Features:

- Target property form with unit labels and range validation (server-side)
- Beam parameters: Depth, Width, Top K, MAPE threshold (%), literature-properties and novelty-gate toggles
- Live progress: iteration, beam size, generated/rejected, best MAPE/score
- Scoring explanation panel at the top: MAPE, SA, minimal-edit, combined formula, and the ranking weights (0.65 / 0.25 / 0.10)
- Candidate cards with RDKit 2D structures, SMILES, InChIKey, property table (value + source), and score component values (MAPE, SA, edit, total) — **without** listing score weights
- Modification lineage depicted as a chain of 2D structures
- CSV export of ranked candidates (no safety / novelty / uncertainty columns)

---


## 18. Default parameters and validation ranges


### 18.1 GUI / API defaults


| Parameter      | Default                         |
| -------------- | ------------------------------- |
| Depth          | 10                              |
| Width          | 5                               |
| Top K          | 3 (GUI) / 5 (API model default) |
| MAPE threshold | 1%                              |
| Literature properties | on                         |
| Novelty gate   | off                             |
| Random seed    | 42                              |


Server clamps: depth ≤ 20, width ≤ 50, top K ≤ 25 (configurable via environment).

### 18.2 Property ranges

See §2.1. Out-of-range targets are rejected with HTTP 422.

---


## 19. Limitations and ethical considerations

1. **Model error.** XGBoost predictions are approximate and trained on a finite domain; OOD molecules may be poorly predicted.
2. **Weak Δ-labels.** Synthetic parent→child deltas are not experimental ground truth.
3. **Novelty gate incompleteness.** Missing API keys or provider outages do not prove novelty; they only fail to reject.
4. **Safety heuristics.** Computational motif and SAScore checks are internal filters, not laboratory hazard assessments.
5. **Dual-use.** Energetic materials research has dual-use implications; this software is intended for controlled research contexts and must not be treated as a production recipe generator.
6. **No analogue property borrowing.** Exact-molecule policy improves provenance clarity but may reduce coverage for rare structures (XGBoost fills the gap).

---


## 20. Glossary


| Term                 | Definition                                                                    |
| -------------------- | ----------------------------------------------------------------------------- |
| MAPE                 | Mean Absolute Percentage Error vs user targets                                |
| Beam depth X         | Max modification iterations                                                   |
| Beam width N         | Expansions per parent per iteration                                           |
| Top K                | Survivors retained each iteration                                             |
| SAScore              | Synthetic Accessibility Score (~1–10)                                         |
| SMARTS               | Pattern language for chemical reactions/substructures                         |
| Exact literature hit | Property or database record for the same canonical SMILES / resolved identity |
| Explore mode         | Temporary reweighting to admit generated children while above MAPE threshold  |


---


## 21. References within the repository


| Resource             | Path                                                      |
| -------------------- | --------------------------------------------------------- |
| High-level diagram   | `docs/architecture/IGNITE_High_Level_Architecture.png`    |
| Modification diagram | `docs/architecture/IGNITE_Modification_Architecture.png`  |
| Module traceability  | `docs/architecture/TRACEABILITY.md`                       |
| Licensing notes      | `docs/LICENSING.md`                                       |
| Threat model         | `docs/THREAT_MODEL.md`                                    |
| Beam controller      | `backend/app/services/beam_search.py`                     |
| Scoring              | `backend/app/services/scoring.py`                         |
| Property estimation  | `backend/app/services/property_estimator.py`              |
| Novelty / identity   | `backend/app/services/novelty.py`, `molecule_identity.py` |
| Reaction primitives  | `backend/app/services/reaction_primitives.py`             |
| Descriptors          | `backend/app/services/descriptors.py`                     |
| Domain contracts     | `backend/app/domain/models.py`                            |


---

*Document version: aligned with IGNITE scoring weights MAPE 0.65 / SA 0.25 / minimal-edit 0.10 (shown only in the UI explanation panel); no client reporting of safety, uncertainty, or novelty (novelty gate input retained); RDKit depiction UI.*