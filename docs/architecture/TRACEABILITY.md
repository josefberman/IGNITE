# Architecture Traceability

Maps every architecture-diagram box to an owned IGNITE module.

## High-level architecture

| Diagram box | Module |
|-------------|--------|
| Target properties from user | [`backend/app/domain/models.py`](../../backend/app/domain/models.py) `TargetProperties` + GUI form |
| Input parameters (X, N, K, MAPE) | `BeamParameters` + API validation in [`backend/app/main.py`](../../backend/app/main.py) |
| Seed molecules file | [`sample_start_molecules.csv`](../../sample_start_molecules.csv) + [`seed_selector.py`](../../backend/app/services/seed_selector.py) |
| Seed Selector (MAPE) | [`seed_selector.py`](../../backend/app/services/seed_selector.py) `select_seeds` |
| Beam Search Controller | [`beam_search.py`](../../backend/app/services/beam_search.py) |
| Modification Mechanism | [`modification.py`](../../backend/app/services/modification.py) + [`reaction_primitives.py`](../../backend/app/services/reaction_primitives.py) + [`ml/models/architectures.py`](../../ml/models/architectures.py) |
| Property Estimation (literature + XGBoost) | [`property_estimator.py`](../../backend/app/services/property_estimator.py), [`prediction.py`](../../backend/app/services/prediction.py), [`novelty.py`](../../backend/app/services/novelty.py) literature adapters |
| Uncertainty Estimate | [`uncertainty.py`](../../backend/app/services/uncertainty.py) |
| Candidate Scoring and Ranking | [`scoring.py`](../../backend/app/services/scoring.py) |
| Top Scored Molecule | API `SearchResult.best` + GUI |

## Modification architecture

| Diagram box | Module |
|-------------|--------|
| Parent Molecule Encoder | [`ml/models/architectures.py`](../../ml/models/architectures.py) `GraphTransformerEncoder` |
| Target Delta Encoder | `TargetDeltaEncoder` |
| Reaction Family / Site / Reagent selectors | `ModificationPlanner` (+ rule fallback in `ModificationMechanism`) |
| Reaction Executor | RDKit SMARTS in `reaction_primitives.py` / `modification.py` |
| Chemical Validity Filter | [`chemistry.py`](../../backend/app/services/chemistry.py) `validate_molecule` |
| Delta-Property Predictor | `DeltaPropertyPredictor` + absolute XGBoost baseline |
| Uncertainty Estimator | `uncertainty.py` + ensemble hooks |
| Safety Estimator | [`safety.py`](../../backend/app/services/safety.py) |
| Modification Ranker | `scoring.py` `score_candidate` |

## Diagram assets

- [High_level_architecture](./High_level_architecture-e5afa91f-2ff4-4c81-b303-625372a960ab.png)
- [Modification_architecture](./Modification_architecture-d6644a13-4211-4706-8302-c14b69f3eb04.png)
