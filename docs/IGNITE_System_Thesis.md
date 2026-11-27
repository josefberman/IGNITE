# IGNITE: A Computational Framework for Target-Driven Energetic Molecular Design

**Intelligent Generation and Navigation through Iterative Transformations of Energetics**

*This is the only project document that explains the system and reports results. Numbers below are the ones measured in this repository. Where a study has not been run, the section says so and gives the protocol, not a result.*

---

## How this document maps onto a paper

| Paper role | Section |
| --- | --- |
| Need for a path designer on four properties | [§1](#1-need-for-a-reaction-path-designer) |
| Why energetic-materials design is a distinct problem | [§2](#2-why-energetic-materials-design-is-difficult) |
| Beam search as the search framework | [§4](#4-beam-search) |
| Three planners that propose the next reaction and the next molecule | [§6](#6-three-planner-architectures) |
| Databases | [§7](#7-databases) |
| Training | [§8](#8-training) |
| Comparison with the rule-based modifier | [§9](#9-results) |
| Ablation of the family head and the site head | [§10](#10-ablation-protocol-not-yet-run) |
| Methods a paper or supplement still needs (catalog, properties, filters, software) | [§3](#3-task), [§5](#5-reaction-catalog-and-the-rule-based-proposer), [§11](#11-properties-validity-safety-and-novelty), [§12](#12-implementation) |

Claims that are safe to carry into a paper are marked as measured. The family-head and site-head lesion study in §10 has not been implemented.

---

## Abstract

IGNITE designs a short reaction path from a known energetic molecule toward four user-specified properties: crystal density, detonation velocity, detonation pressure, and solid heat of formation. The search is a beam search. At each node a proposer names a reagent and a reaction site; RDKit applies that reagent's SMARTS; an XGBoost model estimates the four properties of the product; invalid and critically hazardous structures are rejected; survivors are ranked by property error and synthetic accessibility.

Two proposers share one catalog of 90 reagents in 12 families. The rule-based proposer ranks reagents by the sign of their design effect and picks a site by a seeded shuffle. The learned proposer is a multitask message-passing network with a family head, a reagent head, and a site head, in three layouts (§6). Training uses recorded reactions only (§7). On 3,420 recorded evaluation rows, architecture 3 recovers the recorded reagent on 88.8% of rows, against 2.9% for the sign ranker. Given the recorded reagent, its site head recovers the recorded product on 91.2% of 3,416 rows, against 69.0% for the shuffle.

The head ablations in §10 have not been run.

Outputs are computational candidates. They are not experimental validations, hazard assessments, or legal novelty opinions.

---

## 1. Need for a reaction-path designer

An energetic formulation is specified by how the material performs, not by a target structure. The performance quantities used here are four:

| Symbol | Key | Meaning | Units | Accepted range |
| --- | --- | --- | --- | --- |
| ρ | `density` | Crystal / bulk density | g/cm³ | [0.8, 2.5] |
| D | `det_velocity` | Detonation velocity | m/s | [1000, 12000] |
| P | `det_pressure` | Detonation pressure | GPa | [0.1, 60] |
| ΔHf | `hf_solid` | Solid heat of formation | kJ/mol | [−1000, 1200] |

Density, detonation velocity, and detonation pressure are coupled: velocity and pressure rise with density and with the heat released on decomposition. Heat of formation is the chemical contribution to that heat release. A designer who matches one coordinate and ignores the other three has not matched the specification. When the paper is written, cite the performance equations actually used (Kamlet–Jacobs or the equivalent) at this point; this repository does not re-derive them. IGNITE treats the four numbers as the target vector

```text
t = (t_ρ, t_D, t_P, t_ΔHf)
```

and searches for a molecule whose estimated vector is close to t.

A structure generator that emits a SMILES string does not answer the practical question. A bench chemist needs a path: which known molecule to start from, which reaction to run, and at which atom. IGNITE is that path designer. It keeps a library of published energetic molecules, proposes one catalogued reaction at a time, and stops when the property error is inside a threshold or the depth limit is reached. The product of a run is the best molecule, its score, and the sequence of parents that produced it.

The system is a research tool for controlled settings. It does not claim a synthesis recipe, a measured performance, or a statement that a molecule is new.

---

## 2. Why energetic-materials design is difficult

The difficulty is not generic molecular optimization. It comes from the way energetic performance, chemistry, and the available data interact.

**The four targets move together, and often in opposite directions.** Adding a nitro group, an azide, a fluoro substituent, or a hydrocarbon side chain shifts density, oxygen balance, and heat of formation at once. A reagent that raises density can lower the heat of formation. The catalog therefore labels every reagent with a design direction in {−1, 0, +1}⁴, eighty-one possible sign patterns. Matching the sign of the gap is not the same as choosing the reaction: several reagents share one direction and make different bonds.

**One edit rarely closes the gap.** The residual after a single reaction is still a four-dimensional vector. The search has to commit to a sequence, and an early reaction changes which later reactions are possible. Exhaustive enumeration of sequences is not available: each reagent can match many atoms, and each step multiplies the candidate set.

**The reacting atom matters as much as the reaction.** The same reagent on two atoms of one parent yields two molecules with different properties. Symmetric atoms that give the same product are one site; distinct products are different sites. A proposer that names only the reagent has not named the next molecule.

**Experimental coverage is thin.** Published crystal densities and detonation parameters exist for a small set of known energetics. Most products of a proposed edit have never been measured. IGNITE estimates them with XGBoost models trained on a tabulated set of about 16,000 molecules (`sample_start_molecules.csv`). Those estimates are the labels on every training pair and the scores inside the search. They are not calorimeter or detonation-test values. Out-of-domain products, including unusual cages and high-nitrogen rings, are where the estimator is weakest.

**Reaction corpora are not energetic-materials corpora.** The recorded reactions used here are US patents and a small non-patent set (§7). They are dominated by pharmaceutical and process chemistry. An IGNITE reagent is counted only when its SMARTS reproduces a recorded product. Most patent reactions do not match. Nitro, nitramine, azide, tetrazole, and cage chemistry are a thin slice of those corpora, which is why the merged table has 64,505 matched reactions rather than the millions of raw patent rows.

**Safety and accessibility constrain the same atoms that improve performance.** Nitration, azidation, and N-oxidation are ordinary energetic edits and are allowed. Peroxides and long nitrogen chains are hard rejects. Synthetic accessibility (SAScore) enters the ranking. A path that reaches the property target through an unacceptable structure is discarded. Motif checks are computational filters, not a laboratory hazard assessment.

**There is no inverse map from (ρ, D, P, ΔHf) to a structure.** The operational answer is search: start from a measured molecule that is already near the target, and apply a short sequence of explicit reactions.

---

## 3. Task

### 3.1 What is optimized

Find a molecule m* reached by a path of catalogued reactions from a seed, minimizing

```text
total = 0.65 · mape_norm + 0.25 · s
```

where `mape_norm = min(MAPE / 100, 1)` and s ∈ [0, 1] is the synthetic-accessibility penalty (lower is better). Lower total is better. The edit-size proxy `|n_atoms(child) − n_atoms(parent)|` is stored on the modification record and is not part of the total (`minimal_edit_penalty` is written as 0).

MAPE across the four properties, with equal weights w_i = 0.25:

```text
MAPE(y, t) = Σ_i  w_i · percentage_error_i

percentage_error_i = |y_i − t_i| / |t_i| × 100    if |t_i| > 1e-10
                     |y_i − t_i| × 100            otherwise
```

A missing property is dropped and the remaining weights are renormalized. If no property is present, MAPE is 100%.

While the best MAPE is still above the user threshold, ranking uses explore weights `0.85 · mape_norm + 0.10 · s`, plus 0.04 if the candidate is an unedited generation-0 seed. That term keeps the beam from sitting on the starting library. After the run, survivors are re-scored with the final weights for display and export.

### 3.2 Constraints

- The molecule must parse, sanitize, be a single component, and contain only H, B, C, N, O, F, P, S, Cl, Br, I.
- Critical motifs (peroxide, long nitrogen chain, an internal score ≥ 0.95) are rejected. An azide is not rejected for being an azide.
- An optional novelty gate rejects exact known identities. It does not change the score of an accepted molecule.
- Safety findings, uncertainty intervals, and novelty evidence are internal. They are absent from the API response, the GUI, and the exports.

### 3.3 What one search step must predict

Given the parent and the gap t − y(parent), the proposer returns:

1. a **reagent** (the reaction), optionally via a **family** (the class that contains it);
2. a **site**, realized as one sanitized product of that reagent's SMARTS.

The reagent is the next reaction. The site is the next molecule. Beam search then estimates properties and decides whether the molecule stays in the beam.

---

## 4. Beam search

Module: `backend/app/services/beam_search.py`.

Beam search is the outer algorithm because the decision is sequential and the objective is only known after the product is estimated. A single greedy step cannot revise an early reagent. A full tree of 90 reagents times their sites, to depth 10, is not enumerable. The beam keeps K partial paths, expands each by at most N valid children, and prunes back to K.

### 4.1 Controls

| Parameter | Symbol | Meaning | Default |
| --- | --- | --- | --- |
| Depth | X | Maximum modification iterations | 10 (GUI) |
| Width | N | Valid expansions attempted per parent per iteration | 5 |
| Top K | K | Survivors retained after each iteration | 3 (GUI), 5 (API model default) |
| MAPE threshold | τ | Stop when the best MAPE is at or below this percent | 1% |
| Seed | — | RNG for the rule-based site shuffle | 42 |
| Literature properties | flag | Use exact tabulated values when the canonical SMILES matches | on |
| Novelty gate | flag | Hard-reject exact known molecules | off |

The server clamps depth ≤ 20, width ≤ 50, and top K ≤ 25.

### 4.2 Seeds

Search starts from `data/seeds/energetic_seed_library.csv` (override with `IGNITE_SEED_CSV`): 77 published neutral energetic molecules, rebuilt by `scripts/build_energetic_seed_library.py`. The original core is TNT, RDX, HMX, CL-20, PETN, TATB, HNS, FOX-7, NTO, LLM-105, TNAZ, nitroglycerin, picric acid, DNAN, TNB, tetryl, DATB, nitroguanidine, EGDN, BTF, ETN, MHN, HNB, ONC, TEX, nitromethane, and 2,4-DNT. Later batches add nitroaromatics, nitrate esters, nitroalkanes, nitramines, heterocycles, and the peroxides TATP and HMTD. Salts and ions are excluded. Each of the four properties carries a value, a source, and a provenance (`literature` or `predicted`). Predicted values come from the XGBoost models and are not indexed as literature.

`sample_start_molecules.csv` (~16,359 rows) is the property-model table. The seed loader refuses it.

Selection (`backend/app/services/seed_selector.py`):

1. Keep seeds with all four properties.
2. Compute MAPE against the target.
3. Prefer seeds with MAPE ≤ max(τ, 25%), so a 1% stop threshold still yields a starting pool.
4. If none qualify, take the lowest-MAPE seeds.
5. Return up to `max_seeds` (at least K).

Seed properties are tagged `dataset`.

### 4.3 Iteration

1. Stop on cancellation.
2. Exploring is true while the best MAPE exceeds τ.
3. Stop if the best MAPE is at or below τ and at least one expansion iteration has finished. On the first iteration a passing seed still receives one modification pass.
4. For each parent, request up to N children from the selected proposer (§5 or §6).
5. Drop a child whose canonical SMILES is already in `seen`. Estimate properties. Apply the safety gate and, if enabled, the novelty gate. Score with explore or final weights.
6. Pool the current beam with the children and re-score.
7. If exploring and K > 1, reserve at least ⌈K/2⌉ slots for generation > 0. Fill the rest by ascending total. Keep K.
8. Update the global best: lower total, then lower MAPE. While exploring, prefer a child within about one MAPE point of the best seed.
9. Emit progress. Strip safety, novelty, and uncertainty before the payload leaves the server.

Termination is the MAPE threshold, depth X, or cancellation. The result is `best`, `top_k`, and the parent lineage of every modified candidate.

A 50-target suite of this loop is summarized in §9.3. It measures the search, not the next-reaction accuracy.

---

## 5. Reaction catalog and the rule-based proposer

Modules: `backend/app/services/reaction_primitives.py`, `backend/app/services/reagents.py`. A unit test keeps the primitive deltas and the rule-based list on the same ids.

### 5.1 Catalog

There are **12 families** and **90 reagents**. Families: nitration, azidation, heterocycle, oxidation, amination, halogenation, dehydrogenation, arylation, reduction, alkylation, aromatization, identity.

Each reagent has:

- one design direction (Δρ, ΔD, ΔP, ΔHf) with each component in {−1, 0, +1};
- three SMARTS spellings of the same transformation;
- an effect word per property (`raise`, `lower`, `neutral`) equal to the sign of that component.

The 90 reagents cover all 81 directions. Some directions have more than one reagent, because the reactions differ (aromatic nitration and aliphatic nitration share a sign pattern and are different chemistry). Seventy-eight reagents occur in the recorded USPTO pairs. The other twelve are kept so a direction with no patent hit still has a reagent. The zero direction is the reagent `hold` (a carbon atom unchanged). A separate identity primitive exists for the strategy-pool construction and is not in the proposer vocabulary.

The learned proposer predicts a family and then a reagent inside it. The rule-based proposer never reads the family. Both lists are the same reagent ids.

### 5.2 Rule-based proposer

Given parent properties y and targets t:

1. Score every reagent not yet tried on this parent by how its effect signs agree with the gap, weighted by the size of each gap. Ties break by reagent id (`rank_reagents_for_gap`).
2. Run that reagent's three SMARTS with RDKit `RunReactants`.
3. Sanitize. Drop multi-component products and products identical to the parent.
4. Shuffle the distinct products with the search seed and take the first. That shuffle is the site. The next expansion of the same parent continues at the next reagent.

Accepted children are tagged `planner="rule_based"`. The family field on that record is the reagent id, because no family was chosen.

This proposer is the baseline in §9. It is also a complete search mode: planner mode and rule-based mode are separate runs, selected in the GUI or by `mode` on the API. If the learned checkpoint is missing, planner mode returns an error. It does not switch to the rule-based proposer.

---

## 6. Three planner architectures

Module: `ml/models/architectures.py`, class `MultitaskPlanner`. The three training scripts are `scripts/train_arch1_parallel.py`, `scripts/train_arch2_cascade_fc.py`, and `scripts/train_arch3_cascade.py`.

All three instantiate the same trunk and the same three towers. They differ in whether a head can see the previous head, and in whether two dense layers sit on the shared graph vector.

### 6.1 Shared trunk

The encoder is residual bond message passing.

- Atom projection, then **3 shared message-passing layers**, hidden size 128, dropout 0.15, 5 bond-type embeddings (single, double, triple, aromatic, other).
- Mean-plus-max readout, FiLM from the signed normalized target−parent gap, and a gap-aware attention over atoms, fused into one graph vector.
- Three towers, each one hidden layer of width 128:
  - **family** — which class of reaction;
  - **reagent** — which reaction inside the class;
  - **site** — a logit per atom. At training time the loss is a softmax over the reagent's candidate sites, and every site that yields the labeled child is correct.

The family head and the reagent head together name the next reaction. The site head, followed by execution of the SMARTS, names the candidate molecules. Online, a candidate is returned only when its predicted properties are closer to the target than the parent (§6.4).

### 6.2 The three layouts

| | Conditioning | Shared dense layers | Parameters | Checkpoint |
| --- | --- | ---: | ---: | --- |
| Architecture 1 | parallel | 2 | 302,162 | `output/arch1/planner.pt` |
| Architecture 2 | cascade | 2 | 334,930 | `output/arch2/planner.pt` |
| Architecture 3 | cascade | 0 | 301,394 | `output/arch3/planner.pt` |

**Parallel (architecture 1).** Each tower reads the shared vector and the gap. The reagent tower does not receive the family, and the site tower does not receive the reagent. An error in one head is not an input to the next.

**Cascade (architectures 2 and 3).** The reagent tower reads the family tower's hidden state. The site tower reads the reagent tower's hidden state. Training and inference both use that forward pass, so the later head sees the model's own prediction. Architecture 2 applies two shared dense layers to the graph vector before the fork. Architecture 3 forks from the graph vector directly.

A fourth layout, with a private message-passing stack on the site branch only, was removed from the project.

### 6.3 What was held fixed

Hidden size 128, 3 shared message-passing layers, one-layer towers, dropout 0.15, the loss in §8, and the parent-grouped 80/10/10 split in §8.3. The comparison is of conditioning and of the shared dense layers, not of width or depth.

### 6.4 Online use

`ModificationMechanism.expand` in planner mode:

1. Encode the parent and the signed normalized gap t − y.
2. Take the top families (between 3 and 8, depending on width).
3. For each family, take the top reagents masked to that family (between 2 and 5).
4. Apply each reagent. The site head orders that reagent's products, highest mean site logit first.
5. Predict each product's four properties. Keep a product only when its MAPE to the target is lower than the parent's. Return the lowest MAPE first, up to the beam width.
6. Tag accepted children `planner="learned"` and record the predicted family and reagent.

The true family and the true reagent are not inputs at search time. The reagent is chosen from the family the model just predicted, and the site is chosen for the reagent the model just predicted. The gold labels in §8 are used only while a training pair is being scored.

---

## 7. Databases

Training and evaluation use recorded reactions only: `data/processed/unified_pairs.csv`, with counts and the checksum in `unified_pairs_meta.json`. A row is a reaction that was published, not a SMARTS application invented for the training file. It is kept when one reactant component, passed through one IGNITE reagent SMARTS, sanitizes to one recorded product component. The site is the reacting-atom list of that match. Columns are parent and child SMILES, family, reagent id, reacting atoms, the four property deltas, plus patent, year, and source.

Filters, applied the same way to every source:

- 6 to 80 heavy atoms;
- the molecule contains carbon;
- stereochemistry is removed, so a stereo mark the SMARTS does not write does not reject a real reaction;
- the SMARTS must reproduce a recorded product;
- the first hit per parent and reagent is kept;
- duplicate (parent, child, reagent) rows collapse to one.

`parent_*`, `child_*`, and `delta_*` are XGBoost predictions. The patents do not report the four properties. The file also stores a Bemis–Murcko scaffold label at 70/15/15, seed 42 (46,326 / 7,783 / 10,396 on the non-hold rows). The three architecture runs do not train on that label. They re-split by parent, 80/10/10, seed 42 (§8.3). Hold-out identity rows are added after the scaffold label: 64 parents, child equal to parent, all four deltas 0, reagent `hold`, site label a carbon atom. They are spread across the scaffold label 48 / 9 / 7, and the parent-grouped run counts them inside its own 80/10/10.

| Source | Rows after dedup | License |
| --- | ---: | --- |
| USPTO grants, 1976–Sep 2016 (Lowe, figshare 10.6084/m9.figshare.5104873, file `1976_Sep2016_USPTOgrants_smiles.rsmi`) | 34,271 | CC0-1.0 |
| USPTO applications, 2001–Sep 2016 (same figshare record) | 10,299 | CC0-1.0 |
| van der Lingen reaction SMILES, figshare 22491730 (2023) | 19,871 | CC-BY-4.0 |
| Open Reaction Database, non-patent subset | 64 | CC-BY-SA-4.0 |
| Hold identities | 64 | added so the zero-gap reagent `hold` has rows |
| **Total** | **64,569** | |

Raw reaction counts before the IGNITE filter are much larger (about 1.81 million grant reactions, 1.94 million application reactions, 0.73 million van der Lingen lines, 0.18 million non-patent ORD reactions). The filter is why the table is small. ORD contributes almost nothing because high-throughput coupling screens do not match this catalog.

Not used, and not a substitute: Pistachio, Reaxys, CAS, and the USPTO-50k / USPTO-MIT slices of the same grants.

The table uses the current 12-family, 90-reagent catalog. The counts the architectures train and test on are the parent-grouped ones in §8.3: 51,656 / 6,457 / 6,456.

### 7.1 Property models

`models/{density,det_velocity,det_pressure,hf_solid}.joblib`, each an `XGBRegressor` on a 119-dimensional descriptor (`backend/app/services/descriptors.py`):

1. molecular weight;
2. a zero-point-energy proxy from bond-frequency heuristics;
3. a volumetric heat-capacity stub, fixed at 0, kept so the vector length stays stable;
4. a HOMO–LUMO proxy from the Gasteiger charge span;
5. counts of a fixed SMARTS list;
6. aromatic-ring count, hydrogen-bond acceptor count, hydrogen-bond donor count.

If descriptors fail, the molecule is dropped from expansion. At search time the estimator (`property_estimator.py`) uses, in order, the seed's own tabulated value, an exact canonical-SMILES literature hit when that flag is on, then XGBoost. A nearby molecule does not donate its properties.

---

## 8. Training

Trainer: `ml/training/train_multitask.py`, called from `ml/training/cv_train.py` by the three architecture scripts.

### 8.1 Objective

For each pair the loss is the sum of:

- **family** cross-entropy over families whose SMARTS matches the parent, with the gold family always allowed;
- **reagent** cross-entropy over reagents in the true family that match the parent, with the gold reagent always allowed;
- **site** softmax over the gold reagent's candidate sites. Every site that produces the labeled child is a positive. If no candidate is labeled, the softmax is over all atoms.

Each term has weight 1. There is no property-delta regression in this loss. The supervised reaction is the recorded one, and it is a reaction that decreases the gap to the target shown as input.

During training that target is the recorded child's properties, or a nearby point, kept only when the child's MAPE to it is lower than the parent's (`sample_gap_reducing_target`). The encoded input is the signed, normalized target − parent, the same encoding inference uses for the user's target. Evaluation uses the recorded child as the target, so the reported top-1 does not depend on the draw.

### 8.2 Epoch selection

These three scores are how a training pair is graded. Each head is graded on its own decision, with the previous gold label supplied, because that label is known for the pair. Top-1 is the only cutoff: the argmax either matches the label or it does not.

```text
score = 0.25 × (family top-1) + (site top-1 | true reagent) + (reagent top-1 | true family)
```

Family is down-weighted because a correct reagent already implies a family. Early stopping keeps the epoch with the highest validation score. Patience is 7. The test split is scored once, on that epoch.

| Name | What is supplied | Definition |
| --- | --- | --- |
| Family top-1 | the parent and the gap | The top family is the labeled family |
| Reagent top-1 | the true family | The top reagent, among reagents in that family, is the labeled reagent |
| Site top-1 | the true reagent | The top site, among that reagent's candidate sites, is a site that yields the labeled child |

Supplying the true family to the reagent score, and the true reagent to the site score, is the same conditioning the loss uses: the reagent loss is over the true family's reagents, and the site loss is a softmax over the true reagent's sites. It measures that head, not the chain.

At inference those labels do not exist (§6.4). The reagent head is not given the true family, and the site head is not given the true reagent. Search uses the family the model just predicted, then the reagent predicted from that family, then the site predicted from that reagent. A correct site score in this table does not mean the search would have selected that reagent.

### 8.3 Run

| | |
| --- | --- |
| Data | `unified_pairs.csv`, recorded reactions, 12 families, 90 reagents |
| Split | Parent-grouped, 80/10/10, seed 42: 51,656 train / 6,457 validation / 6,456 test. Every row of one parent stays in one split. The scaffold label stored in the CSV is not used. |
| Optimizer | AdamW, learning rate 2×10⁻⁴, weight decay 10⁻⁴ |
| Batch size | 16 |
| Epochs | 30, patience 7 |
| Applicability masks | on |

---

## 9. Results

### 9.1 The three architectures

Held-out test from each run's model card (`output/arch1/model_card.json`, `output/arch2/model_card.json`, `output/arch3/model_card.json`). The split recorded there is parent-grouped, 80% train / 10% validation / 10% test, seed 42, on the 64,569 recorded pairs: 6,456 test pairs, of which 6,441 have a site label. The columns are the §8.2 scores. Higher is better. The epoch is the one early stopping kept.

| | Epoch | Family top-1 | Reagent top-1, true family | Site top-1, true reagent | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Architecture 1 | 18 | 0.921 | 0.908 | 0.912 | 2.0498 |
| Architecture 2 | 29 | 0.918 | 0.899 | 0.915 | 2.0430 |
| Architecture 3 | 20 | 0.920 | 0.906 | 0.913 | 2.0495 |

Architecture 1 has the highest selection score. Architecture 3 is 0.0003 behind it. Architecture 2 is lower on family top-1 and on reagent top-1, and higher on site top-1. Architecture 3 is the checkpoint used for the rule-based comparison in §9.2.

### 9.2 Architecture 3 against the rule-based proposer

Recorded evaluation rows. A row is kept when both methods return a decision. Figures: `docs/figures/reagent_confusion_planner.svg`, `docs/figures/reagent_confusion_rule_based.svg`, `docs/figures/site_confusion_planner.svg`, `docs/figures/site_confusion_rule_based.svg`.

| Decision | Rows | Architecture 3 | Rule-based |
| --- | ---: | ---: | ---: |
| Reagent | 3,420 | 0.888 | 0.029 |
| Site, given the recorded reagent | 3,416 | 0.912 | 0.690 |
| Gap direction | 3,420 | 0.891 | 0.376 |

**Reagent.** Architecture 3 takes its top family, then its top reagent inside that family, among reagents whose SMARTS matches the parent. The rule-based proposer takes the first gap-ranked reagent that matches the parent. The label is the reagent that produced the recorded child. The sign table has 81 directions and several reagents share one, so a ranker that sees only signs cannot identify which of those reagents was used. The learned head can, on this set, for 88.8% of rows.

**Site.** Both methods are given the recorded reagent, so the comparison is the site decision alone. A site is one distinct product; symmetric atoms that give the same product count as one site, numbered within that parent by the lowest reacting-atom index. Architecture 3 picks the candidate with the highest mean site-head score. The rule-based proposer picks the first product of a seeded shuffle. The head matches the recorded product on 91.2% of rows. The shuffle matches on 69.0%, which is what a shuffle does when many reagents have only one or two products.

**Gap direction.** Same reagent choices as the reagent row. A choice is correct when its design signs (Δρ, ΔD, ΔP, ΔHf) equal the recorded reagent's design signs. The reagent id may differ. Architecture 3 matches the direction on 89.1% of rows, only 0.3 points above its exact-reagent accuracy, so when it names the right direction it almost always names the recorded reagent. The rule-based ranker matches the direction on 37.6% of rows. That is the task it is built for, and it is much higher than its 2.9% exact-reagent accuracy: it often points the same way as the recorded reaction and still names a different reagent.

This is a next-reaction and next-product comparison. It is not a claim about the beam-search score in §9.3.

### 9.3 Search trajectories

`docs/figures/benchmark_s50_d20_w5_k3/` plots, over 50 targets, the mean best-path score and the mean normalized MAPE against generation. The beam is width 5, depth 20, top K 3. Lower is better. A run that ends early keeps its final value at later generations (carry-forward), and each mode is extended only to the longest path observed for that mode.

| Mode | Last generation reported | Mean score ± s.d. | Mean MAPE/100 ± s.d. |
| --- | ---: | ---: | ---: |
| Learned planner | 19 | 0.055 ± 0.065 | 0.052 ± 0.090 |
| Rule-based | 14 | 0.055 ± 0.047 | 0.049 ± 0.063 |

On this suite the two modes finish at the same mean ranking score. The rule-based paths are shorter, and the standard deviation of the planner's final score is larger. The figure CSV does not record which checkpoint was loaded. Do not cite these curves as an architecture-3 result until that checkpoint is identified. Next-reaction accuracy (§9.2) and closed-loop score (§9.3) are different measurements; the first favors architecture 3 by a wide margin, and the second, on this 50-target suite, does not.

### 9.4 What a paper can and cannot claim from the present numbers

Can claim, with the dataset named:

- On the recorded rows behind the confusion figures, architecture 3 names the reagent on 88.8% of rows and the gap direction on 89.1%, against 2.9% and 37.6% for the sign ranker. Given the recorded reagent, its site head matches on 91.2%, against 69.0% for the shuffle.

Cannot claim yet:

- that architecture 3 improves the beam-search score over the rule-based proposer;
- any result from the ablations in §10.

---

## 10. Ablation protocol (not yet run)

No family-head ablation and no site-head ablation has been trained or evaluated. The protocol below is what the paper's ablation section should run. Do not fill the tables until the runs exist.

The question is whether each auxiliary head earns its place inside architecture 3, which is the selected model. The trunk, the optimizer, the split, and the reagent loss stay fixed. Only the lesion changes.

### 10.1 Family head

The family head is a coarse label over the twelve classes, and a correct reagent already determines the family. Architecture 3 still feeds the family hidden state into the reagent tower. The reagent top-1 in §8.2 supplies the true family, and the site top-1 supplies the true reagent. Those are training scores. Inference does not supply either label (§6.4). Removing the family head could hurt the cascade, because the reagent tower would lose that hidden state, or it could be neutral, because the reagent head already solves the finer task.

Three runs, same architecture-3 trunk:

| Run | Family loss | What the reagent tower reads |
| --- | --- | --- |
| A0, reference | on | family hidden state (architecture 3 as published) |
| A1, lesion | off | shared vector and gap only; no family tower |
| A2, auxiliary | on | shared vector and gap only; family is trained but not fed forward |

A1 asks whether family supervision is necessary. A2 asks whether the benefit, if any, is the cascade connection rather than the extra gradient.

Report, on the same test split: reagent top-1, all-three accuracy (site still required; family counted correct only for A0 and A2), selection score, and parameter count. Then run the beam-search suite of §9.3 with each checkpoint and report final mean score and mean MAPE. The search suite is required, because §9.3 already shows that a large reagent-accuracy gap need not appear as a score gap.

### 10.2 Site head

The site head is the step from "which reaction" to "which molecule." The offline comparison in §9.2 shows 0.912 against 0.690 for a shuffle when both methods are handed the recorded reagent. Search still uses that head to name a reagent's candidate products, then keeps the candidate whose predicted properties are closest to the target (§6.4). A lesion has to say whether that site choice is what improves the beam.

Three runs. The reagent tower is architecture 3 in every case. The site policy changes.

| Run | Site head | How the product is chosen |
| --- | --- | --- |
| B0, reference | trained, cascade from the reagent hidden | site head names the products; the one with the lowest MAPE to the target is kept (§6.4) |
| B1, lesion | not trained; its loss weight is 0 | seeded shuffle of the chosen reagent's products, as in §5.2 |
| B2, estimator site | not trained | rank the chosen reagent's products by alignment of the XGBoost delta with the gap, with no site head |

Report site top-1 given the gold reagent, site top-1 given the model's own reagent, and the beam-search final score and MAPE. B1 versus B0 is the shuffle lesion. B2 versus B0 asks whether the property models alone, which the search already trusts for ranking, are enough to pick the atom.

### 10.3 Controls

- Same seed, recorded-reaction split, and epoch-selection rule as §8. For A1 and B1 the selection score drops the missing term; write the modified formula into the model card.
- Train the reference and every lesion on `unified_pairs.csv`. Report them on that test split.
- One seed is not a variance estimate. If the paper reports a difference of a few thousandths in beam score, repeat the training seed. A lesion smaller than the reagent gap in §9.2 should be described as small.

### 10.4 Status

| Study | Status |
| --- | --- |
| Architecture comparison on the recorded test split, §9.1 | defined, scores not yet in this document |
| Reagent and site comparison with the rule-based proposer, §9.2 | measured, recorded evaluation rows |
| Search trajectories, §9.3 | measured, checkpoint not recorded in the figure |
| Family-head lesion A1, A2 | not run |
| Site-head lesion B1, B2 | not run |

---

## 11. Properties, validity, safety, and novelty

### 11.1 Validity

`backend/app/services/chemistry.py`. A candidate is rejected for: SMILES parse failure (`parse_failed`), sanitize or valency failure (`sanitize_failed`), more than one fragment (`multi_component`), an element outside H, B, C, N, O, F, P, S, Cl, Br, I (`disallowed_element_*`), or a canonical SMILES equal to the parent (`unchanged`). The beam's `seen` set is canonical SMILES.

Reaction products additionally pass `sanitize_reaction_product` (kekulization, extra hydrogen on ring nitrogen, shedding of excess hydrogen, drop of a second N-oxide).

### 11.2 Safety

`backend/app/services/safety.py`. Findings stay inside the process. They feed the hard gate and the accessibility term.

SAScore uses RDKit Contrib `SA_Score` or `BRSAScore` when installed, otherwise a heuristic on rings, rotatable bonds, size, heteroatoms, and motif strings (about 1–10), mapped smoothly into [0, 1]. Motifs scanned: peroxide, long nitrogen chain, azide, diazo, nitro together with peroxide.

```text
safety_score = min(1, f(SA) + min(0.4, 0.1 · (motif count)))
```

Hard reject: peroxide, `invalid`, `long_n_chain`, or `safety.score ≥ 0.95`. The soft score is the accessibility penalty s in §3.1. It is not a separate safety term in the published total.

### 11.3 Novelty gate

`molecule_identity.py`, `novelty.py`. Providers run only when `novelty_as_gate` is set. Resolution is canonical SMILES, InChIKey, PubChem CID, IUPAC name and synonyms, the seed table, OpenAlex, and configured patent providers. Exact identity only. There is no analogue or fingerprint neighbour test.

| Condition | Status | Gate |
| --- | --- | --- |
| Exact seed or a confirmed database hit | `known` | reject |
| A provider is missing or the search is incomplete | `incomplete_search` | keep |
| Checked providers return no exact hit | `no_match_found` | keep |

A missing API key does not establish that a molecule is new. It only fails to reject it.

### 11.4 Uncertainty

`uncertainty.py`. Source tags (`dataset`, `literature`, `xgboost`) are kept internally. Confidence intervals are not computed for the client and do not appear in progress, results, or exports.

---

## 12. Implementation

### 12.1 Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI, Pydantic, port 8080 |
| Jobs | SQLite, background threads |
| Chemistry | RDKit |
| Property models | XGBoost, joblib |
| Planner | PyTorch, optional |
| Frontend | React, TypeScript, Vite, port 5173, proxying `/api` |
| Containers | Docker Compose under `infra/` |

Port 8000 is avoided because it collides with the HPLIP web UI on some Linux machines.

### 12.2 API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Liveness |
| GET | `/api/v1/models` | Model metadata, including whether the learned planner loaded |
| POST | `/api/v1/search` | Start a beam search (`mode` is `planner` or `rule_based`) |
| GET | `/api/v1/search/{id}/progress` | Poll |
| GET | `/api/v1/search/{id}/events` | Server-sent progress |
| GET | `/api/v1/search/{id}` | Final result |
| POST | `/api/v1/search/{id}/cancel` | Cancel |
| GET | `/api/v1/search/{id}/export.csv` | CSV |
| GET | `/api/v1/search/{id}/export.sdf` | SDF |
| GET | `/api/v1/molecule/depict?smiles=&size=` | RDKit 2D PNG |

Out-of-range targets return HTTP 422.

### 12.3 Interface

`frontend/src/App.tsx`. Target form with units; depth, width, top K, MAPE threshold, literature toggle, novelty toggle, and the modifier (learned or rule-based). Live progress shows iteration, beam size, generated and rejected counts, and the best MAPE and score. A scoring panel states `total = 0.65 × mape_norm + 0.25 × sascore`. Candidate cards show the 2D structure, SMILES, InChIKey, the four properties with their source, and the score components. The lineage is a chain of structures. CSV export omits safety, novelty, and uncertainty.

---

## 13. Limitations and dual-use

1. **Estimator error.** The four properties of proposed molecules are XGBoost predictions except where a seed or an exact literature hit exists. The training pairs use those same predictions as deltas.
2. **Recorded reactions only.** Training pairs are published reactions that an IGNITE SMARTS reproduces. The four property deltas on those rows are still XGBoost predictions. The patents do not report them.
3. **Search versus next-reaction.** Reagent accuracy and beam score did not move together on the 50-target suite, and that suite's checkpoint is not identified in the figure.
4. **Corpus bias.** Matched patent reactions are whatever the SMARTS happened to reproduce. Energetic-specific reactions are a small fraction of USPTO.
5. **Novelty.** An incomplete provider search is not evidence of novelty.
6. **Safety.** Motif lists and SAScore are not a hazard assessment and not a permit to synthesize.
7. **Dual-use.** The software proposes modifications to energetic molecules. It is a research instrument for controlled use. It is not a production recipe generator.
8. **Ablations.** §10 is a protocol. It is not a result.

---

## 14. Glossary

| Term | Meaning |
| --- | --- |
| MAPE | Mean absolute percentage error against the four targets |
| Gap | Target minus the parent's estimated properties, or child minus parent in a training pair |
| Direction | The sign pattern of a reagent's design effect, in {−1, 0, +1}⁴ |
| Family | One of 12 reaction classes. Predicted by the learned proposer only |
| Reagent | One catalogued transformation, with three SMARTS spellings |
| Site | One distinct product of a reagent on a parent |
| Beam depth X | Maximum number of modification iterations |
| Beam width N | Expansions requested per parent per iteration |
| Top K | Survivors kept after each iteration |
| Explore mode | Reweighting used while the best MAPE is above the threshold, including a penalty on unedited seeds |
| Hold | An identity training row: child equals parent, gap is zero, reagent is `hold` |
| Architecture 3 | Cascade family → reagent → site, 3 shared message-passing layers, no shared dense layers. The selected learned proposer |

---

## 15. Repository map

| Resource | Path |
| --- | --- |
| High-level diagram | `docs/architecture/IGNITE_High_Level_Architecture.png` |
| Modification diagram | `docs/architecture/IGNITE_Modification_Architecture.png` |
| Beam search | `backend/app/services/beam_search.py` |
| Ranking | `backend/app/services/scoring.py` |
| Rule-based proposer | `backend/app/services/reagents.py` |
| SMARTS catalog | `backend/app/services/reaction_primitives.py` |
| Learned expansion | `backend/app/services/modification.py` |
| Planner model | `ml/models/architectures.py` |
| Training | `ml/training/train_multitask.py`, `ml/training/cv_train.py` |
| Architecture scripts | `scripts/train_arch1_parallel.py`, `scripts/train_arch2_cascade_fc.py`, `scripts/train_arch3_cascade.py` |
| Recorded pairs | `ml/data_gen/uspto_pairs.py`, `data/processed/unified_pairs.csv` |
| Architecture-3 confusion figures | `docs/figures/reagent_confusion_*.svg`, `docs/figures/site_confusion_*.svg` |
| Search-trajectory figures | `docs/figures/benchmark_s50_d20_w5_k3/` |
| Seeds | `data/seeds/energetic_seed_library.csv` |
| Property models | `models/*.joblib` |
| License | `LICENCE.md`, `docs/LICENSING.md` |

---

*Paper source. Ranking is 0.65 MAPE + 0.25 accessibility; edit size is recorded and unweighted. The training table is recorded reactions only. The family-head and site-head ablations are specified in §10 and have not been run.*
