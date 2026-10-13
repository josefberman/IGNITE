# IGNITE Modification Planner — Model Card

- **Name:** ignite_modification_planner
- **Version:** 0.4.0
- **Hidden size:** 32 (`DEFAULT_HIDDEN`)
- **Planner gap input:** signed normalized `(target − parent)` per property (`ml/planner_gap.py`); training adds target jitter
- **Components:** GraphTransformerEncoder, TargetDeltaEncoder, hierarchical family / site / reagent heads, DeltaPropertyPredictor
- **Training data:** `data/processed/modification_pairs.csv` (`weak_xgb_delta` + `reacting_atom_sites` + `reagent_variant`)
- **Early stopping:** best checkpoint by validation **family top‑1** (unmasked)
- **Key metrics:** `fam` (family), `reag` (reagent unmasked), `reag_mask` (family-masked), `site`
- **Fallback:** Rule-based SMARTS strategy pool when checkpoint unavailable
