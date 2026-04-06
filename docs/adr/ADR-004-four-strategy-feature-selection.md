# ADR-004: Three-Stage Feature Selection and Consensus Workflow

**Status:** Accepted | **Date:** 2026-01-26 | **Updated:** 2026-02-09

## Decision

**Three-stage workflow for feature selection and cross-model consensus:**

| Stage | Component | Purpose | Output |
|-------|-----------|---------|--------|
| **1. Model Gate** | Permutation test | Filter models with real signal | p-value per model |
| **2. Per-Model Evidence** | OOF importance (primary), stability (hard filter), RFE (sizing), drop-column (post-hoc) | Ranking and interpretation per model | Ranked feature lists per model |
| **3. Consensus** | Geometric mean rank aggregation | Cross-model robust biomarkers | Final deployment panel |

**Other methods:**
- Nested RFECV (during training): RFECV per fold -> consensus
- Fixed Panel (validation): train on predetermined panel

## Rationale

**Stage 1: Model Gate**
- Formal statistical test: classifier performance > chance
- Prevents consensus aggregation across noise models
- Pooled-null aggregation: increases power by combining evidence across folds/seeds

**Stage 2: Per-Model Evidence**
- **OOF importance (primary ranking signal):** Held-out importance closer to generalization; correlation grouping prevents twin-feature artifacts
- **Stability (hard filter):** Proteins must meet a minimum selection frequency threshold (e.g., 0.90-0.95) to enter ranking -- removes noisy, inconsistently selected features
- **RFE (independent sizing):** Panel sizing / selection path via Pareto curve analysis. Not an input to consensus ranking
- **Drop-column (post-hoc interpretation):** Refit-and-ablate on the final consensus panel to measure each cluster's contribution (delta-AUROC primary, delta-PR-AUC, delta-Brier). Not an input to ranking

**Stage 3: Cross-Model Consensus (Geometric Mean Rank Aggregation)**
- Per-model: contribute OOF importance ranks (after stability filter)
- Geometric mean of normalized reciprocal ranks across models (missing = bottom rank)
- Correlation clustering on top candidates, select representatives by consensus score
- Top-N selection for final panel
- Post-hoc drop-column on final panel for interpretation

## Typical Workflow

```bash
# Stage 1: Train models with feature selection
ced train --model LR_EN,RF,XGBoost --split-seed 0,1,2
ced aggregate-splits --run-id <RUN_ID>

# Stage 2: Model gate (permutation testing)
ced permutation-test --run-id <RUN_ID> --model LR_EN --n-jobs 4
ced permutation-test --run-id <RUN_ID> --model RF --n-jobs 4
ced permutation-test --run-id <RUN_ID> --model XGBoost --n-jobs 4

# Per-model evidence (automatically computed during training/aggregation):
# - OOF grouped importance: feature_importance/oof_grouped_importance.csv
# - Stability frequency: feature_selection/stability/feature_stability.csv
# - RFE rank: optimize_panel/{model}/rfe_ranking.json (independent sizing)

# Stage 3: Cross-model consensus (only for significant models, e.g., p < 0.05)
ced consensus-panel --run-id <RUN_ID> --models LR_EN,RF

# Validation (new seed critical)
ced train --fixed-panel consensus_panel/final_panel.txt --split-seed 10
```

## Stage Details

### Stage 1: Model Gate (Permutation Test)

- **Algorithm:** Label permutation test of AUROC (per ADR-011)
- **Output:** `permutation_test/{model}/aggregated_results.json`, p-value
- **Decision:** Keep only models with `p < alpha` (default: 0.05)
- **Runtime:** ~1-4 hrs per model (HPC, B=200)
- **Implementation:** [permutation_test.py](../../src/ced_ml/significance/permutation_test.py), [aggregation.py](../../src/ced_ml/significance/aggregation.py)

### Stage 2: Per-Model Evidence

**Input 1 (Primary): OOF Grouped Importance**
- **Method:** OOF grouped permutation importance (trees) or standardized |coef| (linear)
- **Output:** `feature_importance/oof_grouped_importance.csv` (rank per model)
- **Implementation:** [grouped_importance.py](../../src/ced_ml/features/grouped_importance.py)

**Input 2 (Post-hoc): Drop-Column Essentiality**
- **Method:** Remove cluster -> refit (fixed hyperparams) -> delta-AUROC (primary), delta-PR-AUC, delta-Brier
- **Role:** Post-hoc interpretation on the final consensus panel. NOT an input to ranking
- **Output:** `consensus/essentiality/within_panel_essentiality.csv`
- **Implementation:** [drop_column.py](../../src/ced_ml/features/drop_column.py)

**Input 3 (Sizing): RFE Rank**
- **Method:** RFE with CV-based stopping (panel sizing via Pareto curve)
- **Role:** Independent panel size optimization. NOT an input to consensus ranking
- **Output:** `optimize_panel/{model}/rfe_ranking.json` (elimination order)
- **Implementation:** [rfe.py](../../src/ced_ml/features/rfe.py)

**Input 4 (Hard Filter): Stability Frequency**
- **Method:** Selection frequency across CV folds (>= threshold, e.g. 0.90-0.95)
- **Role:** Hard pre-filter for consensus entry. Proteins below threshold are excluded
- **Output:** `feature_selection/stability/feature_stability.csv`
- **Implementation:** [stability.py](../../src/ced_ml/features/stability.py)

### Stage 3: Cross-Model Consensus

**Algorithm:**
```
Step 1 -- Per-model ranking:
    For each significant model:
        1. Hard filter: keep proteins with stability_freq >= threshold
        2. Rank survivors by OOF grouped importance (descending)
        3. If OOF unavailable, fall back to stability frequency ranking

Step 2 -- Cross-model aggregation:
    Aggregate per-model OOF importance ranks via geometric mean
    of normalized reciprocal ranks.
    - Missing proteins penalized (assigned bottom rank)
    - Correlation-cluster top candidates, select representatives
    - Extract top-N panel

Step 3 -- Post-hoc drop-column (interpretation only):
    On the final consensus panel:
    - Refit model on panel features only
    - Run drop-column per cluster across all CV folds
    - Report delta-AUROC (primary), delta-PR-AUC, delta-Brier
    - Saved as interpretation artifact, NOT used for ranking
```

**Output:** `consensus/`
- `final_panel.txt` - Top-N proteins for deployment
- `final_panel.csv` - Panel with consensus scores and uncertainty metrics
- `consensus_ranking.csv` - All proteins with consensus scores
- `uncertainty_summary.csv` - Per-protein uncertainty metrics
- `per_model_rankings.csv` - Per-model OOF importance rankings
- `correlation_clusters.csv` - Cluster assignments
- `consensus_metadata.json` - Run parameters and statistics
- `essentiality/within_panel_essentiality.csv` - Post-hoc drop-column results

**Note:** Uses geometric mean rank aggregation, not the formal Kolde RRA with beta-model p-values. See FEATURE_SELECTION.md for details.

**Implementation:** [consensus/](../../src/ced_ml/features/consensus/) (ranking, aggregation, clustering, builder), [consensus_panel.py (CLI)](../../src/ced_ml/cli/consensus_panel.py)

## Runtime

(LR_EN, 43k samples, 2920 proteins, 5x10 CV):

| Component | Time | Notes |
|-----------|------|-------|
| Model gate (permutation test) | 1-4 hrs | HPC, B=200, per model |
| OOF grouped importance | <5 min | Computed during aggregation |
| Drop-column essentiality | ~10 min | Post-hoc on final panel only |
| RFE rank | ~5 min | Per model, start size ~100 |
| Stability frequency | <1 min | Computed during training |
| Cross-model consensus | ~15 min | All models |

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| No model gate (use all models) | Consensus aggregation across noise models reduces signal |
| Fisher/Stouffer aggregation (vs pooled-null) | Assumes independence across CV folds (violated by resampling) |
| Composite weighting (OOF + essentiality + stability) | Drop-column is interpretability, not ranking; stability is better as hard filter; silently degraded when signals absent |
| Formal Kolde RRA with beta-model p-values | Geometric mean is simpler, sufficient for our use case, and avoids distributional assumptions |

## Evidence

**Code:**
- Model gate: [permutation_test.py](../../src/ced_ml/significance/permutation_test.py), [aggregation.py](../../src/ced_ml/significance/aggregation.py), [permutation_test.py (CLI)](../../src/ced_ml/cli/permutation_test.py)
- OOF importance: [grouped_importance.py](../../src/ced_ml/features/grouped_importance.py), [importance.py](../../src/ced_ml/features/importance.py)
- Essentiality (post-hoc): [drop_column.py](../../src/ced_ml/features/drop_column.py)
- RFE: [rfe.py](../../src/ced_ml/features/rfe.py), [optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py)
- Stability: [stability.py](../../src/ced_ml/features/stability.py)
- Consensus: [consensus/](../../src/ced_ml/features/consensus/) (ranking, aggregation, clustering, builder), [consensus_panel.py](../../src/ced_ml/cli/consensus_panel.py)
- Legacy: [screening.py](../../src/ced_ml/features/screening.py), [kbest.py](../../src/ced_ml/features/kbest.py), [nested_rfe.py](../../src/ced_ml/features/nested_rfe.py)
- Fixed panel: [train.py](../../src/ced_ml/cli/train.py) `--fixed-panel`

**Tests:** `tests/significance/test_permutation_test.py`, `tests/significance/test_aggregation.py`, `tests/features/test_grouped_importance.py`, `tests/features/test_drop_column.py`, `tests/features/test_rfe.py`, `tests/features/test_stability.py`, `tests/features/test_consensus.py`, `tests/cli/test_consensus_panel.py`

**Docs:** [FEATURE_SELECTION.md](../reference/FEATURE_SELECTION.md), [CLI_REFERENCE.md](../reference/CLI_REFERENCE.md)

**Refs:**
- Ojala & Garriga (2010). Permutation tests for studying classifier performance. JMLR 11:1833-1863.
- Breiman (2001). Random Forests. Machine Learning 45(1).
- Strobl et al. (2008). Conditional variable importance for random forests.
- Lei et al. (2018). Distribution-Free Predictive Inference For Regression. JASA 113(523).
- Guyon et al. (2002). Gene selection for cancer classification using support vector machines. Machine Learning 46(1-3).
- Meinshausen & Buhlmann (2010). Stability selection. JRSS-B 72(4):417-473.
- Kolde et al. (2012). Robust rank aggregation for gene list integration. Bioinformatics 28(4):573-580.

## Related

- Extends: ADR-011 (permutation testing, now Stage 1 model gate)
- Depends: ADR-005 (nested CV provides folds for OOF importance, stability)
- Complements: ADR-006 (Optuna tuning), ADR-007 (ensemble after feature selection)
