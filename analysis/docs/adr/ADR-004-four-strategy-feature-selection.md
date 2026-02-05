# ADR-004: Three-Stage Feature Selection and Consensus Workflow

**Status:** Accepted | **Date:** 2026-01-26 | **Updated:** 2026-02-04

## Decision

**Three-stage workflow for feature selection and cross-model consensus:**

| Stage | Component | Purpose | Output |
|-------|-----------|---------|--------|
| **1. Model Gate** | Permutation test | Filter models with real signal | p-value per model |
| **2. Per-Model Evidence** | OOF importance (primary), drop-column (secondary), RFE (tertiary), stability (filter) | Four complementary importance ranks | Ranked feature lists per model |
| **3. Consensus** | Multi-list RRA + FDR | Cross-model robust biomarkers | Final deployment panel |

**Other methods:**
- Nested RFECV (during training): RFECV per fold → consensus
- Fixed Panel (validation): train on predetermined panel

## Rationale

**Stage 1: Model Gate**
- Formal statistical test: classifier performance > chance
- Prevents consensus aggregation across noise models
- Pooled-null aggregation: increases power by combining evidence across folds/seeds

**Stage 2: Per-Model Evidence (Four Inputs)**
- **OOF importance (primary):** Held-out importance closer to generalization; correlation grouping prevents twin-feature artifacts
- **Drop-column (secondary):** More faithful necessity test than single-feature PFI under correlation
- **RFE (tertiary):** Panel sizing / selection path, tie-breaker
- **Stability (filter/tie-break):** Filter noisy features, resolve ties, ensure generalization

**Stage 3: Sequential Filtering + RRA Consensus**
- Multi-list RRA: each model contributes OOF importance + essentiality + RFE ranks
- FDR correction (Benjamini-Hochberg): controls false positives across biomarkers
- Stability post-filter: ensures robust generalization
- Missing features = bottom rank (conservative)

## Typical Workflow

```bash
# Stage 1: Train models with feature selection
ced train --model LR_EN,RF,XGBoost --split-seed 0,1,2
ced aggregate-splits --run-id <RUN_ID>

# Stage 2: Model gate (permutation testing)
ced permutation-test --run-id <RUN_ID> --model LR_EN --hpc
ced permutation-test --run-id <RUN_ID> --model RF --hpc
ced permutation-test --run-id <RUN_ID> --model XGBoost --hpc

# Stage 3: Per-model evidence (automatically computed during training/aggregation)
# - OOF grouped importance: feature_importance/oof_grouped_importance.csv
# - Drop-column essentiality: drop_column/{model}/aggregated_results.csv
# - RFE rank: optimize_panel/{model}/rfe_ranking.json
# - Stability frequency: feature_selection/stability/feature_stability.csv

# Stage 4: Cross-model consensus (only for significant models, e.g., p < 0.05)
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

**Input 2 (Secondary): Drop-Column Essentiality**
- **Method:** Remove cluster → refit (fixed hyperparams) → ΔAUROC
- **Output:** `drop_column/{model}/aggregated_results.csv` (rank per model)
- **Implementation:** [drop_column.py](../../src/ced_ml/features/drop_column.py)

**Input 3 (Tertiary): RFE Rank**
- **Method:** RFE with CV-based stopping (panel sizing / tie-breaker)
- **Output:** `optimize_panel/{model}/rfe_ranking.json` (elimination order)
- **Implementation:** [rfe.py](../../src/ced_ml/features/rfe.py)

**Input 4 (Filter/Tie-break): Stability Frequency**
- **Method:** Selection frequency across CV folds (≥ 0.75 threshold)
- **Output:** `feature_selection/stability/feature_stability.csv`
- **Implementation:** [stability.py](../../src/ced_ml/features/stability.py)

### Stage 3: Sequential Filtering + RRA Consensus

**Algorithm:**
```
For each significant model:
    1. Filter by stability: keep blocks with stability_freq >= 0.60-0.75
    2. Rank by OOF importance: keep top K_1 blocks (e.g., 150)
    3. Run RFE inside shortlist to pick panel size (e.g., 25-40)
    4. Run drop-column on chosen panel: keep blocks with ΔAUROC above noise or top K_2

Multi-list RRA:
    - Per model: contribute OOF importance + essentiality + RFE ranks
    - RRA across all lists (missing = bottom rank)
    - FDR correction (Benjamini-Hochberg) → rra_q
    - Select by rra_q < alpha or top-K
```

**Output:** `consensus_panel/`
- `final_panel.txt` - Top-N proteins for deployment
- `consensus_ranking.csv` - RRA scores, p-values, FDR-adjusted q-values
- `uncertainty_summary.csv` - Per-protein uncertainty metrics
- `metadata.json` - Run parameters

**Implementation:** [consensus.py](../../src/ced_ml/features/consensus.py), [consensus_panel.py (CLI)](../../src/ced_ml/cli/consensus_panel.py)

## Runtime

(LR_EN, 43k samples, 2920 proteins, 5×10 CV):

| Component | Time | Notes |
|-----------|------|-------|
| Model gate (permutation test) | 1-4 hrs | HPC, B=200, per model |
| OOF grouped importance | <5 min | Computed during aggregation |
| Drop-column essentiality | ~10 min | Per model, panel size ~50 |
| RFE rank | ~5 min | Per model, start size ~100 |
| Stability frequency | <1 min | Computed during training |
| Multi-list RRA consensus | ~15 min | All models |

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| No model gate (use all models) | Consensus aggregation across noise models reduces signal |
| Fisher/Stouffer aggregation (vs pooled-null) | Assumes independence across CV folds (violated by resampling) |
| Traditional RRA (vs multi-list) | Each model contributes only one rank list (less evidence) |
| No FDR correction | High false positive rate across 2,920 proteins |

## Evidence

**Code:**
- Model gate: [permutation_test.py](../../src/ced_ml/significance/permutation_test.py), [aggregation.py](../../src/ced_ml/significance/aggregation.py), [permutation_test.py (CLI)](../../src/ced_ml/cli/permutation_test.py)
- OOF importance: [grouped_importance.py](../../src/ced_ml/features/grouped_importance.py), [importance.py](../../src/ced_ml/features/importance.py)
- Essentiality: [drop_column.py](../../src/ced_ml/features/drop_column.py)
- RFE: [rfe.py](../../src/ced_ml/features/rfe.py), [optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py)
- Stability: [stability.py](../../src/ced_ml/features/stability.py)
- Consensus: [consensus.py](../../src/ced_ml/features/consensus.py), [consensus_panel.py](../../src/ced_ml/cli/consensus_panel.py)
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
- Meinshausen & Bühlmann (2010). Stability selection. JRSS-B 72(4):417-473.
- Kolde et al. (2012). Robust rank aggregation for gene list integration. Bioinformatics 28(4):573-580.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. JRSS-B 57(1):289-300.

## Related

- Extends: ADR-011 (permutation testing, now Stage 1 model gate)
- Depends: ADR-005 (nested CV provides folds for OOF importance, essentiality, stability)
- Complements: ADR-006 (Optuna tuning), ADR-007 (ensemble after feature selection)
