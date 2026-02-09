# Feature Selection and Consensus Workflow

**Status:** Current | **Updated:** 2026-02-09

## Overview

Feature selection workflow comprises three sequential stages:

1. **Model Gate** - Permutation testing to identify models with real signal
2. **Per-Model Evidence** - Four complementary importance measures per model
3. **Sequential Filtering** - RRA consensus across significant models only

---

## Stage 1: Model Gate (Permutation Test)

**Goal:** Decide which models exhibit real signal vs. noise.

**Method:** Label permutation test of classifier AUROC (per ADR-011).

```bash
# Run permutation test (HPC recommended for B >= 200)
ced permutation-test --run-id <RUN_ID> --model LR_EN --hpc

# After completion, view results
ced permutation-test --run-id <RUN_ID> --model LR_EN
```

**Algorithm:**
- For each permutation b = 0..B-1:
  1. Shuffle y_train only (keep X fixed, held-out y unchanged)
  2. Re-run FULL inner pipeline (screening, feature selection, hyperparameter tuning)
  3. Predict on held-out fold (original X, original y)
  4. Record AUROC
- Compute empirical p-value: `p = (1 + #{null >= observed}) / (1 + B)`
- Pooled-null aggregation: pool all null AUROCs across folds/seeds (default)

**Output:**
- `permutation_test/{model}/aggregated_results.json` - Pooled p-value, observed/null statistics
- `permutation_test/{model}/null_distribution.png` - Histogram with observed AUROC line

**Decision:** Keep only models with `p < alpha` (default: 0.05) for downstream panel work.

**Implementation:**
- [permutation_test.py](../../src/ced_ml/significance/permutation_test.py) - Core algorithm
- [aggregation.py](../../src/ced_ml/significance/aggregation.py) - Pooled-null aggregation
- [permutation_test.py (CLI)](../../src/ced_ml/cli/permutation_test.py) - CLI
- [ADR-011](../adr/ADR-011-permutation-testing.md) - Design rationale

**References:**
- Ojala & Garriga (2010). Permutation tests for studying classifier performance. JMLR 11:1833-1863.
- Phipson & Smyth (2010). Permutation P-values should never be zero. Stat Appl Genet Mol Biol 9(1).

---

## Stage 2: Per-Model Feature Evidence

Four complementary importance measures per model (ranked independently):

### Input 1 (Primary): OOF Grouped Importance

**Rationale:** Held-out importance closer to generalization; correlation grouping prevents twin-feature artifacts.

**Method:**
- **Trees:** OOF grouped permutation importance on held-out folds
- **Linear:** Standardized |coef| on standardized inputs + stability across repeats/seeds

**Output:** `feature_importance/oof_grouped_importance.csv`

**Columns:**
- `feature/cluster` - Feature or cluster ID
- `importance/mean_importance` - Mean importance across CV folds
- `importance_std` - Standard deviation across folds
- `rank` - Rank by mean importance (1 = most important)

**Implementation:**
- [grouped_importance.py](../../src/ced_ml/features/grouped_importance.py) - OOF grouped PFI
- [importance.py](../../src/ced_ml/features/importance.py) - Per-fold importance extraction

**References:**
- Breiman (2001). Random Forests. Machine Learning 45(1).
- Strobl et al. (2008). Conditional variable importance for random forests.

---

### Input 2 (Post-hoc): Drop-Column Essentiality

**Rationale:** More faithful necessity test than single-feature PFI, especially under correlation. Used as **post-hoc interpretation** on the final consensus panel, not as an input to ranking.

**Method:** For each cluster in the final consensus panel:
1. Remove cluster and refit model (fixed hyperparams, refit-only)
2. Compute metrics on held-out fold
3. delta-AUROC = `original_auroc - ablated_auroc` (primary)
4. delta-PR-AUC = `original_pr_auc - ablated_pr_auc` (secondary)
5. delta-Brier = `original_brier - ablated_brier` (calibration impact)

**When to run:** Automatically as part of `ced consensus-panel` (post-hoc validation).

**Default mode:** Fixed hyperparams (refit-only) to answer: "Is this cluster essential under this modeling recipe?"

**Output:** `consensus/essentiality/within_panel_essentiality.csv`

**Columns:**
- `cluster_id` - Cluster ID
- `representative` - Cluster representative feature
- `cluster_features` - Comma-separated protein list
- `mean_delta_auroc` - Mean ΔAUROC across folds (primary)
- `std_delta_auroc` - Standard deviation
- `mean_delta_pr_auc` - Mean ΔPR-AUC across folds
- `std_delta_pr_auc` - Standard deviation
- `mean_delta_brier` - Mean ΔBrier across folds
- `std_delta_brier` - Standard deviation

**Implementation:**
- [drop_column.py](../../src/ced_ml/features/drop_column.py) - Drop-column validation
- Supports both individual features and correlation clusters

**References:**
- Lei et al. (2018). Distribution-Free Predictive Inference For Regression. JASA 113(523).

---

### Input 3 (Tertiary): RFE Rank

**Rationale:** Treat RFE as panel sizing/selection path, not primary scientific ranking signal. Use as tie-breaker or selection prior.

**Method:** Recursive Feature Elimination (RFE) with CV-based stopping.

**Output:** `optimize_panel/{model}/rfe_ranking.json` (dict: protein → elimination_order)

**Use cases:**
- Tie-breaking among features with similar OOF importance
- Panel size selection (Pareto curve analysis)
- Selection prior for consensus

**Implementation:**
- [rfe.py](../../src/ced_ml/features/rfe.py) - Aggregated RFE post-training
- [nested_rfe.py](../../src/ced_ml/features/nested_rfe.py) - Nested RFECV during training (deprecated for routine use)

**References:**
- Guyon et al. (2002). Gene selection for cancer classification using support vector machines. Machine Learning 46(1-3).

---

### Input 4 (Filter/Tie-break): Stability Frequency

**Rationale:** Filter noisy features; resolve ties; ensure generalization.

**Method:** For each feature, compute selection frequency across CV folds:
```
stability_freq = (# folds selecting feature) / (total folds)
```

**Output:** `feature_selection/stability/feature_stability.csv`

**Columns:**
- `protein` - Protein name
- `selection_fraction` - Stability frequency [0, 1]
- `n_selections` - Number of folds selecting feature

**Use cases:**
- (a) Filter out noisy features (min stability threshold, e.g., 0.60-0.75)
- (b) Resolve ties in OOF importance rankings
- (c) Ensure robust generalization (stable features generalize better)

**Implementation:**
- [stability.py](../../src/ced_ml/features/stability.py) - Stability tracking

**References:**
- Meinshausen & Bühlmann (2010). Stability selection. JRSS-B 72(4):417-473.

---

## Stage 3: Cross-Model Consensus

**Input:** Significant models from Stage 1, OOF importance ranks from Stage 2.

**Output:** Cross-model consensus panel for clinical deployment.

### Three-Step Consensus Workflow

```
Step 1 -- Per-model ranking:
    For each significant model:
        1. Hard filter: Keep proteins with stability_freq >= threshold (e.g., 0.90-0.95)
        2. Rank survivors by OOF grouped importance (descending)
        3. If OOF unavailable, fall back to stability frequency ranking

Step 2 -- Cross-model RRA:
    Aggregate per-model OOF importance ranks via geometric mean of
    normalized reciprocal ranks.
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

**Implementation:**
- [consensus/](../../src/ced_ml/features/consensus/) - Consensus package (ranking, aggregation, clustering, builder)
- [consensus_panel.py (CLI)](../../src/ced_ml/cli/consensus_panel.py) - CLI

**CLI:**
```bash
# Cross-model consensus panel (RRA)
ced consensus-panel --run-id <RUN_ID>
```

**Output:** `consensus/`
- `final_panel.txt` - Top-N proteins for deployment
- `final_panel.csv` - Panel with consensus scores and uncertainty metrics
- `consensus_ranking.csv` - All proteins with RRA scores and uncertainty
- `uncertainty_summary.csv` - Per-protein uncertainty metrics (rank_std, rank_cv, n_models_present, agreement_strength)
- `per_model_rankings.csv` - Per-model OOF importance rankings
- `correlation_clusters.csv` - Cluster assignments
- `consensus_metadata.json` - Run parameters and statistics
- `essentiality/within_panel_essentiality.csv` - Post-hoc drop-column results

**Note:** Uses geometric mean rank aggregation, not the formal Kolde RRA with
beta-model p-values. See ADR-004 for rationale.

**References:**
- Kolde et al. (2012). Robust rank aggregation for gene list integration. Bioinformatics 28(4):573-580.

---

## Typical Workflow

```bash
# Step 1: Train models with feature selection
ced train --model LR_EN,RF,XGBoost --split-seed 0,1,2

# Step 2: Aggregate results across splits
ced aggregate-splits --run-id <RUN_ID>

# Step 3: Test model significance (model gate)
ced permutation-test --run-id <RUN_ID> --model LR_EN --hpc
ced permutation-test --run-id <RUN_ID> --model RF --hpc
ced permutation-test --run-id <RUN_ID> --model XGBoost --hpc

# Step 4: Compute per-model evidence (OOF importance, essentiality, RFE)
# (automatically computed during training and aggregation)

# Step 5: Cross-model consensus (only for significant models)
ced consensus-panel --run-id <RUN_ID> --models LR_EN,RF  # exclude XGBoost if p >= 0.05

# Step 6: Deploy consensus panel (validation with new seed)
ced train --fixed-panel consensus_panel/final_panel.txt --split-seed 10
```

---

## Legacy Feature Selection Methods (Deprecated for Routine Use)

The following methods remain available but are superseded by the three-stage workflow above.

### Nested RFECV

**Status:** Deprecated for routine use. Superseded by aggregated RFE in Stage 3 (tertiary evidence).

**Original use case:** Scientific discovery, automatic panel sizing during training.

**Method:** RFECV per fold with consensus aggregation.

**Output:** `cv/rfecv/consensus_panel.csv`, `feature_stability.csv`

**Config:** `feature_selection_strategy: rfecv` in `training_config.yaml`.

**Runtime:** ~5-22 hours.

**Implementation:**
- [nested_rfe.py](../../src/ced_ml/features/nested_rfe.py)

**References:**
- Guyon et al. (2002). Gene selection for cancer classification using support vector machines. Machine Learning 46(1-3).

---

### Fixed Panel

**Status:** Retained for validation and benchmarking.

**Use case:** Validate discovery panels on new splits, benchmark literature panels, regulatory submission.

**Method:** Train on predetermined panel (skip feature selection).

**CLI:**
```bash
# CRITICAL: Use NEW split seed to prevent peeking
ced train --fixed-panel panel.csv --split-seed 10
```

**Runtime:** ~30 min (no feature selection overhead).

---

## Summary Table

| Stage | Component | Input | Output | Use Case |
|-------|-----------|-------|--------|----------|
| **1. Model Gate** | Permutation test | Trained models | p-value per model | Filter models with real signal |
| **2A. Primary** | OOF grouped importance | Held-out folds | Rank per model | Generalization-focused ranking (consensus input) |
| **2B. Post-hoc** | Drop-column essentiality | Final panel | delta-AUROC/PR-AUC/Brier per cluster | Interpretation, not ranking input |
| **2C. Sizing** | RFE rank | Shortlist | Elimination order | Panel size optimization (independent) |
| **2D. Filter** | Stability frequency | CV folds | Selection fraction | Hard filter for consensus entry |
| **3. Consensus** | Geometric mean RRA | OOF ranks × N models | Final panel | Cross-model robust biomarkers |

---

## Configuration

**File:** `configs/training_config.yaml`

```yaml
feature_selection:
  # Correlation clustering (Stage 2: OOF grouped importance and drop-column)
  corr_threshold: 0.85

  # Stability tracking (Stage 2 Input 4)
  stability_thresh: 0.75
  stability_min_features: 20

  # RFE (Stage 2 Input 3: independent panel sizing)
  rfe_start_size: 100  # Start RFE from top-K by OOF importance
  rfe_step: 0.1

  # Consensus (Stage 3)
  consensus_top_k: 40  # Target panel size (after clustering)
```

**File:** `configs/consensus_panel.yaml`

```yaml
# Consensus panel (Stage 3)
stability_threshold: 0.95    # Hard filter: min selection fraction
corr_threshold: 0.75         # Clustering threshold
target_size: 25              # Final panel size
rra_method: geometric_mean   # Aggregation method

essentiality:
  enabled: true              # Post-hoc drop-column on final panel
  include_brier: true        # Report delta-Brier
  include_pr_auc: true       # Report delta-PR-AUC
```

**File:** `configs/permutation_test.yaml`

```yaml
permutation_test:
  n_permutations: 200  # Recommended: 200 (publication), 10-50 (CI/quick check)
  aggregation_method: "pooled_null"  # Pooled-null (default) vs Fisher/Stouffer
  alpha: 0.05  # Significance threshold
  hpc_walltime: "04:00:00"  # Per permutation
  hpc_memory: "16GB"
```

---

## Related Documentation

- [ADR-004](../adr/ADR-004-four-strategy-feature-selection.md) - Three-stage workflow (current)
- [ADR-011](../adr/ADR-011-permutation-testing.md) - Permutation testing (Stage 1 model gate)
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Command-line interface documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Package architecture
