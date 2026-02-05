# Feature Selection and Consensus Workflow

**Status:** Current | **Updated:** 2026-02-04

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

### Input 2 (Secondary): Drop-Column Essentiality

**Rationale:** More faithful necessity test than single-feature PFI, especially under correlation.

**Method:** For each cluster in candidate panel:
1. Remove cluster and refit model (fixed hyperparams, refit-only)
2. Compute AUROC on held-out fold
3. Essentiality = `original_auroc - ablated_auroc`

**When to run:** On final candidate panel (or shortlist) after feature selection.

**Default mode:** Fixed hyperparams (refit-only) to answer: "Is this cluster essential under this modeling recipe?"

**Output:** `drop_column/{model}/aggregated_results.csv`

**Columns:**
- `cluster_id` - Cluster ID
- `representative` - Cluster representative feature
- `features` - All features in cluster (JSON list)
- `mean_delta_auroc` - Mean ΔAUROC across CV folds
- `std_delta_auroc` - Standard deviation
- `rank` - Rank by mean ΔAUROC (1 = most essential)

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

## Stage 3: Sequential Filtering

**Input:** Significant models from Stage 1, four evidence types from Stage 2.

**Output:** Cross-model consensus panel for clinical deployment.

### Algorithm

```
For each significant model:
    1. Filter by stability: Keep blocks with stability_freq >= s (e.g., 0.60-0.75)
    2. Rank by OOF grouped importance: Keep top K_1 blocks (e.g., 150)
    3. Run RFE inside shortlist to pick panel size (e.g., 25-40)
    4. Run grouped LOCO/drop-column on chosen panel
       Keep blocks with ΔAUROC above noise floor or top K_2 (e.g., 25)
```

### RRA Consensus (Multi-List)

**Preferred approach:** Contribute multiple rank lists per model.

```
For each significant model:
    - List 1: OOF grouped importance ranks (primary)
    - List 2: Essentiality ranks (secondary, if available)
    - List 3: RFE ranks (tertiary, if available)

Run RRA across all lists:
    - Missing features = bottom rank (conservative)
    - Compute rra_score (geometric mean of reciprocal ranks)
    - Compute rra_p (p-value from RRA null distribution)
    - Apply Benjamini-Hochberg FDR correction → rra_q

Select blocks by:
    - Option A: rra_q < alpha (e.g., 0.05)
    - Option B: Top-K for fixed panel size (e.g., 25-40)

Post-filter by stability as tie-breaker.
```

**Implementation:**
- [consensus.py](../../src/ced_ml/features/consensus.py) - RRA aggregation
- [consensus_panel.py (CLI)](../../src/ced_ml/cli/consensus_panel.py) - CLI

**CLI:**
```bash
# Cross-model consensus panel (RRA)
ced consensus-panel --run-id <RUN_ID>
```

**Output:** `consensus_panel/`
- `final_panel.txt` - Top-N proteins for deployment
- `consensus_ranking.csv` - All proteins with RRA scores, p-values, FDR-adjusted q-values
- `uncertainty_summary.csv` - Per-protein uncertainty metrics (rank_std, rank_cv, n_models_present, agreement_strength)
- `metadata.json` - Run parameters and statistics

**References:**
- Kolde et al. (2012). Robust rank aggregation for gene list integration. Bioinformatics 28(4):573-580.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. JRSS-B 57(1):289-300.

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
| **2A. Primary** | OOF grouped importance | Held-out folds | Rank per model | Generalization-focused ranking |
| **2B. Secondary** | Drop-column essentiality | Candidate panel | Rank per model | Necessity under modeling recipe |
| **2C. Tertiary** | RFE rank | Shortlist | Elimination order | Tie-breaker, panel sizing |
| **2D. Filter** | Stability frequency | CV folds | Selection fraction | Filter noisy features, resolve ties |
| **3. Consensus** | Multi-list RRA | 4 ranks × N models | Final panel | Cross-model robust biomarkers |

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

  # RFE (Stage 2 Input 3 and Stage 3 sequential filtering)
  rfe_start_size: 100  # Start RFE from top-K by OOF importance
  rfe_step: 0.1

  # Consensus (Stage 3)
  consensus_top_k: 40  # Final panel size
  consensus_fdr_alpha: 0.05  # FDR threshold for RRA q-values
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
