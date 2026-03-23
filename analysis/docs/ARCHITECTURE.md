# cel-risk Architecture

**Version:** 2.2
**Date:** 2026-02-03
**Status:** Streamlined algorithmic documentation with consensus panel integration and permutation significance testing

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Algorithm](#2-core-algorithm)
3. [Data Contracts](#3-data-contracts)
4. [Module Reference](#4-module-reference)
5. [Extension Points](#5-extension-points)

---

## 1. System Overview

### Purpose

ML pipeline for predicting **incident Celiac Disease (CeD)** risk from proteomics biomarkers. Generates calibrated risk scores for clinical screening.

**Dataset:**
- 43,960 samples (43,662 controls, 148 incident cases, 150 prevalent)
- 2,920 protein features (`*_resid` suffix)
- Prevalence: ~1:300 (0.33%)

**Models:**
- Base: LR_EN, RF, XGBoost, LinSVM_cal
- Ensemble: Stacking meta-learner (L2 logistic regression)

**Architecture Decision Records:** See [docs/adr/](adr/) for detailed design rationale.

---

## 2. Core Algorithm

### 2.1 Data Split Strategy

**Three-way split:** TRAIN / VAL / TEST

**Stratification:** By target to preserve class balance.

**Control downsampling:** case:control ratio customizable for data imbalance.

**Prevalent case handling:** Prevalent cases can be added to TRAIN only. VAL/TEST remain prospective (incident-only).

**Persistence:** Split indices saved as CSV files (`{scenario}_{split}_idx_seed{N}.csv`) for reproducibility.

**See ADRs:**
- [ADR-001: Split Strategy](adr/ADR-001-split-strategy.md)
- [ADR-002: Prevalent→TRAIN](adr/ADR-002-prevalent-train-only.md)
- [ADR-003: Control Downsampling](adr/ADR-003-control-downsampling.md)

### 2.2 Feature Selection Strategies

The pipeline provides **five distinct strategies**, each optimized for different phases and objectives. No single approach satisfies all needs:
- Production models need fast, reproducible selection with explicit tuning
- Scientific papers require feature stability analysis across folds
- Single-model deployment demands panel size optimization balancing cost vs. performance
- Cross-model deployment requires consensus across multiple algorithms
- Regulatory validation requires unbiased estimates on predetermined panels

See [ADR-004: Three-Stage Feature Selection and Consensus Workflow](adr/ADR-004-four-strategy-feature-selection.md) for the architectural decision documenting the rationale, alternatives considered, and trade-offs.

| Stage | Component | Phase | Runtime | Output |
|-------|-----------|-------|---------|--------|
| **Stage 1** | Model gate (permutation test) | After training | 1-4 hrs per model (HPC) | p-value per model |
| **Stage 2** | Per-model evidence | During training + aggregation | Computed alongside training | OOF importance (primary), stability (hard filter), RFE (sizing), drop-column (post-hoc) |
| **Stage 3** | Geometric mean rank aggregation | After aggregation | ~15 min | Cross-model consensus panel |

**Stage 1: Model Gate (CLI: `ced permutation-test --run-id <RUN_ID> --model <MODEL>`)**
- Label permutation test of classifier AUROC using same CV/training recipe
- Pooled-null aggregation across CV folds/seeds
- Empirical p-value with +1 correction
- **Output:** Keep only significant models (p < α) for downstream consensus

**Stage 2: Per-model Feature Evidence (4 complementary inputs)**

Input 1 (primary): **OOF grouped importance**
- Trees: OOF grouped permutation importance on held-out folds
- Linear: standardized |coef| on standardized inputs + stability across repeats
- Correlation grouping prevents "twin feature" artifacts

Input 2 (post-hoc): **Drop-column / LOCO essentiality**
- Post-hoc interpretation on the final consensus panel (NOT an input to ranking)
- Fixed hyperparams (refit-only) to measure "is this block essential?"
- Reports delta-AUROC (primary), delta-PR-AUC, delta-Brier

Input 3 (tertiary): **RFE rank**
- Panel sizing / selection path, not primary scientific ranking
- Use as tie-breaker / selection prior

Input 4 (filter/tie-break): **Stability frequency**
- Filter out noisy blocks (min stability threshold)
- Resolve ties when other inputs are similar

**Stage 3: Cross-Model Consensus (CLI: `ced consensus-panel --run-id <RUN_ID>`)**

1. **Per-model ranking:** Hard filter by stability (>= threshold), rank survivors by OOF importance
2. **Cross-model aggregation:** Geometric mean of normalized reciprocal ranks across models (missing = bottom rank)
3. **Correlation clustering:** Cluster top candidates, select representatives by consensus score
4. **Top-N selection:** Extract final panel of target size
5. **Post-hoc drop-column:** Refit on panel features, run drop-column per cluster -- interpretation artifact only

**Other methods**:
- Nested RFECV (during training, slow 5-22 hrs)
- Fixed Panel validation (`ced train --fixed-panel panel.csv`)

**Code pointers:**
- [significance/permutation_test.py](../src/ced_ml/significance/permutation_test.py) - Stage 1: Model gate
- [features/grouped_importance.py](../src/ced_ml/features/grouped_importance.py) - Stage 2 Input 1: OOF importance
- [features/drop_column.py](../src/ced_ml/features/drop_column.py) - Stage 2 Input 2: LOCO essentiality
- [features/rfe.py](../src/ced_ml/features/rfe.py) - Stage 2 Input 3: RFE rank
- [features/stability.py](../src/ced_ml/features/stability.py) - Stage 2 Input 4: Stability frequency
- [features/consensus/](../src/ced_ml/features/consensus/) - Stage 3: Geometric mean rank aggregation consensus
- [cli/permutation_test.py](../src/ced_ml/cli/permutation_test.py) - Stage 1 CLI
- [cli/consensus_panel.py](../src/ced_ml/cli/consensus_panel.py) - Stage 3 CLI

**See ADRs:**
- [ADR-004: Three-Stage Feature Selection and Consensus Workflow](adr/ADR-004-four-strategy-feature-selection.md) - Model gate, evidence, geometric mean rank consensus

**See detailed guide:**
- [FEATURE_SELECTION.md](reference/FEATURE_SELECTION.md) - Consolidated guide covering the three-stage workflow

### 2.3 Nested Cross-Validation

**Structure:**  #n outer folds × #n repeats × #n inner folds

**Outer CV:**
- Generates out-of-fold (OOF) predictions for TRAIN set
- Repeated #n times for robust estimates

**Inner CV:**
- Hyperparameter optimization (RandomizedSearchCV or OptunaSearchCV)
- Optimizes AUROC (discrimination-focused)
- Selects best hyperparameters per outer fold

**Hyperparameter search:**
- **OptunaSearchCV** (default): Bayesian TPE sampling with pruning (median/percentile/hyperband)
- **RandomizedSearchCV** (optional): 200 iterations per fold

**OOF prediction tracking:** Each sample's prediction comes from a fold where it was held out (no leakage).

**See ADRs:**
- [ADR-005: Nested CV Structure](adr/ADR-005-nested-cv.md)
- [ADR-006: Optuna Hyperparameter Optimization](adr/ADR-006-optuna-hyperparameter-optimization.md)

### 2.4 Stacking Ensemble

**Purpose:** Combine base model predictions to improve performance (+2-5% AUROC).

**Architecture:**
```
Base Models → OOF Predictions → Meta-Learner (L2 LR) → Calibrated Ensemble
```

**Training:**
1. Base models trained independently with nested CV → OOF predictions
2. Meta-learner (L2 logistic regression) trained on stacked OOF predictions
3. No leakage: meta-learner never sees predictions from samples in training fold

**Configuration:** Opt-in via `ensemble.enabled=true` in config.

**Ensemble calibration:** The meta-learner supports optional post-hoc calibration via `calibrate_meta: true` (default: `false`). When enabled, the LR meta-learner is wrapped in `CalibratedClassifierCV` using isotonic calibration (configurable via `meta_calibration_method`). The regularization strength defaults to `meta_c: 1.0` but can be set to higher values (e.g., `100.0`) to avoid probability compression when few base-model features are present. These settings are configurable via `EnsembleConfig` in `config/ensemble_schema.py`.

**See ADR:**
- [ADR-007: OOF Stacking Ensemble](adr/ADR-007-oof-stacking-ensemble.md)

### 2.5 Calibration

**Methods (OOF posthoc):**
- `logistic_full` - Two-parameter Platt scaling: logit(Y=1) = a + b*logit(p)
- `logistic_intercept` - Intercept-only recalibration (lowest variance, default)
- `isotonic` - Isotonic regression (non-parametric, monotonic)
- `beta` - Beta calibration (Kull et al. 2017, three parameters)

**Methods (per-fold via sklearn):**
- `sigmoid` - Platt scaling (sklearn CalibratedClassifierCV)
- `isotonic` - Isotonic regression (sklearn CalibratedClassifierCV)

**Strategies:**
- `per_fold` (default): Apply `CalibratedClassifierCV` inside each CV fold
- `oof_posthoc`: Fit single calibrator on pooled OOF predictions (eliminates ~0.5-1% optimism bias)
- `none`: No calibration

**Strategy comparison:**

| Strategy | Data Efficiency | Leakage Risk | Optimism Bias | Stability |
|----------|-----------------|--------------|---------------|-----------|
| `per_fold` | Full | Subtle (~0.5-1%) | ~0.5-1% | Lower |
| `oof_posthoc` | Full | None | None | Higher |

**See ADR:**
- [ADR-008: OOF Posthoc Calibration](adr/ADR-008-oof-posthoc-calibration.md)

### 2.6 Prevalence Configuration

**Configuration:**
- TRAIN: 5:1 case:control (16.7% prevalence) - `train_control_per_case: 5`
- VAL/TEST: 5:1 case:control (16.7% prevalence) - `eval_control_per_case: 5`
- All three sets share the same prevalence → valid threshold selection and calibration

**Key observation:** Since TRAIN, VAL, and TEST all operate at 16.7% prevalence, there is no mismatch. This means:
- Threshold selection on VAL is unbiased (same prevalence as TRAIN)
- Calibration is valid (OOF calibrator fit on 16.7% data, applied to 16.7% test data)
- No in-pipeline prevalence adjustment is needed

**Note on real-world deployment:** Real-world incident Celiac Disease prevalence is ~0.34% (1:300). If models are deployed for clinical screening, predicted probabilities will need adjustment to account for this 50× prevalence difference. This is a future concern outside the current training pipeline.

**See speculative deployment guidance:**
- [DEPLOYMENT.md](development/DEPLOYMENT.md) - Speculative best-practices guide

### 2.7 Threshold Selection

**Objectives:**
- `youden` - Youden's J (sensitivity + specificity - 1) [default]
- `max_f1` - Maximize F1 score
- `fixed_spec` - Achieve fixed specificity (e.g., 0.95 for high specificity screening)
- `fixed_ppv` / `fixed_sens` - Fixed positive predictive value / sensitivity

**Source:** Threshold selected on VAL set, never on TEST (prevents leakage).

**See ADRs:**
- [ADR-011: Threshold on VAL](adr/ADR-011-threshold-on-val.md)
- [ADR-012: Fixed Spec 95%](adr/ADR-012-fixed-spec-95.md)

### 2.8 Leakage Prevention

**Critical rules enforced:**
1. Prevalent cases never in VAL/TEST (only TRAIN)
2. Threshold selected on VAL, never on TEST
3. Hyperparameter tuning only on TRAIN (via inner CV)
4. OOF predictions: each sample's prediction from fold where it was held out

### 2.9 Significance Testing

**Purpose:** Test the null hypothesis that model performance is no better than chance.

**Method:** Label permutation testing with full pipeline re-execution.

**Algorithm:**
```
For each permutation b in 0..B-1:
    1. Shuffle y_train only (keep X fixed, held-out y unchanged)
    2. Re-run FULL inner pipeline on permuted labels:
       - Screening (Mann-Whitney/t-test)
       - Inner CV k-best tuning
       - Hyperparameter optimization
       - Fit final model
    3. Predict on held-out fold (original X, original y)
    4. Record AUROC

p-value = (1 + #{null >= observed}) / (1 + B)
```

**Why re-run full pipeline?** Avoids data leakage in null distribution. Features selected on real labels would inflate null performance.

**CLI:** `ced permutation-test --run-id <RUN_ID> [--model <MODEL>] [--n-perms 200]`

**HPC support:** Orchestrator submits one full-command job per seed via `_build_permutation_test_full_command`.

**Code pointers:**
- [significance/permutation_test.py](../src/ced_ml/significance/permutation_test.py) - Core algorithm
- [cli/permutation_test.py](../src/ced_ml/cli/permutation_test.py) - CLI implementation

**See ADR:**
- [ADR-011: Permutation Testing](adr/ADR-011-permutation-testing.md)

### 2.10 Feature Importance

**Purpose:** Understand which features drive model predictions.

**Methods:**

| Method | Model Type | Approach |
|--------|------------|----------|
| Coefficient magnitude | Linear (LR_EN, SVM) | Standardized |coef| |
| Built-in importance | Tree (RF, XGBoost) | Gini/gain importance |
| Permutation importance | All | OOF permutation feature importance |
| Grouped permutation | All | Correlation-robust cluster-level PFI |
| Drop-column validation | All | Refit without feature/cluster |

**Correlation-robust grouping:** Highly correlated features (r > 0.85) are clustered. Permutation applied to entire cluster to avoid underestimating correlated feature importance.

**Code pointers:**
- [features/importance.py](../src/ced_ml/features/importance.py) - Linear/tree importance extraction
- [features/grouped_importance.py](../src/ced_ml/features/grouped_importance.py) - Cluster-based PFI
- [features/drop_column.py](../src/ced_ml/features/drop_column.py) - Drop-column validation

---

## 3. Data Contracts

### 3.1 Input Format

**Required columns:**
- `eid` - Sample identifier
- `incident_CeD` - Binary target (0/1)
- `{protein}_resid` - Protein features (must end with `_resid` suffix)

**Optional columns:**
- `prevalent_CeD` - Prevalent case flag
- Metadata: age, BMI, sex, ethnicity (auto-detected or explicit via `ColumnsConfig`)

**Supported formats:** CSV, Parquet (auto-detected by extension)

### 3.2 Output Artifacts

**Directory structure:** `{outdir}/split_seed{N}/`

```
split_seed42/
  core/
    final_model.pkl           # Trained sklearn model (calibrated if enabled)
    oof_predictions.csv       # OOF predictions (TRAIN)
    val_predictions.csv       # VAL predictions
    test_predictions.csv      # TEST predictions
    *_metrics.json            # Performance metrics
    run_settings.json         # Full config + metadata
    stable_features.txt       # Stability panel
  cv/
    cv_repeat_metrics.csv     # Per-repeat OOF metrics
    best_params.csv           # Best hyperparameters
  plots/
    roc_pr.png, calibration.png, risk_dist.png, dca.png
```

**Prediction CSV format:**
```csv
eid,y_true,y_pred_proba,y_pred,fold,repeat
1001,0,0.012,0,0,0
```

**Metrics JSON format:**
```json
{
  "auroc": 0.85,
  "prauc": 0.42,
  "brier": 0.08,
  "threshold": 0.35,
  "sensitivity": 0.78,
  "specificity": 0.82
}
```

See [ARTIFACTS.md](ARTIFACTS.md) for detailed artifact documentation.

---

## 4. Module Reference

### 4.1 Package Structure

```
src/ced_ml/
  cli/          # Command-line interface
  config/       # Pydantic configuration system
  data/         # Data I/O, splits, persistence
  features/     # Feature selection, importance, drop-column
  models/       # Model training, calibration, stacking
  metrics/      # Performance metrics (AUROC, Brier, DCA)
  significance/ # Permutation testing for model significance
  evaluation/   # Prediction, reporting, scoring
  plotting/     # Visualization
  utils/        # Logging, random, serialization
```

### 4.2 Key Modules

**Data layer:**
- `data/splits.py` - Stratified splitting, downsampling, prevalent handling
- `data/persistence.py` - Split index CSV I/O
- `data/columns.py` - Metadata column resolution

**Feature selection:**
- `features/grouped_importance.py` - Stage 2 Input 1: OOF grouped permutation importance
- `features/drop_column.py` - Stage 2 Input 2: Drop-column / LOCO essentiality
- `features/rfe.py` - Stage 2 Input 3: RFE rank (panel sizing)
- `features/stability.py` - Stage 2 Input 4: Stability frequency tracking
- `features/consensus/` - Stage 3: Geometric mean rank aggregation consensus (ranking, aggregation, clustering, builder)
- `features/corr_prune.py` - Correlation clustering for grouped importance / drop-column
- `features/importance.py` - Linear coefficient and tree importance extraction (base for grouped importance)
- `features/nested_rfe.py` - Legacy: Nested RFECV (deprecated)
- `features/panels.py` - Fixed panel loading (validation/benchmarking)

**Model training:**
- `models/training.py` - Nested CV orchestration, OOF predictions
- `models/stacking.py` - Stacking ensemble (StackingEnsemble, BaseModelBundle)
- `models/calibration.py` - Calibration wrappers, prevalence adjustment, OOF calibration
- `models/optuna_search.py` - Optuna hyperparameter optimization

**Evaluation:**
- `metrics/discrimination.py` - AUROC, PR-AUC, Brier score
- `metrics/thresholds.py` - Threshold selection objectives
- `metrics/dca.py` - Decision curve analysis

**Significance testing:**
- `significance/permutation_test.py` - Label permutation testing for model significance

### 4.3 Configuration Schema

**Top-level:** `TrainingConfig`

**Sub-configs:**
- `CVConfig` - Cross-validation structure (folds, repeats, scoring)
- `FeatureConfig` - Feature selection methods (screening, k-best, stability, correlation)
- `CalibrationConfig` - Calibration settings (method, strategy)
- `ThresholdConfig` - Threshold selection (objective, source)
- `OptunaConfig` - Optuna hyperparameter optimization
- `EnsembleConfig` - Stacking ensemble configuration
- `ColumnsConfig` - Metadata column configuration

See `config/schema.py` for complete schema.

---

## 5. Extension Points

### 5.1 Add New Model

1. Create builder function in `models/registry.py`
2. Add hyperparameter grid in `models/hyperparams.py`
3. Add config class in `config/schema.py` (if model-specific params needed)
4. Add tests in `tests/test_models_*.py`

### 5.2 Add New Feature Selection Method

1. Implement in `features/` module
2. Add config option in `FeatureConfig`
3. Integrate in training pipeline (`cli/train.py`)
4. Add tests in `tests/test_features_*.py`

### 5.3 Add New Metric

1. Implement in `metrics/` module
2. Add config option in `EvaluationConfig` (if needed)
3. Integrate in evaluation pipeline (`evaluation/reports.py`)
4. Add tests in `tests/test_metrics_*.py`

---

## Reproducibility

**All runs are deterministic via:**
- Fixed RNG seeds (`split_seed`, `random_state`)
- Persisted split indices (CSV files)
- Full config logging (`run_settings.json`)
- Git commit hash tracking
- Package version recording

---

**End of ARCHITECTURE.md**
