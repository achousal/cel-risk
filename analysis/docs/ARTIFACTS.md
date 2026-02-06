# CeliacRisks Output Artifacts

**Version:** 3.0
**Date:** 2026-01-29

**Key structural features:**
1. **Run-first grouping** - Everything from one experiment lives under a single `run_{id}/` directory
2. **Model isolation** - Each model has its own subdirectory within the run
3. **Split separation** - Individual splits live under `splits/split_seed{N}/`
4. **Flat predictions** - All prediction types (test, val, train_oof, controls) in single `preds/` directory (no subdirectories)
5. **Flat diagnostics** - All diagnostic CSVs in single `diagnostics/` directory (no subdirectories)
6. **Flat panels** - All panel reports in single `panels/` directory (no subdirectories)
7. **Auto-detection support** - `run_metadata.json` at run level enables zero-config CLI commands

## 1. Directory Structure

### 1.1 Overview

All outputs follow a **run-first hierarchical structure**:

```
results/
  run_{run_id}/                   # Level 1: Run ID (timestamped, e.g., 20260127_115115)
    run_metadata.json             # Shared auto-detection metadata (all models)
    {model}/                      # Level 2: Model name (LR_EN, RF, XGBoost)
      splits/                     # Split container directory
        split_seed{N}/            # Level 3: Split seed (0, 1, 2, ...)
          core/                   # Metrics and settings
          cv/                     # Cross-validation artifacts
          plots/                  # Visualizations
          preds/                  # Predictions (all types in flat structure)
          panels/                 # Feature panels and reports
          diagnostics/            # CSV data exports
      aggregated/                 # Cross-split aggregation (created by aggregate-splits)
        metrics/
        panels/
        plots/
        cv/
        preds/
        diagnostics/
    ENSEMBLE/                     # Ensemble meta-learner (if trained)
      splits/
        split_seed{N}/
    consensus/                    # Cross-model consensus panel (if generated)
```

### 1.2 Single Split Output

**Path pattern:** `results/run_{run_id}/{model}/splits/split_seed{N}/`

**Example:** `results/run_20260127_115115/LR_EN/splits/split_seed0/`

```
results/run_{run_id}/{model}/splits/split_seed{N}/
  core/
    final_model.pkl               # Trained sklearn model (pickled)
    run_settings.json             # Full config + metadata
    test_metrics.csv              # TEST metrics
    val_metrics.csv               # VAL metrics
  cv/
    cv_repeat_metrics.csv         # Per-repeat OOF metrics
    best_params_per_split.csv     # Best hyperparameters per fold
    optuna_config.json            # (if Optuna enabled) Optuna settings
    best_params_optuna.csv        # (if Optuna enabled) Best params with trial metadata
    optuna_trials.csv             # (if Optuna enabled) All trial results
  plots/
    roc_pr.png                    # ROC + PR curves
    calibration.png               # Calibration plot
    risk_dist.png                 # Risk distribution
    dca.png                       # Decision curve analysis
    oof_roc.png                   # OOF ROC with confidence bands
    oof_pr.png                    # OOF PR with confidence bands
    oof_calibration.png           # OOF calibration plot
  preds/
    test_preds__*.csv             # TEST set predictions (flat structure)
    val_preds__*.csv              # VAL set predictions (flat structure)
    train_oof__*.csv              # OOF predictions TRAIN set (flat structure)
    controls_risk__*__oof_mean.csv # Control subjects OOF predictions (flat structure)
  panels/
    {model}__feature_report_train.csv    # Feature importance on TRAIN set
    stable_panel__{panel_type}.csv       # Stable features (threshold 0.75+)
    {model}__N{size}__panel_manifest.json # Panel at specific size (N=10, 25, 50, etc.)
    {model}__final_test_panel.json       # Final deployment-ready panel
    {model}__test_subgroup_metrics.csv   # Subgroup performance metrics
  diagnostics/                    # Flat directory - all diagnostic CSVs in one place
    train_test_split_trace.csv    # Split indices trace
    {model}__calibration.csv      # Calibration curve data
    {model}__learning_curve.csv   # Learning curve data
    {model}__test__dca_results.csv # DCA results (test set)
    {model}__val__dca_results.csv  # DCA results (val set)
    {model}__screening_results.csv # Feature screening statistics
    {model}__test_bootstrap_ci.csv # Bootstrap confidence intervals
```

### 1.3 Run-Level Metadata

**Shared run_metadata.json:** All models in a run share a single metadata file at the run level.

**Path pattern:** `results/run_{run_id}/`

**Example:** `results/run_20260127_115115/`

```
results/run_{run_id}/
  run_metadata.json               # Shared metadata (all models in this run)
  LR_EN/                          # Model outputs
    splits/
      split_seed0/
      split_seed1/
    aggregated/
  RF/
    splits/
    aggregated/
  ENSEMBLE/
    splits/
  consensus/
```

**run_metadata.json contents:**
```json
{
  "run_id": "20260127_115115",
  "models": {
    "LR_EN": {
      "infile": "../data/Celiac_dataset_proteomics_w_demo.parquet",
      "split_dir": "../splits",
      "split_seed": 0,
      "scenario": "IncidentOnly",
      "timestamp": "2026-01-27T11:51:15"
    },
    "RF": {
      "infile": "../data/Celiac_dataset_proteomics_w_demo.parquet",
      "split_dir": "../splits",
      "split_seed": 0,
      "scenario": "IncidentOnly",
      "timestamp": "2026-01-27T11:52:30"
    }
  }
}
```

**Purpose:** Cross-split aggregation and model comparison.

```
results/run_{run_id}/{model}/aggregated/
  aggregation_metadata.json       # Full aggregation metadata
  all_test_metrics.csv            # Per-split test metrics (all splits)
  all_val_metrics.csv             # Per-split val metrics (all splits)
  metrics/
    pooled_test_metrics.csv       # Pooled test metrics by model
    pooled_val_metrics.csv        # Pooled val metrics by model
    test_metrics_summary.csv      # Summary stats across splits
    val_metrics_summary.csv       # Val summary stats
    model_comparison.csv          # Model comparison report
  panels/
    feature_stability.csv         # Aggregated feature stability across all splits
    consensus_panel_N{size}.json  # Consensus panel at specific size (e.g., N=25)
    consensus_panel_metadata.json # Consensus aggregation metadata
    {model}__rfe_panel_N{size}.json # RFE-optimized panel (from optimize-panel)
    uncertainty_summary.csv       # Uncertainty metrics for panel optimization
  plots/
    test_roc.png                  # Aggregated test ROC
    test_pr.png                   # Aggregated test PR
    calibration.png               # Aggregated calibration
    risk_distribution.png         # Aggregated risk distributions
    dca.png                       # Aggregated DCA
    oof_combined.png              # Combined OOF plots
    learning_curve.png            # Aggregated learning curves
    ensemble_weights_aggregated.png  # (if ensemble) Meta-learner coefficients
    model_comparison.png          # (if multi-model) Model comparison chart
  cv/
    all_cv_repeat_metrics.csv     # CV metrics from all splits
    cv_metrics_summary.csv        # CV summary stats
    all_best_params_per_split.csv # Best hyperparameters across splits
    hyperparams_summary.csv       # Hyperparameter summary
    ensemble_config_per_split.csv # (if ensemble) Ensemble configs
    optuna_trials.csv             # (if Optuna enabled) Combined trials from all splits
  preds/
    pooled_test_preds.csv         # All test predictions pooled across splits
    pooled_test_preds__*.csv      # Per-model pooled test predictions
    pooled_val_preds.csv          # All val predictions pooled across splits
    pooled_val_preds__*.csv       # Per-model pooled val predictions
    pooled_train_oof.csv          # All OOF predictions pooled across splits
    pooled_train_oof__*.csv       # Per-model pooled OOF predictions
  diagnostics/                    # Flat directory - aggregated diagnostic CSVs
    {model}__aggregated_calibration.csv  # Aggregated calibration curves
    {model}__aggregated_dca.csv          # Aggregated DCA results
    {model}__aggregated_screening.csv    # Aggregated screening statistics
    {model}__aggregated_learning_curve.csv # Aggregated learning curves
```

## 2. Core Artifacts

### 2.1 final_model.pkl

**Type:** Pickled Python object

**Contents:** Trained sklearn-compatible model (e.g., sklearn.linear_model.LogisticRegression, xgboost.XGBClassifier) with calibration wrapper if enabled.

**Metadata stored separately in:**
- `run_settings.json` - Full config, hyperparameters, feature names
- `reports/stable_panel/` - Selected feature panel

### 2.2 Prediction CSVs

**Format:** CSV with headers

**Columns:**
- `idx` - Sample index
- `y_true` - True label (0/1)
- `y_prob` - Predicted probability
- `category` - Sample category (Controls, Incident, Prevalent)
- `split_seed` - Split seed used
- `model` - Model name

**Files (per split):**
- `preds/test_preds__*.csv` - TEST set predictions (flattened to preds/)
- `preds/val_preds__*.csv` - VAL set predictions (flattened to preds/)
- `preds/train_oof__*.csv` - TRAIN set OOF predictions (with per-repeat probabilities, flattened to preds/)

### 2.3 Metrics CSVs

**Format:** CSV with headers

**Common columns:**
- `model` - Model name
- `scenario` - Training scenario
- `AUROC` - Area under ROC curve
- `PR_AUC` - Area under Precision-Recall curve
- `Brier` - Brier score (lower is better)
- `Sensitivity`, `Specificity` - At selected threshold
- `PPV`, `NPV` - Predictive values
- `calibration_slope`, `calibration_intercept` - Calibration metrics

### 2.4 run_settings.json

**Format:** JSON with nested structure

**Contents:**
- Full resolved configuration (all parameters)
- Resolved metadata columns (auto-detected or explicit)
- Split seed, random state
- Model hyperparameters (selected via CV)
- Feature selection parameters
- Threshold selection settings
- Software versions (Python, numpy, sklearn, xgboost, etc.)
- Timestamp, runtime
- Git commit hash (if available)

**Purpose:** Complete provenance for reproducibility.

---

## 3. Cross-Validation Artifacts

### 3.1 cv_repeat_metrics.csv

**Format:** CSV with headers

**Columns:**
- `repeat` - CV repeat index (0-2)
- `outer_split` - CV fold index (0-4)
- `AUROC` - OOF AUROC for this fold
- `PR_AUC` - OOF PR-AUC for this fold
- `Brier` - OOF Brier score for this fold
- Additional metrics...

**Purpose:** Per-fold performance for stability analysis.

### 3.2 best_params_per_split.csv

**Format:** CSV with headers

**Columns:**
- `repeat` - CV repeat index
- `outer_split` - CV fold index
- Model-specific hyperparameters (e.g., `C`, `max_depth`, `learning_rate`)
- `best_score_inner` - Best inner CV score

**Purpose:** Track hyperparameter selection across CV folds.

### 3.3 Optuna Artifacts (if enabled)

Generated flat in `cv/` directory:
- `optuna_trials.csv` - All trial results with parameters and scores
- `optuna_config.json` - Optuna settings

---

## 4. Plots

### 4.1 roc_pr.png

**Type:** PNG image (matplotlib figure)

**Contents:** Dual-panel plot:
- Left: ROC curve with AUROC annotation
- Right: Precision-Recall curve with PR-AUC annotation

**Colors:** TRAIN (blue), VAL (orange), TEST (green)

### 4.2 calibration.png

**Type:** PNG image (matplotlib figure)

**Contents:** Calibration plot (predicted vs. observed probabilities) with:
- Perfect calibration line (diagonal)
- Observed calibration curve (binned)
- Calibration slope/intercept annotations

**Purpose:** Assess probability calibration quality.

### 4.3 risk_dist.png

**Type:** PNG image (matplotlib figure)

**Contents:** Histogram of predicted probabilities:
- Separate distributions for cases (red) and controls (blue)
- Selected threshold vertical line (dashed)

**Purpose:** Visualize risk score separation.

### 4.4 dca.png

**Type:** PNG image (matplotlib figure)

**Contents:** Decision Curve Analysis plot:
- Net benefit vs. threshold probability
- Model curve vs. "treat all" and "treat none" strategies
- Auto-ranged threshold axis based on prevalence

**Purpose:** Evaluate clinical utility at different decision thresholds.

### 4.5 OOF Plots (oof_roc.png, oof_pr.png, oof_calibration.png)

**Type:** PNG images (matplotlib figures)

**Contents:** Out-of-fold predictions across CV repeats:
- Mean curve across repeats
- 95% confidence bands (shaded region)
- Individual repeat curves (faint lines)

**Purpose:** Assess model stability across CV repeats.

---

**Consensus Panel (RRA aggregation across models):**
```
consensus_panel_N25.json    # 25-protein consensus panel with metadata
consensus_panel_N50.json    # 50-protein consensus panel with metadata
consensus_panel_N100.json   # 100-protein consensus panel with metadata
consensus_panel_metadata.json # Consensus aggregation statistics
uncertainty_summary.csv     # Cross-model agreement and uncertainty metrics
```

**RFE-Optimized Panels (per model):**
```
LR_EN__rfe_panel_N25.json   # 25-protein RFE panel for LR_EN
RF__rfe_panel_N50.json      # 50-protein RFE panel for RF
LR_EN__uncertainty_metadata.json # Uncertainty quantification for RFE panels
```

**Purpose:**
- `feature_stability.csv`: Identify proteins consistently selected across all splits (0.75+ threshold)
- Consensus panels: Cross-model agreement via Robust Rank Aggregation
- RFE panels: Aggregated recursive feature elimination for single-model deployment
- Uncertainty metrics: Bootstrap CIs and cross-model agreement for deployment decisions

---

## 6. Ensemble Artifacts

### 6.1 Ensemble Directory Structure

**Path pattern:** `results/ENSEMBLE/run_{run_id}/splits/split_seed{N}/`

**Example:** `results/ENSEMBLE/run_20260127_115115/splits/split_seed0/`

**Purpose:** Stacking ensemble meta-learner outputs (requires base models trained first).

```
results/ENSEMBLE/run_{run_id}/splits/split_seed{N}/
  core/
    final_model.pkl               # Meta-learner (L2 logistic regression)
    run_settings.json             # Ensemble config + base model references
    test_metrics.csv              # TEST metrics
    val_metrics.csv               # VAL metrics
  cv/
    cv_repeat_metrics.csv         # OOF meta-learner metrics
    ensemble_config.json          # Base model paths and settings
  plots/
    ensemble_weights.png          # Meta-learner coefficients
    ensemble_roc_comparison.png   # Ensemble vs. base models ROC
    ensemble_calibration.png      # Ensemble calibration plot
  preds/
    test_preds__ENSEMBLE.csv      # TEST predictions
    val_preds__ENSEMBLE.csv       # VAL predictions
    train_oof__ENSEMBLE.csv       # OOF predictions
```

---

## 7. Log Artifacts

All CLI commands produce structured log files under `logs/` at the project root. Each command writes to a dedicated subdirectory organized by run ID.

### 7.1 Directory Structure

```
logs/
  training/run_{ID}/              # Model training logs
    {model}_seed{N}.log           # Per-model per-seed training log
  ensemble/run_{ID}/              # Ensemble meta-learner training logs
    ENSEMBLE_seed{N}.log          # Per-seed ensemble log
  aggregation/run_{ID}/           # Cross-split aggregation logs
    {model}.log                   # Per-model aggregation log
  optimization/run_{ID}/          # Panel optimization (RFE) logs
    {model}_seed{N}.log           # Per-model per-seed optimization log
  permutation/run_{ID}/            # Permutation significance testing logs
    perm_{model}_seed{N}.log      # Permutation test log (per model per seed)
  consensus/run_{ID}/             # Cross-model consensus logs
    consensus.log                 # RRA consensus panel log
  pipeline/                       # Pipeline orchestration logs
    run_{ID}.log                  # Full pipeline run log
```

### 7.2 Log Format

All log files use a consistent format:

```
[YYYY-MM-DD HH:MM:SS] LEVEL - message
```

### 7.3 HPC Logs

When running on HPC via LSF, job stderr is captured to `{job_name}.{JOB_ID}.err` in the HPC logs directory. These `.err` files are automatically removed on successful job completion (warnings-only content is not actionable; real errors cause non-zero exit). Job stdout is redirected to `/dev/null` because `ced` commands write their own structured log files.
