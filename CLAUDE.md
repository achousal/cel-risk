# CeliacRisks Project Documentation

**Project**: Machine Learning Pipeline for Incident Celiac Disease Risk Prediction
**Version**: 1.0.0
**Updated**: 2026-02-03
**Primary Package**: ced-ml
**Python**: 3.10+
**Project Owner**: Andres Chousal (Chowell Lab)
**Status**: Production-ready with stacking ensemble, OOF-posthoc calibration, temporal validation, panel optimization, cross-model consensus, and permutation significance testing

## Dataset

| Attribute | Value |
|-----------|-------|
| **Total samples** | 43,960 |
| **Controls** | 43,662 |
| **Incident CeD** | 148 (0.34%) - biomarkers BEFORE diagnosis |
| **Prevalent CeD** | 150 - used in TRAIN only (50% sampling) |
| **Proteins** | 2,920 (`*_resid` columns) |
| **Demographics** | age, BMI, sex, Genetic ethnic grouping (configurable via `ColumnsConfig`) |
| **Missing proteins** | Zero |
| **Missing ethnicity** | 17% (handled as "Missing" category) |

---

## Project Mission

Build calibrated ML models to predict **incident Celiac Disease (CeD) risk** from proteomics biomarkers measured **before clinical diagnosis**. Generate continuous risk scores for apparently healthy individuals to inform follow-up testing decisions.

### Clinical Workflow
```
Blood proteomics panel → ML risk score → [High risk?] → Anti-tTG antibody test → Endoscopy
```

---

**Important**: All `ced` commands must be run from the project root (`CeliacRisks/`).

```bash
ced --help
```

### 1. Complete Pipeline (Recommended)

## Core Workflows

```bash
# Auto-detects data, runs full pipeline
ced run-pipeline

# Or specify models and splits
ced run-pipeline --models LR_EN,RF,XGBoost --split-seeds 0,1,2
```

### 2. Individual Steps (using --run-id)
```bash
# Aggregate results across splits
ced aggregate-splits --run-id 20260127_115115

# Optimize panel size
ced optimize-panel --run-id 20260127_115115

# Cross-model consensus panel
ced consensus-panel --run-id 20260127_115115

# Test model significance (permutation testing)
ced permutation-test --run-id 20260127_115115 --model LR_EN --n-perms 200
```

**Customize via configs**:
- `configs/training_config.yaml` - Models, CV, feature selection
- `configs/output_config.yaml` - Artifact and plot generation controls
- `configs/pipeline_local.yaml` - Execution settings

### 3. Feature Selection

**Workflow (three-stage):**

| Stage | Component | Purpose | Runtime |
|-------|-----------|---------|---------|
| **1. Model Gate** | Permutation test | Filter models with real signal (p < 0.05) | ~1-4 hrs per model (HPC) |
| **2. Per-Model Evidence** | OOF importance (primary), stability (hard filter), RFE (sizing), drop-column (post-hoc) | Ranking and interpretation per model | Computed during training/aggregation |
| **3. Consensus** | Geometric mean rank aggregation | Cross-model robust biomarkers | ~15 min |

**Typical workflow:**
```bash
# Train models
ced train --model LR_EN,RF,XGBoost --split-seed 0,1,2
ced aggregate-splits --run-id <RUN_ID>

# Model gate (keep only significant models)
ced permutation-test --run-id <RUN_ID> --model LR_EN --n-jobs 4

# Cross-model consensus (significant models only)
ced consensus-panel --run-id <RUN_ID> --models LR_EN,RF
```

**Other methods**
- Nested RFECV (during training): RFECV per fold with consensus aggregation
- Fixed Panel (validation): Train on predetermined panel

**For detailed documentation, see [docs/reference/FEATURE_SELECTION.md](analysis/docs/reference/FEATURE_SELECTION.md)**

### 4. Train Ensemble
```bash
ced train-ensemble --run-id 20260127_115115
```

## Configuration System

**Config hierarchy** (lower overrides higher):
1. `configs/training_config.yaml` (model settings, feature selection)
2. `configs/output_config.yaml` (artifact and plot generation controls)
3. `configs/splits_config.yaml` (CV split settings)
4. `configs/pipeline_local.yaml` or `pipeline_hpc.yaml` (execution settings)
5. Environment variables (e.g., `RUN_MODELS`, `DRY_RUN`)
6. CLI flags (e.g., `--model`, `--split-seed`)

## CLI Reference

For complete CLI documentation, see [analysis/docs/reference/CLI_REFERENCE.md](analysis/docs/reference/CLI_REFERENCE.md)

### CLI Commands
| Command | Module | Purpose |
|---------|--------|---------|
| `ced run-pipeline` | `cli/run_pipeline.py` | **Full end-to-end workflow orchestration (RECOMMENDED)** |
| `ced save-splits` | `cli/save_splits.py` | Split generation |
| `ced train` | `cli/train.py` | Model training |
| `ced train-ensemble` | `cli/train_ensemble.py` | Ensemble meta-learner training |
| `ced optimize-panel` | `cli/optimize_panel.py` | Panel optimization (aggregated RFE) |
| `ced consensus-panel` | `cli/consensus_panel.py` | Cross-model consensus panel (RRA) |
| `ced aggregate-splits` | `cli/aggregate_splits.py` | Results aggregation |
| `ced permutation-test` | `cli/permutation_test.py` | Label permutation testing for model significance |
| `ced eval-holdout` | `cli/eval_holdout.py` | Holdout evaluation |
| `ced config` | `cli/config_tools.py` | Config validation and diff |
| `ced convert-to-parquet` | `cli/main.py` | Convert CSV to Parquet format |

## Testing

```bash
# Run all tests
pytest tests/ -v
```

## Package Architecture

For detailed architecture with code pointers, see [docs/ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md).

### Library Modules
| Layer | Modules | Purpose |
|-------|---------|---------|
| Data | `io`, `splits`, `persistence`, `filters`, `schema`, `columns` | Data loading, split generation, column resolution |
| Features | `screening`, `kbest`, `stability`, `corr_prune`, `panels`, `rfe`, `nested_rfe`, `consensus`, `importance`, `grouped_importance`, `drop_column` | Feature selection, importance, and cross-model consensus |
| Models | `registry`, `hyperparams`, `optuna_search`, `training`, `calibration`, `prevalence`, `stacking` | Model training, hyperparameter optimization, and ensemble learning |
| Metrics | `discrimination`, `thresholds`, `dca`, `bootstrap` | Performance metrics |
| Significance | `permutation_test` | Label permutation testing for model significance |
| Evaluation | `predict`, `reports`, `holdout` | Prediction and reporting |
| Plotting | `roc_pr`, `calibration`, `risk_dist`, `dca`, `learning_curve`, `oof`, `optuna_plots`, `panel_curve`, `ensemble` | Visualization |

For output structure details, see [docs/reference/ARTIFACTS.md](analysis/docs/reference/ARTIFACTS.md).

---

## Key Architecture Decisions

The [docs/adr/](analysis/docs/adr/) directory contains 11 Architecture Decision Records documenting critical statistical and methodological design choices, organized by pipeline stage:

**Stage 1: Data Preparation**
- [ADR-001](analysis/docs/adr/ADR-001-split-strategy.md): 50/25/25 train/val/test split strategy
- [ADR-002](analysis/docs/adr/ADR-002-prevalent-train-only.md): Prevalent cases in training only
- [ADR-003](analysis/docs/adr/ADR-003-control-downsampling.md): Control downsampling ratio

**Stage 2: Feature Selection**
- [ADR-004](analysis/docs/adr/ADR-004-four-strategy-feature-selection.md): Three-stage feature selection and consensus workflow (model gate, evidence, RRA)

**Stage 3: Model Training & Ensembling**
- [ADR-005](analysis/docs/adr/ADR-005-nested-cv.md): Nested cross-validation structure
- [ADR-006](analysis/docs/adr/ADR-006-optuna-hyperparameter-optimization.md): Optuna Bayesian hyperparameter optimization
- [ADR-007](analysis/docs/adr/ADR-007-oof-stacking-ensemble.md): Out-of-fold stacking ensemble (implemented 2026-01-22)

**Stage 4: Calibration**
- [ADR-008](analysis/docs/adr/ADR-008-oof-posthoc-calibration.md): OOF-posthoc calibration strategy

**Stage 5: Evaluation & Thresholds**
- [ADR-009](analysis/docs/adr/ADR-009-threshold-on-val.md): Threshold optimization on validation set
- [ADR-010](analysis/docs/adr/ADR-010-fixed-spec.md): Fixed specificity 0.95 for high-specificity screening

**Stage 6: Significance Testing**
- [ADR-011](analysis/docs/adr/ADR-011-permutation-testing.md): Label permutation testing for model significance

---
