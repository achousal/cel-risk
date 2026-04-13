# cel-risk Project Documentation

**Project**: Machine Learning Pipeline for Incident Celiac Disease Risk Prediction
**Version**: 1.0.0
**Updated**: 2026-04-12
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

**Important**: All `ced` commands must be run from the project root (`cel-risk/`).

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

**Customize via configs** (paths relative to `analysis/`):
- `configs/training_config.yaml` - Models, CV, feature selection (canon)
- `configs/output_config.yaml` - Artifact and plot generation controls (canon)
- `configs/pipeline_local.yaml` / `configs/pipeline_hpc.yaml` - Execution settings (canon)
- `configs/variants/val_consensus/pipeline_hpc_val_consensus.yaml` - Phase 2 validation override (inherits canon via `_base: ../../pipeline_hpc.yaml`)

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
4. `configs/pipeline_local.yaml` or `pipeline_hpc.yaml` (execution settings; supports `_base` inheritance)
5. Environment variables (e.g., `RUN_MODELS`, `DRY_RUN`)
6. CLI flags (e.g., `--model`, `--split-seed`)

**Pipeline config resolution:** `--pipeline-config` > `--hpc-config` (when `--pipeline-config` is omitted, `--hpc-config` doubles as the pipeline config) > auto-detect. Config files support `_base` key for YAML inheritance (base loaded first, current file deep-merged on top).

## Operations & Experiments

`ced_ml` (under `analysis/`) is the **library**. **Operations** (orchestration code, configs, scripts, post-hoc analysis) live under `operations/` and consume the library via the CLI or Python API. **Experiments** are the concrete *outputs* of those operations — runs that materialize under `results/` with their own metadata, registered in `results/experiment_registry.csv`.

In short:
- `operations/` = how to run things (versioned, code).
- `results/` = the experiments themselves (gitignored, runtime).

```
cel-risk/
├── analysis/                          # ced_ml library — pure, experiment-agnostic
│   ├── src/ced_ml/
│   └── configs/                       # canon base configs + scenario variants
│       ├── README.md                  # canon/variants policy
│       ├── training_config.yaml       # CANON (10 files, file names frozen)
│       ├── splits_config.yaml
│       ├── pipeline_local.yaml
│       ├── pipeline_hpc.yaml
│       ├── output_config.yaml
│       ├── aggregate_config.yaml
│       ├── consensus_panel.yaml
│       ├── optimize_panel.yaml
│       ├── permutation_test.yaml
│       ├── holdout_config.yaml
│       ├── variants/                  # scenario overrides via `_base:` inheritance
│       │   ├── val_consensus/         #   phase 2 cross-model validation
│       │   ├── holdout/               #   holdout evaluation
│       │   ├── 4protein/              #   fixed 4-protein panel validation
│       │   └── local/                 #   local-mode training overrides
│       └── _archive/                  # deprecated (e.g. pipeline_hpc_phase2.yaml)
├── operations/
│   ├── README.md
│   ├── cellml/                        # CellML — factorial recipe sweep (was: optimal-setup/factorial)
│   │   ├── MASTER_PLAN.md
│   │   ├── DESIGN.md
│   │   ├── configs/manifest.yaml      # declarative recipe + factorial source of truth
│   │   ├── analysis/                  # post-hoc R/Python analysis
│   │   ├── sweeps/                    # sweep orchestration engine
│   │   └── submit_experiment.sh
│   ├── incident-validation/           # Incident Validation — pre-diagnostic case validation
│   │   ├── README.md
│   │   ├── RESULTS_LR_EN.md
│   │   ├── scripts/                   # run_lr.py, run_svm.py, submit_*.sh
│   │   └── analysis/                  # calibration, DCA, SHAP, saturation
│   └── _archive/gen1/                 # frozen first-generation experiments
├── results/                           # gitignored — namespaced by experiment
│   ├── experiment_registry.csv        # append-only run log (see ced_ml/utils/registry.py)
│   ├── cellml/{discovery,v0_gate,main,holdout,compiled,figures}/
│   ├── incident-validation/{lr,linsvm_cal,compiled,figures}/
│   ├── pipeline/                      # ad-hoc pipeline runs (no experiment tag)
│   └── _archive/                      # legacy artifacts (gen1, pre-restructure)
└── logs/                              # gitignored — mirrors results/ namespace
    ├── cellml/, incident-validation/, pipeline/
```

**Tagging runs**: `ced run-pipeline --experiment cellml_v0` (same option on `ced train`) prefixes the auto-generated run_id (e.g. `cellml_v0_20260412_123456`) and records the run in `results/experiment_registry.csv` via `_register_run_safe()` (best-effort; failures do not block). When `outdir` lands under `results/<exp>/<phase>/`, `derive_logs_dir()` in `utils/paths.py` mirrors that namespace into `logs/<exp>/<phase>/run_<id>/` so logs co-locate with their result artifacts.

**Config canon**: the 10 files at the root of `analysis/configs/` are the single source of truth and their filenames are referenced by name from inside `ced_ml/cli/` — do not rename or move them. Scenario-specific overrides live under `analysis/configs/variants/<scenario>/` and inherit via `_base:` (paths relative to the file containing the declaration). See `analysis/configs/README.md` for the full policy.

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
