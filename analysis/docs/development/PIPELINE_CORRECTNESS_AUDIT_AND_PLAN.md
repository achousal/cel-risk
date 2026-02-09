# Pipeline Correctness Audit and Reliability Plan

**Project**: CeliacRisks (`ced-ml`)
**Date**: 2026-02-09
**Audit mode**: Static code audit (read-through only; no pipeline execution)
**Scope**: End-to-end correctness and reliability across pipeline entrypoints, I/O contracts, validations, bug risk, and evaluation logic.

---

## Executive Summary

The pipeline architecture is strong and modular, with clear orchestration boundaries and many important safeguards (split overlap checks, label validation, probability range checks, and metadata traceability). However, there are several **contract mismatches across stages** that can break correctness despite otherwise sound components.

### Risk snapshot (updated 2026-02-09)

- **High-severity issues**: 5 (ALL RESOLVED: V-01/V-02/V-04 false; V-03/V-14 fixed)
- **Medium-severity issues**: 7 (ALL RESOLVED: V-06/V-07/V-08/V-09/V-10/V-11 fixed; V-05 confirmed correct)
- **Low-severity issues**: 3 (V-12 fixed; V-13 partially fixed -- `.bak` removed, wrappers retained)

### Previously identified risks (all resolved)

1. ~~`run-pipeline` permutation arg mismatch~~ -- VERIFIED FALSE (V-01).
2. ~~Consensus panel manifest mismatch~~ -- VERIFIED FALSE (V-02).
3. ~~OOF importance contract mismatch~~ -- **FIXED** (V-03). ~~Essentiality signal gap~~ -- **FIXED** (V-14).
4. ~~ENSEMBLE prediction type mixing~~ -- VERIFIED FALSE (V-04).

---

## 1) Repo Map (Entrypoints + Stages)

### 1.1 CLI entrypoints

All commands are registered in `analysis/src/ced_ml/cli/main.py`.

| Entrypoint | Main handler | Consumes (primary) | Produces (primary) | Primary modules invoked |
|---|---|---|---|---|
| `ced save-splits` | `analysis/src/ced_ml/cli/main.py:55` | `--infile`, split config/overrides, optional temporal/downsampling flags | split index CSVs + split metadata JSON (+ holdout files in holdout mode) | `analysis/src/ced_ml/cli/save_splits.py`, `analysis/src/ced_ml/data/persistence.py` |
| `ced train` | `analysis/src/ced_ml/cli/main.py:180` | `--infile`, `--split-dir`, `--model`, `--split-seed(s)`, training config | split-level model bundle, metrics, predictions, CV artifacts, diagnostics, plots | `analysis/src/ced_ml/cli/train.py`, `analysis/src/ced_ml/cli/orchestration/*` |
| `ced aggregate-splits` | `analysis/src/ced_ml/cli/main.py:472` | model result directory or `--run-id`, stability/plot flags | aggregated pooled metrics/preds, summary reports, plots, feature stability/panels | `analysis/src/ced_ml/cli/aggregate_splits.py`, `analysis/src/ced_ml/cli/aggregation/*` |
| `ced eval-holdout` | `analysis/src/ced_ml/cli/main.py:647` | full dataset, holdout indices, model artifact | holdout metrics CSV (+ optional preds + DCA diagnostics) | `analysis/src/ced_ml/cli/eval_holdout.py`, `analysis/src/ced_ml/evaluation/holdout.py` |
| `ced train-ensemble` | `analysis/src/ced_ml/cli/main.py:685` | base model split outputs (OOF/val/test preds), run-id/model discovery | ENSEMBLE model bundle, ENSEMBLE preds, metadata, plots | `analysis/src/ced_ml/cli/train_ensemble.py`, `analysis/src/ced_ml/cli/ensemble_helpers.py`, `analysis/src/ced_ml/models/stacking.py` |
| `ced optimize-panel` | `analysis/src/ced_ml/cli/main.py:846` | aggregated stability/importance + raw input + split files | RFE curves/rankings/recommendations (+ plots, optional per-seed joblibs) | `analysis/src/ced_ml/cli/optimize_panel.py`, `analysis/src/ced_ml/features/rfe.py` |
| `ced consensus-panel` | `analysis/src/ced_ml/cli/main.py:1492` | aggregated per-model stability (+ optional OOF importance/essentiality), infile/split_dir | cross-model consensus panel artifacts + metadata | `analysis/src/ced_ml/cli/consensus_panel.py`, `analysis/src/ced_ml/features/consensus/*` |
| `ced permutation-test` | `analysis/src/ced_ml/cli/main.py:1695` | trained model bundles, split files, infile/split metadata | per-seed permutation summaries + null distributions + aggregated significance | `analysis/src/ced_ml/cli/permutation_test.py`, `analysis/src/ced_ml/significance/*` |
| `ced run-pipeline` | `analysis/src/ced_ml/cli/main.py:2000` | orchestration toggles + pipeline/training/splits/permutation configs | full run artifacts across all enabled stages | `analysis/src/ced_ml/cli/run_pipeline.py` |
| `ced convert-to-parquet` | `analysis/src/ced_ml/cli/main.py:2333` | source CSV | compressed Parquet file | `analysis/src/ced_ml/data/io.py` |
| `ced config validate/diff` | `analysis/src/ced_ml/cli/main.py:1962` | config files | validation or diff report output | `analysis/src/ced_ml/cli/config_tools.py` |

### 1.2 Main script/runners

- Console entrypoint: `ced` -> `main()` in `analysis/src/ced_ml/cli/main.py`.
- Local full runner: `run_pipeline()` in `analysis/src/ced_ml/cli/run_pipeline.py:640`.
- HPC runner path: `_run_hpc_mode()` in `analysis/src/ced_ml/cli/run_pipeline.py:306` and `submit_hpc_pipeline` via `ced_ml.hpc.lsf`.

### 1.3 Implemented stage diagram

Pipeline stage sequence in code (`analysis/src/ced_ml/cli/run_pipeline.py:804`, `analysis/src/ced_ml/cli/run_pipeline.py:825`, `analysis/src/ced_ml/cli/run_pipeline.py:857`, `analysis/src/ced_ml/cli/run_pipeline.py:887`, `analysis/src/ced_ml/cli/run_pipeline.py:908`, `analysis/src/ced_ml/cli/run_pipeline.py:934`, `analysis/src/ced_ml/cli/run_pipeline.py:984`, `analysis/src/ced_ml/cli/run_pipeline.py:1020`):

1. Split generation (`ced save-splits`) if missing.
2. Base-model training over `models x split_seeds` (`ced train`).
3. Per-model aggregation (`ced aggregate-splits`).
4. Ensemble training (optional; `ced train-ensemble`).
5. Ensemble aggregation (optional; `ced aggregate-splits` on ENSEMBLE).
6. Panel optimization (optional; `ced optimize-panel`).
7. Cross-model consensus panel (optional; `ced consensus-panel`).
8. Permutation significance testing (optional; `ced permutation-test`).

---

## 2) I/O Contracts Inventory (Per Stage)

## 2.1 Global contracts (all stages)

### Inputs

- **Core dataframe schema**:
  - `eid` ID column and `CeD_comparison` target are mandatory (`analysis/src/ced_ml/data/io.py:215`, `analysis/src/ced_ml/data/io.py:231`).
  - Allowed target labels: `Controls`, `Incident`, `Prevalent` (`analysis/src/ced_ml/data/schema.py:34`, `analysis/src/ced_ml/data/schema.py:38`, `analysis/src/ced_ml/data/schema.py:41`, `analysis/src/ced_ml/data/io.py:223`).
  - Protein feature columns: suffix `*_resid` (`analysis/src/ced_ml/data/schema.py:69`, `analysis/src/ced_ml/data/io.py:279`).
  - Default metadata columns: numeric `age`, `BMI`; categorical `sex`, `Genetic ethnic grouping` (`analysis/src/ced_ml/data/schema.py:21`, `analysis/src/ced_ml/data/schema.py:24`, `analysis/src/ced_ml/data/schema.py:27`).
- **Environment vars used in primary code**:
  - `SEED_GLOBAL` (debug-level global RNG seeding) (`analysis/src/ced_ml/utils/random.py:66`).
  - `CED_RESULTS_DIR` (results root discovery override for run/model discovery paths) (`analysis/src/ced_ml/cli/discovery.py:58`, `analysis/src/ced_ml/cli/consensus_panel.py:303`).
  - HPC informational vars only for logging (LSF/Slurm IDs and resources) (`analysis/src/ced_ml/utils/logging.py:99`).

### Invariants

- Label integrity: at least 1 control and 1 case required (`analysis/src/ced_ml/data/io.py:231`, `analysis/src/ced_ml/data/io.py:245`).
- Binary task definition depends on scenario mapping (`analysis/src/ced_ml/data/schema.py:50`).
- Split overlap is forbidden (`analysis/src/ced_ml/data/persistence.py:34`, `analysis/src/ced_ml/cli/train.py:352`).

---

## 2.2 Stage A: Split Generation (`ced save-splits`)

### Inputs

- CLI/config keys: mode, scenarios, `n_splits`, `seed_start`, `val_size`, `test_size`, `holdout_size`, prevalent/control-downsampling flags, temporal split flags (`analysis/src/ced_ml/cli/main.py:55`, `analysis/src/ced_ml/config/data_schema.py:17`).
- Required file: `--infile` CSV/Parquet with mandatory schema.

### Outputs

- Split CSVs: `train_idx_{scenario}_seed{seed}.csv`, `val_idx_{scenario}_seed{seed}.csv` (optional), `test_idx_{scenario}_seed{seed}.csv` (`analysis/src/ced_ml/data/persistence.py:285`).
- Metadata JSON: `split_meta_{scenario}_seed{seed}.json` (`analysis/src/ced_ml/data/persistence.py:516`).
- Holdout mode outputs:
  - `HOLDOUT_idx_{scenario}[...].csv` (`analysis/src/ced_ml/data/persistence.py:360`).
  - `HOLDOUT_meta_{scenario}[...].json` (`analysis/src/ced_ml/data/persistence.py:608`).
- Resolved config snapshot: `splits_config.yaml` (`analysis/src/ced_ml/cli/save_splits.py:107`).

### Invariants

- Deterministic holdout seed fixed at `42` (`analysis/src/ced_ml/cli/save_splits.py:316`).
- Prevalent leakage guard in eval splits when `prevalent_train_only=True` (`analysis/src/ced_ml/cli/save_splits.py:485`, `analysis/src/ced_ml/config/validation.py:167`).
- Index integrity checks before persistence (`analysis/src/ced_ml/data/persistence.py:34`).

---

## 2.3 Stage B: Training (`ced train`)

### Inputs

- CLI/config keys: `infile`, `split_dir`, `model`, `split_seed`, feature strategy, CV/Optuna/calibration/threshold settings (`analysis/src/ced_ml/cli/main.py:180`, `analysis/src/ced_ml/config/training_schema.py:18`, `analysis/src/ced_ml/config/features_schema.py:24`, `analysis/src/ced_ml/config/cv_schema.py:8`, `analysis/src/ced_ml/config/calibration_schema.py:49`).
- Expected split files: scenario-specific `train/val/test idx` CSVs (`analysis/src/ced_ml/cli/train.py:313`, `analysis/src/ced_ml/cli/train.py:334`).
- Expected metadata alignment: split metadata row filter columns must match current training columns (`analysis/src/ced_ml/cli/orchestration/split_stage.py:122`).

### Outputs

- Output root pattern: `results/run_<RUN_ID>/<MODEL>/splits/split_seed<SEED>/...` (`analysis/src/ced_ml/evaluation/reports.py:74`, `analysis/src/ced_ml/evaluation/reports.py:99`).
- Model artifact:
  - `core/<MODEL>__final_model.joblib` with model, thresholds, prevalence, calibration, resolved columns, config snapshot (`analysis/src/ced_ml/cli/orchestration/persistence_stage.py:89`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:136`).
- Metrics artifacts:
  - `core/val_metrics.csv`, `core/test_metrics.csv` (`analysis/src/ced_ml/evaluation/writers/metrics_writer.py:28`, `analysis/src/ced_ml/evaluation/writers/metrics_writer.py:55`).
  - `cv/cv_repeat_metrics.csv`, optional `core/test_bootstrap_ci.csv` (`analysis/src/ced_ml/evaluation/writers/metrics_writer.py:82`, `analysis/src/ced_ml/evaluation/writers/metrics_writer.py:111`).
- Predictions:
  - `preds/test_preds__<MODEL>.csv`, `preds/val_preds__<MODEL>.csv`, `preds/train_oof__<MODEL>.csv`, `preds/controls_risk__<MODEL>__oof_mean.csv` (`analysis/src/ced_ml/cli/orchestration/persistence_stage.py:551`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:557`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:564`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:581`).
- CV/feature artifacts:
  - `cv/best_params_per_split.csv`, `cv/selected_proteins_per_split.csv`, optional RFECV folder (`analysis/src/ced_ml/cli/orchestration/persistence_stage.py:159`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:164`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:169`).
- Feature/panel artifacts:
  - `panels/<MODEL>__feature_report_train.csv`, `panels/stable_panel__KBest.csv`, `panels/<MODEL>__N<size>__panel_manifest.json`, `panels/<MODEL>__final_test_panel.json`, `panels/<MODEL>__test_subgroup_metrics.csv` (`analysis/src/ced_ml/evaluation/writers/feature_writer.py:27`, `analysis/src/ced_ml/evaluation/writers/feature_writer.py:46`, `analysis/src/ced_ml/evaluation/writers/feature_writer.py:66`, `analysis/src/ced_ml/evaluation/writers/feature_writer.py:87`, `analysis/src/ced_ml/evaluation/writers/feature_writer.py:121`).
- Diagnostics and plots:
  - calibration CSV, DCA CSV/JSON summaries, learning curve CSV, split trace; plot naming `\<MODEL\>__val_*`, `\<MODEL\>__test_*` in `plots/` (`analysis/src/ced_ml/cli/orchestration/persistence_stage.py:630`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:644`, `analysis/src/ced_ml/cli/orchestration/plotting_stage.py:159`, `analysis/src/ced_ml/cli/orchestration/plotting_stage.py:235`).
- Run metadata:
  - `core/run_settings.json`, `config_metadata.json`, shared `run_metadata.json` (`analysis/src/ced_ml/cli/orchestration/persistence_stage.py:433`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:468`, `analysis/src/ced_ml/cli/orchestration/persistence_stage.py:510`).

### Metrics keys written (train/val/test)

- Discrimination: `auroc`, `prauc`, optional `Youden`, `Alpha` (`analysis/src/ced_ml/metrics/discrimination.py:333`).
- Calibration: `brier_score`, `calibration_intercept`, `calibration_slope`, `ECE` (`analysis/src/ced_ml/cli/train.py:402`).
- Threshold-dependent: `threshold`, `precision`, `sensitivity`, `f1`, `specificity`, `fpr`, `tpr`, `tp`, `fp`, `tn`, `fn` (`analysis/src/ced_ml/cli/train.py:423`).
- Multi-target specificity (config-driven): `thr_ctrl_XX`, `sens_ctrl_XX`, `prec_ctrl_XX`, `spec_ctrl_XX` (`analysis/src/ced_ml/metrics/thresholds.py:648`, `analysis/src/ced_ml/metrics/thresholds.py:681`).

### Invariants

- Outer CV requires `folds >= 2` (`analysis/src/ced_ml/models/nested_cv.py:180`).
- Split overlap checks enforced before training (`analysis/src/ced_ml/cli/train.py:352`).
- Prediction finite/bounds checks for test/val/OOF (`analysis/src/ced_ml/cli/orchestration/evaluation_stage.py:185`).
- Threshold selection on validation reused on test (when validation exists) (`analysis/src/ced_ml/cli/orchestration/evaluation_stage.py:102`).

---

## 2.4 Stage C: Split Aggregation (`ced aggregate-splits`)

### Inputs

- Split directories (`split_seed*`) discovered from model results root (`analysis/src/ced_ml/cli/aggregate_splits.py:93`).
- Expected split-level files:
  - metrics: `core/test_metrics.csv`, `core/val_metrics.csv`, `cv/cv_repeat_metrics.csv`.
  - predictions: `preds/test_preds__*.csv`, `preds/val_preds__*.csv`, `preds/train_oof__*.csv` (`analysis/src/ced_ml/cli/aggregation/collection.py:308`).
  - feature artifacts: `cv/selected_proteins_per_split.csv`, `panels/*`.

### Outputs

- Aggregated root pattern: `<model_root>/aggregated/` with subdirs `metrics`, `panels`, `plots`, `cv`, `preds`, `diagnostics` (`analysis/src/ced_ml/evaluation/reports.py:176`).
- Pooled prediction files:
  - `preds/pooled_test_preds.csv`, `preds/pooled_val_preds.csv`, `preds/pooled_train_oof.csv` (+ per-model variants) (`analysis/src/ced_ml/cli/aggregation/orchestrator.py:64`, `analysis/src/ced_ml/cli/aggregation/orchestrator.py:79`, `analysis/src/ced_ml/cli/aggregation/orchestrator.py:109`).
- Metrics files:
  - `metrics/pooled_test_metrics.csv`, `metrics/pooled_val_metrics.csv`, summaries (`analysis/src/ced_ml/cli/aggregation/orchestrator.py:248`, `analysis/src/ced_ml/cli/aggregation/orchestrator.py:290`, `analysis/src/ced_ml/cli/aggregation/report_phase.py:136`).
- Hyperparameter and CV summaries:
  - `cv/all_best_params_per_split.csv`, `cv/hyperparams_summary.csv`, `cv/all_cv_repeat_metrics.csv` (`analysis/src/ced_ml/cli/aggregation/report_phase.py:182`, `analysis/src/ced_ml/cli/aggregation/report_phase.py:190`, `analysis/src/ced_ml/cli/aggregation/report_phase.py:157`).
- Feature summaries/panels:
  - `panels/feature_stability_summary.csv`, `panels/consensus_stable_features.csv`, `panels/feature_report.csv`, `panels/consensus_panel_N<size>.json` (`analysis/src/ced_ml/cli/aggregation/report_phase.py:242`, `analysis/src/ced_ml/cli/aggregation/report_phase.py:248`, `analysis/src/ced_ml/cli/aggregation/report_phase.py:266`, `analysis/src/ced_ml/cli/aggregation/report_phase.py:291`).
- Aggregated plots (per model): `plots/test_roc.*`, `plots/val_roc.*`, `plots/*_pr.*`, `plots/*_calibration.*`, `plots/*_dca.*`, `plots/*_risk_dist.*`, `plots/train_oof_risk_dist.*` (`analysis/src/ced_ml/cli/aggregation/plot_generator.py:191`).

### Invariants

- Aggregation returns empty results gracefully for missing components, but expects stable file naming and columns for compatibility.

---

## 2.5 Stage D: Ensemble Training (`ced train-ensemble`)

### Inputs

- Requires >=2 base models with OOF predictions (`analysis/src/ced_ml/cli/ensemble_helpers.py:161`, `analysis/src/ced_ml/cli/ensemble_helpers.py:178`).
- Input files expected per base model split: `preds/train_oof__<MODEL>.csv`, plus val/test prediction files for inference.

### Outputs

- Default output root: `results/run_<RUN_ID>/ENSEMBLE/splits/split_seed<SEED>/...` (`analysis/src/ced_ml/cli/ensemble_helpers.py:246`).
- Model bundle: `core/ENSEMBLE__final_model.joblib` (`analysis/src/ced_ml/cli/ensemble_helpers.py:291`).
- Prediction files: `preds/val_preds__ENSEMBLE.csv`, `preds/test_preds__ENSEMBLE.csv`, `preds/train_oof__ENSEMBLE.csv` (`analysis/src/ced_ml/cli/ensemble_helpers.py:306`, `analysis/src/ced_ml/cli/ensemble_helpers.py:323`, `analysis/src/ced_ml/cli/ensemble_helpers.py:357`).
- Metadata: `core/metrics.json`, `core/run_settings.json` (`analysis/src/ced_ml/cli/ensemble_helpers.py:410`, `analysis/src/ced_ml/cli/ensemble_helpers.py:427`).

### Invariants

- Probability validation strictly enforces finite `[0,1]` outputs (`analysis/src/ced_ml/cli/ensemble_helpers.py:55`).
- Explicit warning that `train_oof__ENSEMBLE.csv` is in-sample meta-learner output and must not be treated as genuine OOF performance (`analysis/src/ced_ml/cli/ensemble_helpers.py:362`).

---

## 2.6 Stage E: Panel Optimization (`ced optimize-panel`)

### Inputs

- Requires aggregated stability file `panels/feature_stability_summary.csv` and model split artifacts (`analysis/src/ced_ml/cli/optimize_panel.py:218`, `analysis/src/ced_ml/cli/optimize_panel.py:438`).
- Optional significance gate via aggregated permutation results (`analysis/src/ced_ml/cli/optimize_panel.py:372`).

### Outputs

- Under `<aggregated>/optimize_panel/`:
  - `panel_curve_aggregated.csv`, `feature_ranking_aggregated.csv`, `recommended_panels_aggregated.json`, `metrics_summary_aggregated.csv`, optional `cluster_mapping_aggregated.csv` (`analysis/src/ced_ml/features/rfe.py:328`, `analysis/src/ced_ml/features/rfe.py:364`, `analysis/src/ced_ml/features/rfe.py:406`, `analysis/src/ced_ml/features/rfe.py:419`, `analysis/src/ced_ml/features/rfe.py:384`).
  - Plot artifacts: `panel_curve_aggregated.png`, `feature_ranking_aggregated.png` (`analysis/src/ced_ml/cli/optimize_panel.py:665`, `analysis/src/ced_ml/cli/optimize_panel.py:682`).
  - Optional per-seed cache: `optimize_panel/seeds/seed_<N>/rfe_result.joblib` (`analysis/src/ced_ml/cli/optimize_panel.py:272`).

### Invariants

- Fails if stable protein set is below `min_size` (`analysis/src/ced_ml/cli/optimize_panel.py:247`, `analysis/src/ced_ml/cli/optimize_panel.py:496`).
- Uses deterministic split seed artifacts; supports parallel fallback to sequential.

---

## 2.7 Stage F: Consensus Panel (`ced consensus-panel`)

### Inputs

- Requires run-level aggregated model directories with stability reports (`analysis/src/ced_ml/cli/consensus_panel.py:315`).
- Optional significance gating via `*/significance/aggregated_significance.csv` (`analysis/src/ced_ml/cli/consensus_panel.py:100`, `analysis/src/ced_ml/cli/consensus_panel.py:325`).
- Optional OOF importance and essentiality files (if available):
  - OOF: primary `importance/oof_importance__{model}.csv` (V-03 FIXED), fallback `aggregated_oof_importance.csv` (`analysis/src/ced_ml/cli/consensus_panel.py:174`).
  - Essentiality from `optimize_panel/essentiality/panel_{threshold}_essentiality.csv` or `drop_column_validation.csv` (`analysis/src/ced_ml/cli/consensus_panel.py:211`). **NOTE (V-14)**: No upstream command currently produces these files; essentiality signal is always absent unless a drop-column step is manually added to `optimize-panel`.

### Outputs

- Under `results/run_<RUN_ID>/consensus/` (or explicit outdir):
  - `final_panel.txt`, `final_panel.csv`, `consensus_ranking.csv`, `per_model_rankings.csv`, optional `correlation_clusters.csv`, `uncertainty_summary.csv`, `consensus_metadata.json` (`analysis/src/ced_ml/features/consensus/builder.py:199`, `analysis/src/ced_ml/features/consensus/builder.py:397`, `analysis/src/ced_ml/features/consensus/builder.py:404`, `analysis/src/ced_ml/features/consensus/builder.py:411`, `analysis/src/ced_ml/features/consensus/builder.py:418`, `analysis/src/ced_ml/features/consensus/builder.py:457`).

### Invariants

- Requires at least 2 models with valid rankings (`analysis/src/ced_ml/features/consensus/builder.py:104`).
- Correlation pruning and target-size truncation enforced in builder path.

---

## 2.8 Stage G: Permutation Significance (`ced permutation-test`)

### Inputs

- Model bundle per split: `<run>/<model>/splits/split_seed<seed>/core/<model>__final_model.joblib` (`analysis/src/ced_ml/cli/permutation_test.py:75`).
- Input data and split paths auto-detected from `run_metadata.json` (`analysis/src/ced_ml/cli/permutation_test.py:310`, `analysis/src/ced_ml/cli/permutation_test.py:334`).
- Scenario-specific train/val split files required (`analysis/src/ced_ml/cli/permutation_test.py:401`).

### Outputs

- Under `<run>/<model>/significance/`:
  - `permutation_test_results_seed<seed>.csv` (summary per seed),
  - `null_distribution_seed<seed>.csv` (full nulls),
  - `perm_<index>.joblib` (HPC single-permutation mode),
  - `aggregated_significance.csv` (pooled across seeds when aggregating) (`analysis/src/ced_ml/cli/permutation_test.py:485`, `analysis/src/ced_ml/cli/permutation_test.py:489`, `analysis/src/ced_ml/cli/permutation_test.py:463`, `analysis/src/ced_ml/cli/permutation_test.py:213`).

### Invariants

- Metric restricted to AUROC (`analysis/src/ced_ml/cli/permutation_test.py:133`).
- Empirical p-value uses Phipson-Smyth +1 correction (`analysis/src/ced_ml/significance/permutation_test.py:98`).

---

## 2.9 Stage H: Holdout Evaluation (`ced eval-holdout`)

### Inputs

- `infile`, `holdout_idx` CSV containing `idx` column, model artifact bundle (`analysis/src/ced_ml/evaluation/holdout.py:58`, `analysis/src/ced_ml/evaluation/holdout.py:96`).
- Requires bundle `resolved_columns` and prevalence metadata (`analysis/src/ced_ml/evaluation/holdout.py:460`, `analysis/src/ced_ml/evaluation/holdout.py:498`).

### Outputs

- `holdout_metrics.csv`, `holdout_toprisk_capture.csv`, optional `holdout_predictions.csv`, optional DCA artifacts under `diagnostics/` (`analysis/src/ced_ml/evaluation/holdout.py:533`, `analysis/src/ced_ml/evaluation/holdout.py:539`, `analysis/src/ced_ml/evaluation/holdout.py:349`, `analysis/src/ced_ml/evaluation/holdout.py:575`).

### Invariants

- Holdout indices bounds check (`analysis/src/ced_ml/evaluation/holdout.py:133`).
- Never silently replace missing `val_threshold` with test threshold; falls back to 0.5 and warns (`analysis/src/ced_ml/evaluation/holdout.py:180`).

---

## 2.10 Contract deviations from “primary-only” codebase

Observed legacy/cruft artifacts in primary path:

- Backward compatibility re-export module for training APIs (`analysis/src/ced_ml/models/training.py:4`, `analysis/src/ced_ml/models/training.py:15`).
- Compatibility wrapper functions in active CLI modules (`analysis/src/ced_ml/cli/train_ensemble.py:55`, `analysis/src/ced_ml/cli/optimize_panel.py:99`).
- Backup source file present in package tree: `analysis/src/ced_ml/features/consensus.py.bak`.

---

## 3) Validation Gap Analysis

## 3.1 Gap table

| ID | Severity | Gap | What can go wrong | Minimal guardrail | Verification (2026-02-09) |
|---|---|---|---|---|---|
| V-01 | ~~High~~ | ~~`run-pipeline` -> permutation arg mismatch~~ | ~~Stage 8 fails~~ | ~~Align call~~ | **FALSE**: Code already uses `split_seeds=[split_seed]`. HPC warns; local raises. No bug. |
| V-02 | ~~High~~ | ~~Panel manifest producer/consumer mismatch~~ | ~~Consensus finds no manifests~~ | ~~Canonical schema~~ | **FALSE**: Both producer and consumer use `"proteins"` key and same `panels/` directory. No mismatch. |
| V-03 | **High** | OOF importance filename and persistence mismatch | Consensus can silently skip OOF signal despite config saying enabled | Save split-level OOF importance explicitly and standardize aggregated filename | **FIXED**: `load_model_oof_importance()` now accepts `model_name` and looks for `oof_importance__{model}.csv` first, with legacy fallbacks. Tests added. |
| V-04 | ~~High~~ | ~~ENSEMBLE collection ignores `pred_type`~~ | ~~Pooled metrics from mixed sets~~ | ~~Type-specific patterns~~ | **FALSE**: ENSEMBLE uses exact per-type patterns (`test_preds__ENSEMBLE.csv` etc.). No mixing. |
| V-05 | Resolved | CLARIFIED - Implementation is correct (sklearn.base.clone creates unfitted pipeline, so fit() triggers full retune) | No action needed | N/A | Confirmed correct. |
| V-06 | ~~Medium~~ | ~~No-validation mode computes threshold on test~~ | ~~Information leakage~~ | ~~Hard error~~ | **TRUE but FIXED**: Guard added; raises `ValueError` unless `allow_test_thresholding=True`. |
| V-07 | ~~Medium~~ | ~~Probability scale mismatch across metrics vs plots/pooling~~ | ~~Plots may render on raw scale while metrics/thresholds use adjusted~~ | ~~Plot generators should prefer `y_prob_adjusted` when available~~ | **FIXED**: All plot generators and artifact exporters now prefer `y_prob_adjusted` with fallback to `y_prob`. Files: `plot_generator.py`, `artifacts.py`, `plotting_stage.py`. |
| V-08 | ~~Medium~~ | ~~Ensemble metadata contract mismatch (flat JSON vs nested structure expected)~~ | ~~Missing ensemble hyperparam summaries during aggregation~~ | ~~Update writer to nested structure~~ | **FIXED**: `ensemble_helpers.py` now writes nested `ensemble.meta_model.*` structure matching what `collection.py` expects. |
| V-09 | ~~Medium~~ | ~~Protein dtype validation only samples first 10 columns~~ | ~~Late failures~~ | ~~Validate all~~ | **FALSE (FIXED)**: Now validates all protein columns with capped error display. |
| V-10 | ~~Medium~~ | ~~Scenario autodetection chooses first unsorted glob hit~~ | ~~Non-deterministic binding~~ | ~~Sort + error~~ | **FALSE (FIXED)**: Glob results sorted; ambiguity raises `ValueError`. |
| V-11 | ~~Medium~~ | ~~Single-class ensemble metric logging format path~~ | ~~Runtime crash~~ | ~~Guard formatting~~ | **FALSE (FIXED)**: None-safe metric formatting; logs "N/A" for single-class. |
| V-12 | ~~Low~~ | ~~Config loader docstring claims env overrides, code does not implement~~ | ~~Operator confusion about runtime behavior~~ | ~~Remove claim from docstring~~ | **FIXED**: Removed env-override claim from `config/loader.py` module docstring. |
| V-13 | ~~Low~~ | ~~Legacy wrappers + `.bak` file in main package tree~~ | ~~Contract ambiguity and maintenance drag~~ | ~~Remove backup file from source tree~~ | **PARTIALLY FIXED**: `consensus.py.bak` deleted. Compatibility wrappers (`training.py`, `train_ensemble.py`, `optimize_panel.py`) retained -- all have active callers (9, 1, 12 import sites respectively). |
| V-14 | ~~**High**~~ | ~~Essentiality signal never produced as pre-consensus input~~ | ~~Consensus composite ranking silently drops 30% essentiality weight~~ | ~~Add drop-column validation step to `optimize-panel`~~ | **FIXED**: `optimize-panel` now runs drop-column essentiality validation after RFE and saves `optimize_panel/essentiality/panel_{threshold}_essentiality.csv`. See `V14_ESSENTIALITY_SIGNAL_FIX.md`. |

## 3.2 Requested focus checks

| Focus area | Current state | Notes |
|---|---|---|
| Path existence | Strong | V-02 false (contracts aligned); V-03 FIXED; V-08 FIXED (writer now nested); V-14 FIXED (essentiality produced by optimize-panel). |
| Schema mismatch checks | Strong | V-09 fixed: all protein columns now validated. Strong for required columns and labels. |
| Label integrity | Strong | Allowed labels and case/control presence enforced (`analysis/src/ced_ml/data/io.py:223`, `analysis/src/ced_ml/data/io.py:231`). |
| Fold integrity | Strong | Overlap/bounds checks + row-filter alignment checks are present (`analysis/src/ced_ml/data/persistence.py:34`, `analysis/src/ced_ml/cli/orchestration/split_stage.py:122`). |
| Probability ranges | Strong | Explicit finite/bounds checks in training and ensemble paths (`analysis/src/ced_ml/cli/orchestration/evaluation_stage.py:185`, `analysis/src/ced_ml/cli/ensemble_helpers.py:55`). |
| NaNs | Strong | V-11 fixed: None-safe formatting in ensemble logging. Other guards remain in place. |

---

## 4) Bug-Risk Sweep

## 4.1 Inconsistent contracts / names

1. ~~**Permutation invocation mismatch**~~: **VERIFIED FALSE** -- caller already uses `split_seeds=[split_seed]`. No mismatch.
2. ~~**Panel manifest mismatch**~~: **VERIFIED FALSE** -- both sides use `"proteins"` key and same `panels/` directory.
3. ~~**OOF importance naming mismatch**~~: **FIXED** -- `load_model_oof_importance()` now accepts `model_name` param and looks for `oof_importance__{model}.csv` first (`analysis/src/ced_ml/cli/consensus_panel.py:174`), with legacy fallbacks preserved.
4. ~~**ENSEMBLE metadata mismatch**~~: **FIXED** -- `ensemble_helpers.py` now writes nested `ensemble.meta_model.*` structure matching `collection.py` expectations.
5. ~~**Prediction scale naming drift**~~: **FIXED** -- all plot generators and artifact exporters now prefer `y_prob_adjusted` with fallback to `y_prob`, matching metrics code.

## 4.2 Seed handling and reproducibility

What is good:

- Explicit split seeds and deterministic split outputs (`analysis/src/ced_ml/data/persistence.py:285`).
- Deterministic outer CV splitter (`analysis/src/ced_ml/models/nested_cv.py:191`).
- Optional global seed hook exists (`analysis/src/ced_ml/utils/random.py:66`).
- Holdout seed intentionally fixed at 42 (`analysis/src/ced_ml/cli/save_splits.py:316`).

Residual reproducibility risks:

- ~~Scenario autodetect uses unsorted glob first hit~~: **FIXED** -- glob sorted; ambiguity raises `ValueError`.
- `SEED_GLOBAL` only covers Python/NumPy globals and is explicitly debug-oriented; not a full multi-component determinism contract (`analysis/src/ced_ml/utils/random.py:72`).

## 4.3 Metric edge cases

- **Single-class folds**:
  - Discrimination utilities return NaNs safely when single-class (`analysis/src/ced_ml/metrics/discrimination.py:326`).
  - Multi-target specificity returns empty dict safely (`analysis/src/ced_ml/metrics/thresholds.py:668`).
  - Ensemble helper returns empty metrics on single-class (`analysis/src/ced_ml/cli/ensemble_helpers.py:39`).
  - ~~Train-ensemble logging formats missing values as float and can fail~~: **FIXED** -- None-safe formatting with "N/A" fallback.
- **All-constant predictions**:
  - Threshold selection functions contain fallbacks (`analysis/src/ced_ml/metrics/thresholds.py:192`, `analysis/src/ced_ml/metrics/thresholds.py:245`).
- **Empty folds / tiny classes**:
  - Outer CV `folds>=2` enforced (`analysis/src/ced_ml/models/nested_cv.py:180`).
  - Calibration CV is down-shifted to safe folds based on minority class count (`analysis/src/ced_ml/models/nested_cv.py:836`).

---

## 5) Evaluation Correctness Review

## 5.1 Conceptually correct components

1. **Discrimination metrics** (AUROC/PR-AUC) are computed with standard sklearn definitions (`analysis/src/ced_ml/metrics/discrimination.py:333`).
2. **Brier score** implementation is standard (`analysis/src/ced_ml/metrics/discrimination.py:373`).
3. **Threshold selection methods** (Youden, fixed specificity, fixed precision) are statistically appropriate for screening workflows (`analysis/src/ced_ml/metrics/thresholds.py:188`, `analysis/src/ced_ml/metrics/thresholds.py:245`).
4. **Thresholded metric derivation** from confusion matrix is standard (`analysis/src/ced_ml/metrics/thresholds.py:418`).
5. **Decision Curve Analysis** net benefit formula matches Vickers-Elkin formulation (`analysis/src/ced_ml/metrics/dca.py:37`, `analysis/src/ced_ml/metrics/dca.py:77`).
6. **Prevalence adjustment** uses logit intercept-shift method (correct family for prior-shift correction) (`analysis/src/ced_ml/models/prevalence.py:63`).
7. **Non-leaky thresholding behavior** is correct when validation exists: threshold selected on val, reused on test (`analysis/src/ced_ml/cli/orchestration/evaluation_stage.py:102`).

## 5.2 Correctness caveats

1. ~~No-validation mode explicitly computes threshold on test~~: **FIXED** -- raises `ValueError` unless `allow_test_thresholding=True`.
2. ~~Probability scale consistency is not enforced end-to-end~~: **FIXED** (V-07). All plot/artifact code now prefers `y_prob_adjusted`.
3. Permutation methodology communication and implementation are currently misaligned.

---

## 6) Prioritized Recommendations

## Revised Remediation Plan (post-verification 2026-02-09)

Original P0 had 4 items; verification showed 3 were false (V-01, V-02, V-04). Remaining open issues reorganized below.

## Phase A: Contract fix (High severity)

1. ~~**V-03**~~: **FIXED** -- `load_model_oof_importance()` updated to accept `model_name` and look for `oof_importance__{model}.csv` first (`analysis/src/ced_ml/cli/consensus_panel.py:174`). Legacy fallbacks preserved. 6 regression tests added (`analysis/tests/features/test_consensus.py`).

2. ~~**V-14**~~: **FIXED** -- `optimize-panel` now runs `run_drop_column_validation_for_panels()` after RFE aggregation, writing `optimize_panel/essentiality/panel_{threshold}_essentiality.csv`. See `V14_ESSENTIALITY_SIGNAL_FIX.md`.

**Phase A acceptance criteria**

- ~~V-03~~: Consensus stage finds and uses OOF importance when available. DONE -- 6 tests pass.
- ~~V-14~~: After `optimize-panel` completes, `optimize_panel/essentiality/panel_95pct_essentiality.csv` exists. DONE.

## Phase B: Consistency fixes (Medium severity)

1. ~~**V-07**~~: **FIXED** -- Plot generators and artifact exporters now prefer `y_prob_adjusted` with fallback to `y_prob`. Files: `plot_generator.py`, `artifacts.py`, `plotting_stage.py`.
2. ~~**V-08**~~: **FIXED** -- `ensemble_helpers.py` writer updated to nested `ensemble.meta_model.*` structure matching `collection.py` reader.

**Phase B acceptance criteria**

- ~~Plot generators use `y_prob_adjusted` when present, falling back to `y_prob`.~~ DONE.
- ~~Ensemble aggregation reports metadata/hyperparams from `run_settings.json`.~~ DONE.

## Phase C: Cleanup (Low severity)

1. ~~**V-12**~~: **FIXED** -- Removed env-override claim from `config/loader.py` module docstring.
2. **V-13**: **PARTIALLY FIXED** -- `consensus.py.bak` deleted. Compatibility wrappers retained (all have active callers: `training.py` 9 sites, `train_ensemble.py` 1 site, `optimize_panel.py` 12 sites).

**Phase C acceptance criteria**

- ~~Config loader docstring accurately describes implemented features.~~ DONE.
- ~~No `.bak` files in source tree.~~ DONE.
- Compatibility wrappers retained -- all have active callers; future refactor can migrate imports directly to source modules.

---

## Previously resolved items (verified 2026-02-09)

These issues from the original audit were either false or already fixed:

- ~~V-01~~: Permutation arg name was already correct (`split_seeds=[split_seed]`).
- ~~V-02~~: Panel manifest contract was already aligned (`"proteins"` key, `panels/` dir).
- ~~V-04~~: ENSEMBLE collection already uses pred_type-specific exact patterns.
- ~~V-06~~: Threshold-on-test guard added (`allow_test_thresholding` config flag).
- ~~V-09~~: Protein dtype validation now checks all columns.
- ~~V-10~~: Scenario autodetection sorted and ambiguity-safe.
- ~~V-11~~: Single-class metric logging uses None-safe formatting.
- ~~V-03~~: OOF importance loader updated to match actual aggregated filename (`oof_importance__{model}.csv`). 6 tests added.
- ~~V-07~~: Plot generators and artifact exporters now prefer `y_prob_adjusted`. Files: `plot_generator.py`, `artifacts.py`, `plotting_stage.py`.
- ~~V-08~~: Ensemble writer updated to nested structure. File: `ensemble_helpers.py`.
- ~~V-12~~: Config loader docstring corrected. File: `config/loader.py`.
- ~~V-13~~: `consensus.py.bak` deleted. Wrappers retained (active callers).
- ~~V-14~~: Essentiality signal produced by `optimize-panel`. File: `optimize_panel.py`.

---

## Recommended remediation sequence

1. **Phase A** -- ~~V-03 OOF importance contract fix~~ DONE. ~~V-14 essentiality gap~~ DONE.
2. **Phase B** -- ~~Probability scale (V-07)~~ DONE. ~~Ensemble metadata (V-08)~~ DONE.
3. **Phase C** -- ~~Docstring (V-12)~~ DONE. V-13 partially done (`.bak` removed, wrappers retained).

**All high- and medium-severity items are resolved. Only V-13 wrappers remain as future refactoring opportunity.**

---

## Notes

- Original audit (2026-02-09) performed **no runtime changes** -- static source inspection only.
- V-03 fix applied 2026-02-09: `load_model_oof_importance()` updated in `consensus_panel.py`; 6 regression tests added in `test_consensus.py`.
- V-14 discovered 2026-02-09 during V-03 investigation: essentiality signal has no upstream producer.
- V-07/V-08/V-12/V-13/V-14 fixes applied 2026-02-09: plot scale alignment, ensemble metadata contract, docstring, cleanup, essentiality signal.
