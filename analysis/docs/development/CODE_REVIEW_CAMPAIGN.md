# Code Review & Refactoring Campaign

## Context

The CeliacRisks codebase (46K lines, 136 modules) is in good shape: no TODOs, proper logging, consistent type hints, centralized constants. However, 19 files exceed the 400-line guideline, test coverage is 15% overall (core modules 40-82%), and critical statistical/security logic has not had a formal review. This campaign systematically reviews correctness, security, and modularity, then applies targeted refactoring.

## Campaign Structure: 3 Waves

### Wave 1 -- Code Reviews (read-only, all parallel)

Eight `code-reviewer` agents analyzing correctness, security, and modularity across all layers. No code changes.

| # | Focus | Model | Files | Looking for |
|---|-------|-------|-------|-------------|
| 1.1 | Statistical logic (nested CV, calibration) | **opus** | `models/nested_cv.py`, `models/calibration.py`, `models/calibration_strategy.py`, `models/prevalence.py` | Train/val/test leakage, OOF calibration bugs, fold isolation, seed propagation |
| 1.2 | Feature selection correctness | **sonnet** | `features/rfe_engine.py`, `features/nested_rfe.py`, `features/consensus.py`, `features/importance.py`, `features/drop_column.py` | Feature leakage across folds, RFE elimination edge cases, RRA math, stability metrics |
| 1.3 | Data pipeline + config security | **sonnet** | `data/` (all 7 modules), `config/loader.py`, `config/validation.py` | Path injection, YAML injection, schema validation gaps, secrets exposure |
| 1.4 | CLI monolith analysis | **sonnet** | `cli/main.py` (2,385 lines), `cli/run_pipeline.py` (1,056 lines) | Natural split boundaries, duplication, error handling gaps, modularity violations |
| 1.5 | Metrics & thresholds correctness | **sonnet** | `metrics/thresholds.py`, `metrics/dca.py`, `metrics/discrimination.py`, `metrics/bootstrap.py` | DCA calculation bugs, threshold leakage (val vs test), bootstrap CI edge cases |
| 1.6 | HPC + orchestration safety | **sonnet** | `hpc/lsf.py`, `cli/orchestration/` (all 10 modules) | Shell injection in bsub, job dependency logic, error recovery, log secrets |
| 1.7 | Stacking & ensemble correctness | **opus** | `models/stacking.py`, `models/stacking_utils.py`, `cli/train_ensemble.py`, `cli/ensemble_helpers.py` | OOF stacking leakage (ADR-007), meta-learner fold correctness, 13% coverage gaps |
| 1.8 | Plotting & output safety | **haiku** | `plotting/` (11 modules), `evaluation/writers/` (4 modules) | File overwrite safety, error handling (plots must not crash pipeline), memory limits |

**Deliverable**: Each agent produces a findings report with issues ranked by severity.

---

### Wave 2 -- Targeted Refactoring (parallel, based on Wave 1 findings)

Six `refactor-cleaner` agents addressing highest-impact issues. All changes preserve existing behavior.

| # | Target | Model | Scope | Goal |
|---|--------|-------|-------|------|
| 2.1 | CLI monolith decomposition | **sonnet** | `cli/main.py` (2,385 lines) | Split into 5-7 subcommand modules, each <500 lines |
| 2.2 | Nested CV modularization | **sonnet** | `models/nested_cv.py` (1,177 lines) | Extract fold execution + protein selection, keep orchestrator <500 lines |
| 2.3 | RFE engine split | **sonnet** | `features/rfe_engine.py` (1,015 lines) | Extract panel eval + elimination loop, keep orchestrator <400 lines |
| 2.4 | Large feature modules | **sonnet** | `features/consensus.py` (838), `features/importance.py` (804), `features/drop_column.py` (692) | Split each below 500 lines by responsibility |
| 2.5 | Large CLI modules | **sonnet** | `cli/optimize_panel.py` (718), `cli/consensus_panel.py` (661) | Extract shared validation, reduce to <450 lines each |
| 2.6 | Security hardening | **sonnet** | Files flagged by agents 1.3 + 1.6 | Fix all security findings (path validation, shell sanitization, schema validation) |

**Acceptance**: All existing tests pass. No files >600 lines in modified set. `ruff` + `black` clean.

---

### Wave 3 -- Verification & Cleanup (parallel)

Four agents verifying Wave 2 changes and closing gaps.

| # | Focus | Model | Scope | Goal |
|---|-------|-------|-------|------|
| 3.1 | Run test suite + lint | **haiku** | Full project | Verify all tests pass, ruff/black/mypy clean |
| 3.2 | Re-review refactored statistical modules | **sonnet** | Wave 2.2 + 2.3 output | Confirm no regressions in nested CV, RFE correctness |
| 3.3 | Re-review refactored CLI | **sonnet** | Wave 2.1 + 2.5 output | Confirm CLI behavior unchanged, imports correct |
| 3.4 | Final dead code sweep | **haiku** | Full `src/ced_ml/` | Remove any orphaned functions/imports introduced by refactoring |

**Acceptance**: All tests green. Zero ruff/black violations. No dead code.

---

## Critical Files

- `analysis/src/ced_ml/cli/main.py` -- 2,385 lines, monolithic CLI (Wave 2.1)
- `analysis/src/ced_ml/models/nested_cv.py` -- 1,177 lines, statistical core (Wave 1.1 + 2.2)
- `analysis/src/ced_ml/features/rfe_engine.py` -- 1,015 lines, feature selection (Wave 1.2 + 2.3)
- `analysis/src/ced_ml/hpc/lsf.py` -- 794 lines, shell command construction (Wave 1.6)
- `analysis/src/ced_ml/models/stacking.py` -- 466 lines, ensemble logic (Wave 1.7)

## Existing Utilities to Reuse

- `utils/logging.py` -- logging setup, reuse for new modules
- `utils/paths.py` -- path resolution, reuse for validation
- `utils/constants.py` -- centralized constants
- `data/schema.py` -- ModelName enum, column schemas
- `config/defaults.py` -- default values

## Verification

After all waves:
1. `pytest tests/ -v` -- all tests pass
2. `ruff check analysis/src/` -- zero violations
3. `black --check analysis/src/` -- zero formatting issues
4. Manual: verify `ced --help` and `ced run-pipeline --help` unchanged
5. Manual: spot-check that no file in `src/ced_ml/` exceeds 600 lines

## Execution Notes

- Wave 1 agents are all read-only (code-reviewer) and run in parallel
- Wave 2 agents depend on Wave 1 findings; each gets a summary of relevant review findings
- Wave 3 agents depend on Wave 2 completion
- Total: 18 agents across 3 waves
- Model budget: 2 opus, 13 sonnet, 3 haiku

Unresolved questions: none.

---

## Wave 1 Results (2026-02-07)

**Status**: COMPLETE. All 8 agents finished. 97 findings across all severity levels.

### Severity Summary

| Severity | Count | Key Areas |
|----------|-------|-----------|
| CRITICAL | 8 | Feature leakage (RFE, consensus), threshold leakage (holdout), bootstrap edge cases, CLI error handling, plotting failures |
| HIGH | 23 | Calibration metric optimism, RRA penalties, early stopping, HPC duplication, test coverage gaps, writer overwrites |
| MEDIUM | 31 | NaN safeguards, schema validation, path injection, resource validation, double calibration, metadata inconsistency |
| LOW | 21 | Determinism, error messages, DPI, colorblind palette |
| INFO | 14 | Confirmed-correct items (prevalence math, OOF stacking, screening cache) |

### Confirmed Correct (positive findings)

- OOF predictions are genuinely out-of-fold (no leakage in core CV path)
- Prevalence adjustment math correct (Saerens 2002 intercept shift)
- DCA net benefit formula matches Vickers & Elkin (2006)
- Phipson-Smyth p-value correction correctly prevents p=0
- YAML uses `safe_load` throughout (no code injection)
- No pickle/eval/exec usage anywhere
- Bootstrap stratification preserves class ratio
- Screening cache keys include data hashes (no cross-fold leakage)

---

### Agent 1.1: Statistical Logic (opus) -- 0C / 2H / 3M / 3L / 3I

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| H-1 | HIGH | `models/nested_cv.py` | 490-502 | OOF calibrator fitted then applied to same OOF predictions; reported calibration metrics (Brier, ECE) are optimistically biased |
| H-2 | HIGH | `models/nested_cv.py` | 816-823 | `per_fold` CalibratedClassifierCV with `cv=int` discards tuned best_estimator_ and re-fits from scratch on smaller internal splits |
| M-1 | MED | `models/nested_cv.py` | 826-865 | `_get_model_feature_count` reads unfitted template from CalibratedClassifierCV, always returning `features=0` in fold logs |
| M-2 | MED | `models/nested_cv.py` | 817-819 | CalibratedClassifierCV constructed without explicit random_state/CV splitter; deterministic only by current sklearn defaults |
| M-3 | MED | `models/nested_cv.py` | 392 | RFECV code path accesses `search.best_estimator_` without null check; crashes when hyperparameter tuning disabled |
| L-1 | LOW | `models/nested_cv.py` | 567-597 | `_scoring_to_direction` silently defaults to "maximize" for unknown metrics |
| L-2 | LOW | `models/prevalence.py` | 282-295 | `__getattr__` delegation risks infinite recursion during unpickling |
| L-3 | LOW | `models/calibration.py` | 303-353 | `OOFCalibrator.fit()` logs Brier improvement computed in-sample |
| I-1 | INFO | `features/screening_cache.py` | 152 | Global screening cache verified: keys include data hashes, no cross-fold leakage |
| I-2 | INFO | `models/nested_cv.py` | 636 | Same random_state for all outer folds' inner CV; correct (different data subsets) |
| I-3 | INFO | `models/prevalence.py` | 54-114 | Prevalence adjustment math verified correct (Saerens 2002) |

---

### Agent 1.2: Feature Selection (sonnet) -- 2C / 3H / 4M / 3L / 3I

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| C-1 | CRIT | `features/rfe_engine.py` | 774-827 | Elimination loop ranks features on full training fold without inner CV, leaking information used for selection decisions |
| C-2 | CRIT | `features/consensus.py` | 391-423 | `cluster_and_select_representatives` accepts `df_train` with no enforcement that val/test data is excluded from correlation computation |
| H-1 | HIGH | `features/rfe_engine.py` | 1007-1013 | Early stopping fires on single noisy AUROC drop with no minimum-evaluations guard |
| H-2 | HIGH | `features/consensus.py` | 313-321 | Missing-protein penalty in RRA uses per-model `max_rank + 1`, creating inconsistent penalties across models with different panel sizes |
| H-3 | HIGH | `features/drop_column.py` | 254-260 | Random state only set on outermost pipeline step; nested estimators may remain unseeded |
| M-1 | MED | `features/rfe_engine.py` | 815-820 | When all importances tied, `min()` breaks ties alphabetically, risking arbitrary elimination |
| M-2 | MED | `features/nested_rfe.py` | 146-153 | `min_features_to_select` passed to RFECV without lower-bound validation |
| M-3 | MED | `features/consensus.py` | 331-345 | `rank_std`/`rank_cv` computed only on present models, making low-std proteins that are missing in most models appear falsely stable |
| M-4 | MED | `features/importance.py` | 584-591 | Grouped importance clusters features on validation data rather than training data |
| L-1 | LOW | `features/stability.py` | 164-170 | Fallback `fallback_top_n=20` may not match intended downstream panel size |
| L-2 | LOW | `features/rfe_engine.py` | 354-409 | RFE correlation clustering uses `selection_freq` from previous CV folds |
| L-3 | LOW | `features/rfe_engine.py` | 301-351 | Knee-point detection returns arbitrary point for monotonically declining curves |
| I-1 | INFO | `features/consensus.py` | 278-283 | `robust_rank_aggregate` does not validate required columns |
| I-2 | INFO | `features/rfe_engine.py` | 573-588 | `quick_tune_at_k` defaults to 3-fold CV; may be unstable for rare outcomes |
| I-3 | INFO | `features/grouped_importance.py` | 263-274 | SE of importance not reported; users may confuse `std_importance` with SE |

---

### Agent 1.3: Data Pipeline + Config Security (sonnet) -- 0C / 1H / 3M / 2L / 1I

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| H-1 | HIGH | `config/loader.py` | 58-69 | `_base` YAML reference resolves paths without containment check, allowing traversal to arbitrary files |
| M-1 | MED | `data/io.py`, `data/columns.py` | io:95-97, columns:159-166 | User-supplied file paths accepted without directory boundary validation |
| M-2 | MED | `data/io.py` | 103, 175 | Post-load DataFrame validation missing checks for duplicate IDs, protein column count, unexpected columns |
| M-3 | MED | `config/loader.py` | 92-110 | `resolve_paths_relative_to_config()` resolves paths without verifying they stay within project root |
| L-1 | LOW | `data/columns.py`, `data/io.py`, `data/splits.py` | various | Error messages expose full filesystem paths, risky on shared HPC |
| L-2 | LOW | `data/persistence.py`, `data/io_helpers.py` | various | No atomic writes; append mode bypasses overwrite guards |
| I-1 | INFO | `config/loader.py` | 189-246 | CLI override parser lacks early type validation (Pydantic catches it later) |

---

### Agent 1.4: CLI Monolith Analysis (sonnet) -- 2C / 6H / 7M / 4L / 3I

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| C-1 | CRIT | `cli/main.py` | 2374-2376 | Bare `except Exception` catches SystemExit/KeyboardInterrupt |
| C-2 | CRIT | `cli/main.py` | 427-428, 1305 | HPC submission failure logged but pipeline continues silently |
| H-1 | HIGH | `cli/main.py` | 316-437, 1198-1313, 1814-1915 | HPC submission logic duplicated verbatim across three commands |
| H-2 | HIGH | `cli/main.py` | 1154-1164, 1380-1393 | Missing try-except around `json.load()` on run_metadata.json |
| H-3 | HIGH | `cli/main.py` | mixed | Inconsistent exception types: mix of `click.UsageError`, `click.ClickException`, raw Python exceptions |
| H-4 | HIGH | `cli/main.py` | 2210-2226 | Business logic (`_derive_split_seeds_from_config`) embedded in CLI layer |
| H-5 | HIGH | `cli/main.py` | 1054-1095, 1617-1653 | Config-file-then-CLI-override merging duplicated between optimize_panel and consensus_panel |
| H-6 | HIGH | `cli/main.py` | entire (2385) | File exceeds modularity threshold by 4x; 11 commands in single file |
| M-1 | MED | `cli/main.py` | 296-314, 554-572, 1097-1104 | Mutually-exclusive argument validation copy-pasted three times |
| M-2 | MED | `cli/main.py` | 302-314, 786-812 | Seed-list parsing duplicated between train and train_ensemble |
| M-3 | MED | `cli/main.py` | 1621-1624 | Broad `except RuntimeError` swallowed silently without logging |
| M-4 | MED | `cli/main.py` | 1124-1125, 1244-1245, 1319-1328 | `CED_RESULTS_DIR` env-var access scattered across three locations |
| M-5 | MED | `cli/main.py` | 994-1477 | optimize_panel command is 483 lines with three modes inlined |
| M-6 | MED | `cli/run_pipeline.py` | 630-1056 | `run_pipeline()` is 426 lines orchestrating 8 stages; should decompose |
| M-7 | MED | `cli/main.py` | 578-633, 1315-1443 | Run-ID auto-discovery duplicated instead of using `cli/discovery.py` consistently |
| L-1 | LOW | `cli/main.py` | scattered | Error output format inconsistent (no standard prefix) |
| L-2 | LOW | `cli/main.py` | scattered | Section header formatting repeated ~12 times |
| L-3 | LOW | `cli/main.py` | ~41 locations | Nearly all imports inline within functions |
| L-4 | LOW | `cli/main.py` | various | Same modules re-imported in multiple function bodies |
| I-1 | INFO | `cli/main.py` | 316-437 | HPC commands tightly coupled to LSF; no abstraction for Slurm/PBS |
| I-2 | INFO | `cli/main.py` | N/A | No tests for HPC submission or config merging logic |
| I-3 | INFO | `cli/discovery.py` | entire | Good discovery module exists but 3 commands bypass it |

**Proposed split for `main.py`** (from agent):
```
cli/commands/
  core.py              -- cli group, convert-to-parquet, eval-holdout
  data_prep.py         -- save-splits
  training.py          -- train, train-ensemble
  aggregation.py       -- aggregate-splits
  panel_optimization.py -- optimize-panel, consensus-panel
  significance.py      -- permutation-test
  config_tools.py      -- config validate, config diff
  orchestration.py     -- run-pipeline

cli/utils/
  hpc_helpers.py       -- shared HPC submission logic
  config_merge.py      -- config priority resolution
  validation.py        -- argument validation helpers
```

---

### Agent 1.5: Metrics & Thresholds (sonnet) -- 2C / 7H / 2M / 2L

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| C-1 | CRIT | `evaluation/holdout.py` | 189-191 | Silent fallback from `val_threshold` to `test_threshold` risks test-set leakage (ADR-009 violation) |
| C-2 | CRIT | `metrics/bootstrap.py` | 38-119 | No guard for constant/near-zero-variance predictions; bootstrap CI meaningless or degenerate |
| H-1 | HIGH | `metrics/thresholds.py` | 219-268 | `threshold_for_specificity` silently returns different specificity than requested without reporting achieved value |
| H-2 | HIGH | `metrics/dca.py` | 269-279 | DCA zero-crossing uses linear interpolation on nonlinear curve; ~10-50% relative error at low thresholds |
| H-3 | HIGH | `metrics/bootstrap.py` | 111-113 | Default `min_valid_frac=0.1` too permissive; CI from 10% valid samples is unstable for imbalanced data |
| H-4 | HIGH | `metrics/thresholds.py` | 174-216 | Youden's J returns first optimal threshold on plateau instead of midpoint |
| H-5 | HIGH | `metrics/dca.py` | 147-149 | No warning when `prevalence_adjustment` deviates significantly from observed prevalence |
| H-6 | HIGH | `metrics/thresholds.py` | 398-419 | `binary_metrics_at_threshold` returns `precision=0` via `zero_division=0` when no positive predictions, hiding degenerate state |
| H-7 | HIGH | `metrics/thresholds.py` | 405 | Specificity returns NaN when TN+FP=0 without warning; downstream may not handle NaN |
| M-1 | MED | `metrics/dca.py` | 621-675 | DCA auto-range may be too narrow for low-prevalence clinical utility |
| M-2 | MED | `metrics/thresholds.py` | 611-614 | `compute_multi_target_specificity_metrics` returns empty `{}` for single-class input; callers may KeyError |
| L-1 | LOW | `metrics/bootstrap.py` | 78 | Uses legacy `np.random.RandomState` instead of `np.random.default_rng` |
| L-2 | LOW | `metrics/dca.py` | 172-175 | Relative utility unstable when `nb_all` very small (no tolerance guard) |

---

### Agent 1.6: HPC + Orchestration Safety (sonnet) -- 0C / 0H / 4M / 4L

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| M-1 | MED | `hpc/lsf.py` | 251-252, 282, 520 | `run_id` and `model` names interpolated into shell commands without character validation |
| M-2 | MED | `utils/serialization.py` | 14-16 | No atomic writes; concurrent HPC jobs risk partial/corrupt output |
| M-3 | MED | `config/compute_schema.py` | 18-54 | `cores`/`mem_per_core` have no upper bounds; `walltime` has no format validation |
| M-4 | MED | `hpc/lsf.py` | 169-177 | Successful jobs auto-delete `.err` logs, destroying audit trail |
| L-1 | LOW | `hpc/lsf.py` | 551, 762 | Dependencies use `done()` which fires on completion regardless of exit code |
| L-2 | LOW | `hpc/lsf.py` | 523-527 | Config/data paths resolved but not validated for existence before submission |
| L-3 | LOW | `hpc/lsf.py` | 68-75 | Venv activation script not checked for ownership or write permissions |
| L-4 | LOW | `cli/consensus_panel.py` | 306-316 | `joblib.load()` in loop with no per-seed exception handling |

---

### Agent 1.7: Stacking & Ensemble (opus) -- 0C / 2H / 4M / 2L / 2I

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| H-1 | HIGH | `models/stacking.py` | 240-246 | `CalibratedClassifierCV(method="isotonic")` on meta-learner is degenerate at 0.34% prevalence (~15 positives per calibration fold) |
| H-2 | HIGH | `tests/models/test_stacking_*.py` | all | No tests for calibration chain with extreme imbalance, double calibration, ensemble OOF integrity, y_true consistency |
| M-1 | MED | `models/nested_cv.py` + `stacking.py` | 488-502, 240-254 | Base model OOF predictions already calibrated before meta-learner applies its own calibration (double calibration) |
| M-2 | MED | `models/stacking_utils.py` | 280-292, 359-375 | `y_true` values not validated across base models (only indices checked) |
| M-3 | MED | `models/stacking.py` | 222-230 | NaN sample dropping has no minimum positive-count safeguard |
| M-4 | MED | `cli/ensemble_helpers.py` | 348-363 | Saved `train_oof__ENSEMBLE.csv` contains in-sample meta-learner predictions, not genuine ensemble OOF |
| L-1 | LOW | `models/stacking.py` | 240-246 | CalibratedClassifierCV CV splitter determinism relies on implicit sklearn defaults |
| L-2 | LOW | `cli/ensemble_helpers.py` | 154-198 | No warning when exactly 2 base models available (degenerate ensemble) |
| I-1 | INFO | `cli/train_ensemble.py` | 189-204 | Calibration asymmetry between OOF/val/test paths correct but undocumented |
| I-2 | INFO | `models/stacking.py` + `stacking_utils.py` | entire | Core OOF stacking design confirmed correct -- no leakage |

---

### Agent 1.8: Plotting & Output Safety (haiku) -- 2C / 2H / 4M / 3L / 2I

| # | Sev | File | Lines | Finding |
|---|-----|------|-------|---------|
| C-1 | CRIT | `plotting/roc_pr.py`, `dca.py`, `calibration.py`, `risk_dist.py` | various | Silent plot failures: missing deps or empty data cause silent returns with no WARNING log |
| C-2 | CRIT | `plotting/optuna_plots.py` | various | Plotly figures never explicitly closed; memory accumulation risk on runs with many studies |
| H-1 | HIGH | `evaluation/writers/*.py` | all | All writers silently overwrite existing files with no `--force` guard |
| H-2 | HIGH | `evaluation/writers/artifacts_writer.py` | various | `joblib.dump` without atomic write pattern; interrupted writes leave corrupt artifacts |
| M-1 | MED | `plotting/roc_pr.py`, `dca.py`, `risk_dist.py` | various | Parent directory not created before `plt.savefig()` (calibration.py does this correctly) |
| M-2 | MED | `plotting/dca.py` vs `calibration.py` | various | Two different `apply_plot_metadata` implementations with inconsistent text positioning |
| M-3 | MED | All plot functions | various | No validation that prediction inputs are in [0,1] |
| M-4 | MED | `plotting/ensemble.py`, `learning_curve.py` | various | Hardcoded figure sizes do not scale with variable content |
| L-1 | LOW | `plotting/optuna_plots.py` | various | Format fallback (PNG to HTML) not recorded in run metadata |
| L-2 | LOW | `plotting/style.py` | 11 | Fixed DPI=150 for all plots; no adaptation for large datasets |
| L-3 | LOW | `plotting/style.py` | various | Color palette not colorblind-safe |
| I-1 | INFO | All plotting | various | No timing instrumentation on plot generation |
| I-2 | INFO | All plotting | various | No `plt.ioff()` context manager; relies on Agg backend |

---

### Wave 2 Input: Priority Issues for Refactoring

The following CRITICAL and HIGH findings feed into Wave 2 agents:

**For 2.1 (CLI decomposition)**: 1.4-C1, 1.4-C2, 1.4-H1 through H6, 1.4-M1 through M7
**For 2.2 (Nested CV)**: 1.1-H1, 1.1-H2, 1.1-M1 through M3
**For 2.3 (RFE engine)**: 1.2-C1, 1.2-H1, 1.2-M1, 1.2-L3
**For 2.4 (Feature modules)**: 1.2-C2, 1.2-H2, 1.2-H3, 1.2-M3, 1.2-M4
**For 2.5 (CLI modules)**: 1.4-H5, 1.4-M5, 1.4-M7
**For 2.6 (Security)**: 1.3-H1, 1.3-M1 through M3, 1.6-M1 through M4, 1.5-C1
