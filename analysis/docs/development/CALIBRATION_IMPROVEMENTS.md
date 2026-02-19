# Calibration Improvements

**Date**: 2026-02-18
**Status**: Implemented
**Audit Reference**: `docs/development/MATHEMATICAL_VALIDITY_AUDIT.md` (FLAGS: M4, stacking double-calibration)

---

## Summary

This changeset addresses the calibration-related findings from the mathematical validity audit
and extends the calibration subsystem with new methods, metrics, and visualizations.

**Workstreams completed:**

| ID | Description | Key Files |
|----|-------------|-----------|
| P0 | Fix ensemble double-calibration | `models/stacking.py`, `cli/train_ensemble.py`, `config/ensemble_schema.py` |
| P1 | Add new calibration methods | `models/calibration.py`, `config/calibration_schema.py` |
| P3 | Add calibration assessment metrics | `models/calibration.py` |
| P3b | Integrate metrics into pipeline | `evaluation/holdout.py`, `cli/train.py`, `cli/orchestration/persistence_stage.py` |
| P4 | Enhanced calibration plots | `plotting/calibration.py` |

---

## P0: Fix Ensemble Double-Calibration

**Problem**: When `calibrate_meta=True` (the previous default), the stacking meta-learner was
wrapped in `CalibratedClassifierCV` on top of base models that were already OOF-calibrated.
This double calibration can distort probability estimates (audit FLAG M4/stacking.py:240).

**Fix**:
- Changed `calibrate_meta` default from `True` to `False` in `StackingEnsemble.__init__`
  and `train_meta_learner`.
- Added `meta_calibration_method` parameter (configurable: `"isotonic"` or `"sigmoid"`).
- Created `EnsembleConfig` Pydantic schema in `config/ensemble_schema.py` with fields:
  - `calibrate_meta: bool = False`
  - `meta_calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"`
  - `calibration_cv: int = 5`
- Wired `EnsembleConfig` into `TrainingConfig` and `schema.py` re-exports.
- Added a `logger.warning()` when `calibrate_meta=True` to flag potential double calibration.
- Updated `training_config.yaml` with commented `ensemble:` section for discoverability.

**Rationale**: Base models are already OOF-calibrated (via `OOFCalibrator`), and the logistic
regression meta-learner is inherently calibrated through its logistic link function. The extra
`CalibratedClassifierCV` layer is unnecessary and potentially harmful.

---

## P1: New Calibration Methods

**Added methods** to `OOFCalibrator` (in addition to existing `isotonic`):

| Method | Parameters | Description |
|--------|------------|-------------|
| `logistic_full` | a, b | Two-parameter Platt scaling: `logit(q) = a + b * logit(p)` |
| `logistic_intercept` | a | Intercept-only: `logit(q) = a + logit(p)`. Lowest variance. |
| `beta` | a, b, c | Beta calibration (Kull et al. 2017): `logit(q) = a*log(p) + b*log(1-p) + c` |
| `sigmoid` | -- | Alias for `logistic_full` (backward compatible) |

**Schema changes** (`config/calibration_schema.py`):
- `CalibrationMethodLiteral` type alias: `Literal["sigmoid", "isotonic", "logistic_full", "logistic_intercept", "beta"]`
- `CalibrationStrategyLiteral` type alias: `Literal["per_fold", "oof_posthoc", "none"]`
- `PerModelCalibrationConfig`: optional per-model `strategy` and `method` overrides.
- `CalibrationConfig.get_method_for_model(model_name)`: resolves per-model method or global default.
- Backward-compatible `_coerce_per_model` validator: accepts legacy string format (e.g., `{"LR_EN": "oof_posthoc"}`).

**Safety guardrails**:
- Isotonic regression minimum samples raised from 10 to 50.
- Warning emitted when `n_positive < 30` for isotonic method.

---

## P3: Calibration Assessment Metrics

New metrics in `models/calibration.py`:

| Metric | Function | Description |
|--------|----------|-------------|
| ICI | `integrated_calibration_index(y, p)` | Integrated Calibration Index via LOWESS smoothing |
| E50, E90 | `calibration_error_quantiles(y, p)` | Median and 90th percentile of absolute calibration errors |
| Spiegelhalter z-test | `spiegelhalter_z_test(y, p)` | Tests H0: model is well-calibrated. Returns z-stat and p-value |
| Adaptive ECE | `adaptive_expected_calibration_error(y, p)` | ECE with data-adaptive bin widths (equal-frequency) |
| Brier decomposition | `brier_score_decomposition(y, p)` | Murphy (1973) decomposition: reliability + resolution + uncertainty |

**Dataclasses**: `CalibrationQuantiles`, `SpiegelhalterResult`, `BrierDecomposition`.

**Bootstrap-compatible scalar wrappers** (for `stratified_bootstrap_ci`):
`ici_metric`, `adaptive_ece_metric`, `brier_reliability_metric`, `brier_resolution_metric`,
`spiegelhalter_z_metric`, `calibration_slope_metric` (pre-existing, now used in bootstrap).

---

## P3b: Pipeline Integration

New metrics are computed and persisted in:

- **`cli/train.py`** (~line 409): ICI, Spiegelhalter z/p, adaptive ECE, Brier decomposition
  components added to per-split metrics dict.
- **`evaluation/holdout.py`** (~line 185): Same metrics added to holdout evaluation output.
- **`cli/orchestration/persistence_stage.py`**: Bootstrap CIs added for `calibration_slope`,
  `ICI`, and `adaptive_ece` alongside existing AUROC/PR-AUC CIs.

**New columns in metrics output**:
`ICI`, `ECE_adaptive`, `spiegelhalter_z`, `spiegelhalter_p`,
`brier_reliability`, `brier_resolution`, `brier_uncertainty`.

---

## P4: Enhanced Calibration Plots

Two new overlays added to probability-calibration panels in `plotting/calibration.py`:

1. **LOWESS overlay** (`_add_lowess_overlay`):
   - Cubic `UnivariateSpline` fit (smoothed calibration curve).
   - Dashed `COLOR_SECONDARY` line.
   - Skips gracefully when `n < 50` or `< 10` unique predictions.

2. **Metrics annotation** (`_add_calibration_metrics_annotation`):
   - Text box in upper-left showing: ICI, E50, E90, Spiegelhalter p-value.
   - Auto-formats p-value with scientific notation when small.

Both are called from `_plot_prob_calibration_panel` for probability-space panels only
(not logit-space panels).

---

## Configuration

### training_config.yaml

The `calibration:` section now documents all five methods:
```yaml
calibration:
  method: isotonic  # isotonic, sigmoid, logistic_full, logistic_intercept, beta
  per_model:        # Optional per-model overrides (strategy + method)
    LR_EN: oof_posthoc                    # Legacy string format (strategy only)
    RF:
      strategy: per_fold
      method: logistic_intercept          # Dict format (strategy + method)
```

The `ensemble:` section is now documented (commented out with defaults):
```yaml
ensemble:
  calibrate_meta: false
  meta_calibration_method: isotonic  # isotonic or sigmoid
  calibration_cv: 5
```

---

## Testing

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_models_calibration.py` | 124 | OOFCalibrator + new methods + config schema |
| `test_advanced_calibration_metrics.py` | 42 | ICI, E50/E90, Spiegelhalter, adaptive ECE, Brier decomposition |
| `test_stacking_core.py` | 48 | Stacking ensemble including new calibrate_meta defaults |

All 214 calibration-related tests pass. Full test suite (excluding e2e): pre-existing
failures only (see audit doc section 5A for known pre-existing issues).

---

## Files Modified

| File | Change Type |
|------|-------------|
| `models/stacking.py` | Modified (calibrate_meta default, meta_calibration_method) |
| `cli/train_ensemble.py` | Modified (reads config.ensemble, passes to train_meta_learner) |
| `config/ensemble_schema.py` | **New** (EnsembleConfig Pydantic model) |
| `config/training_schema.py` | Modified (added ensemble field) |
| `config/schema.py` | Modified (re-exports EnsembleConfig) |
| `models/calibration.py` | Modified (new methods, metrics, dataclasses) |
| `config/calibration_schema.py` | Modified (new literals, PerModelCalibrationConfig) |
| `evaluation/holdout.py` | Modified (new metric computations) |
| `cli/train.py` | Modified (new metric computations) |
| `cli/orchestration/persistence_stage.py` | Modified (bootstrap CIs for new metrics) |
| `plotting/calibration.py` | Modified (LOWESS overlay, metrics annotation) |
| `tests/models/test_models_calibration.py` | Modified (new method + schema tests) |
| `tests/models/test_stacking_core.py` | Modified (new ensemble config tests) |
| `tests/models/test_advanced_calibration_metrics.py` | **New** (42 metric tests) |
| `configs/training_config.yaml` | Modified (updated comments, ensemble section) |

---

## References

- Kull, M., Silva Filho, T., & Flach, P. (2017). Beta calibration: a well-founded and
  easily implemented improvement on logistic calibration for binary classifiers. AISTATS.
- Murphy, A. H. (1973). A new vector partition of the probability score. Journal of
  Applied Meteorology and Climatology, 12(4), 595-600.
- Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and
  clinical trials. Statistics in Medicine, 5(5), 421-433.
- Austin, P. C., & Steyerberg, E. W. (2019). The Integrated Calibration Index (ICI)
  and related metrics for quantifying the calibration of logistic regression models.
  Statistics in Medicine, 38(21), 4051-4065.
