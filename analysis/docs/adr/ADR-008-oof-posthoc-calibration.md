# ADR-008: OOF Posthoc Calibration

**Status:** Accepted | **Date:** 2026-01-22

## Decision

**Add `oof_posthoc` calibration strategy** as alternative to `per_fold`.

**Strategies:**
1. **`per_fold`** (default): CalibratedClassifierCV inside each CV fold
2. **`oof_posthoc`** (new): Fit single calibrator on aggregated OOF predictions
3. **`none`**: No calibration

| Strategy | Data Efficiency | Leakage Risk | Optimism Bias | Stability |
|----------|-----------------|--------------|---------------|-----------|
| `per_fold` | Full | Subtle | Subtle | Lower |
| `oof_posthoc` | Full | None | None | Higher |
| 4-way split | Reduced | None | None | Medium |

## Rationale

**Problem with `per_fold`:**
- Calibrator fitted on same data used for hyperparameter selection
- Subtle optimistic bias in Brier score
- Calibrator indirectly "sees" validation data

**OOF posthoc advantages:**
- Eliminates leakage (calibrator on pure OOF predictions)
- Higher stability (single calibrator vs per-fold)
- Full data efficiency (no additional holdout)

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| 4-way split | Reduces training data (problematic for small datasets) |
| Nested calibration in inner CV | Excessive complexity |
| Always use oof_posthoc | Per_fold has lower variance for some models |
| Temperature scaling | Less flexible, doesn't address leakage |

## Consequences

| Positive | Negative |
|----------|----------|
| Eliminates optimistic bias | Slightly higher variance (small datasets) |
| Higher stability (single calibrator) | Additional pipeline complexity |
| Full data efficiency | Worse calibration for well-calibrated base models |
| Per-model flexibility | |

## Evidence

**Code:** [calibration.py](../../src/ced_ml/models/calibration.py) - `OOFCalibrator`, `OOFCalibratedModel`
[schema.py:301-337](../../src/ced_ml/config/schema.py#L301-L337) - `CalibrationConfig.strategy`
**Tests:** `test_models_calibration.py` - 24 tests for strategies
**Refs:** Guo (2017) ICML. Van Calster (2019) BMC Medicine. Steyerberg (2019) Ch 15

## Related

- Depends: ADR-005 (provides OOF predictions)
