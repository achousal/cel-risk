# P1 Fixes: Threshold-on-Test Validation (V-06) and Probability Scale Preference (V-07)

**Date**: 2026-02-09
**Status**: Implemented and Tested

## Overview

This document describes two P1 (high-priority) fixes that improve evaluation correctness and data flow transparency in the CeD-ML pipeline.

## Fix 1: Threshold-on-Test Explicit Override (V-06)

### Problem
When validation set is unavailable (`val_size=0`), the pipeline silently falls back to computing thresholds on the test set, which can cause leakage and inflate performance metrics. The original implementation only logged a warning.

### Solution
Added an explicit config flag `allow_test_thresholding` (default: `False`) that must be set to `True` to allow threshold computation on test set. When validation set is unavailable and this flag is `False`, the pipeline now raises a hard error instead of proceeding with a silent warning.

### Changes

#### 1. Config Schema (`analysis/src/ced_ml/config/training_schema.py`)
```python
# Added new field to TrainingConfig
allow_test_thresholding: bool = False  # Explicit override for threshold-on-test
```

#### 2. Evaluation Stage (`analysis/src/ced_ml/cli/orchestration/evaluation_stage.py:55-57`)
```python
if not ctx.has_validation_set:
    if not config.allow_test_thresholding:
        raise ValueError(
            "No validation set available (val_size=0) and threshold-on-test not explicitly allowed. "
            "Set allow_test_thresholding=True in config to proceed (not recommended for production)."
        )
    logger.warning("No validation set available (val_size=0). Skipping validation evaluation.")
    logger.warning("Threshold will be computed on test set (allow_test_thresholding=True).")
    val_metrics = None
    val_threshold = None
    val_target_prev = train_prev
```

### Behavior
- **Default (allow_test_thresholding=False)**: Raises `ValueError` when no validation set available
- **Explicit override (allow_test_thresholding=True)**: Allows threshold computation on test set with warning
- **Normal flow (validation set exists)**: No change in behavior

### Testing
Added comprehensive tests in `analysis/tests/cli/test_threshold_validation.py`:
- `test_threshold_on_test_requires_explicit_flag`: Verifies error is raised when flag is False
- `test_threshold_on_test_allowed_with_flag`: Verifies flow works when flag is True
- `test_normal_validation_threshold_flow`: Verifies normal validation threshold flow unchanged

All tests pass.

---

## Fix 2: Probability Scale Preference (V-07)

### Problem
The `compute_pooled_metrics` function in aggregation did not have a clear preference order for probability columns. When both `y_prob` (raw) and `y_prob_adjusted` (prevalence-adjusted) were present, the function would use whichever appeared first in the DataFrame columns, leading to inconsistent behavior.

### Solution
Implemented explicit priority order: `y_prob_adjusted` > `y_prob` > `risk_score`. The function now:
1. Checks columns in priority order
2. Selects the first available column
3. Logs which scale is used when both raw and adjusted probabilities are present

### Changes

#### Aggregation Module (`analysis/src/ced_ml/cli/aggregation/aggregation.py:148-160`)
```python
# Prefer adjusted probabilities (prevalence-adjusted), fall back to raw
# Check in priority order
preferred_cols = ["y_prob_adjusted", "y_prob", "risk_score"]
actual_pred_col = None
for col in preferred_cols:
    if col in pooled_df.columns:
        actual_pred_col = col
        break

if actual_pred_col is None:
    if logger:
        logger.warning(f"No standard prediction columns found in {pooled_df.columns}")
    return {}

# Log which scale is being used when both are present
if "y_prob_adjusted" in pooled_df.columns and "y_prob" in pooled_df.columns:
    if logger:
        logger.info(f"Using {actual_pred_col} for pooled metrics (both scales present)")
```

### Behavior
- Prefers prevalence-adjusted probabilities (`y_prob_adjusted`) when available
- Falls back to raw probabilities (`y_prob`) if adjusted not present
- Falls back to `risk_score` if neither probability column present
- Logs the choice when both scales are present for transparency

### Testing
Added comprehensive tests in `analysis/tests/cli/test_probability_scale.py`:
- `test_prefers_adjusted_probabilities`: Verifies adjusted probabilities are preferred
- `test_falls_back_to_raw_probabilities`: Verifies fallback to raw when adjusted missing
- `test_falls_back_to_risk_score`: Verifies fallback to risk_score
- `test_warns_when_no_standard_columns`: Verifies warning when no standard columns
- `test_logging_when_both_scales_present`: Verifies logging behavior

All tests pass.

---

## Impact

### Benefits
1. **V-06**: Prevents silent leakage from threshold-on-test
2. **V-06**: Makes threshold computation data flow explicit and auditable
3. **V-07**: Ensures consistent probability scale usage across pipeline stages
4. **V-07**: Improves debugging and auditability via explicit logging

### Backward Compatibility
- **V-06**: Breaking change for workflows that rely on `val_size=0` (requires explicit override)
- **V-07**: Non-breaking (only changes internal priority order, not API)

### Migration Guide

#### For V-06
If you have workflows with `val_size=0` or `cv.folds=1` (no validation set):

1. **Production workflows**: Add validation set (recommended)
   ```yaml
   cv:
     folds: 5  # Ensures validation set exists
   ```

2. **Exploratory workflows only**: Explicitly allow threshold-on-test
   ```yaml
   allow_test_thresholding: true  # NOT recommended for production
   ```

#### For V-07
No migration needed. The change improves consistency automatically.

---

## Test Results

All new and existing tests pass:
- 8 new tests added (3 for V-06, 5 for V-07)
- 7 existing aggregation tests pass
- No regressions detected

```bash
# Run new tests
python -m pytest analysis/tests/cli/test_threshold_validation.py -v
python -m pytest analysis/tests/cli/test_probability_scale.py -v

# Run existing tests
python -m pytest analysis/tests/cli/test_aggregation_discovery.py -v
```

---

## Files Modified

### Source Code
1. `analysis/src/ced_ml/config/training_schema.py` (V-06)
2. `analysis/src/ced_ml/cli/orchestration/evaluation_stage.py` (V-06)
3. `analysis/src/ced_ml/cli/aggregation/aggregation.py` (V-07)

### Tests (New)
1. `analysis/tests/cli/test_threshold_validation.py` (V-06)
2. `analysis/tests/cli/test_probability_scale.py` (V-07)

### Documentation
1. `analysis/docs/development/P1_FIXES_V06_V07.md` (this file)

---

## References

- Original audit: `analysis/docs/development/MATHEMATICAL_VALIDITY_AUDIT.md`
- Related ADRs:
  - ADR-009: Threshold optimization on validation set
  - ADR-008: OOF-posthoc calibration strategy
