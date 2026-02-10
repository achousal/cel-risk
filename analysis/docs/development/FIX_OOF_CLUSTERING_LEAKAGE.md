# Fix: OOF Importance Clustering Data Leakage

**Date**: 2026-02-09
**Status**: Fixed
**Severity**: Medium-High (data hygiene violation)

## Problem

During OOF importance computation in nested CV, feature clustering (for grouped importance) was performed on **validation fold data** instead of **training fold data**. This introduced information leakage because:

1. Clustering determines which features are grouped together
2. Grouped importance affects feature ranking in consensus panel selection
3. Using held-out fold data for clustering decisions violates proper data hygiene

### Code Location

- **Function**: `extract_importance_from_model()` ([importance.py:490](../src/ced_ml/features/importance.py#L490))
- **Call site**: `oof_predictions_with_nested_cv()` ([nested_cv.py:307](../src/ced_ml/models/nested_cv.py#L307))

### Original Warning

```
WARNING - Grouped importance is clustering features using validation data.
This may introduce information from the validation set into feature grouping.
For stricter data hygiene, consider clustering on training data instead.
```

## Root Cause

The code conflated two different uses of validation data:

1. **Permutation importance (legitimate)**: Requires `X_val/y_val` to evaluate performance drop (correct)
2. **Feature clustering (problematic)**: Used `X_val` for correlation-based grouping (leakage)

## Solution

### Changes Made

1. **Added `X_train` parameter** to `extract_importance_from_model()`:
   ```python
   def extract_importance_from_model(
       ...,
       X_val: pd.DataFrame | None = None,
       y_val: np.ndarray | None = None,
       X_train: pd.DataFrame | None = None,  # NEW
       ...
   )
   ```

2. **Updated clustering logic** to prefer training data:
   ```python
   # Cluster on training data to avoid leakage
   X_for_clustering = X_train if X_train is not None else X_val

   if X_train is None:
       logger.warning(
           "X_train not provided for clustering, falling back to X_val. "
           "This may introduce information leakage..."
       )
   ```

3. **Updated call site** in `nested_cv.py`:
   ```python
   # Prepare training data for clustering (avoid leakage)
   X_train_fold = X.iloc[train_idx][protein_cols] if oof_grouped else None

   fold_importance = extract_importance_from_model(
       ...,
       X_train=X_train_fold,  # NEW
       ...
   )
   ```

### Files Modified

- `analysis/src/ced_ml/features/importance.py` (lines 490-603)
- `analysis/src/ced_ml/models/nested_cv.py` (lines 303-315)

## Impact Assessment

### Practical Impact

**Likely small** because:
- Feature correlations are usually stable between train/val folds
- Clustering is primarily for importance aggregation, not direct feature selection
- Most proteins have consistent correlation structures across folds

### Methodological Impact

**High** in terms of correctness principles:
- Violates strict data hygiene (held-out data influencing feature decisions)
- Affects consensus panel ranking (which features make it into final panel)
- Contradicts project emphasis on "correctness and scientific validity"

## Validation

### Tests

All existing tests pass (42 tests in `test_importance.py`, 32 tests in `test_training.py`):

```bash
python -m pytest analysis/tests/features/test_importance.py -v
# 42/42 passed

python -m pytest analysis/tests/models/test_training.py -v
# 32/32 passed
```

### Backward Compatibility

- **API change**: New optional parameter `X_train` (defaults to `None`)
- **Behavior**: Falls back to old behavior if `X_train` not provided (with warning)
- **Migration**: Existing code continues to work, but gets warning to update

## Recommendation

**Action**: Accept fix and re-run production pipeline

**Rationale**:
1. Corrects data hygiene violation
2. Aligns with audit emphasis on correctness
3. Minimal practical impact expected
4. Tests confirm no regressions

**Next Steps**:
- Monitor clustering stability across train/val in next run
- Compare consensus panel before/after fix (should be similar but not identical)
- Update any external callers to pass `X_train` parameter

## Related

- **Audit**: Mathematical Validity Audit (2026-02-08)
- **Memory Note**: FLAG items related to methodological choices (calibration, zero-fill, etc.)
- **ADR**: ADR-004 (Three-stage feature selection workflow)
