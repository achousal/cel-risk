# ADR-001: Split Strategy (50/25/25)

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**3-way stratified split: 50% TRAIN / 25% VAL / 25% TEST**

- **TRAIN (50%)**: Nested CV hyperparameter tuning + feature selection
- **VAL (25%)**: Threshold selection (avoid TEST leakage)
- **TEST (25%)**: Final unbiased evaluation

Stratified by `incident_CeD` to preserve class balance.

## Rationale

- 2-way split forces threshold selection on TEST → optimistic bias
- Need held-out VAL for calibration verification
- Limited sample size (148 incident cases) precludes 4-way split

## Consequences

| Positive | Negative |
|----------|----------|
| No TEST leakage | Smaller TRAIN (50% vs 75%) |
| Stable threshold estimates | Reduced tuning power |
| VAL enables calibration checks | |

## Evidence

**Code:** [splits.py:374-438](../../src/ced_ml/data/splits.py#L374-L438) - `stratified_train_val_test_split`
**Config:** [schema.py](../../src/ced_ml/config/schema.py) - `SplitsConfig.validate_split_sizes`
**Tests:** `test_data_splits.py` - stratified split validation
**Refs:** Steyerberg (2019). *Clinical Prediction Models*

## Related

- Supports: ADR-011 (threshold on VAL)
- Supports: ADR-002 (prevalent→TRAIN)
