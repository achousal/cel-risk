# ADR-011: Threshold Selection on VAL

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Select decision threshold on VAL set, never TEST.**

TEST remains held-out for unbiased final evaluation.

## Rationale

- TRAIN: Biased (model trained on it)
- VAL: Unbiased for threshold selection
- TEST: Must remain completely held-out
- Threshold on TEST → optimistic bias (leakage)

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Threshold on TEST | Optimistic bias (tuned to TEST) |
| Threshold on TRAIN (OOF) | TRAIN overfitted, may not generalize |
| Fixed threshold (0.5) | Arbitrary, ignores imbalance/costs |
| Nested threshold (inner CV) | Couples threshold to hyperparameters |

## Consequences

| Positive | Negative |
|----------|----------|
| Unbiased threshold selection | Requires 3-way split |
| TEST held-out (no leakage) | VAL must be large enough (25%) |
| Post-hoc threshold adjustment | |

## Evidence

**Code:** [schema.py:198-208](../../src/ced_ml/config/schema.py#L198-L208) - `ThresholdConfig.threshold_source`
[train.py](../../src/ced_ml/cli/train.py) - threshold selection
[thresholds.py:326-377](../../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective`
**Tests:** `test_choose_threshold_on_val`, `test_threshold_source_validation`
**Refs:** Steyerberg (2019) Ch 11

## Related

- Depends: ADR-001 (provides VAL set)
- Supports: ADR-012 (threshold objective: fixed spec 95%)
