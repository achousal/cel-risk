# ADR-010: Fixed Specificity

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Support fixed specificity objective (default 95%) for threshold selection.**

Threshold chosen to achieve target specificity (minimize false positives).

**Note:** Youden and max_f1 also supported for comparison.

## Rationale

- Clinical screening prioritizes high specificity
- False positives → unnecessary tests, anxiety, costs
- Configurable target (90%, 95%, 99%)

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Youden only | May yield lower specificity (e.g., 85%) |
| Max F1 only | Emphasizes precision/recall, not specificity |

## Consequences

| Positive | Negative |
|----------|----------|
| Minimizes false positives | Lower sensitivity vs Youden/max_f1 |
| Configurable target | Target may not be achievable |
| Clinical interpretation clear | |

## Evidence

**Code:** [schema.py:198-208](../../src/ced_ml/config/schema.py#L198-L208) - `ThresholdConfig.fixed_spec`
[thresholds.py:326-377](../../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective`
**Tests:** `test_choose_threshold_fixed_spec`, `test_threshold_config_fixed_spec`

## Related

- Depends: ADR-011 (threshold on VAL)
