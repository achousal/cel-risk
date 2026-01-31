# ADR-005: Stability Panel (0.75 Threshold)

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Stability threshold = 0.75** (protein selected in ≥37.5 of 50 CV folds)

**Fallback:** If no proteins meet threshold, keep top 20 by frequency.

## Rationale

- 50 CV folds (5 outer × 10 repeats) generate 50 feature sets
- 0.75 balances robustness vs panel size
- Frequently selected features generalize better
- Fallback prevents empty panels

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| 0.5 threshold | Includes unstable features |
| 0.9 threshold | Too strict, often 0-5 proteins |
| Fixed panel size (top K) | Ignores stability |
| No fallback | Risk of 0-feature panels |

## Consequences

| Positive | Negative |
|----------|----------|
| Balances robustness + size | Threshold somewhat arbitrary |
| Fallback ensures non-empty panels | Adds complexity |
| Better generalization | |

## Evidence

**Code:** [stability.py:124-216](../../src/ced_ml/features/stability.py#L124-L216) - `extract_stable_panel`
**Config:** [schema.py](../../src/ced_ml/config/schema.py) - `FeatureConfig.stability_thresh`
**Tests:** `test_extract_stable_panel_threshold`, `test_extract_stable_panel_fallback`
**Refs:** Meinshausen & Bühlmann (2010). Stability selection. *JRSS-B*, 72(4), 417-473

## Related

- Part of: ADR-013 (used by Strategies 1, 2, 3)
- Depends: ADR-004 (hybrid feature selection)
- Depends: ADR-006 (nested CV provides folds)
