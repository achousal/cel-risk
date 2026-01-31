# ADR-004: Hybrid Feature Selection

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Hybrid pipeline: Screening → KBest (tuned) → Stability**

1. **Screening** (Mann-Whitney U or F-stat) → top 1,000 proteins
2. **KBest** (SelectKBest with f_classif) → tune k via inner CV
3. **Stability** → extract proteins selected in ≥75% of CV folds

Order configurable via `hybrid_kbest_first` flag (default: True).

## Rationale

- Screening reduces search space (2,920 → 1,000)
- KBest provides tunable k optimization
- Stability ensures robust panels across folds
- Balances speed, tunability, robustness

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| KBest only | Less stable, overfitting risk |
| Stability only | Slower (requires full CV), no k tuning |
| Screening only | No multivariate optimization |
| L1 (Lasso) | Model-specific, not model-agnostic |

## Consequences

| Positive | Negative |
|----------|----------|
| Fast (2,920 → 1,000 → k → stable) | More complex than single method |
| Tunable k via CV | Stability tracking overhead |
| Robust panels (≥75% threshold) | |

## Evidence

**Code:** [schema.py:83-105](../../src/ced_ml/config/schema.py#L83-L105) - `FeatureConfig`
[screening.py](../../src/ced_ml/features/screening.py), [kbest.py](../../src/ced_ml/features/kbest.py), [stability.py:124-216](../../src/ced_ml/features/stability.py#L124-L216)
**Tests:** `test_features_screening.py`, `test_features_kbest.py`, `test_features_stability.py`
**Refs:** Meinshausen & Bühlmann (2010). Stability selection. *JRSS-B*, 72(4), 417-473

## Related

- Part of: ADR-013 (Strategy 1: Hybrid Stability)
- Supports: ADR-005 (stability panel extraction)
- Depends: ADR-006 (nested CV provides folds)
