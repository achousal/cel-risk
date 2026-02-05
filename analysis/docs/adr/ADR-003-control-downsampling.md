# ADR-003: Control Downsampling (5:1)

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Downsample controls to 5:1 case:control ratio** via random sampling, stratified by split.

- Original: 148 incident cases, 43,662 controls (~1:300)
- Downsampled: ~740 controls (148 × 5) → 16.7% prevalence

## Rationale

- Reduces computational cost 60× (300 → 5 controls per case)
- Preserves adequate negative signal for discrimination
- Enables feasible hyperparameter tuning (50k fits per model)

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| No downsampling (1:300) | 300× longer training, minimal gain |
| 1:10 ratio | 2× slower than 1:5, similar discrimination |
| 1:2 ratio | Insufficient negative signal |
| SMOTE oversampling | Synthetic proteomics data may not generalize |

## Consequences

| Positive | Negative |
|----------|----------|
| 60× computational savings | Distribution shift (1:300 → 1:5) |
| Faster tuning (50k fits feasible) | Requires prevalence adjustment |
| Preserves negative signal | Some control variability loss |

## Evidence

**Code:** [splits.py:193-250](../../src/ced_ml/data/splits.py#L193-L250) - `downsample_controls`
**Config:** [schema.py](../../src/ced_ml/config/schema.py) - `SplitsConfig.train_control_per_case`
**Tests:** `test_downsample_controls`, `test_case_control_ratio`

## Related

- Depends: ADR-001 (split strategy)
