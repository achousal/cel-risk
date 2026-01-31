# ADR-015: Model Selection Scoring

**Status:** Accepted | **Date:** 2026-01-22

## Decision

**Composite score for model ranking:**

```
score = (AUROC × 0.5) + ((1 - Brier) × 0.3) + ((1 - |slope - 1|) × 0.2)
```

- **AUROC (50%)**: Discrimination ability
- **Brier (30%)**: Prediction quality (inverted)
- **Calibration slope (20%)**: Calibration quality

**Range:** [0, 1] where higher is better. Perfect = 1.0, Random ~ 0.475.

## Rationale

- Single metric (AUROC only) incomplete for clinical decision support
- Need holistic comparison: discrimination + calibration
- Configurable weights for different use cases
- Consistent ranking methodology

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| AUROC only | Ignores calibration |
| Brier only | Doesn't distinguish discrimination vs calibration |
| Pareto ranking | No single ranking, harder to interpret |
| Net benefit | Requires threshold a priori, too narrow |
| Log loss | Correlated with Brier, no unique info |

## Consequences

| Positive | Negative |
|----------|----------|
| Holistic comparison | Added complexity |
| Configurable weights | Default weights (50/30/20) arbitrary |
| Single interpretable score | May not align with all clinical frameworks |
| Robust to metric key variations | Composite scores harder to explain |
| Handles missing metrics (NaN) | |

## Evidence

**Code:** [scoring.py:34-130](../../src/ced_ml/evaluation/scoring.py#L34-L130) - `compute_selection_score`
[scoring.py:180-220](../../src/ced_ml/evaluation/scoring.py#L180-L220) - `rank_models_by_selection_score`
**Tests:** `test_evaluation_scoring.py` - 22 tests (perfect model, random model, custom weights, ranking)
**Refs:** Collins (2015) TRIPOD. Van Calster (2019). Steyerberg (2019) Ch 15

## Related

- Complements: ADR-007 (AUROC optimization)
- Related: ADR-014 (calibration affects slope metric)
- Related: ADR-009 (ensemble selection via this score)
