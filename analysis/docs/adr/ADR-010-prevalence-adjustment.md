# ADR-010: Prevalence Adjustment (Deployment-Only)

**Status:** Accepted (deployment concern, not integrated) | **Date:** 2026-01-20

## Decision

**NO adjustment during training/validation/testing.** All splits at same prevalence (16.7%).

**Plan logit-scale adjustment for deployment** if used at real-world prevalence (0.34%).

```
P_adjusted = sigmoid(logit(p) + logit(π_deployment) - logit(π_training))

Where:
  p = model prediction (at 16.7%)
  π_training = 0.167 (5:1 case:control)
  π_deployment = 0.0034 (real-world ~1:300)
```

## Rationale

- TRAIN/VAL/TEST all at 16.7% → no prevalence mismatch → no adjustment needed
- Threshold selection on VAL unbiased (same prevalence as TEST)
- Calibration valid (OOF calibrator fit and applied at same prevalence)
- Adjustment only needed if deployed at different prevalence

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| No adjustment ever | Predictions 50× too high at deployment (0.34%) |
| Platt scaling on deployment data | Requires labeled deployment cohort upfront |
| Sample weights to match 0.34% | Discards controls → worse performance |
| Threshold-only adjustment | Doesn't fix probability calibration |

## Consequences

| Positive | Negative |
|----------|----------|
| Clean training logic (no no-ops) | Requires future deployment work |
| Unbiased threshold selection | Target prevalence must be known |
| Preserves AUROC | External deployment code needed |
| Mathematically principled (Bayes) | |

## Deployment Checklist

When deploying at 0.34% prevalence:

1. Implement logit-scale adjustment wrapper
2. Load trained model from `results/{MODEL}/split_seed*/core/final_model.pkl`
3. Wrap predictions: `adjust_for_deployment(p, 0.167, 0.0034)`
4. Validate on labeled deployment cohort (AUROC, calibration)
5. Re-optimize thresholds on deployment data
6. Monitor drift and performance

## Evidence

**Refs:** Steyerberg (2019) Ch 13 (logit-shift method)
**Docs:** `DEPLOYMENT.md` (speculative workflow)

## Related

- Depends: ADR-001 (uniform 16.7% prevalence)
- Depends: ADR-003 (5:1 control ratio)
- Supports: ADR-011 (threshold on VAL unbiased)
