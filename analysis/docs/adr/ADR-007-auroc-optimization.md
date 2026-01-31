# ADR-007: AUROC Optimization

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Use AUROC (roc_auc) as primary optimization metric in RandomizedSearchCV.**

Optimize discrimination during training, calibrate post-hoc for deployment.

## Rationale

- Clinical workflow: rank patients → threshold → test (prioritizes discrimination)
- AUROC invariant to prevalence (robust across deployment scenarios)
- Calibration context-dependent (prevalence, cost/benefit vary by setting)
- Post-hoc calibration effective without sacrificing discrimination
- Separation of concerns: optimize discrimination, calibrate for context

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| Brier score | Confounds discrimination + calibration; may sacrifice ranking |
| Average precision (PR-AUC) | Prevalence-sensitive, less stable |
| Multi-objective (AUROC+calibration) | Complex weight tuning; clearer to separate |

## Consequences

| Positive | Negative |
|----------|----------|
| Maximum discrimination | Requires post-hoc calibration |
| Prevalence-invariant | Two-stage process |
| Flexible deployment calibration | Prevalence adjustment needed |
| Clinical alignment (rank-then-test) | |

## Evidence

**Code:** [schema.py](../../src/ced_ml/config/schema.py) - `CVConfig.scoring` default
[training.py](../../src/ced_ml/models/training.py) - `RandomizedSearchCV(scoring=...)`
[calibration.py](../../src/ced_ml/models/calibration.py) - post-hoc methods
**Tests:** `test_nested_cv_auroc_optimization`, `test_cv_config_defaults`, `test_models_calibration.py`
**Refs:** Steyerberg (2019) Ch 10. Van Calster (2016). Huang (2020)

## Related

- Supports: ADR-010 (prevalence adjustment)
- Supports: ADR-006 (nested CV framework)
- Supports: ADR-011 (threshold on VAL)
