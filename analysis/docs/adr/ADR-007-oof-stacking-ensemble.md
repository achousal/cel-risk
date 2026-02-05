# ADR-007: OOF Stacking Ensemble

**Status:** Accepted | **Date:** 2026-01-22

## Decision

**Stacking ensemble using out-of-fold (OOF) predictions**

```
Base Models (trained independently)
    ↓
OOF Predictions (n_samples × n_models)
    ↓
Meta-Learner (LogisticRegression L2)
    ↓
Calibrated Ensemble Probability
```

- **OOF predictions:** Each sample predicted by fold where it was held out
- **Meta-learner:** Logistic Regression (L2) trained on stacked OOF predictions
- **Calibration:** Optional isotonic calibration of meta-learner output
- **No leakage:** Meta-learner never sees in-fold predictions

## Rationale

- OOF approach prevents meta-learner overfitting
- Reuses existing OOF predictions from nested CV
- Interpretable meta-learner weights

## Evidence

**Code:** [stacking.py:61-160](../../src/ced_ml/models/stacking.py#L61-L160) - `StackingEnsemble`
[stacking.py:200-280](../../src/ced_ml/models/stacking.py#L200-L280) - `fit_from_oof`
[train_ensemble.py](../../src/ced_ml/cli/train_ensemble.py) - CLI
**Tests:** `test_models_stacking.py` - 23 tests for OOF handling, fitting, prediction
**Refs:** Wolpert (1992) Stacked Generalization. Breiman (1996) Stacked Regressions. Van der Laan (2007) Super Learner

## Related

- Depends: ADR-005 (nested CV provides OOF predictions)
- Related: ADR-006 (Optuna optimizes base models)
