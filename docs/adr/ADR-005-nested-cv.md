# ADR-005: Nested CV (5×10×5)

**Status:** Accepted | **Date:** 2026-01-20

## Decision

**Nested CV: 5 outer folds × 10 repeats × 5 inner folds**

- **Outer CV:** 5-fold × 10 repeats = 50 total outer folds → OOF predictions
- **Inner CV:** 5-fold RandomizedSearchCV × 200 iterations → hyperparameter tuning

**Total fits:** 5 × 10 × 5 × 200 = **50,000 fits per model**

## Rationale

- Outer CV generates unbiased OOF predictions for evaluation
- Inner CV tunes hyperparameters independently per outer fold
- Prevents optimistic bias in hyperparameter selection
- 50 outer folds provide robust OOF estimates

## Inner Fold Positive Balance

**Constraint:** Each inner CV fold must contain sufficient positive samples for stable tuning.

**Logic:** With 5 inner folds, the minority class needs ~10+ samples per inner validation fold:
- Outer train fold: ~80% of data → ~118 cases (at 0.34% prevalence)
- Inner 5-fold split: each inner val fold → ~24 cases
- StratifiedKFold preserves class proportions across inner folds

**Safeguard:** If inner folds would have <2 positives per fold, reduce inner_folds dynamically (used in calibration CV). For hyperparameter search, 5 folds is safe given expected case counts.

**Code:** [nested_cv.py:636](../src/ced_ml/models/nested_cv.py#L636) - `StratifiedKFold(n_splits=inner_folds, ...)`

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| 5-fold (no repeats) | Less stable OOF predictions |
| 10-fold × 5 repeats | Same 50 folds, worse balance |
| 3-fold inner CV | Too few for stable tuning |
| Grid search | Computationally infeasible |

## Consequences

| Positive | Negative |
|----------|----------|
| 50 folds → robust OOF | 50k fits → 12-hour HPC runtime |
| Thorough hyperparameter search | High memory (128 GB/job) |
| No optimistic bias | |

## Evidence

**Code:** [training.py:29-192](../../src/ced_ml/models/training.py#L29-L192) - `oof_predictions_with_nested_cv`
**Config:** [schema.py](../../src/ced_ml/config/schema.py) - `CVConfig` (folds=5, repeats=10, inner_folds=5)
**Tests:** `test_training.py::test_oof_predictions_with_nested_cv`, `test_nested_cv_structure`
**Refs:** Varma & Simon (2006). Bias in error estimation. *BMC Bioinformatics*, 7(1), 91

## Related

- Supports: ADR-004 (provides CV folds for Stage 2 evidence)
