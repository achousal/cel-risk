# ADR-016: Permutation Testing for Model Significance

**Status:** Accepted | **Date:** 2026-02-03

## Decision

**Use label-permutation testing to assess model generalization above chance level.**

For each permutation, re-run the FULL inner pipeline (screening, feature selection, hyperparameter optimization) on permuted training labels to obtain a valid null distribution.

## Rationale

- Tests the null hypothesis: H0: model performance = chance performance
- Accounts for overfitting by re-running full pipeline (avoids data leakage in null)
- Provides p-value for statistical significance of AUROC
- Proteomic data with 2,920 features needs formal overfitting assessment

## Algorithm

```
For each permutation b in 0..B-1:
    1. Shuffle y_train only (keep X fixed, held-out y unchanged)
    2. Run FULL inner pipeline on permuted labels:
        - Screening (Mann-Whitney/t-test)
        - Inner CV k-best tuning
        - Hyperparameter optimization
        - Fit final model
    3. Predict on held-out fold (original X, original y)
    4. Record AUROC

Compute p-value: p = (1 + #{null >= observed}) / (1 + B)
```

The +1 correction ensures p-values are never exactly zero (per Phipson & Smyth 2010).

## Alternatives

| Alternative | Rejected Because |
|-------------|------------------|
| sklearn `permutation_test_score` | Doesn't re-run feature selection; invalid null distribution |
| Bootstrap confidence intervals | Tests different hypothesis (CI width vs. chance level) |
| Cross-validation variance | Measures split stability, not chance vs. real signal |
| Fixed model permutation | Data leakage: features selected on real labels inform null |

## Consequences

| Positive | Negative |
|----------|----------|
| Valid null distribution | Computationally expensive (B x pipeline time) |
| Robust p-values | HPC parallelization needed for B=200+ |
| Detects overfitting | Requires trained model artifacts |
| Formal statistical evidence | Only tests AUROC (per ADR-007) |

## Implementation

**HPC Parallelization:**
- `--hpc` flag submits LSF/Slurm job array (consistent with other CLI commands)
- Each job runs single permutation via `--perm-index`, saves to `perm_{i}.joblib`
- Aggregation collects results post-hoc

```bash
# Submit job array to HPC (recommended)
ced permutation-test --run-id <RUN_ID> --model LR_EN --hpc

# Preview without submitting
ced permutation-test --run-id <RUN_ID> --model LR_EN --hpc --dry-run

# After completion, aggregate results
ced permutation-test --run-id <RUN_ID> --model LR_EN
```

**Recommended B values:**
- CI/quick check: B = 10-50
- Publication: B >= 200
- Final validation: B >= 1000

## Evidence

**Code:**
- [permutation_test.py](../../src/ced_ml/significance/permutation_test.py) - Core algorithm
- [permutation_test.py (CLI)](../../src/ced_ml/cli/permutation_test.py) - CLI implementation

**Tests:**
- `tests/significance/test_permutation_test.py` - 30 unit tests
- `tests/e2e/test_pipeline_significance.py` - E2E workflow tests

**References:**
- Ojala & Garriga (2010). Permutation tests for studying classifier performance. JMLR 11:1833-1863.
- Phipson & Smyth (2010). Permutation P-values should never be zero. Stat Appl Genet Mol Biol 9(1).
- Golland & Fischl (2003). Permutation tests for classification. MICCAI.

## Related

- Supports: ADR-007 (AUROC as primary metric - only AUROC tested)
- Supports: ADR-006 (nested CV - respects outer fold structure)
- Supports: ADR-004 (hybrid feature selection - re-run in permutation loop)
