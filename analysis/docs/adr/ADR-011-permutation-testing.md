# ADR-011: Permutation Testing for Model Significance (Stage 1 Model Gate)

**Status:** Accepted | **Date:** 2026-02-03 | **Updated:** 2026-02-04

## Decision

**Use label-permutation testing as Stage 1 model gate to filter models with real signal before consensus aggregation (ADR-004).**

For each permutation, re-run the FULL inner pipeline (screening, feature selection, hyperparameter optimization) on permuted training labels to obtain a valid null distribution.

**Workflow integration:**
- **Before consensus:** Only models with `p < alpha` (default 0.05) proceed to Stage 2-3 (per-model evidence + RRA consensus)
- **Prevents noise aggregation:** Ensures consensus panel aggregates only models with statistically significant signal

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

## Implementation

**HPC Parallelization:**
- Orchestrator submits one full-command job per (model, seed) pair
- Each job runs all permutations internally with `--n-jobs -1` (all cores)
- Produces `null_distribution_seed{N}.csv` per seed
- `--aggregate-only` pools per-seed CSVs into a single significance result

```bash
# Local: run all permutations with internal parallelism
ced permutation-test --run-id <RUN_ID> --model LR_EN --n-jobs 4

# Aggregate existing per-seed results
ced permutation-test --run-id <RUN_ID> --model LR_EN --aggregate-only
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

- Part of: ADR-004 (Stage 1 model gate in three-stage workflow)
- Supports: ADR-005 (nested CV - respects outer fold structure)
- Enables: Stage 2-3 of ADR-004 (filters significant models before per-model evidence + consensus)
