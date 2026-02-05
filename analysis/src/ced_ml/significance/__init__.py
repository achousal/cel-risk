"""Statistical significance testing for model performance.

This module provides permutation-based hypothesis testing to determine whether
trained models generalize above chance level. The null hypothesis is that the
model's predictive performance is no better than random chance.

Main Functions
--------------
run_permutation_test
    Test a single fold against a null distribution from label permutations.
aggregate_permutation_results
    Combine results across multiple folds/splits.
compute_p_value
    Calculate p-value from observed score and null distribution.
pool_null_distribution
    Pool null distributions across seeds/folds for model-level significance.
load_hpc_permutation_results
    Load permutation results from HPC job array runs.
detect_and_aggregate
    Auto-detect and aggregate permutation results for a run.

Classes
-------
PermutationTestResult
    Container for permutation test results from a single fold.
PooledNullResult
    Container for pooled-null aggregation results across seeds/folds.

Example
-------
>>> from ced_ml.significance import run_permutation_test, aggregate_permutation_results
>>> result = run_permutation_test(
...     pipeline=my_pipeline,
...     X=X_train_val,
...     y=y_train_val,
...     train_idx=train_idx,
...     test_idx=val_idx,
...     config=config,
...     n_perms=200,
...     n_jobs=4,
...     random_state=42
... )
>>> print(f"p-value: {result.p_value:.4f}")
>>> df = aggregate_permutation_results([result1, result2, result3])

>>> # Pooled null aggregation
>>> from ced_ml.significance import pool_null_distribution, detect_and_aggregate
>>> pooled = pool_null_distribution(results_df, model='LR_EN', alpha=0.05)
>>> print(f"Pooled p-value: {pooled.empirical_p_value:.4f}")
"""

from ced_ml.significance.aggregation import (
    PooledNullResult,
    compute_pooled_p_value,
    detect_and_aggregate,
    load_hpc_permutation_results,
    pool_null_distribution,
)
from ced_ml.significance.permutation_test import (
    PermutationTestResult,
    aggregate_permutation_results,
    compute_p_value,
    run_permutation_test,
)

__all__ = [
    "PermutationTestResult",
    "compute_p_value",
    "run_permutation_test",
    "aggregate_permutation_results",
    "PooledNullResult",
    "compute_pooled_p_value",
    "pool_null_distribution",
    "load_hpc_permutation_results",
    "detect_and_aggregate",
]
