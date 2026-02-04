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

Classes
-------
PermutationTestResult
    Container for permutation test results from a single fold.

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
"""

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
]
