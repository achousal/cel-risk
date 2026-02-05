"""Permutation testing for model generalization above chance.

This module implements label permutation testing to test the null hypothesis
that a model's predictive performance is no better than random chance.

For each permutation, the full pipeline (screening, feature selection,
hyperparameter optimization, training) is re-run on permuted labels to avoid
data leakage and obtain a valid null distribution.

Algorithm (per outer fold):
    1. Load trained model artifacts and split indices
    2. For each permutation b in 0..B-1:
        a. Shuffle y_train only (keep X fixed, held-out y unchanged)
        b. Run FULL inner pipeline:
            - Screening on permuted labels
            - Inner CV k-best tuning
            - Hyperparameter optimization (Optuna/RandomizedSearchCV)
            - Fit final model on permuted train
        c. Predict on held-out fold (original X)
        d. Record AUROC
    3. Aggregate null scores across folds/permutations
    4. Compute p-value: p = (1 + #{null >= observed}) / (1 + B)

References
----------
Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier
performance. Journal of Machine Learning Research, 11, 1833-1863.

Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero:
calculating exact P-values when permutations are randomly drawn. Statistical
Applications in Genetics and Molecular Biology, 9(1).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class PermutationTestResult:
    """Results from permutation testing on a single fold.

    Attributes
    ----------
    model : str
        Model name (e.g., 'LR_EN', 'RF', 'XGBoost').
    split_seed : int
        Random seed for the train/val/test split.
    outer_fold : int
        Outer fold index (0-based).
    observed_auroc : float
        AUROC from the original (unpermuted) model.
    null_aurocs : list[float]
        AUROC values from B permutations (null distribution).
    p_value : float
        One-sided p-value testing H0: model performance <= chance.
        Computed as: p = (1 + #{null >= observed}) / (1 + B)
    n_perms : int
        Number of permutations performed.
    random_state : int
        Random seed used for permutations.
    """

    model: str
    split_seed: int
    outer_fold: int
    observed_auroc: float
    null_aurocs: list[float]
    p_value: float
    n_perms: int
    random_state: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes null_aurocs for summary)."""
        result = asdict(self)
        result.pop("null_aurocs")
        return result

    def summary_stats(self) -> dict[str, float]:
        """Compute summary statistics of null distribution."""
        return {
            "null_mean": float(np.mean(self.null_aurocs)),
            "null_std": float(np.std(self.null_aurocs)),
            "null_min": float(np.min(self.null_aurocs)),
            "null_max": float(np.max(self.null_aurocs)),
            "null_median": float(np.median(self.null_aurocs)),
        }


def compute_p_value(observed: float, null_distribution: list[float]) -> float:
    """Compute one-sided permutation p-value.

    Tests the null hypothesis that the observed score is no better than
    the null distribution (chance performance).

    Uses the formula: p = (1 + #{null >= observed}) / (1 + B)
    where B is the number of permutations.

    The +1 in numerator and denominator prevents p-values of exactly 0,
    which are theoretically incorrect for finite permutation tests.

    Parameters
    ----------
    observed : float
        Observed performance metric (e.g., AUROC) from unpermuted data.
    null_distribution : list[float]
        Performance metrics from B label permutations.

    Returns
    -------
    float
        One-sided p-value in [0, 1]. Lower values indicate stronger evidence
        against the null hypothesis.

    References
    ----------
    Phipson & Smyth (2010). Permutation P-values should never be zero.
    Statistical Applications in Genetics and Molecular Biology, 9(1).

    Examples
    --------
    >>> observed = 0.75
    >>> null = [0.48, 0.52, 0.55, 0.49, 0.51]
    >>> p = compute_p_value(observed, null)
    >>> print(f"{p:.3f}")
    0.167
    """
    null_array = np.asarray(null_distribution)
    n_perms = len(null_array)

    if n_perms == 0:
        raise ValueError("Null distribution is empty")

    n_greater_equal = np.sum(null_array >= observed)
    p_value = (1 + n_greater_equal) / (1 + n_perms)

    return float(p_value)


def run_permutation_for_fold(
    pipeline: Any,
    X_train: "np.ndarray | pd.DataFrame",
    y_train: np.ndarray,
    X_test: "np.ndarray | pd.DataFrame",
    y_test: np.ndarray,
    random_state: int,
    perm_idx: int,
) -> float:
    """Run a single permutation: shuffle labels, fit pipeline, evaluate.

    This function re-runs the FULL pipeline (screening, feature selection,
    hyperparameter optimization, training) on permuted labels to ensure
    the null distribution is valid and avoids data leakage.

    Parameters
    ----------
    pipeline : sklearn.base.BaseEstimator
        Pipeline to clone and fit. Must implement fit() and predict_proba().
    X_train : np.ndarray or pd.DataFrame
        Training features (not permuted).
    y_train : np.ndarray
        Training labels (will be permuted).
    X_test : np.ndarray or pd.DataFrame
        Test features (not permuted).
    y_test : np.ndarray
        Test labels (not permuted).
    random_state : int
        Base random seed.
    perm_idx : int
        Permutation index (added to random_state for reproducibility).

    Returns
    -------
    float
        AUROC on held-out test set using permuted training labels.

    Notes
    -----
    - Each permutation uses a different random seed: random_state + perm_idx
    - Held-out labels (y_test) are never permuted
    - Pipeline is cloned to ensure independence between permutations
    """
    perm_seed = random_state + perm_idx
    rng = np.random.RandomState(perm_seed)

    y_train_permuted = rng.permutation(y_train)

    pipeline_clone = clone(pipeline)

    try:
        pipeline_clone.fit(X_train, y_train_permuted)
        y_pred_proba = pipeline_clone.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        logger.warning(f"Permutation {perm_idx} failed (seed={perm_seed}): {e}. Returning NaN.")
        auroc = np.nan

    return float(auroc)


def run_permutation_test(
    pipeline: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    split_seed: int,
    outer_fold: int,
    n_perms: int = 200,
    n_jobs: int = 1,
    random_state: int = 42,
) -> PermutationTestResult:
    """Run permutation test for a single outer fold.

    Tests the null hypothesis that the trained model performs no better
    than chance by comparing observed AUROC against a null distribution
    from B label permutations.

    Parameters
    ----------
    pipeline : sklearn.base.BaseEstimator
        Fitted pipeline or unfitted pipeline that will be cloned.
        Must implement fit() and predict_proba().
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (train + test combined).
    y : np.ndarray, shape (n_samples,)
        Label vector (train + test combined).
    train_idx : np.ndarray
        Indices for training set.
    test_idx : np.ndarray
        Indices for held-out test set.
    model_name : str
        Model identifier (e.g., 'LR_EN', 'RF', 'XGBoost').
    split_seed : int
        Random seed used for train/val/test split.
    outer_fold : int
        Outer fold index (0-based).
    n_perms : int, default=200
        Number of label permutations. Larger B gives more precise p-values.
        Recommended: B >= 200 for publication, B >= 1000 for final validation.
    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all CPUs.
        Note: Memory usage scales with n_jobs.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    PermutationTestResult
        Result object containing observed AUROC, null distribution,
        p-value, and metadata.

    Notes
    -----
    - Computation time: Each permutation re-runs the full pipeline
      (screening, feature selection, hyperparameter optimization).
      Expected runtime: ~B * single_pipeline_time.
    - Memory: Each parallel job requires a full pipeline clone.
    - P-value interpretation: p < 0.05 provides evidence against H0
      (model generalizes above chance).

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('classifier', LogisticRegression(random_state=42))
    ... ])
    >>> result = run_permutation_test(
    ...     pipeline=pipeline,
    ...     X=X_train_val,
    ...     y=y_train_val,
    ...     train_idx=train_idx,
    ...     test_idx=val_idx,
    ...     model_name='LR',
    ...     split_seed=0,
    ...     outer_fold=0,
    ...     n_perms=200,
    ...     n_jobs=4,
    ...     random_state=42
    ... )
    >>> print(f"Observed AUROC: {result.observed_auroc:.3f}")
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    logger.info(
        f"Starting permutation test: model={model_name}, split={split_seed}, "
        f"fold={outer_fold}, n_perms={n_perms}, n_jobs={n_jobs}"
    )

    # Handle both DataFrame and numpy array inputs
    if hasattr(X, "iloc"):
        # DataFrame: use iloc for integer position indexing
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        # NumPy array: use direct indexing
        X_train = X[train_idx]
        X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Extract base model if wrapped in OOFCalibratedModel
    # Calibration is irrelevant for permutation testing (testing discrimination, not calibration)
    from ced_ml.models.calibration import OOFCalibratedModel

    base_pipeline = pipeline
    if isinstance(pipeline, OOFCalibratedModel):
        base_pipeline = pipeline.base_model
        logger.info("Extracted base model from OOFCalibratedModel for permutation testing")

    pipeline_clone = clone(base_pipeline)
    pipeline_clone.fit(X_train, y_train)
    y_pred_proba = pipeline_clone.predict_proba(X_test)[:, 1]
    observed_auroc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Observed AUROC: {observed_auroc:.4f}")
    logger.info(f"Running {n_perms} permutations with {n_jobs} parallel jobs...")

    null_aurocs = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(run_permutation_for_fold)(
            pipeline=base_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
            perm_idx=i,
        )
        for i in range(n_perms)
    )

    null_aurocs_clean = [x for x in null_aurocs if not np.isnan(x)]
    n_failed = len(null_aurocs) - len(null_aurocs_clean)

    if n_failed > 0:
        logger.warning(
            f"{n_failed}/{n_perms} permutations failed. "
            f"Using {len(null_aurocs_clean)} valid permutations."
        )

    if len(null_aurocs_clean) == 0:
        raise RuntimeError("All permutations failed. Cannot compute p-value.")

    p_value = compute_p_value(observed_auroc, null_aurocs_clean)

    result = PermutationTestResult(
        model=model_name,
        split_seed=split_seed,
        outer_fold=outer_fold,
        observed_auroc=observed_auroc,
        null_aurocs=null_aurocs_clean,
        p_value=p_value,
        n_perms=len(null_aurocs_clean),
        random_state=random_state,
    )

    logger.info(
        f"Permutation test complete: p-value={p_value:.4f}, "
        f"null_mean={np.mean(null_aurocs_clean):.4f}, "
        f"null_std={np.std(null_aurocs_clean):.4f}"
    )

    return result


def aggregate_permutation_results(
    results: list[PermutationTestResult],
) -> pd.DataFrame:
    """Aggregate permutation test results across folds/splits.

    Parameters
    ----------
    results : list[PermutationTestResult]
        List of permutation test results from multiple folds/splits.

    Returns
    -------
    pd.DataFrame
        Aggregated results with columns:
        - model: Model name
        - split_seed: Split random seed
        - outer_fold: Fold index
        - observed_auroc: Observed AUROC
        - null_mean: Mean of null distribution
        - null_std: Standard deviation of null distribution
        - null_min: Minimum of null distribution
        - null_max: Maximum of null distribution
        - null_median: Median of null distribution
        - p_value: Permutation p-value
        - n_perms: Number of permutations
        - random_state: Random seed used

    Examples
    --------
    >>> results = [result_fold0, result_fold1, result_fold2]
    >>> df = aggregate_permutation_results(results)
    >>> print(df[['model', 'outer_fold', 'observed_auroc', 'p_value']])
    """
    if not results:
        raise ValueError("Results list is empty")

    records = []
    for result in results:
        record = result.to_dict()
        record.update(result.summary_stats())
        records.append(record)

    df = pd.DataFrame.from_records(records)

    df = df.sort_values(["model", "split_seed", "outer_fold"]).reset_index(drop=True)

    return df


def save_null_distributions(
    results: list[PermutationTestResult],
    output_path: str,
) -> None:
    """Save full null distributions to CSV for downstream analysis.

    Parameters
    ----------
    results : list[PermutationTestResult]
        List of permutation test results.
    output_path : str
        Path to output CSV file.

    Notes
    -----
    Output columns: model, split_seed, outer_fold, perm_index, null_auroc
    """
    records = []
    for result in results:
        for perm_idx, auroc in enumerate(result.null_aurocs):
            records.append(
                {
                    "model": result.model,
                    "split_seed": result.split_seed,
                    "outer_fold": result.outer_fold,
                    "perm_index": perm_idx,
                    "null_auroc": auroc,
                }
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["model", "split_seed", "outer_fold", "perm_index"])
    df.to_csv(output_path, index=False)

    logger.info(f"Saved null distributions to {output_path}")
