"""Pooled-null permutation aggregation for model-level significance testing.

This module provides functions to aggregate permutation test results across
seeds and folds into a single pooled null distribution for testing model-level
significance. This approach increases statistical power by combining evidence
across multiple independent test sets.

Algorithm:
    1. Collect all individual permutation results from splits/folds
    2. Pool null AUROCs: null_pool = [null_0_fold0, ..., null_B_foldK]
    3. Compute mean observed AUROC across folds
    4. Test H0: observed_mean <= null_pool via empirical p-value

The pooled approach is valid when:
    - Folds/seeds represent independent test sets
    - Each permutation uses the full pipeline (no data leakage)
    - Null distributions are comparable (same model, same scenario)

References
----------
Westfall, P. H., & Young, S. S. (1993). Resampling-based multiple testing:
Examples and methods for p-value adjustment. John Wiley & Sons.

Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier
performance. Journal of Machine Learning Research, 11, 1833-1863.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PooledNullResult:
    """Results from pooled-null permutation aggregation.

    This class represents model-level significance testing by pooling null
    distributions across multiple seeds/folds. It provides stronger evidence
    than per-fold tests by aggregating independent test sets.

    Attributes
    ----------
    model : str
        Model name (e.g., 'LR_EN', 'RF', 'XGBoost').
    observed_auroc : float
        Mean observed AUROC across all seeds/folds.
    pooled_null : np.ndarray
        All null AUROCs pooled across seeds/folds.
        Shape: (n_seeds * n_folds * n_perms,)
    empirical_p_value : float
        One-sided p-value: p = (1 + #{null >= observed}) / (1 + n_total).
    n_seeds : int
        Number of split seeds aggregated.
    n_perms_total : int
        Total number of permutations pooled.
    significant : bool
        Whether p_value < alpha.
    alpha : float
        Significance level used (default: 0.05).

    Notes
    -----
    The pooled null distribution increases statistical power compared to
    per-fold tests by combining evidence across independent test sets.
    However, it assumes folds/seeds are exchangeable under H0.

    Examples
    --------
    >>> result = PooledNullResult(
    ...     model='LR_EN',
    ...     observed_auroc=0.75,
    ...     pooled_null=np.array([0.48, 0.52, 0.55, 0.49, 0.51]),
    ...     empirical_p_value=0.05,
    ...     n_seeds=3,
    ...     n_perms_total=600,
    ...     significant=True,
    ...     alpha=0.05
    ... )
    >>> print(f"Model {result.model} significant: {result.significant}")
    """

    model: str
    observed_auroc: float
    pooled_null: np.ndarray
    empirical_p_value: float
    n_seeds: int
    n_perms_total: int
    significant: bool
    alpha: float

    def summary_stats(self) -> dict[str, float]:
        """Compute summary statistics of pooled null distribution.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: null_mean, null_std, null_min, null_max,
            null_median, null_q25, null_q75.

        Examples
        --------
        >>> stats = result.summary_stats()
        >>> print(f"Null mean: {stats['null_mean']:.4f}")
        """
        if len(self.pooled_null) == 0:
            return {
                "null_mean": np.nan,
                "null_std": np.nan,
                "null_min": np.nan,
                "null_max": np.nan,
                "null_median": np.nan,
                "null_q25": np.nan,
                "null_q75": np.nan,
            }

        return {
            "null_mean": float(np.mean(self.pooled_null)),
            "null_std": float(np.std(self.pooled_null)),
            "null_min": float(np.min(self.pooled_null)),
            "null_max": float(np.max(self.pooled_null)),
            "null_median": float(np.median(self.pooled_null)),
            "null_q25": float(np.percentile(self.pooled_null, 25)),
            "null_q75": float(np.percentile(self.pooled_null, 75)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes pooled_null array for summary).

        Returns
        -------
        dict[str, Any]
            Dictionary with scalar fields and summary statistics.

        Examples
        --------
        >>> d = result.to_dict()
        >>> print(d['model'], d['empirical_p_value'])
        """
        summary = {
            "model": self.model,
            "observed_auroc": self.observed_auroc,
            "empirical_p_value": self.empirical_p_value,
            "n_seeds": self.n_seeds,
            "n_perms_total": self.n_perms_total,
            "significant": self.significant,
            "alpha": self.alpha,
        }
        summary.update(self.summary_stats())
        return summary


def compute_pooled_p_value(observed: float, null_pool: np.ndarray) -> float:
    """Compute empirical p-value from observed score and pooled null distribution.

    Tests the null hypothesis that observed performance is no better than
    the pooled null distribution using the formula:
        p = (1 + #{null >= observed}) / (1 + n)

    The +1 correction prevents p-values of exactly 0, which are theoretically
    incorrect for finite permutation tests.

    Parameters
    ----------
    observed : float
        Observed mean performance metric (e.g., AUROC) across folds.
    null_pool : np.ndarray
        Pooled null distribution from all permutations across folds/seeds.

    Returns
    -------
    float
        One-sided empirical p-value in [0, 1]. Lower values indicate
        stronger evidence against H0.

    Raises
    ------
    ValueError
        If null_pool is empty.

    References
    ----------
    Phipson & Smyth (2010). Permutation P-values should never be zero.
    Statistical Applications in Genetics and Molecular Biology, 9(1).

    Examples
    --------
    >>> observed = 0.75
    >>> null = np.array([0.48, 0.52, 0.55, 0.49, 0.51])
    >>> p = compute_pooled_p_value(observed, null)
    >>> print(f"p-value: {p:.3f}")
    0.167
    """
    if len(null_pool) == 0:
        raise ValueError("Null pool is empty. Cannot compute p-value.")

    null_array = np.asarray(null_pool)
    n_total = len(null_array)

    n_greater_equal = np.sum(null_array >= observed)
    p_value = (1 + n_greater_equal) / (1 + n_total)

    return float(p_value)


def pool_null_distribution(
    results_df: pd.DataFrame,
    model: str | None = None,
    alpha: float = 0.05,
) -> PooledNullResult:
    """Pool null AUROCs across seeds/folds into one distribution.

    This function aggregates permutation test results from multiple
    independent test sets (folds/seeds) to test model-level significance
    with increased statistical power.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: model, split_seed, outer_fold, null_auroc.
        Expected format from load_hpc_permutation_results() or
        aggregated per-fold permutation test results.
    model : str, optional
        Model name to filter (if None, uses first model in results_df).
    alpha : float, default=0.05
        Significance level for hypothesis testing.

    Returns
    -------
    PooledNullResult
        Pooled null distribution and significance test results.

    Raises
    ------
    ValueError
        If results_df is empty or missing required columns.

    Notes
    -----
    - Assumes each row represents one null AUROC from one permutation
    - Observed AUROC must be provided separately or computed from training logs
    - Folds/seeds must be independent for valid pooling

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'model': ['LR_EN'] * 600,
    ...     'split_seed': [0]*200 + [1]*200 + [2]*200,
    ...     'outer_fold': [0]*600,
    ...     'null_auroc': np.random.rand(600) * 0.1 + 0.45
    ... })
    >>> result = pool_null_distribution(df, model='LR_EN', alpha=0.05)
    >>> print(f"Pooled p-value: {result.empirical_p_value:.4f}")
    """
    required_cols = ["model", "null_auroc"]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        raise ValueError(f"results_df missing required columns: {missing}")

    if results_df.empty:
        raise ValueError("results_df is empty. Cannot pool null distribution.")

    if model is None:
        available_models = results_df["model"].unique()
        if len(available_models) == 0:
            raise ValueError("No models found in results_df")
        model = available_models[0]
        logger.info(f"No model specified. Using first model: {model}")

    model_df = results_df[results_df["model"] == model].copy()

    if model_df.empty:
        raise ValueError(f"No results found for model: {model}")

    pooled_null = model_df["null_auroc"].values

    valid_null = pooled_null[~np.isnan(pooled_null)]

    if len(valid_null) == 0:
        logger.warning(f"All null AUROCs are NaN for model {model}. Returning empty result.")
        return PooledNullResult(
            model=model,
            observed_auroc=np.nan,
            pooled_null=np.array([]),
            empirical_p_value=1.0,
            n_seeds=0,
            n_perms_total=0,
            significant=False,
            alpha=alpha,
        )

    if len(valid_null) < len(pooled_null):
        n_failed = len(pooled_null) - len(valid_null)
        logger.warning(f"Removed {n_failed} NaN values from pooled null distribution")

    n_seeds = model_df["split_seed"].nunique() if "split_seed" in model_df.columns else 1

    if "observed_auroc" in model_df.columns:
        observed_auroc = model_df["observed_auroc"].mean()
    else:
        observed_auroc = np.nan

    if np.isnan(observed_auroc):
        logger.warning(
            f"Observed AUROC not found in results_df for model {model}. "
            f"Returning p-value=1.0. Load observed AUROC from training logs."
        )
        p_value = 1.0
    else:
        p_value = compute_pooled_p_value(observed_auroc, valid_null)

    significant = p_value < alpha

    result = PooledNullResult(
        model=model,
        observed_auroc=observed_auroc,
        pooled_null=valid_null,
        empirical_p_value=p_value,
        n_seeds=n_seeds,
        n_perms_total=len(valid_null),
        significant=significant,
        alpha=alpha,
    )

    logger.info(
        f"Pooled null distribution: model={model}, n_seeds={n_seeds}, "
        f"n_total={len(valid_null)}, p={p_value:.4f}, significant={significant}"
    )

    return result


def load_hpc_permutation_results(outdir: Path) -> pd.DataFrame:
    """Load individual permutation results from HPC job array runs.

    This function discovers and loads all perm_*.joblib files generated
    by HPC job arrays (via --perm-index flag in CLI).

    Parameters
    ----------
    outdir : Path
        Directory containing perm_*.joblib files.
        Expected location: results/run_{id}/{model}/significance/

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, split_seed, outer_fold, perm_index,
        perm_seed, null_auroc. Returns empty DataFrame if no results found.

    Notes
    -----
    Expected joblib file structure:
        {
            'model': str,
            'split_seed': int,
            'outer_fold': int,
            'perm_index': int,
            'perm_seed': int,
            'null_auroc': float
        }

    Examples
    --------
    >>> outdir = Path('results/run_20260127_115115/LR_EN/significance')
    >>> df = load_hpc_permutation_results(outdir)
    >>> print(f"Loaded {len(df)} permutation results")
    """
    if not outdir.exists():
        logger.warning(f"Output directory does not exist: {outdir}")
        return pd.DataFrame()

    perm_files = sorted(outdir.glob("perm_*.joblib"))

    if not perm_files:
        logger.warning(f"No perm_*.joblib files found in {outdir}")
        return pd.DataFrame()

    logger.info(f"Found {len(perm_files)} permutation result files in {outdir}")

    records = []
    for perm_file in perm_files:
        try:
            result = joblib.load(perm_file)
            if isinstance(result, dict):
                records.append(result)
            else:
                logger.warning(f"Unexpected format in {perm_file.name}: {type(result)}")
        except Exception as e:
            logger.warning(f"Failed to load {perm_file.name}: {e}")

    if not records:
        logger.warning("No valid permutation results loaded")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    expected_cols = ["model", "split_seed", "outer_fold", "perm_index", "null_auroc"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.warning(f"Loaded results missing expected columns: {missing}")

    df = df.sort_values(["model", "split_seed", "outer_fold", "perm_index"]).reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} permutation results: "
        f"models={df['model'].unique().tolist() if 'model' in df.columns else []}, "
        f"n_perms={len(df)}"
    )

    return df


def detect_and_aggregate(
    run_dir: Path,
    model: str | None = None,
    alpha: float = 0.05,
) -> dict[str, PooledNullResult]:
    """Auto-detect permutation results and aggregate if complete.

    This convenience function discovers permutation test results from a run
    directory and aggregates them into pooled null distributions for all
    available models.

    Parameters
    ----------
    run_dir : Path
        Run directory containing model subdirectories.
        Expected structure: run_dir/{model}/significance/perm_*.joblib
    model : str, optional
        Specific model to aggregate (if None, processes all models).
    alpha : float, default=0.05
        Significance level for hypothesis testing.

    Returns
    -------
    dict[str, PooledNullResult]
        Dictionary mapping model names to PooledNullResult objects.
        Returns empty dict if no results found.

    Notes
    -----
    - Scans run_dir for model subdirectories with significance/ folders
    - Loads all perm_*.joblib files for each model
    - Aggregates into pooled null distributions
    - Logs warnings if results incomplete or missing

    Examples
    --------
    >>> run_dir = Path('results/run_20260127_115115')
    >>> results = detect_and_aggregate(run_dir, alpha=0.05)
    >>> for model_name, result in results.items():
    ...     print(f"{model_name}: p={result.empirical_p_value:.4f}")
    """
    if not run_dir.exists():
        logger.warning(f"Run directory does not exist: {run_dir}")
        return {}

    if model is not None:
        model_dirs = [run_dir / model]
    else:
        model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not model_dirs:
        logger.warning(f"No model directories found in {run_dir}")
        return {}

    results = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        sig_dir = model_dir / "significance"

        if not sig_dir.exists():
            logger.info(f"No significance/ directory for model {model_name}")
            continue

        df = load_hpc_permutation_results(sig_dir)

        if df.empty:
            logger.warning(f"No permutation results found for model {model_name}")
            continue

        try:
            pooled_result = pool_null_distribution(df, model=model_name, alpha=alpha)
            results[model_name] = pooled_result

            logger.info(
                f"Aggregated {model_name}: p={pooled_result.empirical_p_value:.4f}, "
                f"significant={pooled_result.significant}"
            )
        except Exception as e:
            logger.error(f"Failed to aggregate {model_name}: {e}")

    if not results:
        logger.warning(f"No permutation results aggregated for run {run_dir.name}")
    else:
        logger.info(f"Successfully aggregated {len(results)} models: {list(results.keys())}")

    return results
