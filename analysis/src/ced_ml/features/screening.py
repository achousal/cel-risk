"""
Feature screening using univariate statistical tests.

Pure functions for Mann-Whitney U and F-statistic screening to identify
discriminative proteins before model training.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif

logger = logging.getLogger(__name__)


def mann_whitney_screen(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
    top_n: int,
    min_n_per_group: int = 10,
) -> tuple[list[str], pd.DataFrame]:
    """
    Screen proteins using Mann-Whitney U test.

    Compares protein distributions between cases and controls. Proteins with
    smaller p-values (more discriminative) are ranked higher.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with protein columns
    y_train : np.ndarray
        Binary labels (0=control, 1=case)
    protein_cols : List[str]
        Protein column names to screen
    top_n : int
        Number of top proteins to return
    min_n_per_group : int, default=10
        Minimum samples per group (cases/controls) required for test

    Returns
    -------
    selected_proteins : List[str]
        Top N proteins ranked by p-value (ascending), then effect_size (descending)
    screening_stats : pd.DataFrame
        Statistics for all tested proteins with columns:
        - protein: protein name
        - p_value: Mann-Whitney p-value (two-sided)
        - effect_size: |mean_cases - mean_controls|
        - nonmissing_frac: fraction of non-missing values

    Notes
    -----
    - Proteins with <min_n_per_group samples in either class are excluded
    - Missing values are ignored (test uses only non-missing observations)
    - Ties in p-value are broken by effect_size (descending)
    - Uses asymptotic method when available (scipy >= 1.4.0)

    Notes (top_n=0 behavior)
    ------------------------
    When top_n=0, computes statistics for all proteins but returns all without filtering.
    This is useful for diagnostics/reporting when you want full screening stats.
    """
    if len(protein_cols) == 0:
        return protein_cols, pd.DataFrame()

    # Handle top_n=0 case: compute stats for all, but don't filter
    effective_top_n = len(protein_cols) if top_n <= 0 else top_n

    y = np.asarray(y_train, dtype=int)
    if np.unique(y).size < 2:
        return protein_cols, pd.DataFrame()

    rows = []
    for p in protein_cols:
        x = pd.to_numeric(X_train[p], errors="coerce")
        ok = x.notna().to_numpy()

        # Skip if insufficient non-missing values
        if ok.sum() < (2 * min_n_per_group):
            continue

        x_ok = x[ok].to_numpy(dtype=float)
        y_ok = y[ok]
        x0 = x_ok[y_ok == 0]  # controls
        x1 = x_ok[y_ok == 1]  # cases

        # Enforce minimum group size
        if len(x0) < min_n_per_group or len(x1) < min_n_per_group:
            continue

        # Compute Mann-Whitney U test
        try:
            # Try asymptotic method (faster, available in scipy >= 1.4.0)
            _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic")
        except TypeError:
            # Fallback for older scipy versions (no 'method' parameter)
            try:
                _, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Mann-Whitney test failed for {p}: {type(e).__name__}: {e}")
                p_mw = np.nan
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Mann-Whitney test failed for {p}: {type(e).__name__}: {e}")
            p_mw = np.nan

        if not isinstance(p_mw, float):
            try:
                p_mw = float(p_mw)
            except (ValueError, TypeError):
                p_mw = np.nan

        # Compute effect size (mean difference)
        delta = float(np.nanmean(x1) - np.nanmean(x0))
        nonmissing_frac = float(np.mean(ok))

        rows.append((p, p_mw, abs(delta), nonmissing_frac))

    if not rows:
        return protein_cols, pd.DataFrame()

    # Build results DataFrame
    df_stats = pd.DataFrame(rows, columns=["protein", "p_value", "effect_size", "nonmissing_frac"])

    # Sort by p-value (ascending), then effect_size (descending)
    df_stats = df_stats.sort_values(
        ["p_value", "effect_size"], ascending=[True, False], na_position="last"
    )

    # Select top N (or all if top_n was 0)
    selected = df_stats["protein"].head(min(effective_top_n, len(df_stats))).tolist()

    return selected, df_stats


def f_statistic_screen(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
    top_n: int,
) -> tuple[list[str], pd.DataFrame]:
    """
    Screen proteins using ANOVA F-statistic.

    Compares protein means between classes using one-way ANOVA. Proteins with
    higher F-scores are ranked higher.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with protein columns
    y_train : np.ndarray
        Binary labels (0=control, 1=case)
    protein_cols : List[str]
        Protein column names to screen
    top_n : int
        Number of top proteins to return

    Returns
    -------
    selected_proteins : List[str]
        Top N proteins ranked by F-score (descending)
    screening_stats : pd.DataFrame
        Statistics for all proteins with columns:
        - protein: protein name
        - F_score: ANOVA F-statistic
        - p_value: F-test p-value
        - nonmissing_frac: fraction of non-missing values

    Notes
    -----
    - Missing values are median-imputed before testing
    - Proteins with non-finite F-scores (e.g., zero variance) are excluded
    - Uses sklearn.feature_selection.f_classif

    Notes (top_n=0 behavior)
    ------------------------
    When top_n=0, computes statistics for all proteins but returns all without filtering.
    This is useful for diagnostics/reporting when you want full screening stats.
    """
    if len(protein_cols) == 0:
        return protein_cols, pd.DataFrame()

    # Handle top_n=0 case: compute stats for all, but don't filter
    effective_top_n = len(protein_cols) if top_n <= 0 else top_n

    y = np.asarray(y_train, dtype=int)
    if np.unique(y).size < 2:
        return protein_cols, pd.DataFrame()

    # Convert to numeric and median-impute
    Xp = X_train[protein_cols].apply(pd.to_numeric, errors="coerce")
    nonmissing_frac = Xp.notna().mean(axis=0)
    med = Xp.median(axis=0, skipna=True)
    Ximp = Xp.fillna(med)

    # Compute F-statistics
    try:
        F, pvals = f_classif(Ximp.to_numpy(dtype=float), y)
    except (ValueError, RuntimeError) as e:
        logger.warning(
            f"F-statistic screening failed (returning all proteins): {type(e).__name__}: {e}"
        )
        return protein_cols, pd.DataFrame()

    # Process results (no exception handling needed - should fail fast on bugs)
    F = np.asarray(F, dtype=float)
    pvals = np.asarray(pvals, dtype=float)

    # Filter out non-finite F-scores
    ok = np.isfinite(F)

    df_stats = pd.DataFrame(
        {
            "protein": np.asarray(protein_cols)[ok],
            "F_score": F[ok],
            "p_value": pvals[ok],
            "nonmissing_frac": nonmissing_frac.to_numpy()[ok],
        }
    )

    # Sort by F-score descending
    df_stats = df_stats.sort_values(["F_score"], ascending=[False], na_position="last")

    # Select top N (or all if top_n was 0)
    selected = df_stats["protein"].head(min(effective_top_n, len(df_stats))).tolist()

    return selected, df_stats


def screen_proteins(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
    method: str = "mannwhitney",
    top_n: int = 1000,
    min_n_per_group: int = 10,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Screen proteins using univariate statistical tests.

    Convenience wrapper for mann_whitney_screen and f_statistic_screen.
    Automatically caches results to avoid redundant computation when the same
    data is screened multiple times (e.g., learning curves, diagnostics export).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with protein columns
    y_train : np.ndarray
        Binary labels (0=control, 1=case)
    protein_cols : List[str]
        Protein column names to screen
    method : str, default="mannwhitney"
        Screening method: "mannwhitney" or "f_classif"
    top_n : int, default=1000
        Number of top proteins to return (0=keep all, no screening)
    min_n_per_group : int, default=10
        Minimum samples per group (Mann-Whitney only)
    use_cache : bool, default=True
        Whether to use caching for screening results
    verbose : bool, default=True
        If False, use debug-level logging (suitable for repeated calls in loops)

    Returns
    -------
    selected_proteins : List[str]
        Top N proteins (or all if top_n=0)
    screening_stats : pd.DataFrame
        Test statistics for screened proteins

    Raises
    ------
    ValueError
        If method is not recognized

    Examples
    --------
    >>> selected, stats = screen_proteins(
    ...     X_train, y_train, protein_cols,
    ...     method="mannwhitney", top_n=1000
    ... )
    >>> print(f"Selected {len(selected)} proteins")
    >>> print(stats.head())
    """
    method = (method or "").strip().lower()

    # Handle empty protein list (no caching needed)
    if len(protein_cols) == 0:
        logger.debug("Screening disabled (no proteins), returning empty list")
        return protein_cols, pd.DataFrame()

    # Try cache lookup (even for top_n=0 to get full stats)
    if use_cache:
        from ced_ml.features.screening_cache import get_screening_cache

        cache = get_screening_cache()
        cached_result = cache.get(X_train, y_train, protein_cols, method, top_n)
        if cached_result is not None:
            selected, stats = cached_result
            logger.info(
                f"Feature Screening ({method}) - CACHED: {len(selected)}/{len(protein_cols)} proteins (top_n={top_n})"
            )
            return selected, stats

    # Run screening (with optional logging level control for repeated CV folds)
    log_level = logger.info if verbose else logger.debug
    if method == "mannwhitney":
        selected, stats = mann_whitney_screen(
            X_train, y_train, protein_cols, top_n, min_n_per_group
        )
    elif method == "f_classif":
        selected, stats = f_statistic_screen(X_train, y_train, protein_cols, top_n)
    else:
        raise ValueError(f"Unknown screen_method='{method}'. Valid: 'mannwhitney', 'f_classif'")

    # Store in cache
    if use_cache and not stats.empty:
        from ced_ml.features.screening_cache import get_screening_cache

        cache = get_screening_cache()
        cache.put(X_train, y_train, protein_cols, method, top_n, selected, stats)

    # Log screening results (compact single-line format)
    if stats.empty:
        logger.warning("Screening failed: no proteins tested (check data quality)")
        return selected, stats

    n_selected = len(selected)

    # Compact summary: method, selection ratio, key statistic
    if "p_value" in stats.columns:
        p_vals = stats["p_value"].dropna()
        if len(p_vals) > 0:
            log_level(
                f"Screened {method}: {n_selected}/{len(protein_cols)} proteins "
                f"(n={len(X_train)}, p_median={p_vals.median():.2e})"
            )
            # Detailed stats at DEBUG level
            logger.debug(
                f"  P-value range: [{p_vals.min():.2e}, {p_vals.max():.2e}], "
                f"sig: p<0.001={int((p_vals < 0.001).sum())}, p<0.01={int((p_vals < 0.01).sum())}"
            )
    elif "F_score" in stats.columns:
        f_scores = stats["F_score"].dropna()
        if len(f_scores) > 0:
            log_level(
                f"Screened {method}: {n_selected}/{len(protein_cols)} proteins "
                f"(n={len(X_train)}, F_median={f_scores.median():.2f})"
            )
            logger.debug(f"  F-score range: [{f_scores.min():.2f}, {f_scores.max():.2f}]")
    else:
        log_level(
            f"Screened {method}: {n_selected}/{len(protein_cols)} proteins (n={len(X_train)})"
        )

    # Warnings for edge cases
    if n_selected == 0:
        logger.warning("Zero proteins passed screening - check data quality or relax top_n")
    elif n_selected < top_n:
        logger.warning(f"Only {n_selected} proteins available (requested top_n={top_n})")

    return selected, stats


def variance_missingness_prefilter(
    X_train: pd.DataFrame,
    protein_cols: list[str],
    min_nonmissing: float = 0.95,
    min_var: float = 1e-6,
    strict: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Pre-filter proteins by missingness and variance.

    Removes proteins with excessive missing values or near-zero variance.
    This is a data quality filter applied before statistical screening.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with protein columns
    protein_cols : List[str]
        Protein column names to filter
    min_nonmissing : float, default=0.95
        Minimum fraction of non-missing values required
    min_var : float, default=1e-6
        Minimum variance required (computed on non-missing values)
    strict : bool, default=True
        If True, raise ValueError if all proteins fail
        If False, disable filter and return all proteins with warning

    Returns
    -------
    kept_proteins : List[str]
        Proteins passing both filters
    filter_report : pd.DataFrame
        Per-protein filter results with columns:
        - protein: protein name
        - nonmissing_frac: fraction of non-missing values
        - passed_nonmissing: bool, passed missingness threshold
        - variance_train: variance (if computed)
        - passed_variance: bool, passed variance threshold

    Raises
    ------
    ValueError
        If strict=True and all proteins fail either filter

    Notes
    -----
    - Variance is computed only on proteins passing missingness filter
    - If all proteins fail, behavior depends on strict flag:
        - strict=True: raises ValueError
        - strict=False: returns all proteins with warning
    """
    if len(protein_cols) == 0:
        return protein_cols, pd.DataFrame()

    Xm = X_train[protein_cols]
    nonmiss = Xm.notna().mean(axis=0)

    # Filter by missingness
    prot_ok = nonmiss[nonmiss >= min_nonmissing].index.tolist()

    if len(prot_ok) == 0:
        msg = (
            f"All {len(protein_cols)} proteins failed non-missing filter "
            f"(min_nonmissing={min_nonmissing})."
        )
        if strict:
            raise ValueError(msg)
        # Non-strict: return all proteins with report
        report = pd.DataFrame(
            {
                "protein": protein_cols,
                "nonmissing_frac": nonmiss.reindex(protein_cols).values,
                "passed_nonmissing": [False] * len(protein_cols),
                "variance_train": np.nan,
                "passed_variance": [False] * len(protein_cols),
            }
        )
        return protein_cols, report

    # Filter by variance (only proteins that passed missingness)
    var = pd.Series(np.nanvar(X_train[prot_ok].to_numpy(dtype=float), axis=0), index=prot_ok)
    prot_keep = var[var >= min_var].index.tolist()

    if len(prot_keep) == 0:
        msg = f"All {len(prot_ok)} proteins failed variance filter (min_var={min_var})."
        if strict:
            raise ValueError(msg)
        # Non-strict: return proteins that passed missingness
        prot_keep = prot_ok

    # Build report
    report = pd.DataFrame(
        {
            "protein": protein_cols,
            "nonmissing_frac": nonmiss.reindex(protein_cols).values,
        }
    )
    report["passed_nonmissing"] = report["protein"].isin(prot_ok)
    report["variance_train"] = var.reindex(report["protein"]).values
    report["passed_variance"] = report["protein"].isin(prot_keep)

    return prot_keep, report
