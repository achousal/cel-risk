"""Panel size derivation rules.

Two strategies:
- three_criterion: smallest p where >= 2 of 3 statistical criteria pass
- stability: size determined by feature consistency (bootstrap + CV sign)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def derive_size_three_criterion(
    sweep_df: pd.DataFrame,
    *,
    order_filter: str | None = None,
    model_filter: str | None = None,
    delta: float = 0.02,
    n_seeds: int = 30,
    min_criteria: int = 2,
    auroc_col: str = "summary_auroc_mean",
    auroc_std_col: str = "summary_auroc_std",
    panel_col: str = "panel_size",
) -> tuple[int, dict[str, Any]]:
    """Derive optimal panel size using the 3-criterion rule.

    For each panel size p (averaged across remaining rows after filtering):
    1. Non-inferior: AUROC(p) >= AUROC(p_best) - delta (one-sided z-test)
    2. Within 1 SE: AUROC(p) >= AUROC(p_best) - SE(p_best)
    3. Marginal gain insignificant: delta(p -> p+1) not significant after Holm

    Select smallest p where >= min_criteria criteria pass.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Aggregated sweep results with per-model, per-panel-size rows.
    order_filter : str, optional
        If provided, filter sweep_df to rows where ``order == order_filter``.
    model_filter : str, optional
        If provided, filter sweep_df to rows where ``model == model_filter``.
        Used for per-model size derivation in model-specific recipes.
    delta : float
        Non-inferiority margin for criterion 1.
    n_seeds : int
        Number of seeds used to compute sweep statistics (for SE calculation).
    min_criteria : int
        Minimum number of criteria that must pass (2 = majority, 3 = unanimous).
    auroc_col : str
        Column name for mean AUROC.
    auroc_std_col : str
        Column name for AUROC standard deviation.
    panel_col : str
        Column name for panel size.

    Returns
    -------
    optimal_p : int
        Smallest panel size where >= 2 criteria pass.
    audit_log : dict
        Per-size criterion results for reproducibility audit.
    """
    df = sweep_df.copy()

    # Apply order filter
    if order_filter is not None:
        if "order" not in df.columns:
            raise ValueError("sweep_df has no 'order' column for filtering")
        df = df[df["order"] == order_filter]
        if df.empty:
            raise ValueError(f"No rows match order='{order_filter}'")

    # Apply model filter
    if model_filter is not None:
        if "model" not in df.columns:
            raise ValueError("sweep_df has no 'model' column for filtering")
        df = df[df["model"] == model_filter]
        if df.empty:
            raise ValueError(f"No rows match model='{model_filter}'")

    # Average per panel size (cross-model if unfiltered, single-model if filtered)
    agg = (
        df.groupby(panel_col)
        .agg(
            auroc_mean=(auroc_col, "mean"),
            auroc_std=(auroc_std_col, "mean"),
        )
        .reset_index()
        .sort_values(panel_col)
    )

    sizes = agg[panel_col].astype(int).tolist()
    means = agg["auroc_mean"].values
    stds = agg["auroc_std"].values

    if len(sizes) == 0:
        raise ValueError("No panel sizes found after aggregation")

    # Best panel
    best_idx = int(np.argmax(means))
    best_auroc = means[best_idx]
    best_se = stds[best_idx] / np.sqrt(n_seeds)

    # Criterion 3: marginal gain significance with Holm correction
    marginal_pvals = _marginal_gain_pvalues(means, stds, n_seeds)

    # Holm correction on marginal p-values
    holm_adjusted = _holm_correction(marginal_pvals)

    # Evaluate criteria per size
    audit: dict[str, Any] = {
        "best_panel_size": sizes[best_idx],
        "best_auroc": float(best_auroc),
        "best_se": float(best_se),
        "delta": delta,
        "n_seeds": n_seeds,
        "min_criteria": min_criteria,
        "model_filter": model_filter,
        "per_size": [],
    }

    optimal_p: int | None = None

    for i, p in enumerate(sizes):
        se_p = stds[i] / np.sqrt(n_seeds)

        # Criterion 1: non-inferior (one-sided z-test)
        pooled_se = np.sqrt(best_se**2 + se_p**2)
        z_stat = (best_auroc - means[i] - delta) / pooled_se if pooled_se > 0 else 0.0
        # H0: AUROC(best) - AUROC(p) >= delta  → reject if z < -z_alpha
        c1_pass = bool(z_stat < stats.norm.ppf(0.05))  # one-sided alpha=0.05

        # Criterion 2: within 1 SE of best
        c2_pass = bool(means[i] >= best_auroc - best_se)

        # Criterion 3: marginal gain from p to p+1 is insignificant
        if i < len(sizes) - 1:
            c3_pass = bool(holm_adjusted[i] > 0.05)
        else:
            # Last size: no next size to compare
            c3_pass = True

        n_pass = sum([c1_pass, c2_pass, c3_pass])

        entry = {
            "panel_size": p,
            "auroc_mean": float(means[i]),
            "auroc_std": float(stds[i]),
            "se": float(se_p),
            "c1_noninferior": c1_pass,
            "c2_within_1se": c2_pass,
            "c3_marginal_insignificant": c3_pass,
            "criteria_passed": n_pass,
        }
        if i < len(sizes) - 1:
            entry["marginal_pval_raw"] = float(marginal_pvals[i])
            entry["marginal_pval_holm"] = float(holm_adjusted[i])

        audit["per_size"].append(entry)

        if n_pass >= min_criteria and optimal_p is None:
            optimal_p = p

    if optimal_p is None:
        # Fallback: use the best-performing size
        optimal_p = sizes[best_idx]
        logger.warning(
            "No panel size met >= %d criteria; falling back to best size %d",
            min_criteria,
            optimal_p,
        )

    audit["optimal_panel_size"] = optimal_p
    return optimal_p, audit


def derive_size_stability(
    feature_df: pd.DataFrame,
    *,
    stability_col: str = "stability_freq",
    stability_threshold: float = 1.0,
    sign_col: str = "sign_consistent",
) -> tuple[int, dict[str, Any]]:
    """Derive panel size from feature stability (bootstrap freq + CV sign consistency).

    Selects features that are bootstrap-stable AND sign-consistent.
    Panel size = count of qualifying features.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature consistency data with stability_freq and sign_consistent columns.
    stability_col : str
        Column for bootstrap stability frequency.
    stability_threshold : float
        Minimum stability frequency to include a feature.
    sign_col : str
        Column for CV sign consistency (boolean).

    Returns
    -------
    panel_size : int
        Number of stable + sign-consistent features.
    audit_log : dict
        Details of which features qualified.
    """
    df = feature_df.copy()

    stable_mask = df[stability_col] >= stability_threshold
    sign_mask = df[sign_col].astype(bool)
    qualifying = df[stable_mask & sign_mask]

    panel_size = len(qualifying)

    audit: dict[str, Any] = {
        "stability_threshold": stability_threshold,
        "total_features": len(df),
        "stable_count": int(stable_mask.sum()),
        "sign_consistent_count": int(sign_mask.sum()),
        "qualifying_count": panel_size,
        "qualifying_features": (
            qualifying["protein"].tolist() if "protein" in qualifying.columns else []
        ),
    }

    if panel_size == 0:
        logger.warning("No features meet stability + sign consistency criteria")

    return panel_size, audit


def derive_size_significance_count(
    proteins_df: pd.DataFrame,
    *,
    q_column: str = "bh_adjusted_p",
    q_threshold: float = 0.05,
) -> tuple[int, dict[str, Any]]:
    """Derive panel size = count of proteins passing significance threshold.

    Used for significance-constrained recipes where the panel is exactly
    the set of significant proteins (no more, no less).

    Parameters
    ----------
    proteins_df : pd.DataFrame
        Protein table with q-value column.
    q_column : str
        BH-adjusted q-value column.
    q_threshold : float
        Maximum q-value for inclusion.

    Returns
    -------
    panel_size : int
        Number of significant proteins.
    audit_log : dict
        Significance filter details.
    """
    sig_mask = proteins_df[q_column] <= q_threshold
    panel_size = int(sig_mask.sum())

    audit: dict[str, Any] = {
        "q_column": q_column,
        "q_threshold": q_threshold,
        "total_proteins": len(proteins_df),
        "significant_count": panel_size,
        "significant_proteins": (
            proteins_df.loc[sig_mask, "protein"].tolist()
            if "protein" in proteins_df.columns
            else []
        ),
    }

    if panel_size == 0:
        logger.warning("No proteins pass q <= %.3f", q_threshold)

    return panel_size, audit


def derive_size_fixed(
    *,
    panel_size: int,
) -> tuple[int, dict[str, Any]]:
    """Fixed panel size — no statistical derivation.

    Used as a bootstrap size rule when sweep data are not yet available.
    Replace with three_criterion after the factorial run produces sweep results.

    Parameters
    ----------
    panel_size : int
        Hard-coded panel size to use.

    Returns
    -------
    panel_size : int
    audit_log : dict
    """
    if panel_size < 1:
        raise ValueError(f"panel_size must be >= 1, got {panel_size}")
    audit: dict[str, Any] = {"rule": "fixed", "panel_size": panel_size}
    return panel_size, audit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _marginal_gain_pvalues(means: np.ndarray, stds: np.ndarray, n_seeds: int) -> list[float]:
    """Compute p-values for marginal AUROC gain from size p to p+1.

    Uses a two-sided z-test for the difference in means.
    """
    pvals = []
    for i in range(len(means) - 1):
        diff = means[i + 1] - means[i]
        se_diff = np.sqrt((stds[i] / np.sqrt(n_seeds)) ** 2 + (stds[i + 1] / np.sqrt(n_seeds)) ** 2)
        if se_diff > 0:
            z = diff / se_diff
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p = 1.0
        pvals.append(float(p))
    return pvals


def _holm_correction(pvals: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    n = len(pvals)
    if n == 0:
        return []

    # Sort indices by p-value
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        corrected = min(corrected, 1.0)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = cummax

    return adjusted
