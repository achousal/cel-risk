"""Ordering dispatch: resolve a RecipeConfig's ordering rule to a ranked protein list.

Each function takes source data and returns proteins in the desired order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ced_ml.recipes.streams import discover_streams, interleave_round_robin

logger = logging.getLogger(__name__)


def consensus_score_descending(
    proteins_df: pd.DataFrame,
    *,
    score_column: str = "observed_rra",
    q_column: str = "bh_adjusted_p",
    q_threshold: float | None = 0.05,
) -> list[str]:
    """Order proteins by consensus score descending, optionally filtering by q-value.

    Parameters
    ----------
    proteins_df : pd.DataFrame
        RRA significance table with protein, score, and q-value columns.
    score_column : str
        Column with consensus score (higher = more important).
    q_column : str
        Column with BH-adjusted p-value.
    q_threshold : float or None
        Maximum q-value for inclusion. None = no filter (use all proteins).

    Returns
    -------
    list[str]
        Proteins ordered by score descending.
    """
    df = proteins_df.copy()

    if q_threshold is not None:
        df = df[df[q_column] <= q_threshold]
        if df.empty:
            logger.warning("No proteins pass q <= %.3f", q_threshold)
            return []
        label = f"q <= {q_threshold:.3f}"
    else:
        label = "unfiltered"

    ordered = df.sort_values(score_column, ascending=False)
    result = ordered["protein"].tolist()
    logger.info("consensus_score_descending: %d proteins (%s)", len(result), label)
    return result


def stream_balanced(
    proteins_df: pd.DataFrame,
    data_df: pd.DataFrame,
    *,
    score_column: str = "observed_rra",
    q_column: str = "bh_adjusted_p",
    q_threshold: float | None = 0.05,
    corr_threshold: float = 0.50,
    corr_method: str = "spearman",
) -> list[str]:
    """Stream-balanced ordering: cluster → within-stream rank → round-robin.

    Parameters
    ----------
    proteins_df : pd.DataFrame
        RRA significance table.
    data_df : pd.DataFrame
        Training data matrix for correlation computation.
    score_column : str
        Consensus score column.
    q_column : str
        BH-adjusted q-value column.
    q_threshold : float or None
        Significance threshold. None = no filter.
    corr_threshold : float
        Correlation threshold for stream discovery.
    corr_method : str
        Correlation method.

    Returns
    -------
    list[str]
        Proteins in stream-balanced order.
    """
    # Filter to significant proteins (or use all)
    if q_threshold is not None:
        sig = proteins_df[proteins_df[q_column] <= q_threshold]
        if sig.empty:
            logger.warning("No proteins pass q <= %.3f", q_threshold)
            return []
    else:
        sig = proteins_df

    protein_list = sig["protein"].tolist()
    scores = dict(zip(sig["protein"], sig[score_column], strict=False))

    # Discover streams
    streams = discover_streams(
        data_df,
        protein_list,
        threshold=corr_threshold,
        method=corr_method,
    )

    # Round-robin interleave
    result = interleave_round_robin(streams, scores)
    logger.info(
        "stream_balanced: %d proteins across %d streams",
        len(result),
        len(streams),
    )
    return result


def abs_coefficient_descending(
    feature_df: pd.DataFrame,
    *,
    coef_column: str = "mean_coef",
    stability_column: str = "stability_freq",
    stability_threshold: float = 1.0,
    sign_column: str = "sign_consistent",
) -> list[str]:
    """Order by |coefficient| descending among stable + sign-consistent features.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature consistency data with coefficient and stability columns.
    coef_column : str
        Mean coefficient column (absolute value used for ranking).
    stability_column : str
        Bootstrap stability frequency column.
    stability_threshold : float
        Minimum stability frequency for inclusion.
    sign_column : str
        CV sign consistency column (boolean).

    Returns
    -------
    list[str]
        Proteins ordered by |coefficient| descending.
    """
    df = feature_df.copy()

    stable = df[df[stability_column] >= stability_threshold]
    consistent = stable[stable[sign_column].astype(bool)]

    if consistent.empty:
        logger.warning(
            "No features meet stability >= %.2f + sign_consistent",
            stability_threshold,
        )
        return []

    consistent = consistent.copy()
    consistent["abs_coef"] = consistent[coef_column].abs()
    ordered = consistent.sort_values("abs_coef", ascending=False)
    result = ordered["protein"].tolist()
    logger.info("abs_coefficient_descending: %d proteins", len(result))
    return result


def oof_importance(
    importance_csv: str | Path,
) -> list[str]:
    """Order proteins by OOF permutation importance (model-specific).

    Parameters
    ----------
    importance_csv : str or Path
        Path to oof_importance__{model}.csv with columns: feature, rank.

    Returns
    -------
    list[str]
        Proteins ordered by importance rank ascending (rank 1 = most important).
    """
    csv_path = Path(importance_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"OOF importance file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ordered = df.sort_values("rank", ascending=True)
    result = ordered["feature"].tolist()
    logger.info("oof_importance: %d proteins from %s", len(result), csv_path.name)
    return result


def rfe_elimination(
    rfe_csv: str | Path,
) -> list[str]:
    """Order proteins by RFE elimination order (model-specific).

    Last eliminated = most important (highest elimination_order).

    Parameters
    ----------
    rfe_csv : str or Path
        Path to rfe_feature_report_aggregated.csv with columns: protein, rank.

    Returns
    -------
    list[str]
        Proteins ordered by RFE rank ascending (rank 1 = last eliminated = most important).
    """
    from pathlib import Path

    csv_path = Path(rfe_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"RFE report file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ordered = df.sort_values("rank", ascending=True)
    result = ordered["protein"].tolist()
    logger.info("rfe_elimination: %d proteins from %s", len(result), csv_path.name)
    return result


def dispatch_ordering(
    ordering_type: str,
    proteins_df: pd.DataFrame,
    params: dict[str, Any],
    data_df: pd.DataFrame | None = None,
) -> list[str]:
    """Dispatch to the appropriate ordering function.

    Parameters
    ----------
    ordering_type : str
        One of: consensus_score_descending, stream_balanced, abs_coefficient_descending.
    proteins_df : pd.DataFrame
        Protein source data.
    params : dict
        Parameters passed to the ordering function.
    data_df : pd.DataFrame, optional
        Training data (required for stream_balanced).

    Returns
    -------
    list[str]
        Ordered protein list.
    """
    if ordering_type == "consensus_score_descending":
        return consensus_score_descending(proteins_df, **params)
    elif ordering_type == "stream_balanced":
        if data_df is None:
            raise ValueError("stream_balanced ordering requires data_df")
        return stream_balanced(proteins_df, data_df, **params)
    elif ordering_type == "abs_coefficient_descending":
        return abs_coefficient_descending(proteins_df, **params)
    elif ordering_type == "oof_importance":
        return oof_importance(**params)
    elif ordering_type == "rfe_elimination":
        return rfe_elimination(**params)
    else:
        raise ValueError(f"Unknown ordering type: {ordering_type}")
