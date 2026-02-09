"""Per-model ranking for consensus panel generation.

Ranks proteins within a single model using OOF importance as the primary signal,
with stability acting as a hard filter (proteins must meet the stability threshold
before entering this function).

Workflow:
    1. Caller filters stability_df to proteins >= stability_threshold
    2. This function ranks the survivors by OOF importance (descending)
    3. If OOF importance is unavailable, falls back to stability frequency ranking
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_per_model_ranking(
    stability_df: pd.DataFrame,
    stability_col: str = "selection_fraction",
    oof_importance_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute ranking for a single model.

    Ranks proteins by OOF importance (primary). If OOF importance is not
    available, falls back to ranking by stability frequency.

    Args:
        stability_df: DataFrame with columns [protein, {stability_col}].
            Expected to be pre-filtered to proteins meeting the stability
            threshold.
        stability_col: Column name for stability frequency.
        oof_importance_df: DataFrame with OOF grouped importance (columns:
            feature/protein, importance/mean_importance). If None, ranking
            falls back to stability frequency.

    Returns:
        DataFrame with columns:
            - protein: Protein name
            - stability_freq: Selection frequency [0, 1]
            - oof_importance: OOF importance score (NaN if unavailable)
            - oof_rank: Rank by OOF importance (1 = best)
            - final_rank: Same as oof_rank (or stability rank if no OOF)
    """
    _validate_input_dataframe(stability_df, stability_col)

    df = stability_df.copy()
    df = df.rename(columns={stability_col: "stability_freq"})

    # Initialize OOF columns
    df["oof_importance"] = np.nan

    # Map OOF importance to proteins
    has_oof = _map_oof_importance(df, oof_importance_df)

    if has_oof:
        # Rank by OOF importance (higher = better)
        df = df.sort_values("oof_importance", ascending=False, na_position="last")
        df["oof_rank"] = range(1, len(df) + 1)
        df["final_rank"] = df["oof_rank"]
    else:
        # Fallback: rank by stability frequency
        logger.warning(
            "OOF importance not available -- falling back to stability frequency ranking"
        )
        df = df.sort_values("stability_freq", ascending=False)
        df["oof_rank"] = np.nan
        df["final_rank"] = range(1, len(df) + 1)

    output_cols = [
        "protein",
        "stability_freq",
        "oof_importance",
        "oof_rank",
        "final_rank",
    ]

    return df[[c for c in output_cols if c in df.columns]].copy()


def _validate_input_dataframe(df: pd.DataFrame, stability_col: str) -> None:
    """Validate input DataFrame has required columns."""
    if "protein" not in df.columns:
        raise ValueError("stability_df must have 'protein' column")
    if stability_col not in df.columns:
        raise ValueError(f"stability_df must have '{stability_col}' column")


def _map_oof_importance(df: pd.DataFrame, oof_importance_df: pd.DataFrame | None) -> bool:
    """Map OOF importance scores onto the protein DataFrame.

    Detects column naming conventions and maps scores. Modifies df in-place.

    Returns:
        True if OOF importance was successfully mapped to at least one protein.
    """
    if oof_importance_df is None or len(oof_importance_df) == 0:
        return False

    # Detect column names (support both 'feature' and 'protein')
    oof_col = "feature" if "feature" in oof_importance_df.columns else "protein"
    imp_col = "mean_importance" if "mean_importance" in oof_importance_df.columns else "importance"

    oof_lookup = dict(zip(oof_importance_df[oof_col], oof_importance_df[imp_col], strict=False))
    df["oof_importance"] = df["protein"].map(oof_lookup)

    n_mapped = df["oof_importance"].notna().sum()
    if n_mapped == 0:
        return False

    logger.debug(f"Mapped OOF importance to {n_mapped}/{len(df)} proteins")
    return True
