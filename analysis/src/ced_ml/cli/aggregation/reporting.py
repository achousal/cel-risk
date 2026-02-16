"""
Feature reporting aggregation for aggregate-splits.

Handles feature stability analysis and feature report aggregation across splits.
"""

import json
import logging
from pathlib import Path

import pandas as pd


def aggregate_feature_stability(
    split_dirs: list[Path],
    stability_threshold: float = 0.75,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate feature selection across splits.

    Args:
        split_dirs: List of split subdirectory paths
        stability_threshold: Fraction of splits a feature must appear in to be "stable"
        logger: Optional logger instance

    Returns:
        Tuple of (feature_stability_df, stable_features_df)
        - feature_stability_df: All features with selection counts
        - stable_features_df: Features meeting stability threshold
    """
    all_selections = []

    for split_dir in split_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))

        cv_path = split_dir / "cv" / "selected_proteins_per_split.csv"
        if not cv_path.exists():
            if logger:
                logger.debug(f"No selected proteins file in {split_dir.name}")
            continue

        try:
            df = pd.read_csv(cv_path)
            proteins_col = None
            for col in ["selected_proteins_split", "selected_proteins", "proteins"]:
                if col in df.columns:
                    proteins_col = col
                    break

            if proteins_col is None:
                continue

            for _, row in df.iterrows():
                proteins_raw = row[proteins_col]
                if pd.isna(proteins_raw):
                    continue

                # Parse protein list from JSON format (standard from training.py:399 and rfe.py:758)
                try:
                    proteins_list = (
                        json.loads(proteins_raw) if isinstance(proteins_raw, str) else proteins_raw
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    if logger:
                        logger.warning(
                            f"Failed to parse protein list in split {seed}: {proteins_raw[:100]} ({e})"
                        )
                    continue

                for protein in proteins_list:
                    if protein:
                        all_selections.append(
                            {
                                "split_seed": seed,
                                "protein": protein,
                            }
                        )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to read {cv_path}: {e}")

    if not all_selections:
        return pd.DataFrame(), pd.DataFrame()

    selection_df = pd.DataFrame(all_selections)

    n_splits = len(split_dirs)

    protein_counts = (
        selection_df.groupby("protein")["split_seed"]
        .nunique()
        .reset_index()
        .rename(columns={"split_seed": "n_splits_selected"})
    )
    protein_counts["selection_fraction"] = protein_counts["n_splits_selected"] / n_splits
    protein_counts = protein_counts.sort_values("selection_fraction", ascending=False)

    stable_features = protein_counts[
        protein_counts["selection_fraction"] >= stability_threshold
    ].copy()

    return protein_counts, stable_features


def aggregate_feature_reports(
    feature_reports_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Aggregate feature reports across splits.

    Computes mean, std, min, max, and count for selection_freq, effect_size, p_value.

    Args:
        feature_reports_df: DataFrame with feature reports from all splits
        logger: Optional logger instance

    Returns:
        DataFrame with aggregated feature statistics
    """
    if feature_reports_df.empty:
        return pd.DataFrame()

    agg_funcs = {
        "selection_freq": ["mean", "std", "min", "max", "count"],
    }

    if "effect_size" in feature_reports_df.columns:
        agg_funcs["effect_size"] = ["mean", "std"]
    if "p_value" in feature_reports_df.columns:
        agg_funcs["p_value"] = ["mean", "std"]

    agg_df = feature_reports_df.groupby("protein").agg(agg_funcs).reset_index()

    agg_df.columns = [
        "_".join(col).strip("_") if col[1] else col[0] for col in agg_df.columns.values
    ]

    if "selection_freq_count" in agg_df.columns:
        agg_df.rename(columns={"selection_freq_count": "n_splits"}, inplace=True)

    agg_df = agg_df.sort_values("selection_freq_mean", ascending=False).reset_index(drop=True)
    agg_df["rank"] = range(1, len(agg_df) + 1)

    col_order = [
        "rank",
        "protein",
        "selection_freq_mean",
        "selection_freq_std",
        "n_splits",
    ]
    if "effect_size_mean" in agg_df.columns:
        col_order.extend(["effect_size_mean", "effect_size_std"])
    if "p_value_mean" in agg_df.columns:
        col_order.extend(["p_value_mean", "p_value_std"])

    remaining_cols = [c for c in agg_df.columns if c not in col_order]
    col_order.extend(remaining_cols)

    agg_df = agg_df[[c for c in col_order if c in agg_df.columns]]

    return agg_df
