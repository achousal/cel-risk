"""
Orchestration helper functions for aggregate_splits.

This module contains the stage-level orchestration. Each function represents
a logical stage in the aggregation pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ced_ml.data.io_helpers import save_metrics, save_predictions
from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER, METRIC_PRAUC
from ced_ml.evaluation.reports import AggregatedOutputDirectories
from ced_ml.features.importance import aggregate_fold_importances

if TYPE_CHECKING:
    from logging import Logger


def setup_aggregation_directories(results_path: Path) -> AggregatedOutputDirectories:
    """
    Create output directory structure for aggregation results.

    Returns an AggregatedOutputDirectories dataclass with paths to:
    - root: aggregated/ base directory
    - metrics: aggregated/metrics/ (CSVs, JSONs)
    - panels: aggregated/panels/ (consensus panels, feature stability)
    - plots: aggregated/plots/ (all visualizations)
    - cv: aggregated/cv/ (CV artifacts)
    - preds: aggregated/preds/ (pooled predictions)
    - diagnostics: aggregated/diagnostics/ (calibration, DCA, screening)

    Args:
        results_path: Root results directory (e.g., results/{model}/run_{id}/)

    Returns:
        AggregatedOutputDirectories dataclass instance with all directory paths
    """
    return AggregatedOutputDirectories.create(root=str(results_path), exist_ok=True)


def save_pooled_predictions(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    pooled_train_oof_df: pd.DataFrame,
    preds_dir: Path,
    logger: Logger,
) -> None:
    """
    Save pooled predictions to CSV files.

    Saves both combined (all models) and per-model prediction files.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        pooled_train_oof_df: Pooled train OOF predictions
        preds_dir: Directory to save predictions
        logger: Logger instance
    """
    # Test predictions (flattened to preds/)
    if not pooled_test_df.empty:
        save_predictions(pooled_test_df, preds_dir / "pooled_test_preds.csv")

        if "model" in pooled_test_df.columns:
            for model_name, model_df in pooled_test_df.groupby("model"):
                save_predictions(model_df, preds_dir / f"pooled_test_preds__{model_name}.csv")

        logger.info(
            f"Pooled test predictions: {len(pooled_test_df)} samples from "
            f"{pooled_test_df['split_seed'].nunique()} splits, "
            f"{pooled_test_df['model'].nunique() if 'model' in pooled_test_df.columns else 1} model(s)"
        )

    # Validation predictions (flattened to preds/)
    if not pooled_val_df.empty:
        save_predictions(pooled_val_df, preds_dir / "pooled_val_preds.csv")

        if "model" in pooled_val_df.columns:
            for model_name, model_df in pooled_val_df.groupby("model"):
                save_predictions(model_df, preds_dir / f"pooled_val_preds__{model_name}.csv")

        logger.info(
            f"Pooled val predictions: {len(pooled_val_df)} samples from "
            f"{pooled_val_df['split_seed'].nunique()} splits, "
            f"{pooled_val_df['model'].nunique() if 'model' in pooled_val_df.columns else 1} model(s)"
        )

    # Train OOF predictions (flattened to preds/)
    if not pooled_train_oof_df.empty:
        # Compute mean across CV repeats for each split
        # Only overwrite y_prob if it doesn't exist (ENSEMBLE models already have y_prob)
        repeat_cols = [c for c in pooled_train_oof_df.columns if c.startswith("y_prob_repeat")]
        if repeat_cols:
            # For rows with repeat columns, compute mean; for rows without (e.g., ENSEMBLE), preserve existing y_prob
            if "y_prob" not in pooled_train_oof_df.columns:
                pooled_train_oof_df["y_prob"] = pooled_train_oof_df[repeat_cols].mean(axis=1)
            else:
                # Compute mean only for rows where y_prob is NaN (base models)
                mask = pooled_train_oof_df["y_prob"].isna()
                pooled_train_oof_df.loc[mask, "y_prob"] = pooled_train_oof_df.loc[
                    mask, repeat_cols
                ].mean(axis=1)

        save_predictions(pooled_train_oof_df, preds_dir / "pooled_train_oof.csv")

        if "model" in pooled_train_oof_df.columns:
            for model_name, model_df in pooled_train_oof_df.groupby("model"):
                save_predictions(model_df, preds_dir / f"pooled_train_oof__{model_name}.csv")

        logger.info(
            f"Pooled train OOF predictions: {len(pooled_train_oof_df)} samples from "
            f"{pooled_train_oof_df['split_seed'].nunique()} splits, "
            f"{pooled_train_oof_df['model'].nunique() if 'model' in pooled_train_oof_df.columns else 1} model(s)"
        )


def aggregate_importance(
    split_dirs: list[Path],
    model_name: str,
    output_dir: Path,
    logger: Logger,
) -> pd.DataFrame | None:
    """
    Aggregate OOF importance across splits.

    Args:
        split_dirs: List of split directories containing importance CSVs
        model_name: Model name (e.g., "LR_EN")
        output_dir: Output directory for aggregated importance
        logger: Logger instance

    Returns:
        Aggregated importance DataFrame, or None if no importance files found
    """
    importance_files = []
    for split_dir in split_dirs:
        cv_dir = split_dir / "cv"
        importance_file = cv_dir / f"oof_importance__{model_name}.csv"
        if importance_file.exists():
            importance_files.append(importance_file)

    if not importance_files:
        logger.debug(f"No importance files found for {model_name}")
        return None

    logger.info(f"Aggregating importance from {len(importance_files)} splits for {model_name}...")

    fold_dfs = [pd.read_csv(f) for f in importance_files]

    agg_df = aggregate_fold_importances(fold_dfs)

    if agg_df.empty:
        logger.warning(f"Aggregated importance is empty for {model_name}")
        return None

    agg_df["stability"] = agg_df["n_folds_nonzero"] / len(importance_files)

    agg_df = agg_df.sort_values("mean_importance", ascending=False, ignore_index=True)
    agg_df["rank"] = range(1, len(agg_df) + 1)

    output_df = agg_df[["feature", "mean_importance", "std_importance", "stability", "rank"]]

    importance_dir = output_dir / "importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    out_path = importance_dir / f"oof_importance__{model_name}.csv"
    output_df.to_csv(out_path, index=False)
    logger.info(f"Saved aggregated importance to {out_path}")

    return output_df


def aggregate_shap_importance(
    split_dirs: list[Path],
    model_name: str,
    output_dir: Path,
    logger: Logger,
) -> pd.DataFrame | None:
    """
    Aggregate OOF SHAP importance across splits.

    Mirrors aggregate_importance() but for SHAP-based importance.
    Reads per-split oof_shap_importance__{model}.csv files and
    computes cross-split mean/std of mean_abs_shap.

    Args:
        split_dirs: List of split directories containing SHAP importance CSVs
        model_name: Model name (e.g., "LR_EN")
        output_dir: Output directory for aggregated SHAP importance
        logger: Logger instance

    Returns:
        Aggregated SHAP importance DataFrame, or None if no SHAP importance files found
    """
    shap_files = []
    for split_dir in split_dirs:
        cv_dir = split_dir / "cv"
        shap_file = cv_dir / f"oof_shap_importance__{model_name}.csv"
        if shap_file.exists():
            shap_files.append(shap_file)

    if not shap_files:
        logger.debug(f"No SHAP importance files found for {model_name}")
        return None

    logger.info(f"Aggregating SHAP importance from {len(shap_files)} splits for {model_name}...")

    dfs = [pd.read_csv(f) for f in shap_files]

    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby("feature", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        std_abs_shap=("mean_abs_shap", "std"),
        n_splits_present=("mean_abs_shap", "count"),
    )
    agg["stability"] = agg["n_splits_present"] / len(shap_files)
    agg = agg.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    agg["rank_pooled"] = range(1, len(agg) + 1)

    # Stability-weighted rank: penalise features absent from some splits
    agg["_sw_score"] = agg["mean_abs_shap"] * agg["stability"]
    agg["rank_stability"] = agg["_sw_score"].rank(ascending=False, method="min").astype(int)
    agg.drop(columns=["_sw_score"], inplace=True)

    # Keep backward-compat 'rank' alias pointing at pooled ranking
    agg["rank"] = agg["rank_pooled"]

    output_cols = [
        "feature",
        "mean_abs_shap",
        "std_abs_shap",
        "stability",
        "rank",
        "rank_pooled",
        "rank_stability",
    ]
    output_df = agg[output_cols]

    importance_dir = output_dir / "importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    out_path = importance_dir / f"oof_shap_importance__{model_name}.csv"
    output_df.to_csv(out_path, index=False)
    logger.info(f"Saved aggregated SHAP importance to {out_path}")

    return output_df


def aggregate_shap_values(
    split_dirs: list[Path],
    model_name: str,
    output_dir: Path,
    logger: Logger,
) -> pd.DataFrame | None:
    """
    Pool per-split test (and optionally val) SHAP value matrices across splits.

    Reads per-split test_shap_values__{model}.parquet.gz files, appends a
    split_seed column, concatenates, and writes a pooled parquet to
    {output_dir}/shap/. Repeats for val_shap_values if any exist.

    Args:
        split_dirs: List of split directories containing shap/ subdirectories
        model_name: Model name (e.g., "LR_EN")
        output_dir: Aggregation output directory (aggregated/)
        logger: Logger instance

    Returns:
        Pooled test SHAP DataFrame, or None if no files found
    """
    shap_dir_out = output_dir / "shap"

    def _pool_parquets(filename_pattern: str, out_name: str) -> pd.DataFrame | None:
        dfs = []
        for split_dir in split_dirs:
            parquet_path = split_dir / "shap" / filename_pattern
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                seed_str = split_dir.name.replace("split_seed", "")
                # Walk up to find split_seed dir when split_dir is model-level
                if not seed_str.isdigit():
                    seed_str = split_dir.parent.name.replace("split_seed", "")
                df["split_seed"] = int(seed_str)
                dfs.append(df)

        if not dfs:
            return None

        shap_dir_out.mkdir(parents=True, exist_ok=True)
        pooled = pd.concat(dfs, ignore_index=True)
        out_path = shap_dir_out / out_name
        pooled.to_parquet(out_path, compression="gzip")
        logger.info(
            f"Pooled {len(dfs)} splits -> {out_path.name} "
            f"({len(pooled)} samples, {len(pooled.columns) - 1} features)"
        )
        return pooled

    # Pool test SHAP values
    test_pooled = _pool_parquets(
        f"test_shap_values__{model_name}.parquet.gz",
        f"test_shap_values_pooled__{model_name}.parquet.gz",
    )

    # Pool val SHAP values (optional -- only if save_val_shap was enabled)
    _pool_parquets(
        f"val_shap_values__{model_name}.parquet.gz",
        f"val_shap_values_pooled__{model_name}.parquet.gz",
    )

    # Pool transformed feature matrices (for aggregate beeswarm/dependence color axis)
    _pool_parquets(
        f"test_shap_features__{model_name}.parquet.gz",
        f"test_shap_features_pooled__{model_name}.parquet.gz",
    )
    _pool_parquets(
        f"val_shap_features__{model_name}.parquet.gz",
        f"val_shap_features_pooled__{model_name}.parquet.gz",
    )

    # Pool sample metadata (y_true, predictions for aggregate waterfall)
    _pool_parquets(
        f"test_shap_sample_meta__{model_name}.parquet.gz",
        f"test_shap_sample_meta_pooled__{model_name}.parquet.gz",
    )
    _pool_parquets(
        f"val_shap_sample_meta__{model_name}.parquet.gz",
        f"val_shap_sample_meta_pooled__{model_name}.parquet.gz",
    )

    # Diagnostics logging
    _log_shap_diagnostics(test_pooled, model_name, split_dirs, shap_dir_out, logger)

    return test_pooled


def _log_shap_diagnostics(
    pooled_df: pd.DataFrame | None,
    model_name: str,
    split_dirs: list[Path],
    shap_dir_out: Path,
    logger: Logger,
) -> None:
    """Emit diagnostic summary after SHAP aggregation."""
    n_splits = len(split_dirs)
    has_pooled = pooled_df is not None and not pooled_df.empty

    if not has_pooled:
        logger.warning(
            f"[SHAP diag] {model_name}: no pooled SHAP values "
            f"(0/{n_splits} splits had test SHAP parquets)"
        )
        return

    feature_cols = [c for c in pooled_df.columns if c != "split_seed"]
    n_samples = len(pooled_df)
    n_features = len(feature_cols)
    nan_frac = float(pooled_df[feature_cols].isna().mean().mean())
    n_seeds = int(pooled_df["split_seed"].nunique())

    # Feature overlap: intersection / union across seeds
    per_seed_feats = []
    for _seed, grp in pooled_df.groupby("split_seed"):
        nonzero_cols = [c for c in feature_cols if grp[c].notna().any()]
        per_seed_feats.append(set(nonzero_cols))

    if per_seed_feats:
        intersection = set.intersection(*per_seed_feats)
        union = set.union(*per_seed_feats)
        overlap = len(intersection) / len(union) if union else 1.0
    else:
        overlap = 0.0

    has_features = (shap_dir_out / f"test_shap_features_pooled__{model_name}.parquet.gz").exists()
    has_meta = (shap_dir_out / f"test_shap_sample_meta_pooled__{model_name}.parquet.gz").exists()

    logger.info(
        f"[SHAP diag] {model_name}: {n_samples} samples from {n_seeds}/{n_splits} seeds, "
        f"{n_features} features, NaN={nan_frac:.4f}, feature_overlap={overlap:.2f}, "
        f"X_transformed={'yes' if has_features else 'no'}, "
        f"sample_meta={'yes' if has_meta else 'no'}"
    )


def aggregate_shap_metadata(
    split_dirs: list[Path],
    model_name: str,
    output_dir: Path,
    logger: Logger,
) -> dict | None:
    """
    Collect per-split SHAP metadata and write an aggregated summary.

    Reads per-split shap_metadata__{model}.json files, checks consistency
    of output scale and explainer type, and writes a summary JSON to
    {output_dir}/shap/.

    Args:
        split_dirs: List of split directories containing cv/ subdirectories
        model_name: Model name (e.g., "LR_EN")
        output_dir: Aggregation output directory (aggregated/)
        logger: Logger instance

    Returns:
        Aggregated metadata dict, or None if no metadata files found
    """
    meta_entries: list[dict] = []
    for split_dir in split_dirs:
        meta_path = split_dir / "cv" / f"shap_metadata__{model_name}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta_entries.append(json.load(f))

    if not meta_entries:
        logger.debug(f"No SHAP metadata files found for {model_name}")
        return None

    # Check consistency
    scales = {m.get("shap_output_scale", "unknown") for m in meta_entries}
    explainers = {m.get("explainer_type", "unknown") for m in meta_entries}
    model_states = {m.get("explained_model_state", "unknown") for m in meta_entries}

    scale_consistent = len(scales) == 1
    explainer_consistent = len(explainers) == 1

    if not scale_consistent:
        logger.warning(
            f"SHAP output scale inconsistent across splits for {model_name}: {scales}. "
            "Aggregated plots may have misleading axis labels."
        )
    if not explainer_consistent:
        logger.warning(
            f"SHAP explainer type inconsistent across splits for {model_name}: {explainers}"
        )

    # Use first entry as reference for consistent fields
    ref = meta_entries[0]

    summary = {
        "model": model_name,
        "n_splits_total": len(split_dirs),
        "n_splits_with_shap": len(meta_entries),
        "shap_output_scale": ref.get("shap_output_scale", "unknown"),
        "scale_consistent": scale_consistent,
        "scales_observed": sorted(scales),
        "explainer_type": ref.get("explainer_type", "unknown"),
        "explainer_consistent": explainer_consistent,
        "explainers_observed": sorted(explainers),
        "explained_model_state": ref.get("explained_model_state", "unknown"),
        "model_states_observed": sorted(model_states),
        "n_features": ref.get("n_features"),
        "n_background": ref.get("n_background"),
        "background_strategy": ref.get("background_strategy"),
        "raw_dtype": ref.get("raw_dtype"),
    }

    shap_dir_out = output_dir / "shap"
    shap_dir_out.mkdir(parents=True, exist_ok=True)
    out_path = shap_dir_out / f"shap_metadata_summary__{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved aggregated SHAP metadata to {out_path.name}")

    return summary


def compute_and_save_pooled_metrics(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    target_specificity: float,
    control_spec_targets: list[float],
    metrics_dir: Path,
    agg_dir: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, Any]]:
    """
    Compute and save pooled metrics for test and validation sets.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        target_specificity: Target specificity for threshold
        control_spec_targets: List of specificity targets for metrics
        metrics_dir: Directory for aggregated metrics
        agg_dir: Aggregation root directory
        logger: Logger instance

    Returns:
        Tuple of (pooled_test_metrics, pooled_val_metrics, threshold_info)
    """
    from ced_ml.cli.aggregation.aggregation import (
        compute_pooled_metrics_by_model,
        compute_pooled_threshold_metrics,
        save_threshold_data,
    )

    pooled_test_metrics: dict[str, dict[str, float]] = {}
    pooled_val_metrics: dict[str, dict[str, float]] = {}
    threshold_info: dict[str, Any] = {}

    # Detect models
    test_models = (
        pooled_test_df["model"].unique().tolist()
        if not pooled_test_df.empty and "model" in pooled_test_df.columns
        else []
    )
    val_models = (
        pooled_val_df["model"].unique().tolist()
        if not pooled_val_df.empty and "model" in pooled_val_df.columns
        else []
    )
    all_models = sorted(set(test_models + val_models))

    if len(all_models) > 1:
        logger.info(f"Multiple models detected: {', '.join(all_models)}")
    elif all_models:
        logger.info(f"Single model: {all_models[0]}")

    # Test metrics
    if not pooled_test_df.empty:
        pooled_test_metrics = compute_pooled_metrics_by_model(
            pooled_test_df, spec_targets=control_spec_targets
        )

        if pooled_test_metrics:
            metrics_rows = list(pooled_test_metrics.values())
            save_metrics(pd.DataFrame(metrics_rows), metrics_dir / "pooled_test_metrics.csv")

            for model_name, metrics in pooled_test_metrics.items():
                auroc = metrics.get(METRIC_AUROC)
                prauc = metrics.get(METRIC_PRAUC)
                brier = metrics.get(METRIC_BRIER)
                logger.info(
                    f"Pooled test [{model_name}] AUROC: {auroc:.4f}"
                    if auroc is not None
                    else f"Pooled test [{model_name}] AUROC: N/A"
                )
                logger.info(
                    f"Pooled test [{model_name}] PR-AUC: {prauc:.4f}"
                    if prauc is not None
                    else f"Pooled test [{model_name}] PR-AUC: N/A"
                )
                logger.info(
                    f"Pooled test [{model_name}] Brier: {brier:.4f}"
                    if brier is not None
                    else f"Pooled test [{model_name}] Brier: N/A"
                )

        # Threshold info
        for model_name in test_models:
            model_df = pooled_test_df[pooled_test_df["model"] == model_name]
            model_threshold = compute_pooled_threshold_metrics(
                model_df, target_spec=target_specificity, logger=logger
            )
            if model_threshold:
                threshold_info[model_name] = model_threshold
                youden_thr = model_threshold.get("youden_threshold")
                spec_thr = model_threshold.get("spec_target_threshold")
                logger.info(
                    f"Youden threshold [{model_name}]: {youden_thr:.4f}"
                    if youden_thr is not None
                    else f"Youden threshold [{model_name}]: N/A"
                )
                logger.info(
                    f"Target spec threshold [{model_name}] (spec={target_specificity}): {spec_thr:.4f}"
                    if spec_thr is not None
                    else f"Target spec threshold [{model_name}] (spec={target_specificity}): N/A"
                )

        if threshold_info:
            save_threshold_data(threshold_info, agg_dir, logger)

    # Validation metrics
    if not pooled_val_df.empty:
        pooled_val_metrics = compute_pooled_metrics_by_model(
            pooled_val_df, spec_targets=control_spec_targets
        )
        if pooled_val_metrics:
            metrics_rows = list(pooled_val_metrics.values())
            save_metrics(pd.DataFrame(metrics_rows), metrics_dir / "pooled_val_metrics.csv")
            for model_name, metrics in pooled_val_metrics.items():
                val_auroc = metrics.get(METRIC_AUROC)
                logger.info(
                    f"Pooled val [{model_name}] AUROC: {val_auroc:.4f}"
                    if val_auroc is not None
                    else f"Pooled val [{model_name}] AUROC: N/A"
                )

    return pooled_test_metrics, pooled_val_metrics, threshold_info


def collect_sample_categories_metadata(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    pooled_train_oof_df: pd.DataFrame,
) -> dict[str, dict[str, int | None]]:
    """
    Collect sample category breakdowns from pooled predictions.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        pooled_train_oof_df: Pooled OOF predictions

    Returns:
        Dictionary mapping split names to category counts
    """
    sample_categories_metadata: dict[str, dict[str, int | None]] = {}

    for split_name, df in [
        ("test", pooled_test_df),
        ("val", pooled_val_df),
        ("train_oof", pooled_train_oof_df),
    ]:
        if df.empty:
            continue

        if "category" in df.columns:
            cat_counts = df["category"].value_counts().to_dict()
            sample_categories_metadata[split_name] = {
                "controls": int(cat_counts.get("Controls", 0)),
                "incident": int(cat_counts.get("Incident", 0)),
                "prevalent": int(cat_counts.get("Prevalent", 0)),
                "total": len(df),
            }
        else:
            # Fallback: just total count
            sample_categories_metadata[split_name] = {
                "total": len(df),
                "controls": None,
                "incident": None,
                "prevalent": None,
            }

    return sample_categories_metadata


def build_aggregation_metadata(
    n_splits: int,
    split_seeds: list[int],
    all_models: list[str],
    n_boot: int,
    stability_threshold: float,
    target_specificity: float,
    sample_categories_metadata: dict[str, Any],
    pooled_test_metrics: dict[str, dict[str, float]],
    pooled_val_metrics: dict[str, dict[str, float]],
    threshold_info: dict[str, Any],
    feature_stability_df: pd.DataFrame,
    stable_features_df: pd.DataFrame,
    agg_feature_report: pd.DataFrame,
    all_feature_reports: pd.DataFrame,
    consensus_panels: dict[int, dict[str, Any]],
    ensemble_metadata: dict[str, Any],
    agg_dir: Path,
) -> dict[str, Any]:
    """
    Build and save aggregation metadata JSON.

    Args:
        n_splits: Number of splits
        split_seeds: List of split seeds
        all_models: List of all models
        n_boot: Number of bootstrap iterations
        stability_threshold: Feature stability threshold
        target_specificity: Target specificity
        sample_categories_metadata: Sample category counts per split
        pooled_test_metrics: Test metrics by model
        pooled_val_metrics: Validation metrics by model
        threshold_info: Threshold info by model
        feature_stability_df: Feature stability dataframe
        stable_features_df: Stable features dataframe
        agg_feature_report: Aggregated feature report
        all_feature_reports: All feature reports
        consensus_panels: Consensus panel manifests
        ensemble_metadata: Ensemble-specific metadata
        agg_dir: Aggregation directory

    Returns:
        Metadata dictionary
    """
    agg_metadata: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "n_splits": n_splits,
        "split_seeds": split_seeds,
        "models": all_models,
        "n_models": len(all_models),
        "n_boot": n_boot,
        "stability_threshold": stability_threshold,
        "target_specificity": target_specificity,
        "sample_categories": sample_categories_metadata,
        "pooled_metrics": {
            "test": pooled_test_metrics,
            "val": pooled_val_metrics,
        },
        "thresholds": threshold_info,
        "ensemble": ensemble_metadata if ensemble_metadata else None,
        "feature_consensus": {
            "n_features_analyzed": (
                len(feature_stability_df) if not feature_stability_df.empty else 0
            ),
            "n_stable_features": (len(stable_features_df) if not stable_features_df.empty else 0),
            "top_10_features": (
                stable_features_df["protein"].head(10).tolist()
                if not stable_features_df.empty
                else []
            ),
        },
        "feature_reports": {
            "n_proteins_in_reports": (
                len(agg_feature_report) if not agg_feature_report.empty else 0
            ),
            "n_splits_with_reports": (
                all_feature_reports["split_seed"].nunique() if not all_feature_reports.empty else 0
            ),
            "top_10_by_selection_freq": (
                agg_feature_report.head(10)["protein"].tolist()
                if not agg_feature_report.empty
                else []
            ),
        },
        "consensus_panels": {
            str(k): {
                "n_proteins": v["n_consensus_proteins"],
                "n_splits_with_panel": v["n_splits_with_panel"],
            }
            for k, v in consensus_panels.items()
        },
        "files_generated": [f.name for f in agg_dir.rglob("*") if f.is_file()],
    }

    agg_metadata_path = agg_dir / "aggregation_metadata.json"
    with open(agg_metadata_path, "w") as f:
        json.dump(agg_metadata, f, indent=2)

    return agg_metadata


def build_return_summary(
    all_models: list[str],
    pooled_test_metrics: dict[str, dict[str, float]],
    threshold_info: dict[str, Any],
    n_splits: int,
    stable_features_df: pd.DataFrame,
    agg_dir: Path,
) -> dict[str, Any]:
    """
    Build summary dictionary for return value.

    Args:
        all_models: List of all models
        pooled_test_metrics: Test metrics by model
        threshold_info: Threshold info by model
        n_splits: Number of splits
        stable_features_df: Stable features dataframe
        agg_dir: Aggregation directory

    Returns:
        Summary dictionary
    """
    per_model_summary = {}
    for model_name in all_models:
        model_test = pooled_test_metrics.get(model_name, {})
        model_threshold = threshold_info.get(model_name, {})
        per_model_summary[model_name] = {
            "pooled_test_auroc": model_test.get(METRIC_AUROC),
            "pooled_test_prauc": model_test.get(METRIC_PRAUC),
            "youden_threshold": model_threshold.get("youden_threshold"),
        }

    return {
        "status": "success",
        "output_dir": str(agg_dir),
        "n_splits": n_splits,
        "models": all_models,
        "per_model": per_model_summary,
        "n_stable_features": (len(stable_features_df) if not stable_features_df.empty else 0),
    }
