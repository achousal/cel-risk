"""
Plot generation for aggregated results.

This module handles generating all diagnostic plots from pooled predictions
across multiple splits, including ROC, PR, calibration, DCA, and risk distribution plots.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER, METRIC_PRAUC
from ced_ml.metrics.thresholds import compute_threshold_bundle


def generate_aggregated_plots(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    pooled_train_oof_df: pd.DataFrame,
    out_dir: Path,
    threshold_info: dict[str, Any],
    plot_formats: list[str],
    meta_lines: list[str] | None = None,
    logger: logging.Logger | None = None,
    plot_roc: bool = True,
    plot_pr: bool = True,
    plot_calibration: bool = True,
    plot_risk_distribution: bool = True,
    plot_dca: bool = True,
    plot_oof_combined: bool = True,
    target_specificity: float = 0.95,
) -> None:
    """
    Generate all aggregated diagnostic plots, separated by model.

    Args:
        pooled_test_df: DataFrame with pooled test predictions
        pooled_val_df: DataFrame with pooled validation predictions
        pooled_train_oof_df: DataFrame with pooled train OOF predictions
        out_dir: Output directory for plots
        threshold_info: Dictionary with threshold information (keyed by model)
        plot_formats: List of plot formats (e.g., ["png", "pdf"])
        meta_lines: Metadata lines to add to plots
        logger: Optional logger instance
        plot_roc: Whether to generate ROC plots
        plot_pr: Whether to generate PR plots
        plot_calibration: Whether to generate calibration plots
        plot_risk_distribution: Whether to generate risk distribution plots
        plot_dca: Whether to generate DCA plots
        plot_oof_combined: Whether to generate OOF combined plots
    """
    try:
        from ced_ml.plotting.calibration import plot_calibration_curve
        from ced_ml.plotting.dca import plot_dca_curve
        from ced_ml.plotting.risk_dist import plot_risk_distribution
        from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve
    except ImportError as e:
        if logger:
            logger.warning(f"Plotting not available: {e}")
        return

    pred_col_names = ["y_prob_adjusted", "y_prob", "y_pred", "risk_score", "prob", "prediction"]

    def get_arrays(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if df.empty:
            return None, None, None, None

        pred_col = None
        for col in pred_col_names:
            if col in df.columns:
                pred_col = col
                break

        if pred_col is None or "y_true" not in df.columns:
            return None, None, None, None

        y_true = df["y_true"].values
        y_pred = df[pred_col].values
        split_ids = df["split_seed"].values if "split_seed" in df.columns else None
        category = df["category"].values if "category" in df.columns else None

        return y_true, y_pred, split_ids, category

    # Detect models present in the data
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
    train_models = (
        pooled_train_oof_df["model"].unique().tolist()
        if not pooled_train_oof_df.empty and "model" in pooled_train_oof_df.columns
        else []
    )
    all_models = sorted(set(test_models + val_models + train_models))

    if not all_models:
        all_models = ["unknown"]

    if logger:
        logger.info(f"Generating plots for {len(all_models)} model(s): {', '.join(all_models)}")

    # Generate plots for each model separately
    for model_name in all_models:
        if logger:
            logger.info(f"Generating plots for model: {model_name}")

        # Create output directories (model name not needed since parent folder already specifies it)
        model_plots_dir = out_dir / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)

        # Filter data for this model
        model_test_df = (
            pooled_test_df[pooled_test_df["model"] == model_name]
            if not pooled_test_df.empty and "model" in pooled_test_df.columns
            else pooled_test_df
        )
        model_val_df = (
            pooled_val_df[pooled_val_df["model"] == model_name]
            if not pooled_val_df.empty and "model" in pooled_val_df.columns
            else pooled_val_df
        )
        model_train_oof_df = (
            pooled_train_oof_df[pooled_train_oof_df["model"] == model_name]
            if not pooled_train_oof_df.empty and "model" in pooled_train_oof_df.columns
            else pooled_train_oof_df
        )

        # Get model-specific threshold info
        model_threshold_info = threshold_info.get(model_name, {})
        dca_thr = model_threshold_info.get("dca_threshold")
        youden_metrics = model_threshold_info.get("youden_metrics", {})
        spec_target_metrics = model_threshold_info.get("spec_target_metrics", {})
        dca_metrics = model_threshold_info.get("dca_metrics", {})

        metrics_at_thresholds = {}
        if youden_metrics:
            metrics_at_thresholds["youden"] = {
                "fpr": youden_metrics.get("fpr"),
                "tpr": youden_metrics.get("tpr"),
            }
        if spec_target_metrics:
            metrics_at_thresholds["spec_target"] = {
                "fpr": spec_target_metrics.get("fpr"),
                "tpr": spec_target_metrics.get("tpr"),
            }
        if dca_metrics:
            metrics_at_thresholds["dca"] = {
                "fpr": dca_metrics.get("fpr"),
                "tpr": dca_metrics.get("tpr"),
            }

        # Add model name to metadata lines
        model_meta_lines = (meta_lines or []) + [f"Model: {model_name}"]

        # Generate test/val plots
        for data_name, df in [("test", model_test_df), ("val", model_val_df)]:
            y_true, y_pred, split_ids, category = get_arrays(df)
            if y_true is None:
                if logger:
                    logger.debug(f"Skipping {data_name} plots for {model_name}: no valid data")
                continue

            if logger:
                logger.info(f"Generating aggregated {data_name} plots for {model_name}")

            # Compute threshold bundle (standardized interface) - always compute fresh
            # for aggregated data to ensure consistency
            local_bundle = compute_threshold_bundle(
                y_true,
                y_pred,
                target_spec=target_specificity,
                dca_threshold=dca_thr,
            )

            # Ensemble models: skip 95% CI band (only use SD) since CI and SD are redundant
            # when there are no CV repeats per split
            skip_ci_band = model_name == "ENSEMBLE"

            for fmt in plot_formats:
                if plot_roc:
                    plot_roc_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=model_plots_dir / f"{data_name}_roc.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set ROC - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        threshold_bundle=local_bundle,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_pr:
                    plot_pr_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=model_plots_dir / f"{data_name}_pr.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set PR Curve - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_calibration:
                    plot_calibration_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=model_plots_dir / f"{data_name}_calibration.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set Calibration - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_dca:
                    plot_dca_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=str(model_plots_dir / f"{data_name}_dca.{fmt}"),
                        title=f"Aggregated {data_name.capitalize()} Set DCA - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_risk_distribution:
                    plot_risk_distribution(
                        y_true=y_true,
                        scores=y_pred,
                        out_path=model_plots_dir / f"{data_name}_risk_dist.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set Risk Distribution - {model_name}",
                        category_col=category,
                        threshold_bundle=local_bundle,
                        meta_lines=model_meta_lines,
                    )

        # Generate train OOF plots if available
        if not model_train_oof_df.empty:
            y_true_train, y_pred_train, _, category_train = get_arrays(model_train_oof_df)
            if y_true_train is not None:
                if logger:
                    logger.info(f"Generating aggregated train OOF plots for {model_name}")

                # Compute threshold bundle for OOF data
                oof_bundle = compute_threshold_bundle(
                    y_true_train,
                    y_pred_train,
                    target_spec=target_specificity,
                    dca_threshold=dca_thr,
                )

                for fmt in plot_formats:
                    if plot_risk_distribution:
                        plot_risk_distribution(
                            y_true=y_true_train,
                            scores=y_pred_train,
                            out_path=model_plots_dir / f"train_oof_risk_dist.{fmt}",
                            title=f"Aggregated Train OOF Risk Distribution - {model_name}",
                            category_col=category_train,
                            threshold_bundle=oof_bundle,
                            meta_lines=model_meta_lines,
                        )

                    # Generate OOF combined plots (ROC, PR, Calibration)
                    if plot_oof_combined:
                        try:
                            from ced_ml.plotting.oof import plot_oof_combined

                            # For OOF plots, we need to use the per-repeat predictions
                            repeat_cols = [
                                c
                                for c in model_train_oof_df.columns
                                if c.startswith("y_prob_repeat")
                            ]

                            if repeat_cols:
                                # Get unique sample indices from FIRST split only
                                # (all splits have identical train set indices, so we just need one)
                                first_seed = model_train_oof_df["split_seed"].iloc[0]
                                seed_df = model_train_oof_df[
                                    model_train_oof_df["split_seed"] == first_seed
                                ]

                                unique_idx = seed_df["idx"].unique()
                                n_samples = len(unique_idx)
                                n_repeats = len(repeat_cols)

                                # Create oof_preds array (n_repeats x n_samples)
                                oof_preds = np.full((n_repeats, n_samples), np.nan)
                                y_true_oof = np.zeros(n_samples)

                                # Map idx to position
                                idx_to_pos = {idx: pos for pos, idx in enumerate(unique_idx)}

                                # Fill in predictions from first split
                                for _, row in seed_df.iterrows():
                                    pos = idx_to_pos[row["idx"]]
                                    y_true_oof[pos] = row["y_true"]
                                    for repeat_idx, col in enumerate(repeat_cols):
                                        oof_preds[repeat_idx, pos] = row[col]

                                # Validate we have valid data (should never be all NaN)
                                if np.all(np.isnan(oof_preds)):
                                    if logger:
                                        logger.warning(
                                            f"Skipping OOF combined plots for {model_name}: "
                                            f"all OOF predictions are NaN"
                                        )
                                    continue

                                plot_oof_combined(
                                    y_true=y_true_oof,
                                    oof_preds=oof_preds,
                                    out_dir=model_plots_dir,
                                    model_name=model_name,
                                    plot_format=fmt,
                                    calib_bins=10,
                                    meta_lines=model_meta_lines,
                                )
                        except Exception as e:
                            if logger:
                                logger.warning(
                                    f"Failed to generate OOF combined plots for {model_name}: {e}"
                                )


def generate_model_comparison_report(
    pooled_test_metrics: dict[str, dict[str, float]],
    pooled_val_metrics: dict[str, dict[str, float]],
    threshold_info: dict[str, dict[str, Any]],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Generate a model comparison report comparing all models including ENSEMBLE.

    Args:
        pooled_test_metrics: Dict mapping model name to test metrics
        pooled_val_metrics: Dict mapping model name to val metrics
        threshold_info: Dict mapping model name to threshold information
        out_dir: Output directory for the report
        logger: Optional logger instance

    Returns:
        DataFrame with model comparison data
    """
    if not pooled_test_metrics:
        return pd.DataFrame()

    rows = []
    for model_name in sorted(pooled_test_metrics.keys()):
        test_metrics = pooled_test_metrics.get(model_name, {})
        val_metrics = pooled_val_metrics.get(model_name, {})
        model_threshold = threshold_info.get(model_name, {})

        row = {
            "model": model_name,
            "is_ensemble": model_name == "ENSEMBLE",
            # Test metrics
            "test_AUROC": test_metrics.get(METRIC_AUROC),
            "test_PR_AUC": test_metrics.get(METRIC_PRAUC),
            "test_Brier": test_metrics.get(METRIC_BRIER),
            "test_n_samples": test_metrics.get("n_samples"),
            "test_n_positive": test_metrics.get("n_positive"),
            "test_prevalence": test_metrics.get("prevalence"),
            # Calibration
            "test_calib_slope": test_metrics.get("calib_slope"),
            # Val metrics
            "val_AUROC": val_metrics.get(METRIC_AUROC),
            "val_PR_AUC": val_metrics.get(METRIC_PRAUC),
            "val_Brier": val_metrics.get(METRIC_BRIER),
            # Thresholds
            "youden_threshold": model_threshold.get("youden_threshold"),
            "spec_target_threshold": model_threshold.get("spec_target_threshold"),
            "dca_threshold": model_threshold.get("dca_threshold"),
        }

        # Add multi-target specificity metrics if present
        for key in test_metrics:
            if key.startswith("sens_ctrl_") or key.startswith("prec_ctrl_"):
                row[f"test_{key}"] = test_metrics[key]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by test AUROC descending
    if "test_AUROC" in df.columns:
        df = df.sort_values("test_AUROC", ascending=False)

    # Save to metrics directory
    comparison_dir = out_dir / "metrics"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    report_path = comparison_dir / "model_comparison.csv"
    df.to_csv(report_path, index=False)

    if logger:
        logger.info(f"Model comparison report saved: {report_path}")

        # Log summary of comparison
        if "ENSEMBLE" in df["model"].values:
            ensemble_row = df[df["model"] == "ENSEMBLE"].iloc[0]
            best_base = df[df["model"] != "ENSEMBLE"].iloc[0] if len(df) > 1 else None

            if best_base is not None:
                ensemble_auroc = ensemble_row.get("test_AUROC")
                best_base_auroc = best_base.get("test_AUROC")
                if ensemble_auroc is not None and best_base_auroc is not None:
                    improvement = (ensemble_auroc - best_base_auroc) * 100
                    logger.info(
                        f"ENSEMBLE vs best base ({best_base['model']}): "
                        f"AUROC {ensemble_auroc:.4f} vs {best_base_auroc:.4f} "
                        f"({improvement:+.2f} pp)"
                    )

    return df
