"""Additional artifact generation for aggregation pipeline.

This module handles the generation of additional CSV exports and diagnostic
artifacts for calibration, DCA, screening results, and learning curves.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def generate_calibration_csv(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Generate calibration CSV from pooled predictions.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        out_dir: Output directory (aggregated/)
        logger: Optional logger instance
    """
    try:
        from sklearn.calibration import calibration_curve

        diag_calibration_dir = out_dir / "diagnostics"
        diag_calibration_dir.mkdir(parents=True, exist_ok=True)

        calib_bins = 10  # Match train.py default
        calib_rows = []

        for split_name, df in [("test", pooled_test_df), ("val", pooled_val_df)]:
            if df.empty:
                continue

            pred_col = None
            for col in ["y_prob_adjusted", "y_prob", "y_pred", "risk_score", "prob", "prediction"]:
                if col in df.columns:
                    pred_col = col
                    break

            if pred_col is None or "y_true" not in df.columns:
                continue

            y_true = df["y_true"].values
            y_pred = df[pred_col].values

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask].astype(int)
            y_pred = y_pred[mask].astype(float)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue

            prob_true, prob_pred = calibration_curve(
                y_true, y_pred, n_bins=calib_bins, strategy="uniform"
            )

            for bin_center, obs_freq in zip(prob_pred, prob_true, strict=False):
                calib_rows.append(
                    {
                        "split": split_name,
                        "bin_center": bin_center,
                        "observed_freq": obs_freq,
                        "scenario": "aggregated",
                        "model": "pooled",
                    }
                )

        if calib_rows:
            calib_df = pd.DataFrame(calib_rows)
            calib_csv_path = diag_calibration_dir / "calibration.csv"
            calib_df.to_csv(calib_csv_path, index=False)
            if logger:
                logger.info(f"Calibration CSV saved: {calib_csv_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save calibration CSV: {e}")


def generate_dca_csv(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Generate DCA CSV from pooled predictions.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        out_dir: Output directory (aggregated/)
        logger: Optional logger instance
    """
    try:
        from ced_ml.metrics.dca import save_dca_results

        diag_dca_dir = out_dir / "diagnostics"
        diag_dca_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in [("test", pooled_test_df), ("val", pooled_val_df)]:
            if df.empty:
                continue

            pred_col = None
            for col in ["y_prob_adjusted", "y_prob", "y_pred", "risk_score", "prob", "prediction"]:
                if col in df.columns:
                    pred_col = col
                    break

            if pred_col is None or "y_true" not in df.columns:
                continue

            y_true = df["y_true"].values
            y_pred = df[pred_col].values

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask].astype(int)
            y_pred = y_pred[mask].astype(float)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue

            dca_result = save_dca_results(
                y_true=y_true,
                y_pred_prob=y_pred,
                out_dir=str(diag_dca_dir),
                prefix=f"{split_name}__",
                thresholds=None,
                report_points=None,
                prevalence_adjustment=None,
            )
            if logger:
                logger.info(f"DCA CSV ({split_name}): {dca_result.get('dca_csv_path', 'N/A')}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save DCA CSV: {e}")


def aggregate_screening_results(
    split_dirs: list[Path],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Aggregate screening results across splits.

    Args:
        split_dirs: List of split_seedX directories
        out_dir: Output directory (aggregated/)
        logger: Optional logger instance
    """
    try:
        diag_screening_dir = out_dir / "diagnostics"
        diag_screening_dir.mkdir(parents=True, exist_ok=True)

        # Concat-as-you-go to avoid accumulating all DataFrames in memory (saves ~500MB)
        combined_screening = None
        for split_dir in split_dirs:
            seed = int(split_dir.name.replace("split_seed", ""))
            screening_path = split_dir / "diagnostics"

            if not screening_path.exists():
                continue

            for csv_file in screening_path.glob("*_screening_results.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    df["split_seed"] = seed
                    if combined_screening is None:
                        combined_screening = df
                    else:
                        combined_screening = pd.concat([combined_screening, df], ignore_index=True)
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to read {csv_file}: {e}")

        if combined_screening is not None:
            screening_csv_path = diag_screening_dir / "all_screening_results.csv"
            combined_screening.to_csv(screening_csv_path, index=False)
            if logger:
                logger.info(f"Screening results aggregated: {screening_csv_path}")

            # Compute summary statistics
            if "protein" in combined_screening.columns:
                protein_cols = [
                    c
                    for c in combined_screening.columns
                    if c not in ["split_seed", "scenario", "model", "protein"]
                ]
                if protein_cols:
                    screening_summary = (
                        combined_screening.groupby("protein")[protein_cols]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    screening_summary.columns = [
                        "_".join(col).strip("_") for col in screening_summary.columns
                    ]
                    screening_summary_path = diag_screening_dir / "screening_summary.csv"
                    screening_summary.to_csv(screening_summary_path, index=False)
                    if logger:
                        logger.info(f"Screening summary saved: {screening_summary_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to aggregate screening results: {e}")


def aggregate_learning_curves(
    split_dirs: list[Path],
    out_dir: Path,
    save_plots: bool,
    plot_learning_curve: bool,
    plot_formats: list[str],
    meta_lines: list[str],
    logger: logging.Logger | None = None,
) -> None:
    """Aggregate learning curve results across splits.

    Args:
        split_dirs: List of split_seedX directories
        out_dir: Output directory (aggregated/)
        save_plots: Whether to save plots
        plot_learning_curve: Whether to generate learning curve plots
        plot_formats: List of plot formats (e.g., ["png"])
        meta_lines: Metadata lines for plot annotations
        logger: Optional logger instance
    """
    try:
        # CSVs go to diagnostics/, plots go to plots/
        diag_learning_dir = out_dir / "diagnostics"
        diag_learning_dir.mkdir(parents=True, exist_ok=True)
        diag_plots_dir = out_dir / "plots"
        diag_plots_dir.mkdir(parents=True, exist_ok=True)

        # Note: Keep list for aggregate_learning_curve_runs (needs individual DFs)
        # Memory impact is lower than screening (smaller DataFrames)
        all_learning_curves = []
        for split_dir in split_dirs:
            seed = int(split_dir.name.replace("split_seed", ""))
            # Individual splits store CSVs in diagnostics/ (flattened)
            lc_path = split_dir / "diagnostics"

            if not lc_path.exists():
                continue

            for csv_file in lc_path.glob("*_learning_curve.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    df["split_seed"] = seed
                    df["run_dir"] = split_dir.name
                    all_learning_curves.append(df)
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to read {csv_file}: {e}")

        if all_learning_curves:
            combined_lc = pd.concat(all_learning_curves, ignore_index=True)
            lc_csv_path = diag_learning_dir / "all_learning_curves.csv"
            combined_lc.to_csv(lc_csv_path, index=False)
            if logger:
                logger.info(f"Learning curves aggregated: {lc_csv_path}")

            # Generate learning curve summary plot
            if save_plots and plot_learning_curve:
                try:
                    from ced_ml.plotting.learning_curve import (
                        aggregate_learning_curve_runs,
                        plot_learning_curve_summary,
                    )

                    if "train_size" in combined_lc.columns:
                        # aggregate_learning_curve_runs expects list[pd.DataFrame]
                        agg_lc = aggregate_learning_curve_runs(all_learning_curves)
                        if not agg_lc.empty:
                            # Save aggregated summary CSV to learning_curve dir
                            agg_lc_path = diag_learning_dir / "learning_curve_summary.csv"
                            agg_lc.to_csv(agg_lc_path, index=False)
                            if logger:
                                logger.info(f"Learning curve summary: {agg_lc_path}")

                            # Save plots to diagnostics/plots/
                            for fmt in plot_formats:
                                plot_learning_curve_summary(
                                    df=agg_lc,
                                    out_path=diag_plots_dir / f"learning_curve.{fmt}",
                                    title="Aggregated Learning Curve",
                                    meta_lines=meta_lines,
                                )
                            if logger:
                                logger.info("Learning curve summary plot saved")
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to generate learning curve plot: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to aggregate learning curves: {e}")


def generate_additional_artifacts(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    split_dirs: list[Path],
    out_dir: Path,
    save_plots: bool,
    plot_learning_curve: bool,
    plot_formats: list[str],
    meta_lines: list[str],
    logger: logging.Logger | None = None,
) -> None:
    """Generate all additional diagnostic artifacts.

    This is the main orchestrator function that calls all artifact generators.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        split_dirs: List of split_seedX directories
        out_dir: Output directory (aggregated/)
        save_plots: Whether to save plots
        plot_learning_curve: Whether to generate learning curve plots
        plot_formats: List of plot formats (e.g., ["png"])
        meta_lines: Metadata lines for plot annotations
        logger: Optional logger instance
    """
    if logger:
        logger.info("Generating additional artifacts...")

    # 1. Calibration CSV
    generate_calibration_csv(pooled_test_df, pooled_val_df, out_dir, logger)

    # 2. DCA CSV
    generate_dca_csv(pooled_test_df, pooled_val_df, out_dir, logger)

    # 3. Screening results
    aggregate_screening_results(split_dirs, out_dir, logger)

    # 4. Learning curves
    aggregate_learning_curves(
        split_dirs,
        out_dir,
        save_plots,
        plot_learning_curve,
        plot_formats,
        meta_lines,
        logger,
    )
