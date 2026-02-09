"""
Aggregation Module: Compute Summary Statistics and Pooled Metrics

This module provides functions for aggregating metrics across splits:
- Summary statistics (mean, std, CI) across split metrics
- Pooled metric computation from concatenated predictions
- Threshold-based metrics on pooled data
- Threshold data persistence

Design Pattern:
- Pure computation functions (no I/O side effects except save_threshold_data)
- Graceful handling of missing/empty data
- Optional logging for diagnostics
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER
from ced_ml.metrics.dca import threshold_dca_zero_crossing
from ced_ml.metrics.discrimination import (
    compute_brier_score,
    compute_discrimination_metrics,
)
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    compute_multi_target_specificity_metrics,
    threshold_for_specificity,
    threshold_youden,
)
from ced_ml.utils.constants import Z_CRITICAL_005


def compute_summary_stats(
    metrics_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    metric_cols: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, std, min, max, count) across splits.

    Args:
        metrics_df: DataFrame with metrics from all splits
        group_cols: Columns to group by (e.g., ["scenario", "model"])
        metric_cols: Numeric columns to summarize (auto-detected if None)
        logger: Optional logger instance

    Returns:
        DataFrame with summary statistics
    """
    if metrics_df.empty:
        if logger:
            logger.debug("No metrics to summarize (empty DataFrame)")
        return pd.DataFrame()

    if group_cols is None:
        group_cols = []
        for col in ["scenario", "model"]:
            if col in metrics_df.columns:
                group_cols.append(col)

    if metric_cols is None:
        exclude = {"split_seed", "scenario", "model", "seed", "random_state"}
        metric_cols = [
            col
            for col in metrics_df.columns
            if metrics_df[col].dtype in [np.float64, np.int64, float, int] and col not in exclude
        ]

    if not group_cols:
        summary_rows = []
        for col in metric_cols:
            values = metrics_df[col].dropna()
            if len(values) > 0:
                summary_rows.append(
                    {
                        "metric": col,
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "count": len(values),
                    }
                )
        return pd.DataFrame(summary_rows)

    summary_rows = []
    for group_vals, group_df in metrics_df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        row = dict(zip(group_cols, group_vals, strict=False))
        row["n_splits"] = len(group_df)

        for col in metric_cols:
            values = group_df[col].dropna()
            if len(values) > 0:
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
                row[f"{col}_ci95_lo"] = values.mean() - Z_CRITICAL_005 * values.std() / np.sqrt(
                    len(values)
                )
                row[f"{col}_ci95_hi"] = values.mean() + Z_CRITICAL_005 * values.std() / np.sqrt(
                    len(values)
                )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if logger:
        logger.info(f"Computed summary stats: {len(summary_df)} groups, {len(metric_cols)} metrics")

    return summary_df


def compute_pooled_metrics(
    pooled_df: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_prob",
    spec_targets: list[float] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Compute metrics on pooled predictions.

    Args:
        pooled_df: DataFrame with pooled predictions
        y_col: Column name for true labels
        pred_col: Column name for predicted probabilities
        spec_targets: List of specificity targets for threshold computation
        logger: Optional logger instance

    Returns:
        Dictionary of computed metrics
    """
    if pooled_df.empty:
        if logger:
            logger.debug("Empty pooled DataFrame, no metrics computed")
        return {}

    # Prefer adjusted probabilities (prevalence-adjusted), fall back to raw
    # Check in priority order
    preferred_cols = ["y_prob_adjusted", "y_prob", "risk_score"]
    actual_pred_col = None
    for col in preferred_cols:
        if col in pooled_df.columns:
            actual_pred_col = col
            break

    if actual_pred_col is None:
        if logger:
            logger.warning(f"No standard prediction columns found in {pooled_df.columns}")
        return {}

    # Log which scale is being used when both are present
    if "y_prob_adjusted" in pooled_df.columns and "y_prob" in pooled_df.columns:
        if logger:
            logger.info(f"Using {actual_pred_col} for pooled metrics (both scales present)")

    y_true = pooled_df[y_col].values
    y_pred = pooled_df[actual_pred_col].values

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {}

    metrics = compute_discrimination_metrics(y_true, y_pred)
    metrics[METRIC_BRIER] = compute_brier_score(y_true, y_pred)
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["prevalence"] = float(y_true.mean())

    # Multi-target specificity metrics
    if spec_targets:
        multi_target_metrics = compute_multi_target_specificity_metrics(
            y_true=y_true, y_pred=y_pred, spec_targets=spec_targets
        )
        metrics.update(multi_target_metrics)

    if logger:
        logger.debug(
            f"Computed pooled metrics: n={len(y_true)}, "
            f"AUROC={metrics.get(METRIC_AUROC, -1):.3f}, "
            f"Brier={metrics.get(METRIC_BRIER, -1):.4f}"
        )

    return metrics


def compute_pooled_metrics_by_model(
    pooled_df: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_prob",
    spec_targets: list[float] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute metrics on pooled predictions, grouped by model.

    Args:
        pooled_df: DataFrame with pooled predictions (must have 'model' column)
        y_col: Column name for true labels
        pred_col: Column name for predicted probabilities
        spec_targets: List of specificity targets for threshold computation
        logger: Optional logger instance

    Returns:
        Dictionary mapping model name to metrics dict
    """
    if pooled_df.empty:
        if logger:
            logger.debug("Empty pooled DataFrame, no model metrics computed")
        return {}

    if "model" not in pooled_df.columns:
        # Fall back to single-model behavior
        if logger:
            logger.debug("No 'model' column found, computing single-model metrics")
        metrics = compute_pooled_metrics(pooled_df, y_col, pred_col, spec_targets, logger=logger)
        return {"unknown": metrics} if metrics else {}

    results = {}
    for model_name, model_df in pooled_df.groupby("model"):
        if logger:
            logger.debug(f"Computing pooled metrics for model: {model_name}")
        metrics = compute_pooled_metrics(model_df, y_col, pred_col, spec_targets, logger=logger)
        if metrics:
            metrics["model"] = model_name
            results[model_name] = metrics

    if logger:
        logger.info(f"Computed pooled metrics for {len(results)} models")

    return results


def compute_pooled_threshold_metrics(
    pooled_df: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_prob",
    target_spec: float = 0.95,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Compute threshold-based metrics from pooled predictions.

    Uses Youden threshold from pooled data.

    Args:
        pooled_df: DataFrame with pooled predictions
        y_col: Column name for true labels
        pred_col: Column name for predicted probabilities
        target_spec: Target specificity for alpha threshold

    Returns:
        Dictionary with thresholds and metrics at each threshold
    """
    if pooled_df.empty:
        return {}

    pred_cols = [
        c
        for c in pooled_df.columns
        if c in ["y_prob", pred_col, "risk_score", "prob", "prediction"]
    ]
    if not pred_cols:
        return {}
    actual_pred_col = pred_cols[0]

    y_true = pooled_df[y_col].values
    y_pred = pooled_df[actual_pred_col].values

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {}

    youden_thr = threshold_youden(y_true, y_pred)
    spec_target_thr = threshold_for_specificity(y_true, y_pred, target_spec=target_spec)
    dca_thr = threshold_dca_zero_crossing(y_true, y_pred)

    youden_metrics = binary_metrics_at_threshold(y_true, y_pred, youden_thr)
    spec_target_metrics = binary_metrics_at_threshold(y_true, y_pred, spec_target_thr)

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    def get_fpr_tpr_at_threshold(threshold: float) -> tuple[float, float]:
        y_hat = (y_pred >= threshold).astype(int)
        tp = np.sum((y_hat == 1) & (y_true == 1))
        fp = np.sum((y_hat == 1) & (y_true == 0))
        fn = np.sum((y_hat == 0) & (y_true == 1))
        tn = np.sum((y_hat == 0) & (y_true == 0))
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return fpr_val, tpr_val

    youden_fpr, youden_tpr = get_fpr_tpr_at_threshold(youden_thr)
    spec_target_fpr, spec_target_tpr = get_fpr_tpr_at_threshold(spec_target_thr)

    result = {
        "youden_threshold": youden_thr,
        "spec_target_threshold": spec_target_thr,
        "target_specificity": target_spec,
        "youden_metrics": {
            "threshold": youden_metrics.threshold,
            "precision": youden_metrics.precision,
            "sensitivity": youden_metrics.sensitivity,
            "f1": youden_metrics.f1,
            "specificity": youden_metrics.specificity,
            "tp": youden_metrics.tp,
            "fp": youden_metrics.fp,
            "tn": youden_metrics.tn,
            "fn": youden_metrics.fn,
            "fpr": youden_fpr,
            "tpr": youden_tpr,
        },
        "spec_target_metrics": {
            "threshold": spec_target_metrics.threshold,
            "precision": spec_target_metrics.precision,
            "sensitivity": spec_target_metrics.sensitivity,
            "f1": spec_target_metrics.f1,
            "specificity": spec_target_metrics.specificity,
            "tp": spec_target_metrics.tp,
            "fp": spec_target_metrics.fp,
            "tn": spec_target_metrics.tn,
            "fn": spec_target_metrics.fn,
            "fpr": spec_target_fpr,
            "tpr": spec_target_tpr,
        },
    }

    if dca_thr is not None:
        dca_metrics = binary_metrics_at_threshold(y_true, y_pred, dca_thr)
        dca_fpr, dca_tpr = get_fpr_tpr_at_threshold(dca_thr)
        result["dca_threshold"] = dca_thr
        result["dca_metrics"] = {
            "threshold": dca_metrics.threshold,
            "precision": dca_metrics.precision,
            "sensitivity": dca_metrics.sensitivity,
            "f1": dca_metrics.f1,
            "specificity": dca_metrics.specificity,
            "tp": dca_metrics.tp,
            "fp": dca_metrics.fp,
            "tn": dca_metrics.tn,
            "fn": dca_metrics.fn,
            "fpr": dca_fpr,
            "tpr": dca_tpr,
        }

    if logger:
        logger.debug(
            f"Computed thresholds: Youden={youden_thr:.4f}, "
            f"SpecTarget(spec={target_spec})={spec_target_thr:.4f}, "
            f"DCA={dca_thr if dca_thr is not None else 'N/A'}"
        )

    return result


def save_threshold_data(
    threshold_info: dict[str, dict[str, Any]],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """
    Save threshold information to CSV files (one per model).

    Args:
        threshold_info: Dictionary mapping model name to threshold data
        out_dir: Output directory for threshold files (expected: core directory)
        logger: Optional logger instance
    """
    if not threshold_info:
        if logger:
            logger.info("No threshold data to save")
        return

    # Save thresholds flat at core level (no thresholds subdirectory)
    core_dir = out_dir / "core"
    core_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_thresholds in threshold_info.items():
        rows = []

        # Youden threshold
        if "youden_threshold" in model_thresholds:
            youden_metrics = model_thresholds.get("youden_metrics", {})
            rows.append(
                {
                    "threshold_type": "youden",
                    "threshold_value": model_thresholds["youden_threshold"],
                    "sensitivity": youden_metrics.get("sensitivity"),
                    "specificity": youden_metrics.get("specificity"),
                    "precision": youden_metrics.get("precision"),
                    "f1": youden_metrics.get("f1"),
                    "fpr": youden_metrics.get("fpr"),
                    "tpr": youden_metrics.get("tpr"),
                    "tp": youden_metrics.get("tp"),
                    "fp": youden_metrics.get("fp"),
                    "tn": youden_metrics.get("tn"),
                    "fn": youden_metrics.get("fn"),
                }
            )

        # Target specificity threshold
        if "spec_target_threshold" in model_thresholds:
            spec_target_metrics = model_thresholds.get("spec_target_metrics", {})
            rows.append(
                {
                    "threshold_type": "spec_target",
                    "threshold_value": model_thresholds["spec_target_threshold"],
                    "target_specificity": model_thresholds.get("target_specificity"),
                    "sensitivity": spec_target_metrics.get("sensitivity"),
                    "specificity": spec_target_metrics.get("specificity"),
                    "precision": spec_target_metrics.get("precision"),
                    "f1": spec_target_metrics.get("f1"),
                    "fpr": spec_target_metrics.get("fpr"),
                    "tpr": spec_target_metrics.get("tpr"),
                    "tp": spec_target_metrics.get("tp"),
                    "fp": spec_target_metrics.get("fp"),
                    "tn": spec_target_metrics.get("tn"),
                    "fn": spec_target_metrics.get("fn"),
                }
            )

        # DCA threshold
        if "dca_threshold" in model_thresholds:
            dca_metrics = model_thresholds.get("dca_metrics", {})
            rows.append(
                {
                    "threshold_type": "dca",
                    "threshold_value": model_thresholds["dca_threshold"],
                    "sensitivity": dca_metrics.get("sensitivity"),
                    "specificity": dca_metrics.get("specificity"),
                    "precision": dca_metrics.get("precision"),
                    "f1": dca_metrics.get("f1"),
                    "fpr": dca_metrics.get("fpr"),
                    "tpr": dca_metrics.get("tpr"),
                    "tp": dca_metrics.get("tp"),
                    "fp": dca_metrics.get("fp"),
                    "tn": dca_metrics.get("tn"),
                    "fn": dca_metrics.get("fn"),
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            csv_path = core_dir / f"thresholds__{model_name}.csv"
            df.to_csv(csv_path, index=False)
            if logger:
                logger.info(f"Thresholds saved for {model_name}: {csv_path}")

    # Also save a combined file with all models
    all_rows = []
    for model_name, model_thresholds in threshold_info.items():
        # Youden
        if "youden_threshold" in model_thresholds:
            youden_metrics = model_thresholds.get("youden_metrics", {})
            all_rows.append(
                {
                    "model": model_name,
                    "threshold_type": "youden",
                    "threshold_value": model_thresholds["youden_threshold"],
                    "sensitivity": youden_metrics.get("sensitivity"),
                    "specificity": youden_metrics.get("specificity"),
                    "precision": youden_metrics.get("precision"),
                    "f1": youden_metrics.get("f1"),
                    "fpr": youden_metrics.get("fpr"),
                    "tpr": youden_metrics.get("tpr"),
                    "tp": youden_metrics.get("tp"),
                    "fp": youden_metrics.get("fp"),
                    "tn": youden_metrics.get("tn"),
                    "fn": youden_metrics.get("fn"),
                }
            )

        # Target specificity
        if "spec_target_threshold" in model_thresholds:
            spec_target_metrics = model_thresholds.get("spec_target_metrics", {})
            all_rows.append(
                {
                    "model": model_name,
                    "threshold_type": "spec_target",
                    "threshold_value": model_thresholds["spec_target_threshold"],
                    "target_specificity": model_thresholds.get("target_specificity"),
                    "sensitivity": spec_target_metrics.get("sensitivity"),
                    "specificity": spec_target_metrics.get("specificity"),
                    "precision": spec_target_metrics.get("precision"),
                    "f1": spec_target_metrics.get("f1"),
                    "fpr": spec_target_metrics.get("fpr"),
                    "tpr": spec_target_metrics.get("tpr"),
                    "tp": spec_target_metrics.get("tp"),
                    "fp": spec_target_metrics.get("fp"),
                    "tn": spec_target_metrics.get("tn"),
                    "fn": spec_target_metrics.get("fn"),
                }
            )

        # DCA
        if "dca_threshold" in model_thresholds:
            dca_metrics = model_thresholds.get("dca_metrics", {})
            all_rows.append(
                {
                    "model": model_name,
                    "threshold_type": "dca",
                    "threshold_value": model_thresholds["dca_threshold"],
                    "sensitivity": dca_metrics.get("sensitivity"),
                    "specificity": dca_metrics.get("specificity"),
                    "precision": dca_metrics.get("precision"),
                    "f1": dca_metrics.get("f1"),
                    "fpr": dca_metrics.get("fpr"),
                    "tpr": dca_metrics.get("tpr"),
                    "tp": dca_metrics.get("tp"),
                    "fp": dca_metrics.get("fp"),
                    "tn": dca_metrics.get("tn"),
                    "fn": dca_metrics.get("fn"),
                }
            )

    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_path = core_dir / "thresholds_all_models.csv"
        combined_df.to_csv(combined_path, index=False)
        if logger:
            logger.info(f"Combined thresholds saved: {combined_path}")
