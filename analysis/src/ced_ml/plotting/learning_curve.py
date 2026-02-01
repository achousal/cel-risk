"""Learning curve computation and plotting.

This module provides functions for generating learning curves that show
model performance as a function of training set size.

Functions:
    compute_learning_curve: Compute learning curve data using sklearn's learning_curve
    save_learning_curve_csv: Compute and save learning curve data to CSV
    plot_learning_curve: Generate single-run learning curve plot
    aggregate_learning_curve_runs: Aggregate learning curves across multiple runs
    plot_learning_curve_summary: Plot aggregated learning curve with CIs
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve

from ced_ml.utils.constants import CI_LOWER_PCT, CI_UPPER_PCT

from .dca import apply_plot_metadata
from .style import (
    BBOX_INCHES,
    COLOR_FILL_ALPHA,
    COLOR_PRIMARY,
    COLOR_REFERENCE,
    COLOR_SECONDARY,
    DPI,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_PRIMARY,
    LW_REFERENCE,
    MARKER_SIZE_SMALL,
    PAD_INCHES,
    configure_backend,
)

logger = logging.getLogger(__name__)


def compute_learning_curve(
    estimator,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    scoring: str,
    cv: int = 5,
    min_frac: float = 0.3,
    n_points: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute learning curve using sklearn's learning_curve.

    Args:
        estimator: Sklearn estimator (pipeline or model)
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        scoring: Scoring metric (e.g., 'neg_brier_score', 'roc_auc')
        cv: Number of CV folds
        min_frac: Minimum fraction of training set (e.g., 0.3 = 30%)
        n_points: Number of points on learning curve
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_sizes, train_scores, val_scores)
        - train_sizes: (n_points,) array of training set sizes
        - train_scores: (n_points, cv) array of train scores
        - val_scores: (n_points, cv) array of validation scores
    """
    train_sizes = np.linspace(float(min_frac), 1.0, int(n_points))

    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=int(seed)),
        scoring=scoring,
        n_jobs=1,
    )

    return sizes, train_scores, val_scores


def _normalize_metric_scores(
    scoring: str,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
) -> tuple[str, bool, np.ndarray, np.ndarray]:
    """Normalize metric scores and extract metric metadata.

    Args:
        scoring: Scoring metric name
        train_scores: Training scores
        val_scores: Validation scores

    Returns:
        Tuple of (metric_label, metric_is_error, train_scores_norm, val_scores_norm)
    """
    metric_label = scoring
    metric_is_error = False

    if str(scoring).startswith("neg_"):
        metric_label = str(scoring).replace("neg_", "", 1)
        metric_is_error = True
        train_scores = -train_scores
        val_scores = -val_scores

    if (train_scores < 0).any() or (val_scores < 0).any():
        logger.warning(
            f"[learning_curve] WARNING: negative values remain after metric normalization ({scoring})."
        )

    return metric_label, metric_is_error, train_scores, val_scores


def save_learning_curve_csv(
    estimator,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    out_csv: Path,
    scoring: str,
    cv: int = 5,
    min_frac: float = 0.3,
    n_points: int = 5,
    seed: int = 0,
    out_plot: Path | None = None,
    meta_lines: Sequence[str] | None = None,
) -> None:
    """Compute learning curve and save results to CSV.

    Args:
        estimator: Sklearn estimator (pipeline or model)
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        out_csv: Output CSV path
        scoring: Scoring metric (e.g., 'neg_brier_score', 'roc_auc')
        cv: Number of CV folds
        min_frac: Minimum fraction of training set
        n_points: Number of points on learning curve
        seed: Random seed
        out_plot: Optional output path for plot
        meta_lines: Optional metadata lines for plot annotation
    """
    logger.info(f"Generating learning curve ({n_points} training sizes, {cv}-fold CV)")

    # Reduce screening log verbosity during learning curve generation
    # by temporarily elevating the screening module's logger level
    screening_logger = logging.getLogger("ced_ml.features.screening")
    old_level = screening_logger.level
    screening_logger.setLevel(logging.WARNING)
    try:
        sizes, train_scores, val_scores = compute_learning_curve(
            estimator, X, y, scoring, cv, min_frac, n_points, seed
        )
    finally:
        screening_logger.setLevel(old_level)

    metric_label, metric_is_error, train_scores, val_scores = _normalize_metric_scores(
        scoring, train_scores, val_scores
    )

    train_mean = train_scores.mean(axis=1)
    train_sd = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_sd = val_scores.std(axis=1)

    n_sizes, n_splits = val_scores.shape
    rows = []
    for i in range(n_sizes):
        for split_idx in range(n_splits):
            rows.append(
                {
                    "train_size": int(sizes[i]),
                    "cv_split": int(split_idx),
                    "train_score": float(train_scores[i, split_idx]),
                    "val_score": float(val_scores[i, split_idx]),
                    "train_score_mean": float(train_mean[i]),
                    "train_score_sd": float(train_sd[i]),
                    "val_score_mean": float(val_mean[i]),
                    "val_score_sd": float(val_sd[i]),
                    "scoring": str(scoring),
                    "error_metric": str(metric_label),
                    "metric_direction": (
                        "lower_is_better" if metric_is_error else "higher_is_better"
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if out_plot:
        try:
            plot_learning_curve(
                sizes,
                train_scores,
                val_scores,
                out_plot,
                metric_label,
                metric_is_error,
                meta_lines=meta_lines,
            )
            logger.info(f"[plot] Saved learning curve plot: {out_plot}")
        except Exception as e:
            logger.warning(f"[plot] WARNING: Failed to generate learning curve plot: {e}")


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    out_path: Path,
    metric_label: str,
    metric_is_error: bool,
    meta_lines: Sequence[str] | None = None,
) -> None:
    """Generate a learning curve plot with per-split scatter and mean lines.

    Args:
        train_sizes: (n_points,) array of training set sizes
        train_scores: (n_points, n_splits) array of train scores
        val_scores: (n_points, n_splits) array of validation scores
        out_path: Output plot path
        metric_label: Metric name for y-axis
        metric_is_error: Whether metric is error-based (lower is better)
        meta_lines: Optional metadata lines for plot annotation
    """

    configure_backend()
    import matplotlib.pyplot as plt

    train_sizes = np.asarray(train_sizes)
    train_scores = np.asarray(train_scores)
    val_scores = np.asarray(val_scores)

    if train_scores.ndim != 2 or val_scores.ndim != 2:
        return

    n_sizes, n_splits = val_scores.shape
    if n_sizes == 0 or n_splits == 0:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Scatter validation scores across splits (main focus)
    for split_idx in range(n_splits):
        label = "Validation split" if split_idx == 0 else None
        ax.scatter(
            train_sizes,
            val_scores[:, split_idx],
            color=COLOR_PRIMARY,
            alpha=0.35,
            s=MARKER_SIZE_SMALL,
            label=label,
        )

    # Mean lines + shaded variability
    val_mean = val_scores.mean(axis=1)
    val_sd = val_scores.std(axis=1)
    train_mean = train_scores.mean(axis=1)

    # Plot validation SD band (main information)
    ax.fill_between(
        train_sizes,
        val_mean - val_sd,
        val_mean + val_sd,
        color=COLOR_PRIMARY,
        alpha=COLOR_FILL_ALPHA,
        label="Validation +/-1 SD",
    )

    # Plot validation mean line (main information)
    ax.plot(
        train_sizes,
        val_mean,
        color=COLOR_PRIMARY,
        linestyle="-",
        linewidth=LW_PRIMARY,
        alpha=0.8,
        label="Validation mean",
    )

    # Plot train mean as thin reference line (minimal visual weight)
    ax.plot(
        train_sizes,
        train_mean,
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=LW_REFERENCE,
        alpha=0.6,
        label="Train (reference)",
    )

    metric_text = metric_label.replace("_", " ").upper()
    if metric_is_error:
        metric_text += " (LOWER IS BETTER)"

    ax.set_xlabel("Training samples", fontsize=FONT_LABEL)
    ax.set_ylabel(metric_text, fontsize=FONT_LABEL)
    ax.set_title("Learning Curve", fontsize=FONT_TITLE)
    ax.grid(True, alpha=GRID_ALPHA)

    # Set x-axis ticks and labels to training sizes
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(
        [str(int(size)) for size in train_sizes], rotation=45, ha="right", fontsize=FONT_LEGEND
    )

    ax.legend(loc="best", fontsize=FONT_LEGEND)

    # Apply metadata annotation to bottom of figure
    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES)
    plt.close()


def aggregate_learning_curve_runs(lc_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate learning curves across multiple runs.

    Args:
        lc_frames: List of learning curve DataFrames from different runs

    Returns:
        Aggregated DataFrame with mean, SD, and 95% CI across runs
    """
    if not lc_frames:
        return pd.DataFrame()

    per_run = []
    metric_label = ""
    metric_direction = ""
    scoring = ""

    for df in lc_frames:
        if df is None or df.empty or "train_size" not in df.columns:
            continue
        run_id = df.get("run_dir")
        run_label = run_id.iloc[0] if run_id is not None else "run"

        if "train_score_mean" in df.columns and "val_score_mean" in df.columns:
            agg = df.groupby("train_size", as_index=False).agg(
                train_mean=("train_score_mean", "mean"),
                val_mean=("val_score_mean", "mean"),
            )
        elif "train_score" in df.columns and "val_score" in df.columns:
            agg = df.groupby("train_size", as_index=False).agg(
                train_mean=("train_score", "mean"), val_mean=("val_score", "mean")
            )
        else:
            continue

        agg["run"] = run_label
        per_run.append(agg)

        if not metric_label and "error_metric" in df.columns:
            metric_label = str(df["error_metric"].iloc[0])
        if not metric_direction and "metric_direction" in df.columns:
            metric_direction = str(df["metric_direction"].iloc[0])
        if not scoring and "scoring" in df.columns:
            scoring = str(df["scoring"].iloc[0])

    if not per_run:
        return pd.DataFrame()

    all_df = pd.concat(per_run, ignore_index=True)

    def _ci_lo(x):
        return float(np.percentile(x, CI_LOWER_PCT)) if len(x) > 1 else np.nan

    def _ci_hi(x):
        return float(np.percentile(x, CI_UPPER_PCT)) if len(x) > 1 else np.nan

    summary = all_df.groupby("train_size", as_index=False).agg(
        train_mean=("train_mean", "mean"),
        train_sd=("train_mean", "std"),
        train_ci_lo=("train_mean", _ci_lo),
        train_ci_hi=("train_mean", _ci_hi),
        val_mean=("val_mean", "mean"),
        val_sd=("val_mean", "std"),
        val_ci_lo=("val_mean", _ci_lo),
        val_ci_hi=("val_mean", _ci_hi),
        n_runs=("run", "nunique"),
    )
    summary["metric_label"] = metric_label or scoring
    summary["metric_direction"] = metric_direction
    summary["scoring"] = scoring
    return summary


def plot_learning_curve_summary(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    meta_lines: Sequence[str] | None = None,
) -> None:
    """Plot aggregated learning curve with confidence intervals.

    Args:
        df: Aggregated learning curve DataFrame (from aggregate_learning_curve_runs)
        out_path: Output plot path
        title: Plot title
        meta_lines: Optional metadata lines for plot annotation
    """
    if df is None or df.empty:
        return

    try:

        configure_backend()
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error(f"[PLOT] Learning curve failed to import dependencies: {e}")
        return

    x = df["train_size"].to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Helper function to safely plot confidence/uncertainty bands
    def _plot_band(
        mean_col: str,
        sd_col: str,
        ci_lo_col: str,
        ci_hi_col: str,
        color: str,
        label: str,
        alpha_ci: float = 0.12,
        alpha_sd: float = 0.20,
    ):
        mean = np.asarray(df[mean_col], dtype=float)
        sd = np.asarray(df[sd_col], dtype=float)
        ci_lo = np.asarray(df[ci_lo_col], dtype=float)
        ci_hi = np.asarray(df[ci_hi_col], dtype=float)

        # Plot 95% CI band if available
        if np.isfinite(ci_lo).any() and np.isfinite(ci_hi).any():
            valid = np.isfinite(ci_lo) & np.isfinite(ci_hi)
            if valid.sum() > 1:
                ax.fill_between(
                    x[valid],
                    ci_lo[valid],
                    ci_hi[valid],
                    color=color,
                    alpha=alpha_ci,
                    label=f"{label} 95% CI",
                )

        # Plot ±1 SD band if available
        if np.isfinite(sd).any() and np.isfinite(mean).any():
            valid = np.isfinite(mean) & np.isfinite(sd)
            if valid.sum() > 1:
                ax.fill_between(
                    x[valid],
                    (mean - sd)[valid],
                    (mean + sd)[valid],
                    color=color,
                    alpha=alpha_sd,
                    label=f"{label} ±1 SD",
                )

    # Plot validation bands (val_ci_lo, val_ci_hi, val_sd)
    _plot_band("val_mean", "val_sd", "val_ci_lo", "val_ci_hi", COLOR_SECONDARY, "Val")

    # Plot individual validation data points if available
    if "val_score" in df.columns:
        val_scores = np.asarray(df["val_score"], dtype=float)
        valid_val = np.isfinite(val_scores)
        if valid_val.any():
            ax.scatter(
                x[valid_val],
                val_scores[valid_val],
                color=COLOR_SECONDARY,
                alpha=0.35,
                s=MARKER_SIZE_SMALL,
                label="Val points",
            )

    # Plot validation mean line with markers
    ax.plot(
        x,
        df["val_mean"],
        color=COLOR_SECONDARY,
        linewidth=LW_PRIMARY,
        label="Val mean",
        marker="s",
        markersize=6,
        markerfacecolor=COLOR_SECONDARY,
        markeredgecolor=COLOR_SECONDARY,
    )

    # Plot train mean as thin reference line (minimal visual weight)
    ax.plot(
        x,
        df["train_mean"],
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=LW_REFERENCE,
        alpha=0.6,
        label="Train (reference)",
    )

    # Format axis labels with metric direction
    metric_label = (
        str(df["metric_label"].iloc[0]) if "metric_label" in df.columns and len(df) else "Score"
    )
    metric_direction = (
        str(df["metric_direction"].iloc[0]) if "metric_direction" in df.columns and len(df) else ""
    )
    ylabel = metric_label.replace("_", " ").upper()
    if metric_direction == "lower_is_better":
        ylabel += " (lower is better)"
    elif metric_direction == "higher_is_better":
        ylabel += " (higher is better)"

    ax.set_xlabel("Training examples", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)

    ax.legend(fontsize=FONT_LEGEND, loc="best")

    # Apply metadata annotation to bottom of figure
    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=DPI, pad_inches=PAD_INCHES)
    plt.close()
