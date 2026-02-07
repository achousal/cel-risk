"""
Decision Curve Analysis (DCA) plotting functions.

This module provides functions for visualizing DCA results to assess clinical utility
of prediction models at different decision thresholds.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ced_ml.metrics.dca import decision_curve_analysis
from ced_ml.utils.constants import CI_LOWER_PCT, CI_UPPER_PCT

from .style import (
    ALPHA_CI,
    ALPHA_SD,
    BBOX_INCHES,
    COLOR_FILL_ALPHA,
    COLOR_PRIMARY,
    DPI,
    FIGSIZE_DCA,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_PRIMARY,
    LW_SECONDARY,
    PAD_INCHES,
    configure_backend,
)

logger = logging.getLogger(__name__)


def apply_plot_metadata(
    fig: matplotlib.figure.Figure, meta_lines: Sequence[str] | None = None
) -> float:
    """
    Apply metadata text to bottom of figure.

    Args:
        fig: matplotlib figure object
        meta_lines: sequence of metadata strings to display

    Returns:
        Required bottom margin as fraction of figure height (0.0 to 1.0)
    """
    lines = [str(line) for line in (meta_lines or []) if line]
    if not lines:
        return 0.12  # Default minimum bottom margin

    # Position metadata at very bottom with fixed offset from edge
    fig.text(0.5, 0.005, "\n".join(lines), ha="center", va="bottom", fontsize=8, wrap=True)

    # Calculate required bottom margin: base + space per line
    # Increased spacing for better separation between metadata and figures
    required_bottom = 0.12 + (0.022 * len(lines))
    return min(required_bottom, 0.30)  # Cap at 30% to avoid excessive margin


def plot_dca(dca_df: pd.DataFrame, out_path: str, meta_lines: Sequence[str] | None = None) -> None:
    """
    Generate DCA plot from pre-computed DCA DataFrame.

    Args:
        dca_df: DataFrame from decision_curve_analysis() with columns:
            - threshold_pct: threshold as percentage (0-100)
            - net_benefit_model: model net benefit
            - net_benefit_all: treat-all net benefit
            - net_benefit_none: treat-none net benefit
        out_path: Path to save plot
        meta_lines: Optional metadata lines to display at bottom
    """
    configure_backend()

    fig, ax = plt.subplots(figsize=FIGSIZE_DCA)

    thresholds = dca_df["threshold_pct"].values
    nb_model = dca_df["net_benefit_model"].values
    nb_all = dca_df["net_benefit_all"].values
    nb_none = dca_df["net_benefit_none"].values

    ax.plot(
        thresholds,
        nb_model,
        color=COLOR_PRIMARY,
        linestyle="-",
        linewidth=LW_PRIMARY,
        label="Model",
    )
    ax.plot(thresholds, nb_all, "r--", linewidth=LW_SECONDARY, label="Treat All")
    ax.plot(thresholds, nb_none, "k:", linewidth=LW_SECONDARY, label="Treat None")

    # Shade region where model is better than alternatives
    ax.fill_between(
        thresholds,
        np.maximum(nb_all, nb_none),
        nb_model,
        where=(nb_model > np.maximum(nb_all, nb_none)),
        alpha=COLOR_FILL_ALPHA,
        color=COLOR_PRIMARY,
        label="Model Benefit",
    )

    ax.set_xlabel("Threshold Probability (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Net Benefit", fontsize=FONT_LABEL)
    ax.set_title("Decision Curve Analysis", fontsize=FONT_TITLE)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    # Set reasonable y-axis limits with consistent padding
    y_min = min(nb_model.min(), nb_all.min(), nb_none.min())
    y_max = max(nb_model.max(), nb_all.max(), nb_none.max())
    y_range = y_max - y_min
    if y_range > 0:
        y_min_padded = y_min - 0.1 * y_range
        y_max_padded = y_max + 0.1 * y_range
    else:
        y_min_padded = min(y_min, -0.05)
        y_max_padded = max(y_max, 0.05)
    ax.set_ylim(y_min_padded, y_max_padded)

    # Set x-axis to start at 0 and extend to data max with small margin
    # Add 2% beyond max threshold for better visualization
    x_max = min(100, thresholds.max() * 1.02)
    ax.set_xlim(0, x_max)

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES)
    plt.close()


def plot_dca_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str,
    subtitle: str = "",
    max_pt: float = 0.20,
    step: float = 0.005,
    split_ids: np.ndarray | None = None,
    meta_lines: Sequence[str] | None = None,
    skip_ci_band: bool = False,
) -> None:
    """
    Generate DCA plot from raw predictions with multi-split averaging.

    Computes Decision Curve Analysis across threshold range, with optional
    multi-split averaging and confidence bands.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Path to save plot
        title: Plot title
        subtitle: Optional subtitle
        max_pt: Maximum threshold (default: 0.20)
        step: Threshold step size (default: 0.005)
        split_ids: Optional array of split IDs for multi-split averaging
        meta_lines: Optional metadata lines to display at bottom
        skip_ci_band: If True, skip rendering 95% CI band (only show ±1 SD).
            Useful for ensemble models where CI and SD are redundant.
    """
    try:
        configure_backend()
    except Exception as e:
        logger.error(f"DCA plot failed to configure backend: {e}")
        return

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    # Validate prediction range
    if len(p) > 0 and (np.min(p) < 0 or np.max(p) > 1):
        logger.warning(f"Predictions outside [0,1] range: min={np.min(p):.4f}, max={np.max(p):.4f}")

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]

    if len(y) == 0:
        logger.warning("No valid data for DCA curve plot after filtering")
        return

    # Generate threshold array
    thresholds = np.arange(0.0005, max_pt + step, step)

    fig, ax = plt.subplots(figsize=FIGSIZE_DCA)

    # Convert thresholds to percentage for consistent x-axis scale
    thresholds_pct = thresholds * 100

    # Multi-split handling
    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    # Initialize for scope
    dca_df = pd.DataFrame()
    nb_model_curves = []

    if len(unique_splits) > 1:
        # Compute DCA per split and average
        nb_model_curves = []
        nb_all_curves = []
        nb_none_curves = []

        for sid in unique_splits:
            m = split_ids == sid
            y_s = y[m]
            p_s = p[m]
            if len(np.unique(y_s)) < 2 or len(y_s) < 2:
                continue

            dca_df = decision_curve_analysis(y_s, p_s, thresholds=thresholds)
            if not dca_df.empty:
                nb_model_curves.append(dca_df["net_benefit_model"].values)
                nb_all_curves.append(dca_df["net_benefit_all"].values)
                nb_none_curves.append(dca_df["net_benefit_none"].values)

        if nb_model_curves:
            nb_model_curves = np.vstack(nb_model_curves)
            nb_all_curves = np.vstack(nb_all_curves)
            nb_none_curves = np.vstack(nb_none_curves)

            nb_model_mean = np.mean(nb_model_curves, axis=0)
            nb_model_sd = np.std(nb_model_curves, axis=0)
            nb_model_lo = np.percentile(nb_model_curves, CI_LOWER_PCT, axis=0)
            nb_model_hi = np.percentile(nb_model_curves, CI_UPPER_PCT, axis=0)

            nb_all_mean = np.mean(nb_all_curves, axis=0)
            nb_none_mean = np.mean(nb_none_curves, axis=0)

            thr_pct = thresholds_pct[: len(nb_model_mean)]

            # Plot with confidence bands
            if not skip_ci_band:
                ax.fill_between(
                    thr_pct,
                    nb_model_lo,
                    nb_model_hi,
                    color=COLOR_PRIMARY,
                    alpha=ALPHA_CI,
                    label="95% CI",
                )
            ax.fill_between(
                thr_pct,
                np.maximum(0, nb_model_mean - nb_model_sd),
                np.minimum(1, nb_model_mean + nb_model_sd),
                color=COLOR_PRIMARY,
                alpha=ALPHA_SD,
                label="±1 SD",
            )
            ax.plot(
                thr_pct,
                nb_model_mean,
                color=COLOR_PRIMARY,
                linestyle="-",
                linewidth=LW_PRIMARY,
                label="Model",
            )
            ax.plot(thr_pct, nb_all_mean, "r--", linewidth=LW_SECONDARY, label="Treat All")
            ax.plot(thr_pct, nb_none_mean, "k:", linewidth=LW_SECONDARY, label="Treat None")

            # Shade region where model is better
            ax.fill_between(
                thr_pct,
                np.maximum(nb_all_mean, nb_none_mean),
                nb_model_mean,
                where=(nb_model_mean > np.maximum(nb_all_mean, nb_none_mean)),
                alpha=0.2,
                color=COLOR_PRIMARY,
                label="Model Benefit",
            )
        else:
            # Fallback to single curve if all splits fail
            dca_df = decision_curve_analysis(y, p, thresholds=thresholds)
            if not dca_df.empty:
                thr_pct = dca_df["threshold"].values * 100
                ax.plot(
                    thr_pct,
                    dca_df["net_benefit_model"].values,
                    color=COLOR_PRIMARY,
                    linestyle="-",
                    linewidth=LW_PRIMARY,
                    label="Model",
                )
                ax.plot(
                    thr_pct,
                    dca_df["net_benefit_all"].values,
                    "r--",
                    linewidth=LW_SECONDARY,
                    label="Treat All",
                )
                ax.plot(
                    thr_pct,
                    dca_df["net_benefit_none"].values,
                    "k:",
                    linewidth=LW_SECONDARY,
                    label="Treat None",
                )
                ax.fill_between(
                    thr_pct,
                    np.maximum(
                        dca_df["net_benefit_all"].values,
                        dca_df["net_benefit_none"].values,
                    ),
                    dca_df["net_benefit_model"].values,
                    where=(
                        dca_df["net_benefit_model"].values
                        > np.maximum(
                            dca_df["net_benefit_all"].values,
                            dca_df["net_benefit_none"].values,
                        )
                    ),
                    alpha=COLOR_FILL_ALPHA,
                    color=COLOR_PRIMARY,
                    label="Model Benefit",
                )
    else:
        # Single split or no split_ids
        dca_df = decision_curve_analysis(y, p, thresholds=thresholds)
        if not dca_df.empty:
            thr_pct = dca_df["threshold"].values * 100
            ax.plot(
                thr_pct,
                dca_df["net_benefit_model"].values,
                color=COLOR_PRIMARY,
                linestyle="-",
                linewidth=LW_PRIMARY,
                label="Model",
            )
            ax.plot(
                thr_pct,
                dca_df["net_benefit_all"].values,
                "r--",
                linewidth=LW_SECONDARY,
                label="Treat All",
            )
            ax.plot(
                thr_pct,
                dca_df["net_benefit_none"].values,
                "k:",
                linewidth=LW_SECONDARY,
                label="Treat None",
            )
            ax.fill_between(
                thr_pct,
                np.maximum(dca_df["net_benefit_all"].values, dca_df["net_benefit_none"].values),
                dca_df["net_benefit_model"].values,
                where=(
                    dca_df["net_benefit_model"].values
                    > np.maximum(
                        dca_df["net_benefit_all"].values,
                        dca_df["net_benefit_none"].values,
                    )
                ),
                alpha=0.2,
                color=COLOR_PRIMARY,
                label="Model Benefit",
            )

    # Compute y-range to include all curves (treat all, treat none, model)
    y_min = 0
    y_max = 0

    if len(unique_splits) > 1 and len(nb_model_curves) > 0:
        y_min = min(
            y_min,
            np.nanmin(nb_model_lo),
            np.nanmin(nb_all_mean),
            np.nanmin(nb_none_mean),
        )
        y_max = max(
            y_max,
            np.nanmax(nb_model_hi),
            np.nanmax(nb_all_mean),
            np.nanmax(nb_none_mean),
        )
    elif not dca_df.empty:
        y_min = min(
            y_min,
            dca_df["net_benefit_model"].min(),
            dca_df["net_benefit_all"].min(),
            dca_df["net_benefit_none"].min(),
        )
        y_max = max(
            y_max,
            dca_df["net_benefit_model"].max(),
            dca_df["net_benefit_all"].max(),
            dca_df["net_benefit_none"].max(),
        )

    # Add 10% padding
    y_range = y_max - y_min
    if y_range > 0:
        y_min_padded = y_min - 0.1 * y_range
        y_max_padded = y_max + 0.1 * y_range
    else:
        y_min_padded = min(y_min, -0.05)
        y_max_padded = max(y_max, 0.05)
    ax.set_ylim([y_min_padded, y_max_padded])

    # Set x-axis to match actual data extent with small margin
    # Add 2% beyond max threshold for better visualization
    x_max_data = max_pt * 100  # Convert to percentage
    x_max_display = min(100, x_max_data * 1.02)
    ax.set_xlim(0, x_max_display)

    ax.set_xlabel("Threshold Probability (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Net Benefit", fontsize=FONT_LABEL)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE)
    else:
        ax.set_title(title, fontsize=FONT_TITLE)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES)
    plt.close()
