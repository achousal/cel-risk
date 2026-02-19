"""
Calibration plotting utilities.

This module provides calibration curve plotting in both probability and logit space:
- Probability-space calibration (observed vs predicted frequencies)
- Logit-space calibration (log-odds)
- Multi-split aggregation with 95% CI and ±1 SD confidence bands
- LOESS smoothing for logit calibration
- Bootstrap confidence intervals (95% CI and ±1 SD)
- LOWESS-smoothed calibration curve overlay on probability-space panels
- ICI, E50, E90, and Spiegelhalter z-test annotation on probability-space panels

References:
    Van Calster et al. (2016). Calibration: the Achilles heel of predictive analytics.
    BMC Medicine.

    Austin & Steyerberg (2019). The Integrated Calibration Index (ICI).
    Statistics in Medicine.
"""

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from ced_ml.models.calibration import (
    calibration_error_quantiles,
    spiegelhalter_z_test,
)
from ced_ml.utils.constants import CI_LOWER_PCT, CI_UPPER_PCT

from .calibration_reliability import (
    binned_logits,
    get_legend_reference_sizes,
    plot_logit_calibration_panel,
)
from .style import (
    ALPHA_LEGEND_MARKER,
    ALPHA_LINE,
    ALPHA_REFERENCE,
    ALPHA_SCATTER,
    ALPHA_SD,
    COLOR_EDGE,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    DPI,
    FIGSIZE_CALIBRATION,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_PRIMARY,
    LW_SECONDARY,
    PAD_INCHES,
    configure_backend,
)

# Minimum sample count required to attempt LOWESS smoothing on a panel.
_LOWESS_MIN_SAMPLES = 50

# Minimum unique prediction values required for spline fitting (matches
# the threshold in models/calibration._loess_calibration_errors).
_LOWESS_MIN_UNIQUE_PREDS = 10

logger = logging.getLogger(__name__)


def _add_lowess_overlay(ax, y: np.ndarray, p: np.ndarray) -> None:
    """
    Add a LOWESS-smoothed calibration curve overlay to a probability-space panel.

    The curve is computed via a cubic UnivariateSpline (same approach used by
    the ICI metric in ced_ml.models.calibration._loess_calibration_errors).
    The overlay is skipped silently when there are too few samples or when
    spline fitting fails.

    Only called for probability-space panels, not logit-space panels.

    Args:
        ax: Matplotlib axis on which to draw the overlay.
        y: True binary labels (0/1), already filtered for finite values.
        p: Predicted probabilities, already filtered for finite values.
    """
    n = len(p)
    if n < _LOWESS_MIN_SAMPLES:
        logger.debug(
            "_add_lowess_overlay: only %d samples (need >= %d); skipping.",
            n,
            _LOWESS_MIN_SAMPLES,
        )
        return

    n_unique = len(np.unique(p))
    if n_unique < _LOWESS_MIN_UNIQUE_PREDS:
        logger.debug(
            "_add_lowess_overlay: only %d unique prediction values " "(need >= %d); skipping.",
            n_unique,
            _LOWESS_MIN_UNIQUE_PREDS,
        )
        return

    sort_idx = np.argsort(p)
    p_sorted = p[sort_idx]
    y_sorted = y[sort_idx]

    try:
        spline = UnivariateSpline(p_sorted, y_sorted, k=3, s=len(y_sorted), ext=3)
        # Evaluate on a dense grid for a smooth visual curve.
        p_grid = np.linspace(float(p_sorted[0]), float(p_sorted[-1]), 200)
        curve = np.clip(spline(p_grid), 0.0, 1.0)
    except Exception as exc:
        logger.debug("_add_lowess_overlay: spline fitting failed: %s", exc)
        return

    ax.plot(
        p_grid,
        curve,
        color=COLOR_SECONDARY,
        linewidth=LW_SECONDARY,
        linestyle="--",
        alpha=0.75,
        label="LOWESS smooth",
        zorder=3,
    )


def _add_calibration_metrics_annotation(ax, y: np.ndarray, p: np.ndarray) -> None:
    """
    Add an ICI / E50 / E90 / Spiegelhalter p-value annotation box to a panel.

    Metrics are computed from the supplied y and p arrays.  When a metric
    cannot be computed (too few samples, NaN) its field is shown as 'N/A'.

    Placed in the upper-left corner of the axes with a semi-transparent
    background box.  Only called for probability-space panels.

    Args:
        ax: Matplotlib axis on which to place the annotation.
        y: True binary labels (0/1), already filtered for finite values.
        p: Predicted probabilities, already filtered for finite values.
    """
    try:
        q = calibration_error_quantiles(y, p)
        spieg = spiegelhalter_z_test(y, p)
    except Exception as exc:
        logger.debug("_add_calibration_metrics_annotation: metric computation failed: %s", exc)
        return

    def _fmt(val: float, digits: int = 3) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.{digits}f}"

    lines = [
        f"ICI: {_fmt(q.ici)}",
        f"E50: {_fmt(q.e50)}",
        f"E90: {_fmt(q.e90)}",
        f"Spieg. p: {_fmt(spieg.p_value, digits=2)}",
    ]
    text = "\n".join(lines)

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "lightgray",
            "linewidth": 0.8,
        },
        zorder=5,
    )


# Re-export extracted functions for backward compatibility
def _get_legend_reference_sizes(actual_sizes: np.ndarray) -> list:
    """Backward compat: see get_legend_reference_sizes in calibration_reliability."""
    return get_legend_reference_sizes(actual_sizes)


def _plot_prob_calibration_panel(
    ax,
    y: np.ndarray,
    p: np.ndarray,
    bins: np.ndarray,
    bin_centers: np.ndarray,
    actual_n_bins: int,
    bin_strategy: str,
    split_ids: np.ndarray | None = None,
    unique_splits: list | None = None,
    panel_title: str = "",
    variable_sizes: bool = True,
    skip_ci_band: bool = False,
) -> None:
    """
    Plot a single probability-space calibration panel.

    Args:
        ax: Matplotlib axis to plot on
        y: True labels (0/1)
        p: Predicted probabilities
        bins: Bin edges
        bin_centers: Center points of bins
        actual_n_bins: Number of bins
        bin_strategy: 'uniform' or 'quantile'
        split_ids: Optional split identifiers
        unique_splits: List of unique split IDs
        panel_title: Title for this panel
        variable_sizes: If True, circle sizes vary with bin sample counts
        skip_ci_band: If True, skip rendering 95% CI band (only show ±1 SD)
    """
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=LW_SECONDARY,
        label="Perfect calibration",
        alpha=ALPHA_REFERENCE,
    )

    if unique_splits is not None and len(unique_splits) > 1:
        curves = []
        counts_all = []
        for sid in unique_splits:
            m_split = (split_ids == sid) if sid is not None else np.isnan(split_ids)
            y_s = y[m_split]
            p_s = p[m_split]
            bin_idx = np.digitize(p_s, bins) - 1
            bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)
            obs = []
            counts = []
            for i in range(actual_n_bins):
                m = bin_idx == i
                obs.append(np.nan if m.sum() == 0 else y_s[m].mean())
                counts.append(int(m.sum()))
            curves.append(obs)
            counts_all.append(counts)
        curves = np.array(curves, dtype=float)
        counts_all = np.array(counts_all, dtype=float)

        # Suppress expected warnings when bins are empty across all splits
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            obs_mean = np.nanmean(curves, axis=0)
            obs_sd = np.nanstd(curves, axis=0)
            obs_lo = np.nanpercentile(curves, CI_LOWER_PCT, axis=0)
            obs_hi = np.nanpercentile(curves, CI_UPPER_PCT, axis=0)
            np.nanmean(counts_all, axis=0)
            sum_counts = np.nansum(counts_all, axis=0)

        if not skip_ci_band:
            ax.fill_between(
                bin_centers,
                np.clip(obs_lo, 0, 1),
                np.clip(obs_hi, 0, 1),
                color=COLOR_PRIMARY,
                alpha=ALPHA_SD,
                label="95% CI",
            )
        ax.fill_between(
            bin_centers,
            np.clip(obs_mean - obs_sd, 0, 1),
            np.clip(obs_mean + obs_sd, 0, 1),
            color=COLOR_PRIMARY,
            alpha=ALPHA_SD,
            label="±1 SD",
        )

        valid = ~np.isnan(obs_mean) & (sum_counts > 0)
        # Only use variable marker sizes for uniform binning; quantile binning gets fixed sizes
        if bin_strategy == "quantile":
            scatter_sizes = 50  # Fixed size for quantile binning
        elif variable_sizes:
            # Use variable marker sizes based on aggregate counts
            # Square root scaling for better visual separation at low counts
            scatter_sizes = np.clip(np.sqrt(sum_counts[valid]) * 15, 30, 350)
        else:
            # Fixed marker size for all points
            scatter_sizes = 30
        ax.scatter(
            bin_centers[valid],
            obs_mean[valid],
            s=scatter_sizes,
            color=COLOR_PRIMARY,
            alpha=ALPHA_SCATTER,
            edgecolors=COLOR_EDGE,
            linewidths=0.5,
        )
        ax.plot(
            bin_centers,
            obs_mean,
            color=COLOR_PRIMARY,
            linewidth=LW_PRIMARY,
            alpha=ALPHA_LINE,
            label=f"Mean (n={len(curves)} splits)",
        )
    else:
        bin_idx = np.digitize(p, bins) - 1
        bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)
        obs = []
        pred_means = []
        sizes = []
        for i in range(actual_n_bins):
            m = bin_idx == i
            if m.sum() == 0:
                obs.append(np.nan)
                pred_means.append(np.nan)
                sizes.append(0)
            else:
                obs.append(y[m].mean())
                pred_means.append(p[m].mean())
                sizes.append(int(m.sum()))
        obs = np.array(obs)
        pred_means = np.array(pred_means)
        sizes = np.array(sizes)
        valid = ~np.isnan(obs)

        # Only use variable marker sizes for uniform binning; quantile binning gets fixed sizes
        if bin_strategy == "quantile":
            scatter_sizes = 60  # Fixed size for quantile binning
        elif variable_sizes:
            # Square root scaling for better visual separation at low counts
            scatter_sizes = np.clip(np.sqrt(sizes[valid]) * 20, 40, 450)
        else:
            scatter_sizes = 60
        ax.scatter(
            pred_means[valid],
            obs[valid],
            s=scatter_sizes,
            color=COLOR_PRIMARY,
            alpha=ALPHA_SCATTER,
            edgecolors=COLOR_EDGE,
            linewidths=0.5,
        )
        ax.plot(
            pred_means[valid],
            obs[valid],
            color=COLOR_PRIMARY,
            linewidth=LW_SECONDARY,
            alpha=ALPHA_LINE,
        )

    # LOWESS overlay and calibration metrics annotation (probability-space only).
    _add_lowess_overlay(ax, y, p)
    _add_calibration_metrics_annotation(ax, y, p)

    bin_label = "quantile" if bin_strategy == "quantile" else "uniform"
    if panel_title:
        title_text = panel_title
    else:
        title_text = f"Calibration ({bin_label} bins, k={actual_n_bins})"
    ax.set_title(title_text, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Predicted probability", fontsize=FONT_LABEL)
    ax.set_ylabel("Expected frequency", fontsize=FONT_LABEL)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

    # Add size legend for uniform binning
    if bin_strategy == "uniform":
        from matplotlib.lines import Line2D

        # Determine the actual sizing formula used in the scatter plot
        if unique_splits is not None and len(unique_splits) > 1:
            # Multi-split case: uses sqrt(sum_counts[valid]) * 15
            sqrt_multiplier = 15
            min_scatter = 30
            max_scatter = 350
            # Use only valid bins (those actually plotted)
            actual_bin_sizes = sum_counts[valid]
        else:
            # Single-split case: uses sqrt(sizes[valid]) * 20
            sqrt_multiplier = 20
            min_scatter = 40
            max_scatter = 450
            # Use only valid bins (those actually plotted)
            actual_bin_sizes = sizes[valid]

        # Get legend reference sizes based on actual data
        reference_sizes = _get_legend_reference_sizes(actual_bin_sizes)

        size_handles = []
        size_labels = []
        for sample_count in reference_sizes:
            # Match the actual scatter plot sizing formula (square root scaling)
            # scatter() 's' parameter is area in points^2
            scatter_area = np.clip(
                np.sqrt(sample_count) * sqrt_multiplier, min_scatter, max_scatter
            )
            # Line2D markersize is the marker width/diameter in points
            # Convert: diameter = sqrt(area) for matplotlib scatter
            markersize = np.sqrt(scatter_area)
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_PRIMARY,
                markersize=markersize,
                markeredgecolor=COLOR_EDGE,
                markeredgewidth=0.5,
                linestyle="None",
                alpha=ALPHA_LEGEND_MARKER,
            )
            size_handles.append(handle)
            size_labels.append(f"{sample_count}")

        # Position legend to the right with adequate spacing
        size_legend = ax.legend(
            size_handles,
            size_labels,
            title="Bin size (n)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
            labelspacing=1.2,  # Increase vertical spacing between legend entries
        )
        ax.add_artist(size_legend)
        # Re-add main legend (was overwritten by size legend)
        ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.9, labelspacing=1.0)
    else:
        ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.9, labelspacing=1.0)


def _binned_logits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
    min_bin_size: int = 1,
    merge_tail: bool = False,
    n_boot: int = 500,
    seed: int | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Backward compat: see binned_logits in calibration_reliability."""
    return binned_logits(
        y_true, y_pred, n_bins, bin_strategy, min_bin_size, merge_tail, n_boot, seed
    )


def _plot_logit_calibration_panel(
    ax,
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
    bin_strategy: str,
    split_ids: np.ndarray | None,
    unique_splits: list | None,
    panel_title: str,
    calib_intercept: float | None,
    calib_slope: float | None,
    eps: float = 1e-7,
    skip_ci_band: bool = False,
) -> None:
    """Backward compat: see plot_logit_calibration_panel in calibration_reliability."""
    return plot_logit_calibration_panel(
        ax,
        y,
        p,
        n_bins,
        bin_strategy,
        split_ids,
        unique_splits,
        panel_title,
        calib_intercept,
        calib_slope,
        eps,
        skip_ci_band,
    )


def _apply_plot_metadata(fig, meta_lines: Sequence[str] | None = None) -> float:
    """
    Calculate bottom margin for metadata text.

    This function calculates the required bottom margin based on the number of metadata
    lines. Must be called BEFORE plt.subplots_adjust().

    Args:
        fig: matplotlib figure object
        meta_lines: sequence of metadata strings to display

    Returns:
        Required bottom margin as fraction of figure height (0.0 to 1.0)
    """
    lines = [str(line) for line in (meta_lines or []) if line]
    if not lines:
        return 0.12  # Default minimum bottom margin

    # Calculate required bottom margin: base + space per line
    # Increased spacing for better separation between metadata and figures
    required_bottom = 0.12 + (0.022 * len(lines))
    return min(required_bottom, 0.30)  # Cap at 30% to avoid excessive margin


def _add_metadata_text(fig, meta_lines: Sequence[str] | None, bottom_margin: float) -> None:
    """
    Add metadata text to figure in the bottom margin.

    MUST be called AFTER plt.subplots_adjust() to position text correctly.

    Args:
        fig: matplotlib figure object
        meta_lines: sequence of metadata strings to display
        bottom_margin: the bottom margin value used in subplots_adjust()
    """
    if not meta_lines:
        return

    lines = [str(line) for line in meta_lines if line]
    if not lines:
        return

    # Position text in center of bottom margin (halfway between 0 and bottom_margin)
    y_pos = bottom_margin / 2
    fig.text(
        0.5,
        y_pos,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=8,
        transform=fig.transFigure,
    )


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    n_bins: int = 10,
    split_ids: np.ndarray | None = None,
    meta_lines: Sequence[str] | None = None,
    calib_intercept: float | None = None,
    calib_slope: float | None = None,
    skip_ci_band: bool = False,
) -> None:
    """
    Generate 4-panel calibration plot.

    Always generates a 2x2 layout:
        Panel 1 (top-left): Calibration curve with quantile binning
        Panel 2 (top-right): Calibration curve with uniform binning
        Panel 3 (bottom-left): Logit calibration curve with quantile binning
        Panel 4 (bottom-right): Logit calibration curve with uniform binning

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Output file path
        title: Plot title
        subtitle: Plot subtitle
        n_bins: Number of bins for calibration curve
        split_ids: Optional split identifiers for multi-split aggregation
        meta_lines: Metadata lines for plot annotation
        calib_intercept: Calibration intercept (alpha) from logistic recalibration
        calib_slope: Calibration slope (beta) from logistic recalibration
        skip_ci_band: If True, skip rendering 95% CI band (only show ±1 SD).
            Useful for ensemble models where CI and SD are redundant.
    """
    try:

        configure_backend()
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error(f"Calibration plot failed to import matplotlib: {e}")
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
        logger.warning("No valid data for calibration plot after filtering")
        return

    # Clip probabilities for numerical stability
    eps = 1e-7
    np.clip(p, eps, 1 - eps)

    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    # Create figure layout: Always 2x2 panels
    fig = plt.figure(figsize=FIGSIZE_CALIBRATION)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Calibration quantile
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right: Calibration uniform
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left: Logit quantile
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right: Logit uniform

    # ========== Panel 1 (top-left): Probability-space calibration curve with quantile binning ==========
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins_quantile = np.percentile(p, quantiles)
    bins_quantile = np.unique(bins_quantile)
    if len(bins_quantile) < 3:
        bins_quantile = np.linspace(0, 1, n_bins + 1)
    actual_n_bins_q = len(bins_quantile) - 1
    bin_centers_q = (bins_quantile[:-1] + bins_quantile[1:]) / 2

    # Compute actual bin sizes from the quantile bins
    bin_idx_q = np.digitize(p, bins_quantile) - 1
    bin_idx_q = np.clip(bin_idx_q, 0, actual_n_bins_q - 1)
    bin_sizes_q = np.array([int((bin_idx_q == i).sum()) for i in range(actual_n_bins_q)])

    # Compute per-bin sample counts
    nonzero_sizes = bin_sizes_q[bin_sizes_q > 0]
    if len(nonzero_sizes) > 0:
        mean_size = int(np.mean(nonzero_sizes))
        min_size = int(np.min(nonzero_sizes))
        max_size = int(np.max(nonzero_sizes))
        if min_size == max_size:
            bin_size_str = f"n={mean_size}/bin"
        else:
            bin_size_str = f"n≈{mean_size}/bin (range {min_size}–{max_size})"
    else:
        bin_size_str = ""

    panel_title_1 = f"Calibration (quantile bins)\nk={actual_n_bins_q}, {bin_size_str}"
    if subtitle:
        panel_title_1 = (
            f"{subtitle} – Calibration (quantile bins)\nk={actual_n_bins_q}, {bin_size_str}"
        )

    _plot_prob_calibration_panel(
        ax1,
        y,
        p,
        bins_quantile,
        bin_centers_q,
        actual_n_bins_q,
        "quantile",
        split_ids=split_ids,
        unique_splits=unique_splits,
        panel_title=panel_title_1,
        variable_sizes=False,
        skip_ci_band=skip_ci_band,
    )

    # ========== Panel 2 (top-right): Probability-space calibration curve with uniform binning ==========
    bins_uniform = np.linspace(0, 1, n_bins + 1)
    actual_n_bins_u = len(bins_uniform) - 1
    bin_centers_u = (bins_uniform[:-1] + bins_uniform[1:]) / 2

    panel_title_2 = f"Calibration (uniform bins)\nk={actual_n_bins_u}"
    if subtitle:
        panel_title_2 = f"{subtitle} – Calibration (uniform bins)\nk={actual_n_bins_u}"

    _plot_prob_calibration_panel(
        ax2,
        y,
        p,
        bins_uniform,
        bin_centers_u,
        actual_n_bins_u,
        "uniform",
        split_ids=split_ids,
        unique_splits=unique_splits,
        panel_title=panel_title_2,
        variable_sizes=True,
        skip_ci_band=skip_ci_band,
    )

    # ========== Panel 3 (bottom-left): Log-odds calibration with quantile binning ==========
    logit_title_q = "Logit calibration (quantile bins)"
    _plot_logit_calibration_panel(
        ax3,
        y,
        p,
        n_bins,
        "quantile",
        split_ids,
        unique_splits,
        logit_title_q,
        calib_intercept,
        calib_slope,
        eps=eps,
        skip_ci_band=skip_ci_band,
    )

    # ========== Panel 4 (bottom-right): Log-odds calibration with uniform binning ==========
    logit_title_u = "Logit calibration (uniform bins)"
    _plot_logit_calibration_panel(
        ax4,
        y,
        p,
        n_bins,
        "uniform",
        split_ids,
        unique_splits,
        logit_title_u,
        calib_intercept,
        calib_slope,
        eps=eps,
        skip_ci_band=skip_ci_band,
    )

    # Add title at the top
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", y=0.98)

    # Apply metadata and adjust layout
    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    # Increase right margin to accommodate size legend in uniform binning panels
    plt.subplots_adjust(left=0.10, right=0.88, top=0.92, bottom=bottom_margin)
    _add_metadata_text(fig, meta_lines, bottom_margin)

    # Save figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, pad_inches=PAD_INCHES)
    plt.close()

    logger.info(f"Calibration plot saved to {out_path}")
