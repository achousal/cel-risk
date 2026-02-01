"""
Calibration plotting utilities.

This module provides calibration curve plotting in both probability and logit space:
- Probability-space calibration (observed vs predicted frequencies)
- Logit-space calibration (log-odds)
- Multi-split aggregation with 95% CI and ±1 SD confidence bands
- LOESS smoothing for logit calibration
- Bootstrap confidence intervals (95% CI and ±1 SD)

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

from ced_ml.utils.constants import CI_LOWER_PCT, CI_UPPER_PCT
from ced_ml.utils.math_utils import jeffreys_smooth, logit

from .style import (
    ALPHA_CI,
    ALPHA_LEGEND_MARKER,
    ALPHA_LINE,
    ALPHA_RECALIBRATION,
    ALPHA_REFERENCE,
    ALPHA_SCATTER,
    ALPHA_SD,
    COLOR_EDGE,
    COLOR_PRIMARY,
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

logger = logging.getLogger(__name__)


def _get_legend_reference_sizes(actual_sizes: np.ndarray) -> list:
    """
    Compute appropriate legend reference sizes based on actual bin sizes.

    Args:
        actual_sizes: Array of actual bin sizes from the data

    Returns:
        List of 3-4 representative sample counts for legend
    """
    if len(actual_sizes) == 0 or actual_sizes.max() == 0:
        return [10, 50, 100, 200]

    min_size = int(actual_sizes.min())
    max_size = int(actual_sizes.max())

    # If range is small, use actual min/max and interpolate
    if max_size - min_size < 50:
        return [min_size, max_size]

    # Generate 3-4 evenly spaced reference points
    # Round to nice numbers (multiples of 10, 50, or 100)
    def round_to_nice(x):
        if x < 50:
            return int(np.round(x / 10) * 10)
        elif x < 200:
            return int(np.round(x / 25) * 25)
        else:
            return int(np.round(x / 50) * 50)

    # Create quartile-based reference points
    q25 = round_to_nice(np.percentile(actual_sizes, 25))
    q50 = round_to_nice(np.percentile(actual_sizes, 50))
    q75 = round_to_nice(np.percentile(actual_sizes, 75))
    q_max = round_to_nice(max_size)

    # Filter duplicates and sort
    sizes = sorted({q25, q50, q75, q_max})

    # Ensure we have at least 2 reference points
    if len(sizes) < 2:
        sizes = [min_size, max_size]

    return sizes


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
    """
    Compute binned logits for calibration plot in logit space with bootstrap 95% CI and ±1 SD.

    Logits are log-odds of probabilities: log(p/(1-p)). Creates calibration curve
    with predicted logits on x-axis and observed logits on y-axis.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping predictions
        bin_strategy: 'uniform' (equal width) or 'quantile' (equal size)
        min_bin_size: Minimum number of samples per bin
        merge_tail: If True, merge small bins with adjacent bins
        n_boot: Number of bootstrap resamples for CI computation
        seed: Random seed for bootstrap resampling (default: 42)

    Returns:
        Tuple of (xs, ys, ys_lo, ys_hi, ys_sd, sizes) where:
        - xs: predicted log-odds (bin centers)
        - ys: observed log-odds (empirical event rates)
        - ys_lo: lower 95% CI bound (log-odds, 2.5th percentile)
        - ys_hi: upper 95% CI bound (log-odds, 97.5th percentile)
        - ys_sd: standard deviation in log-odds (for ±1 SD bands)
        - sizes: bin sizes
        Returns (None, None, None, None, None, None) if insufficient data
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None, None, None, None, None, None

    # Create initial bins
    if bin_strategy == "quantile":
        quantiles = np.linspace(0, 100, int(n_bins) + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, int(n_bins) + 1)
    else:
        bins = np.linspace(0, 1, int(n_bins) + 1)

    # Assign samples to bins
    bin_idx = np.digitize(p, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    # Compute per-bin statistics with bootstrap CI
    xs_list, ys_list, ys_lo_list, ys_hi_list, ys_sd_list, sizes_list = [], [], [], [], [], []
    eps = 1e-7

    for i in range(len(bins) - 1):
        mask_bin = bin_idx == i
        if mask_bin.sum() < min_bin_size and merge_tail:
            continue

        y_bin = y[mask_bin]
        p_bin = p[mask_bin]

        if len(y_bin) == 0:
            continue

        # Predicted logit (mean of bin)
        p_mean = np.mean(p_bin)
        logit_pred = logit(p_mean, eps=eps)

        # Observed proportion with Jeffreys smoothing and bootstrap 95% CI + SD
        n = len(y_bin)
        k = int(y_bin.sum())
        # Apply Jeffreys smoothing to point estimate for consistency with bootstrap
        obs_prop = jeffreys_smooth(k, n)
        logit_obs = logit(obs_prop, eps=eps)

        # Bootstrap resampling for CI and SD
        boot_logits = []
        rng = np.random.RandomState(seed if seed is not None else 42)
        for _ in range(n_boot):
            boot_indices = rng.choice(n, size=n, replace=True)
            y_boot = y_bin[boot_indices]
            k_boot = y_boot.sum()
            # Apply Jeffreys smoothing to avoid 0/1 proportions
            prop_boot = jeffreys_smooth(k_boot, n)
            logit_boot = logit(prop_boot, eps=eps)
            boot_logits.append(logit_boot)

        boot_logits = np.array(boot_logits)
        logit_obs_lo = np.percentile(boot_logits, CI_LOWER_PCT)
        logit_obs_hi = np.percentile(boot_logits, CI_UPPER_PCT)
        logit_obs_sd = np.std(boot_logits)

        xs_list.append(logit_pred)
        ys_list.append(logit_obs)
        ys_lo_list.append(logit_obs_lo)
        ys_hi_list.append(logit_obs_hi)
        ys_sd_list.append(logit_obs_sd)
        sizes_list.append(n)

    if len(xs_list) == 0:
        return None, None, None, None, None, None

    return (
        np.array(xs_list),
        np.array(ys_list),
        np.array(ys_lo_list),
        np.array(ys_hi_list),
        np.array(ys_sd_list),
        np.array(sizes_list),
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
    """
    Plot a single logit-space calibration panel.

    Uses binned observations with bootstrap 95% CI and SD bands (steelblue style)
    consistently for both single-split and multi-split cases.

    Args:
        ax: Matplotlib axis to plot on
        y: True labels (0/1)
        p: Predicted probabilities
        n_bins: Number of bins for binning
        bin_strategy: 'uniform' or 'quantile'
        split_ids: Optional split identifiers
        unique_splits: List of unique split IDs
        panel_title: Title for this panel
        calib_intercept: Calibration intercept (alpha) from logistic recalibration
        calib_slope: Calibration slope (beta) from logistic recalibration
        eps: Small epsilon for clipping probabilities
        skip_ci_band: If True, skip rendering 95% CI band (only show ±1 SD)
    """
    # Clip probabilities for numerical stability
    p_clipped = np.clip(p, eps, 1 - eps)

    # Create bins based on strategy
    if bin_strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, n_bins + 1)
    else:
        bins = np.linspace(0, 1, n_bins + 1)

    actual_n_bins = len(bins) - 1

    # Convert to logit space
    logit_pred = logit(p_clipped, eps=eps)

    # Initialize axis ranges with default values
    logit_range_x = [-5, 5]
    logit_range_y = [-5, 5]

    # Multi-split logit calibration aggregation (using fixed probability bins)
    if unique_splits is not None and len(unique_splits) > 1:
        # Define fixed probability bins
        prob_x_bins = []
        prob_y_bins = []
        bin_sizes_per_split = []

        for sid in unique_splits:
            m_split = (split_ids == sid) if sid is not None else np.isnan(split_ids)
            y_s = y[m_split]
            p_s = p[m_split]

            # Bin predictions in probability space
            bin_idx = np.digitize(p_s, bins) - 1
            bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)

            # Compute observed frequency per bin (BEFORE logit transform)
            prob_x_per_split = []
            prob_y_per_split = []
            bin_sizes_per_bin = []
            for i in range(actual_n_bins):
                m_bin = bin_idx == i
                if m_bin.sum() == 0:
                    prob_x_per_split.append(np.nan)
                    prob_y_per_split.append(np.nan)
                    bin_sizes_per_bin.append(0)
                else:
                    # Mean predicted probability in bin (probability scale)
                    pred_mean = np.mean(p_s[m_bin])
                    prob_x_per_split.append(pred_mean)

                    # Apply Jeffreys smoothing to avoid 0/1
                    n_in_bin = m_bin.sum()
                    k_in_bin = np.sum(y_s[m_bin])
                    obs_freq_smoothed = jeffreys_smooth(k_in_bin, n_in_bin)
                    prob_y_per_split.append(obs_freq_smoothed)
                    bin_sizes_per_bin.append(n_in_bin)

            prob_x_bins.append(prob_x_per_split)
            prob_y_bins.append(prob_y_per_split)
            bin_sizes_per_split.append(bin_sizes_per_bin)

        # Aggregate across splits in PROBABILITY SPACE
        prob_x_bins = np.array(prob_x_bins, dtype=float)
        prob_y_bins = np.array(prob_y_bins, dtype=float)
        bin_sizes_per_split = np.array(bin_sizes_per_split, dtype=int)

        # Aggregate predicted and observed probabilities across splits
        # Suppress expected warnings when bins are empty across all splits
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            prob_x_mean = np.nanmean(prob_x_bins, axis=0)
            prob_y_mean = np.nanmean(prob_y_bins, axis=0)
            prob_y_lo = np.nanpercentile(prob_y_bins, CI_LOWER_PCT, axis=0)
            prob_y_hi = np.nanpercentile(prob_y_bins, CI_UPPER_PCT, axis=0)
            np.nanstd(prob_y_bins, axis=0)

        # Aggregate bin sizes across splits (use sum for consistency with prob-space)
        bin_sizes_sum = np.nansum(bin_sizes_per_split, axis=0)

        # NOW convert aggregated probabilities to logit space
        logit_x_mean = np.log(
            np.clip(prob_x_mean, eps, 1 - eps) / (1 - np.clip(prob_x_mean, eps, 1 - eps))
        )
        logit_y_mean = np.log(
            np.clip(prob_y_mean, eps, 1 - eps) / (1 - np.clip(prob_y_mean, eps, 1 - eps))
        )
        logit_y_lo = np.log(
            np.clip(prob_y_lo, eps, 1 - eps) / (1 - np.clip(prob_y_lo, eps, 1 - eps))
        )
        logit_y_hi = np.log(
            np.clip(prob_y_hi, eps, 1 - eps) / (1 - np.clip(prob_y_hi, eps, 1 - eps))
        )

        # For SD: compute logit of each smoothed split value, then take SD in logit space
        logit_curves_smooth = np.log(
            np.clip(prob_y_bins, eps, 1 - eps) / (1 - np.clip(prob_y_bins, eps, 1 - eps))
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            logit_y_sd = np.nanstd(logit_curves_smooth, axis=0)

        # Plot aggregated logit calibration bands
        valid_logit = ~np.isnan(logit_x_mean) & ~np.isnan(logit_y_mean)
        if valid_logit.sum() > 0:
            if not skip_ci_band:
                ax.fill_between(
                    logit_x_mean[valid_logit],
                    logit_y_lo[valid_logit],
                    logit_y_hi[valid_logit],
                    color=COLOR_PRIMARY,
                    alpha=ALPHA_CI,
                    label="95% CI",
                )
            ax.fill_between(
                logit_x_mean[valid_logit],
                np.clip(logit_y_mean[valid_logit] - logit_y_sd[valid_logit], -20, 20),
                np.clip(logit_y_mean[valid_logit] + logit_y_sd[valid_logit], -20, 20),
                color=COLOR_PRIMARY,
                alpha=ALPHA_SD,
                label="±1 SD",
            )

            # Plot line connecting bin centers
            ax.plot(
                logit_x_mean[valid_logit],
                logit_y_mean[valid_logit],
                "-",
                color=COLOR_PRIMARY,
                linewidth=LW_PRIMARY,
                alpha=ALPHA_LINE,
                zorder=4,
            )

            # Compute marker sizes: only variable for uniform binning
            if bin_strategy == "quantile":
                scatter_area = 36  # Fixed size for quantile binning
            else:
                valid_sizes = bin_sizes_sum[valid_logit]
                if len(valid_sizes) > 0 and valid_sizes.max() > 0:
                    # Use same sqrt-based scaling as probability-space plot
                    # for consistent legend appearance across panels
                    scatter_area = np.clip(np.sqrt(valid_sizes) * 15, 30, 350)
                else:
                    scatter_area = 36

            # Plot markers
            ax.scatter(
                logit_x_mean[valid_logit],
                logit_y_mean[valid_logit],
                s=scatter_area,
                marker="o",
                color=COLOR_PRIMARY,
                alpha=ALPHA_SCATTER,
                edgecolors=COLOR_EDGE,
                linewidth=0.5,
                label=f"Mean logit calib (n={len(unique_splits)} splits)",
                zorder=5,
            )

        # Determine axis ranges from aggregated data
        if valid_logit.sum() > 0:
            logit_range_x = [
                np.nanpercentile(logit_x_mean[valid_logit], 1) - 0.5,
                np.nanpercentile(logit_x_mean[valid_logit], 99) + 0.5,
            ]
            logit_range_y = [
                np.nanpercentile(logit_y_lo[valid_logit], 1) - 0.5,
                np.nanpercentile(logit_y_hi[valid_logit], 99) + 0.5,
            ]
        else:
            logit_range_x = [-5, 5]
            logit_range_y = [-5, 5]

    # Determine axis ranges based on actual data (for single-split mode)
    if not (unique_splits is not None and len(unique_splits) > 1):
        logit_min = np.percentile(logit_pred, 1)
        logit_max = np.percentile(logit_pred, 99)
        logit_range_x = [logit_min - 0.5, logit_max + 0.5]
        logit_range_y = list(logit_range_x)

    # Plot ideal calibration line
    ax.plot(
        logit_range_x,
        logit_range_x,
        "k--",
        linewidth=LW_SECONDARY,
        alpha=ALPHA_REFERENCE,
        label="Ideal (α=0, β=1)",
    )

    # Plot recalibration line if available
    if (
        calib_intercept is not None
        and calib_slope is not None
        and np.isfinite(calib_intercept)
        and np.isfinite(calib_slope)
    ):
        recal_x = np.array(logit_range_x)
        recal_y = calib_intercept + calib_slope * recal_x
        ax.plot(
            recal_x,
            recal_y,
            "r-",
            linewidth=LW_PRIMARY,
            alpha=ALPHA_RECALIBRATION,
            label=f"Recalibration (α={calib_intercept:.2f}, β={calib_slope:.2f})",
        )
        # Extend y-range if recalibration line goes outside
        logit_range_y = [
            min(logit_range_y[0], recal_y.min() - 0.5),
            max(logit_range_y[1], recal_y.max() + 0.5),
        ]

    # Compute binned observations (skip for multi-split, already computed above)
    if not (unique_splits is not None and len(unique_splits) > 1):
        binned_result = _binned_logits(
            y,
            p,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            min_bin_size=1,
            merge_tail=False,
        )
        bx, by, by_lo, by_hi, by_sd, bin_sizes = binned_result

        # Plot binned observations with bootstrap 95% CI and ±1 SD
        method_label = "Binned"
        if (
            bx is not None
            and by is not None
            and by_lo is not None
            and by_hi is not None
            and by_sd is not None
        ):
            # Plot 95% CI band
            if not skip_ci_band:
                ax.fill_between(
                    bx,
                    by_lo,
                    by_hi,
                    color=COLOR_PRIMARY,
                    alpha=ALPHA_CI,
                    label="95% CI",
                    zorder=3,
                )

            # Plot ±1 SD band
            ax.fill_between(
                bx,
                np.clip(by - by_sd, -20, 20),
                np.clip(by + by_sd, -20, 20),
                color=COLOR_PRIMARY,
                alpha=ALPHA_SD,
                label="±1 SD",
                zorder=4,
            )

            # Plot line connecting bin centers
            ax.plot(
                bx, by, "-", color=COLOR_PRIMARY, linewidth=LW_PRIMARY, alpha=ALPHA_LINE, zorder=5
            )

            # Compute marker sizes: use same sqrt scaling as probability plot
            if bin_strategy == "uniform" and bin_sizes is not None and len(bin_sizes) > 0:
                scatter_area = np.clip(np.sqrt(bin_sizes) * 20, 40, 450)
            else:
                scatter_area = 49

            # Plot markers
            ax.scatter(
                bx,
                by,
                s=scatter_area,
                marker="o",
                color=COLOR_PRIMARY,
                alpha=ALPHA_SCATTER,
                edgecolors=COLOR_EDGE,
                linewidth=0.5,
                label=f"Binned logits (n={len(bx)} bins)",
                zorder=6,
            )

            # Extend y-range for binned data
            logit_range_y = [
                min(logit_range_y[0], by_lo.min() - 0.5),
                max(logit_range_y[1], by_hi.max() + 0.5),
            ]
    else:
        # Multi-split mode: aggregated bands already plotted
        method_label = "Multi-split aggregated"

    ax.set_title(panel_title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Predicted logit: logit(p̂)", fontsize=FONT_LABEL)
    ylabel = f"Empirical logit ({method_label})" if method_label else "Empirical logit"
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

    # Add size legend for uniform binning (match probability-space legend style)
    if bin_strategy == "uniform":
        from matplotlib.lines import Line2D

        # Determine actual bin sizes from the data
        # Multi-split case: bin_sizes_sum already computed
        # Single-split case: bin_sizes from _binned_logits
        if unique_splits is not None and len(unique_splits) > 1:
            # Multi-split: use bin_sizes_sum from earlier computation
            actual_bin_sizes = bin_sizes_sum[bin_sizes_sum > 0]
        else:
            # Single-split: use bin_sizes from binned_result
            if bin_sizes is not None and len(bin_sizes) > 0:
                actual_bin_sizes = bin_sizes[bin_sizes > 0]
            else:
                actual_bin_sizes = np.array([])

        # Get legend reference sizes based on actual data
        reference_sizes = _get_legend_reference_sizes(actual_bin_sizes)

        size_handles = []
        size_labels = []

        # Use same sqrt-based sizing as the probability-space legend
        # so that identical reference values render at identical sizes
        if unique_splits is not None and len(unique_splits) > 1:
            sqrt_multiplier = 15
            min_scatter = 30
            max_scatter = 350
        else:
            sqrt_multiplier = 20
            min_scatter = 40
            max_scatter = 450

        for sample_count in reference_sizes:
            scatter_area = np.clip(
                np.sqrt(sample_count) * sqrt_multiplier, min_scatter, max_scatter
            )
            # Line2D markersize is diameter; convert from scatter area
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
        # Re-add main legend
        ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.9, labelspacing=1.0)
    else:
        ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.9, labelspacing=1.0)

    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(logit_range_x)
    ax.set_ylim(logit_range_y)


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
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
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
