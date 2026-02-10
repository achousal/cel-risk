"""
Reliability diagram helpers for logit-space calibration visualization.

This module provides functions for plotting calibration in logit (log-odds) space,
including binned observations with bootstrap confidence intervals and SD bands.

References:
    Van Calster et al. (2016). Calibration: the Achilles heel of predictive analytics.
    BMC Medicine.
"""

import warnings

import numpy as np

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
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_PRIMARY,
    LW_SECONDARY,
)


def get_legend_reference_sizes(actual_sizes: np.ndarray) -> list:
    """
    Compute appropriate legend reference sizes based on actual bin sizes.

    Uses adaptive rounding based on scale to ensure visual separation in legend.
    For large numbers (>1000), uses logarithmic-like scaling.

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

    # Adaptive rounding based on scale
    def round_to_nice(x):
        if x < 50:
            # Round to multiples of 10
            return int(np.round(x / 10) * 10)
        elif x < 200:
            # Round to multiples of 25
            return int(np.round(x / 25) * 25)
        elif x < 1000:
            # Round to multiples of 50
            return int(np.round(x / 50) * 50)
        elif x < 5000:
            # Round to multiples of 250
            return int(np.round(x / 250) * 250)
        else:
            # Round to multiples of 500 or 1000 for very large numbers
            if x < 10000:
                return int(np.round(x / 500) * 500)
            else:
                return int(np.round(x / 1000) * 1000)

    # For large ranges, use log-spaced percentiles for better visual separation
    if max_size > 1000:
        # Log-space percentiles (10th, 30th, 60th, 90th)
        # This gives better separation when sizes span orders of magnitude
        percentiles = [10, 30, 60, 90]
        reference_points = [round_to_nice(np.percentile(actual_sizes, p)) for p in percentiles]
        # Add max if it's significantly larger than 90th percentile
        p90 = reference_points[-1]
        if max_size > p90 * 1.3:
            reference_points.append(round_to_nice(max_size))
    else:
        # Standard quartile-based reference points for moderate ranges
        q25 = round_to_nice(np.percentile(actual_sizes, 25))
        q50 = round_to_nice(np.percentile(actual_sizes, 50))
        q75 = round_to_nice(np.percentile(actual_sizes, 75))
        q_max = round_to_nice(max_size)
        reference_points = [q25, q50, q75, q_max]

    # Filter duplicates and sort
    sizes = sorted(set(reference_points))

    # Ensure we have at least 2 reference points
    if len(sizes) < 2:
        sizes = [min_size, max_size]

    return sizes


def binned_logits(
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
    xs_list, ys_list, ys_lo_list, ys_hi_list, ys_sd_list, sizes_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
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


def plot_logit_calibration_panel(
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
        label="Ideal (alpha=0, beta=1)",
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
            label=f"Recalibration (alpha={calib_intercept:.2f}, beta={calib_slope:.2f})",
        )
        # Extend y-range if recalibration line goes outside
        logit_range_y = [
            min(logit_range_y[0], recal_y.min() - 0.5),
            max(logit_range_y[1], recal_y.max() + 0.5),
        ]

    # Compute binned observations (skip for multi-split, already computed above)
    if not (unique_splits is not None and len(unique_splits) > 1):
        binned_result = binned_logits(
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
                bx,
                by,
                "-",
                color=COLOR_PRIMARY,
                linewidth=LW_PRIMARY,
                alpha=ALPHA_LINE,
                zorder=5,
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
        # Single-split case: bin_sizes from binned_logits
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
        reference_sizes = get_legend_reference_sizes(actual_bin_sizes)

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
