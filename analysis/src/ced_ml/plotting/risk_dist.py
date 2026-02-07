"""Risk distribution plotting for CeliacRiskML.

Creates publication-quality risk score distribution plots with:
- Multi-panel layouts for incident/prevalent/control cases
- Clinical threshold overlays (DCA, Youden, specificity)
- Performance metrics at thresholds
- Density estimation and summary statistics
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from ced_ml.data.schema import CONTROL_LABEL
from ced_ml.utils.math_utils import EPSILON_BOUNDS

from .dca import apply_plot_metadata
from .style import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    DPI,
    FONT_LEGEND,
    FONT_TITLE,
    LW_PRIMARY,
    LW_SECONDARY,
    configure_backend,
)

logger = logging.getLogger(__name__)

# Threshold overlap detection parameters
_OVERLAP_EPSILON = 0.01  # 1% of x-range for overlap detection
_OFFSET_AMOUNT = 0.003  # Tiny offset when lines overlap


def compute_distribution_stats(scores: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a distribution of scores.

    Args:
        scores: Array of numeric scores

    Returns:
        Dictionary with keys: mean, median, iqr, sd
    """
    scores = np.asarray(scores).astype(float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return {"mean": np.nan, "median": np.nan, "iqr": np.nan, "sd": np.nan}

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)

    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "iqr": float(q3 - q1),
        "sd": float(np.std(scores)),
    }


def _normalize_threshold(value: float | None) -> float | None:
    """Normalize a threshold to [0, 1] for plotting, or drop invalid values.

    Handles edge cases where threshold computation may produce values slightly
    outside [0, 1] due to floating point arithmetic (e.g., max(p) + 1e-12).

    Args:
        value: Threshold value to normalize

    Returns:
        Normalized threshold in [0, 1], or None if invalid

    Notes:
        - Allows small epsilon (EPSILON_BOUNDS=1e-6) tolerance outside [0, 1]
        - Clamps values to [0, 1] range
        - Returns None for non-finite or severely out-of-range values
    """
    if value is None:
        return None
    try:
        thresh = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(thresh):
        return None

    # Allow small epsilon outside [0, 1] for floating point precision
    if thresh < 0.0 - EPSILON_BOUNDS or thresh > 1.0 + EPSILON_BOUNDS:
        return None

    # Clamp to [0, 1]
    return float(min(max(thresh, 0.0), 1.0))


def _compute_threshold_offsets(
    spec_target: float | None,
    youden: float | None,
    dca: float | None,
) -> tuple[float, float, float]:
    """Compute visual offsets for overlapping threshold lines.

    When threshold values are very close, apply small horizontal offsets
    to ensure all lines are visible in the plot.

    Args:
        spec_target: Specificity target threshold (0-1)
        youden: Youden index threshold (0-1)
        dca: DCA optimal threshold (0-1)

    Returns:
        Tuple of (spec_target_offset, youden_offset, dca_offset)
    """

    def thresholds_overlap(t1: float | None, t2: float | None) -> bool:
        if t1 is None or t2 is None:
            return False
        return abs(t1 - t2) < _OVERLAP_EPSILON

    spec_target_offset = 0.0
    youden_offset = 0.0
    dca_offset = 0.0

    # Check spec_target vs youden overlap
    if thresholds_overlap(spec_target, youden):
        spec_target_offset = -_OFFSET_AMOUNT
        youden_offset = _OFFSET_AMOUNT

    # Check spec_target vs dca overlap
    if thresholds_overlap(spec_target, dca):
        if spec_target_offset == 0.0:
            spec_target_offset = -_OFFSET_AMOUNT
        dca_offset = _OFFSET_AMOUNT * 1.5

    # Check youden vs dca overlap
    if thresholds_overlap(youden, dca):
        if youden_offset == 0.0:
            youden_offset = -_OFFSET_AMOUNT
        if dca_offset == 0.0:
            dca_offset = _OFFSET_AMOUNT

    return spec_target_offset, youden_offset, dca_offset


def _draw_threshold_lines(
    ax: plt.Axes,
    spec_target: float | None,
    youden: float | None,
    dca: float | None,
    offsets: tuple[float, float, float],
    linewidth: float = LW_PRIMARY,
    alpha: float = 0.7,
) -> None:
    """Draw vertical threshold lines on an axis.

    Args:
        ax: Matplotlib axis to draw on
        spec_target: Specificity target threshold value
        youden: Youden index threshold value
        dca: DCA optimal threshold value
        offsets: Tuple of (spec_target_offset, youden_offset, dca_offset)
        linewidth: Line width for threshold lines
        alpha: Alpha transparency for lines
    """
    spec_target_offset, youden_offset, dca_offset = offsets

    if spec_target is not None:
        ax.axvline(
            spec_target + spec_target_offset,
            color="red",
            linestyle="--",
            linewidth=linewidth,
            alpha=alpha,
            zorder=10,
        )

    if youden is not None:
        ax.axvline(
            youden + youden_offset,
            color="green",
            linestyle="--",
            linewidth=linewidth,
            alpha=alpha,
            zorder=9,
        )

    if dca is not None:
        ax.axvline(
            dca + dca_offset,
            color=COLOR_SECONDARY,
            linestyle="--",
            linewidth=linewidth,
            alpha=alpha,
            zorder=8,
        )


def _build_threshold_legend(
    spec_target: float | None,
    youden: float | None,
    dca: float | None,
    target_spec: float,
    metrics_at_thresholds: dict,
) -> tuple[list, list]:
    """Build legend handles and labels for threshold lines.

    Creates multi-line labels with threshold name and performance metrics
    (sensitivity, PPV, false positives).

    Args:
        spec_target: Specificity target threshold value
        youden: Youden index threshold value
        dca: DCA optimal threshold value
        target_spec: Target specificity value (e.g., 0.95)
        metrics_at_thresholds: Dict with metrics for each threshold type

    Returns:
        Tuple of (handles, labels) for matplotlib legend
    """
    handles = []
    labels = []

    if spec_target is not None:
        m = metrics_at_thresholds.get("spec_target")
        line_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=LW_PRIMARY, alpha=0.7)
        handles.append(line_handle)

        label_text = f"{target_spec*100:.0f}% Spec"
        if m:
            sens = m.get("sensitivity", np.nan)
            ppv = m.get("precision", np.nan)
            fp = m.get("fp", np.nan)
            if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
                label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        labels.append(label_text)

    if youden is not None:
        m = metrics_at_thresholds.get("youden")
        line_handle = Line2D(
            [0], [0], color="green", linestyle="--", linewidth=LW_PRIMARY, alpha=0.7
        )
        handles.append(line_handle)

        label_text = "Youden"
        if m:
            sens = m.get("sensitivity", np.nan)
            ppv = m.get("precision", np.nan)
            fp = m.get("fp", np.nan)
            if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
                label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        labels.append(label_text)

    if dca is not None:
        m = metrics_at_thresholds.get("dca")
        line_handle = Line2D(
            [0], [0], color=COLOR_SECONDARY, linestyle="--", linewidth=LW_PRIMARY, alpha=0.7
        )
        handles.append(line_handle)

        label_text = "DCA"
        if m:
            sens = m.get("sensitivity", np.nan)
            ppv = m.get("precision", np.nan)
            fp = m.get("fp", np.nan)
            if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
                label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        labels.append(label_text)

    return handles, labels


def _plot_main_histogram(
    ax: plt.Axes,
    scores: np.ndarray,
    y_true: np.ndarray | None = None,
    category_col: np.ndarray | None = None,
    pos_label: str = "Incident CeD",
) -> None:
    """Plot main histogram/density on the given axis.

    Supports three modes:
    - Single distribution (no y_true or category_col)
    - Binary split by y_true (Controls vs Incident)
    - Three-way split by category_col (Controls, Incident, Prevalent)

    Args:
        ax: Matplotlib axis to plot on
        scores: Risk scores array
        y_true: Optional binary labels for two-class split
        category_col: Optional category labels for three-class split
        pos_label: Label for positive class
    """
    bins = min(60, max(10, int(np.sqrt(len(scores)))))

    if y_true is None and category_col is None:
        # Single distribution
        ax.hist(
            scores,
            bins=bins,
            density=True,
            alpha=0.7,
            color=COLOR_PRIMARY,
            edgecolor="white",
        )
    elif category_col is not None:
        # Three-way split (Controls, Incident, Prevalent)
        categories = [
            ("Controls", COLOR_PRIMARY, "Controls"),
            ("Incident", COLOR_TERTIARY, "Incident"),
            ("Prevalent", COLOR_SECONDARY, "Prevalent"),
        ]

        for label, color, cat_name in categories:
            vals = scores[category_col == cat_name]
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                edgecolor="white",
                label=label,
            )

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=FONT_LEGEND)
    else:
        # Binary split (Controls vs Incident)
        for label, color, target in [
            (CONTROL_LABEL, COLOR_PRIMARY, 0),
            (pos_label, COLOR_TERTIARY, 1),
        ]:
            vals = scores[y_true == target]
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                edgecolor="white",
                label=label,
            )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=FONT_LEGEND)


def _add_statistics_annotation(ax: plt.Axes, stats: dict[str, float]) -> None:
    """Add statistics text annotation to axis.

    Args:
        ax: Matplotlib axis to annotate
        stats: Statistics dict with keys: mean, median, iqr, sd
    """
    stats_text = (
        f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
        f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )


def _plot_density_subplot(
    ax: plt.Axes,
    scores: np.ndarray,
    color: str,
    ylabel: str,
    xlabel: str,
    threshold_values: tuple[float | None, float | None, float | None],
    threshold_offsets: tuple[float, float, float],
    show_xlabel: bool = False,
) -> None:
    """Plot density (KDE) subplot for a specific category.

    Args:
        ax: Matplotlib axis to plot on
        scores: Score array for this category
        color: Color for density plot
        ylabel: Y-axis label
        xlabel: X-axis label
        threshold_values: Tuple of (spec_target, youden, dca) thresholds
        threshold_offsets: Tuple of (spec_target_offset, youden_offset, dca_offset)
        show_xlabel: Whether to show x-axis label
    """
    # Compute statistics
    stats = compute_distribution_stats(scores)

    # Create KDE density plot
    if len(scores) > 0:
        try:
            kde = gaussian_kde(scores, bw_method="scott")
            x_range = np.linspace(0, 1, 200)
            density = kde(x_range)
            ax.plot(x_range, density, color=color, linewidth=LW_PRIMARY, alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.3, color=color)
        except Exception:
            # Fallback to histogram if KDE fails
            ax.hist(
                scores,
                bins=20,
                density=True,
                alpha=0.7,
                color=color,
                edgecolor="white",
            )

    # Draw threshold lines
    _draw_threshold_lines(
        ax,
        threshold_values[0],
        threshold_values[1],
        threshold_values[2],
        threshold_offsets,
        linewidth=LW_SECONDARY,
        alpha=0.5,
    )

    # Configure axis
    ax.set_xlim(0, 1)
    ax.set_ylabel(ylabel, fontsize=9)
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_yticks([])

    # Add statistics annotation
    _add_statistics_annotation(ax, stats)


def plot_risk_distribution(
    y_true: np.ndarray | None,
    scores: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    xlabel: str = "Predicted risk",
    pos_label: str = "Incident CeD",
    meta_lines: Sequence[str] | None = None,
    category_col: np.ndarray | None = None,
    x_limits: tuple[float, float] | None = None,
    threshold_bundle: dict | None = None,
) -> None:
    """Plot risk score distribution with optional thresholds and case-type subplots.

    Creates a multi-panel plot with:
    - Main histogram/KDE showing overall distribution
    - Optional incident-only density subplot
    - Optional prevalent-only density subplot
    - Threshold lines with performance metrics

    Args:
        y_true: Binary outcome labels (0/1), optional if category_col provided
        scores: Risk scores (0-1 range)
        out_path: Path to save figure
        title: Plot title
        subtitle: Optional subtitle
        xlabel: X-axis label
        pos_label: Label for positive class (e.g., "Incident CeD")
        meta_lines: Metadata lines for bottom of figure
        category_col: Array of category labels ("Controls", "Incident", "Prevalent")
        x_limits: Optional tuple (xmin, xmax) for x-axis range
        threshold_bundle: ThresholdBundle from compute_threshold_bundle().
            Contains threshold values and metrics for Youden, specificity target, and DCA.

    Notes:
        - If category_col is provided, creates three-category KDE plot
        - If y_true is provided without category_col, creates binary histogram
        - If neither is provided, creates single-category histogram
        - Incident/prevalent subplots only shown if category_col includes those categories
    """
    configure_backend()

    # Parse threshold bundle
    spec_target_threshold = None
    youden_threshold = None
    dca_threshold = None
    target_spec = 0.95
    metrics_at_thresholds = {}

    if threshold_bundle is not None:
        spec_target_threshold = _normalize_threshold(threshold_bundle.get("spec_target_threshold"))
        youden_threshold = _normalize_threshold(threshold_bundle.get("youden_threshold"))
        dca_threshold = _normalize_threshold(threshold_bundle.get("dca_threshold"))
        target_spec = threshold_bundle.get("target_spec", 0.95)

        # Extract metrics for each threshold
        for key, threshold_key in [
            ("youden", "youden"),
            ("spec_target", "spec_target"),
            ("dca", "dca"),
        ]:
            if key in threshold_bundle:
                m = threshold_bundle[key]
                metrics_at_thresholds[threshold_key] = {
                    "sensitivity": m.get("sensitivity"),
                    "precision": m.get("precision"),
                    "fp": m.get("fp"),
                    "n_celiac": (m.get("tp", 0) or 0) + (m.get("fn", 0) or 0),
                }

    # Prepare data
    s = np.asarray(scores).astype(float)
    y = np.asarray(y_true).astype(int) if y_true is not None else None

    # Validate prediction range
    if len(s) > 0 and (np.min(s) < 0 or np.max(s) > 1):
        logger.warning(f"Risk scores outside [0,1] range: min={np.min(s):.4f}, max={np.max(s):.4f}")

    # Filter for category_col case
    has_incident = False
    has_prevalent = False
    cat = None
    if category_col is not None:
        cat = np.asarray(category_col)
        mask = np.isfinite(s)
        s = s[mask]
        cat = cat[mask]
        if y is not None:
            y = y[mask]
        has_incident = np.any(cat == "Incident")
        has_prevalent = np.any(cat == "Prevalent")
    elif y is not None:
        # Binary case
        mask = np.isfinite(s) & np.isfinite(y)
        s = s[mask]
        y = y[mask]
    else:
        # Single distribution
        mask = np.isfinite(s)
        s = s[mask]

    if len(s) == 0:
        logger.warning("No valid data for risk distribution plot after filtering")
        plt.close()
        return

    # Compute threshold offsets
    threshold_offsets = _compute_threshold_offsets(
        spec_target_threshold, youden_threshold, dca_threshold
    )

    # Determine subplot layout
    n_subplots = 1 + int(has_incident) + int(has_prevalent)
    height_ratios = [8] + [3] * (n_subplots - 1) if n_subplots > 1 else [1]

    if n_subplots == 1:
        figsize = (12, 3) if category_col is not None else (9, 6)
    elif n_subplots == 2:
        figsize = (9, 8.5)
    elif n_subplots == 3:
        figsize = (9, 11)
    else:
        figsize = (9, 6 + 2.5 * (n_subplots - 1))

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, height_ratios=height_ratios)
    if n_subplots == 1:
        axes = [axes]

    ax_main = axes[0]

    # Plot main histogram
    _plot_main_histogram(ax_main, s, y, cat, pos_label)

    # Draw threshold lines
    _draw_threshold_lines(
        ax_main,
        spec_target_threshold,
        youden_threshold,
        dca_threshold,
        threshold_offsets,
    )

    # Build and apply legend
    handles, labels = ax_main.get_legend_handles_labels()
    threshold_handles, threshold_labels = _build_threshold_legend(
        spec_target_threshold,
        youden_threshold,
        dca_threshold,
        target_spec,
        metrics_at_thresholds,
    )

    all_handles = handles + threshold_handles
    all_labels = labels + threshold_labels

    if all_handles:
        ax_main.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=FONT_LEGEND,
            framealpha=0.9,
        )

    # Configure main axis
    if subtitle:
        ax_main.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE)
    else:
        ax_main.set_title(title, fontsize=FONT_TITLE)
    ax_main.set_ylabel("Density")
    ax_main.grid(True, alpha=0.2)

    if x_limits is not None:
        ax_main.set_xlim(x_limits)
    elif category_col is not None:
        ax_main.set_xlim(0, 1)

    # Plot incident subplot if needed
    subplot_idx = 1
    if has_incident:
        incident_scores = s[cat == "Incident"]
        _plot_density_subplot(
            axes[subplot_idx],
            incident_scores,
            COLOR_TERTIARY,
            "Incident\nDensity",
            xlabel,
            (spec_target_threshold, youden_threshold, dca_threshold),
            threshold_offsets,
            show_xlabel=not has_prevalent,
        )
        subplot_idx += 1

    # Plot prevalent subplot if needed
    if has_prevalent:
        prevalent_scores = s[cat == "Prevalent"]
        _plot_density_subplot(
            axes[subplot_idx],
            prevalent_scores,
            COLOR_SECONDARY,
            "Prevalent\nDensity",
            xlabel,
            (spec_target_threshold, youden_threshold, dca_threshold),
            threshold_offsets,
            show_xlabel=True,
        )
    elif not has_incident:
        # No subplots, add xlabel to main
        ax_main.set_xlabel(xlabel)

    # Apply metadata and save
    bottom_margin = apply_plot_metadata(fig, meta_lines) if meta_lines else 0.1
    plt.subplots_adjust(left=0.12, right=0.70, top=0.92, bottom=bottom_margin, hspace=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
