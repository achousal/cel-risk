"""Risk distribution plotting for CeliacRiskML.

Creates publication-quality risk score distribution plots with:
- Multi-panel layouts for incident/prevalent/control cases
- Clinical threshold overlays (DCA, Youden, specificity)
- Performance metrics at thresholds
- Density estimation and summary statistics
"""

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
    dca_threshold: float | None = None,
    spec_target_threshold: float | None = None,
    youden_threshold: float | None = None,
    metrics_at_thresholds: dict[str, dict[str, float]] | None = None,
    x_limits: tuple[float, float] | None = None,
    target_spec: float = 0.95,
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
        dca_threshold: DCA zero-crossing threshold (0-1) [deprecated, use threshold_bundle]
        spec_target_threshold: Target specificity threshold (0-1) [deprecated, use threshold_bundle]
        youden_threshold: Youden's J statistic threshold (0-1) [deprecated, use threshold_bundle]
        metrics_at_thresholds: Performance metrics at each threshold [deprecated, use threshold_bundle]
        x_limits: Optional tuple (xmin, xmax) for x-axis range
        target_spec: Target specificity value for annotation label (default: 0.95)
        threshold_bundle: ThresholdBundle from compute_threshold_bundle() - preferred interface.
            If provided, overrides individual threshold parameters.

    Notes:
        - If category_col is provided, creates three-category KDE plot
        - If y_true is provided without category_col, creates binary histogram
        - If neither is provided, creates single-category histogram
        - Incident/prevalent subplots only shown if category_col includes those categories
    """
    # If threshold_bundle provided, extract values (preferred interface)
    if threshold_bundle is not None:
        youden_threshold = threshold_bundle.get("youden_threshold")
        spec_target_threshold = threshold_bundle.get("spec_target_threshold")
        dca_threshold = threshold_bundle.get("dca_threshold")
        target_spec = threshold_bundle.get("target_spec", 0.95)

        # Build metrics_at_thresholds from bundle
        metrics_at_thresholds = {}
        if "youden" in threshold_bundle:
            m = threshold_bundle["youden"]
            metrics_at_thresholds["youden"] = {
                "sensitivity": m.get("sensitivity"),
                "precision": m.get("precision"),
                "fp": m.get("fp"),
                "n_celiac": (m.get("tp", 0) or 0) + (m.get("fn", 0) or 0),
            }
        if "spec_target" in threshold_bundle:
            m = threshold_bundle["spec_target"]
            metrics_at_thresholds["spec_target"] = {
                "sensitivity": m.get("sensitivity"),
                "precision": m.get("precision"),
                "fp": m.get("fp"),
                "n_celiac": (m.get("tp", 0) or 0) + (m.get("fn", 0) or 0),
            }
        if "dca" in threshold_bundle:
            m = threshold_bundle["dca"]
            metrics_at_thresholds["dca"] = {
                "sensitivity": m.get("sensitivity"),
                "precision": m.get("precision"),
                "fp": m.get("fp"),
                "n_celiac": (m.get("tp", 0) or 0) + (m.get("fn", 0) or 0),
            }

    # Normalize thresholds to [0, 1] with epsilon tolerance
    # This handles edge cases where threshold computation may produce values
    # slightly outside [0, 1] due to floating point arithmetic (e.g., max(p) + 1e-12)
    spec_target_threshold = _normalize_threshold(spec_target_threshold)
    youden_threshold = _normalize_threshold(youden_threshold)
    dca_threshold = _normalize_threshold(dca_threshold)

    configure_backend()

    s = np.asarray(scores).astype(float)

    # Determine if we have incident/prevalent subplots to show
    has_incident = False
    has_prevalent = False
    if category_col is not None:
        cat = np.asarray(category_col)
        mask = np.isfinite(s)
        s = s[mask]
        cat = cat[mask]
        has_incident = np.any(cat == "Incident")
        has_prevalent = np.any(cat == "Prevalent")

    # Calculate number of subplots needed
    n_subplots = 1
    if has_incident:
        n_subplots += 1
    if has_prevalent:
        n_subplots += 1

    # Create figure with appropriate number of subplots
    # Determine figure size based on plot type (histogram vs KDE)
    if n_subplots == 1:
        height_ratios = [1]
        if category_col is not None:
            # Single KDE plot: 4:1 aspect ratio
            figsize = (12, 3)
        else:
            # Single histogram: 3:2 aspect ratio
            figsize = (9, 6)
    elif n_subplots == 2:
        # Main histogram (3:2) + 1 KDE subplot (4:1)
        # With width=9: histogram needs height=6, KDE needs height=2.25
        # Ratio: 6 / 2.25 = 8 / 3
        height_ratios = [8, 3]
        figsize = (9, 8.5)
    elif n_subplots == 3:
        # Main histogram (3:2) + 2 KDE subplots (4:1 each)
        # Ratio: 6 : 2.25 : 2.25 = 8 : 3 : 3
        height_ratios = [8, 3, 3]
        figsize = (9, 11)
    else:
        # Fallback for more subplots
        height_ratios = [8] + [3] * (n_subplots - 1)
        figsize = (9, 6 + 2.5 * (n_subplots - 1))

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, height_ratios=height_ratios)
    if n_subplots == 1:
        axes = [axes]

    ax_main = axes[0]

    # === MAIN HISTOGRAM (ax_main) ===
    if y_true is None and category_col is None:
        mask = np.isfinite(s)
        s = s[mask]
        if len(s) == 0:
            plt.close()
            return
        bins = min(60, max(10, int(np.sqrt(len(s)))))
        ax_main.hist(
            s,
            bins=bins,
            density=True,
            alpha=0.7,
            color=COLOR_PRIMARY,
            edgecolor="white",
        )
    elif category_col is not None:
        # Use category column for three-way split (Controls, Incident, Prevalent)
        # Note: s and cat already filtered for NaN at lines 146-148
        if len(s) == 0:
            plt.close()
            return
        bins = min(60, max(10, int(np.sqrt(len(s)))))

        # Define three categories with distinct colors
        categories = [
            ("Controls", COLOR_PRIMARY, "Controls"),
            ("Incident", COLOR_TERTIARY, "Incident"),
            ("Prevalent", COLOR_SECONDARY, "Prevalent"),
        ]

        for label, color, cat_name in categories:
            vals = s[cat == cat_name]
            if len(vals) == 0:
                continue
            ax_main.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                edgecolor="white",
                label=label,
            )

        if ax_main.get_legend_handles_labels()[0]:
            ax_main.legend(loc="upper right", fontsize=FONT_LEGEND)
    else:
        y = np.asarray(y_true).astype(int)
        mask = np.isfinite(s) & np.isfinite(y)
        s = s[mask]
        y = y[mask]
        if len(s) == 0:
            plt.close()
            return
        bins = min(60, max(10, int(np.sqrt(len(s)))))
        for label, color, target in [
            (CONTROL_LABEL, COLOR_PRIMARY, 0),
            (pos_label, COLOR_TERTIARY, 1),
        ]:
            vals = s[y == target]
            if len(vals) == 0:
                continue
            ax_main.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                edgecolor="white",
                label=label,
            )
        if ax_main.get_legend_handles_labels()[0]:
            ax_main.legend(loc="upper right", fontsize=FONT_LEGEND)

    # Add threshold lines (without labels - will be added to legend separately)
    # Note: Thresholds are pre-normalized to [0, 1] via _normalize_threshold()

    # Detect overlapping thresholds and apply offsets so all lines are visible
    # Overlap threshold: 1% of x-range (0.01 for [0,1] range)
    overlap_eps = 0.01
    offset_amount = 0.003  # Tiny offset when lines overlap (just enough to see both)

    def _thresholds_overlap(t1: float | None, t2: float | None) -> bool:
        """Check if two thresholds are close enough to visually overlap."""
        if t1 is None or t2 is None:
            return False
        return abs(t1 - t2) < overlap_eps

    # Compute offsets for each threshold to avoid overlap
    spec_target_offset = 0.0
    youden_offset = 0.0
    dca_offset = 0.0

    # Check spec_target vs youden overlap
    if _thresholds_overlap(spec_target_threshold, youden_threshold):
        spec_target_offset = -offset_amount  # Shift spec_target left
        youden_offset = offset_amount  # Shift youden right

    # Check spec_target vs dca overlap (if not already offset)
    if _thresholds_overlap(spec_target_threshold, dca_threshold):
        if spec_target_offset == 0.0:
            spec_target_offset = -offset_amount
        dca_offset = offset_amount * 1.5  # Shift dca further right

    # Check youden vs dca overlap
    if _thresholds_overlap(youden_threshold, dca_threshold):
        if youden_offset == 0.0:
            youden_offset = -offset_amount
        if dca_offset == 0.0:
            dca_offset = offset_amount

    # Draw threshold lines with z-order to ensure visibility
    # Higher zorder = drawn on top
    if spec_target_threshold is not None:
        ax_main.axvline(
            spec_target_threshold + spec_target_offset,
            color="red",
            linestyle="--",
            linewidth=LW_PRIMARY,
            alpha=0.7,
            zorder=10,  # Red (spec_target) on top
        )

    if youden_threshold is not None:
        ax_main.axvline(
            youden_threshold + youden_offset,
            color="green",
            linestyle="--",
            linewidth=LW_PRIMARY,
            alpha=0.7,
            zorder=9,  # Green (youden) middle
        )

    if dca_threshold is not None:
        ax_main.axvline(
            dca_threshold + dca_offset,
            color=COLOR_SECONDARY,
            linestyle="--",
            linewidth=LW_PRIMARY,
            alpha=0.7,
            zorder=8,
        )

    # Create comprehensive legend with threshold metrics
    handles, labels = ax_main.get_legend_handles_labels()
    threshold_handles = []
    threshold_labels = []

    if spec_target_threshold is not None:
        m = metrics_at_thresholds.get("spec_target") if metrics_at_thresholds else None

        line_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=LW_PRIMARY, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format with each metric on separate line
        label_text = f"{target_spec*100:.0f}% Spec"
        if m:
            sens = m.get("sensitivity", np.nan)
            ppv = m.get("precision", np.nan)
            fp = m.get("fp", np.nan)
            if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
                label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        threshold_labels.append(label_text)

    if youden_threshold is not None:
        m = metrics_at_thresholds.get("youden") if metrics_at_thresholds else None

        line_handle = Line2D(
            [0], [0], color="green", linestyle="--", linewidth=LW_PRIMARY, alpha=0.7
        )
        threshold_handles.append(line_handle)

        # Multi-line label format with each metric on separate line
        label_text = "Youden"
        if m:
            sens = m.get("sensitivity", np.nan)
            ppv = m.get("precision", np.nan)
            fp = m.get("fp", np.nan)
            if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
                label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        threshold_labels.append(label_text)

    if dca_threshold is not None:
        m = metrics_at_thresholds.get("dca") if metrics_at_thresholds else None

        line_handle = Line2D(
            [0], [0], color=COLOR_SECONDARY, linestyle="--", linewidth=LW_PRIMARY, alpha=0.7
        )
        threshold_handles.append(line_handle)

        # Multi-line label format with each metric on separate line
        label_text = "DCA"
        if m:
            sens = m.get("sensitivity", np.nan)
            ppv = m.get("precision", np.nan)
            fp = m.get("fp", np.nan)
            if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
                label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        threshold_labels.append(label_text)

    # Combine all handles and labels
    all_handles = handles + threshold_handles
    all_labels = labels + threshold_labels

    # Create legend outside plot area
    if all_handles:
        ax_main.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=FONT_LEGEND,
            framealpha=0.9,
        )

    if subtitle:
        ax_main.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE)
    else:
        ax_main.set_title(title, fontsize=FONT_TITLE)
    ax_main.set_ylabel("Density")
    ax_main.grid(True, alpha=0.2)

    # Apply x-axis limits if provided
    if x_limits is not None:
        ax_main.set_xlim(x_limits)
    elif category_col is not None:
        # When using category_col (3-way split), match subplot xlim for consistency
        ax_main.set_xlim(0, 1)

    # === INCIDENT DENSITY PLOT (if applicable) ===
    subplot_idx = 1
    if has_incident:
        ax_incident = axes[subplot_idx]
        subplot_idx += 1

        incident_scores = s[cat == "Incident"]
        stats = compute_distribution_stats(incident_scores)

        # Create KDE density plot
        if len(incident_scores) > 0:
            try:
                kde = gaussian_kde(incident_scores, bw_method="scott")
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)
                ax_incident.plot(
                    x_range, density, color=COLOR_TERTIARY, linewidth=LW_PRIMARY, alpha=0.8
                )
                ax_incident.fill_between(x_range, density, alpha=0.3, color=COLOR_TERTIARY)
            except Exception:
                # Fallback to histogram if KDE fails (e.g., too few points)
                ax_incident.hist(
                    incident_scores,
                    bins=20,
                    density=True,
                    alpha=0.7,
                    color=COLOR_TERTIARY,
                    edgecolor="white",
                )

        # Add threshold lines (no labels) with same offsets as main plot
        # Note: Thresholds are pre-normalized to [0, 1] via _normalize_threshold()
        if spec_target_threshold is not None:
            ax_incident.axvline(
                spec_target_threshold + spec_target_offset,
                color="red",
                linestyle="--",
                linewidth=LW_SECONDARY,
                alpha=0.5,
                zorder=10,
            )
        if youden_threshold is not None:
            ax_incident.axvline(
                youden_threshold + youden_offset,
                color="green",
                linestyle="--",
                linewidth=LW_SECONDARY,
                alpha=0.5,
                zorder=9,
            )
        if dca_threshold is not None:
            ax_incident.axvline(
                dca_threshold + dca_offset,
                color=COLOR_SECONDARY,
                linestyle="--",
                linewidth=LW_SECONDARY,
                alpha=0.5,
                zorder=8,
            )

        ax_incident.set_xlim(0, 1)
        ax_incident.set_ylabel("Incident\nDensity", fontsize=9)
        ax_incident.grid(True, alpha=0.2, axis="x")
        ax_incident.set_yticks([])

        # Add statistics text
        stats_text = (
            f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
            f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}"
        )
        ax_incident.text(
            0.02,
            0.95,
            stats_text,
            transform=ax_incident.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    # === PREVALENT DENSITY PLOT (if applicable) ===
    if has_prevalent:
        ax_prevalent = axes[subplot_idx]

        prevalent_scores = s[cat == "Prevalent"]
        stats = compute_distribution_stats(prevalent_scores)

        # Create KDE density plot
        if len(prevalent_scores) > 0:
            try:
                kde = gaussian_kde(prevalent_scores, bw_method="scott")
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)
                ax_prevalent.plot(
                    x_range, density, color=COLOR_SECONDARY, linewidth=LW_PRIMARY, alpha=0.8
                )
                ax_prevalent.fill_between(x_range, density, alpha=0.3, color=COLOR_SECONDARY)
            except Exception:
                # Fallback to histogram if KDE fails
                ax_prevalent.hist(
                    prevalent_scores,
                    bins=20,
                    density=True,
                    alpha=0.7,
                    color=COLOR_SECONDARY,
                    edgecolor="white",
                )

        # Add threshold lines (no labels) with same offsets as main plot
        # Note: Thresholds are pre-normalized to [0, 1] via _normalize_threshold()
        if spec_target_threshold is not None:
            ax_prevalent.axvline(
                spec_target_threshold + spec_target_offset,
                color="red",
                linestyle="--",
                linewidth=LW_SECONDARY,
                alpha=0.5,
                zorder=10,
            )
        if youden_threshold is not None:
            ax_prevalent.axvline(
                youden_threshold + youden_offset,
                color="green",
                linestyle="--",
                linewidth=LW_SECONDARY,
                alpha=0.5,
                zorder=9,
            )
        if dca_threshold is not None:
            ax_prevalent.axvline(
                dca_threshold + dca_offset,
                color=COLOR_SECONDARY,
                linestyle="--",
                linewidth=LW_SECONDARY,
                alpha=0.5,
                zorder=8,
            )

        ax_prevalent.set_xlim(0, 1)
        ax_prevalent.set_ylabel("Prevalent\nDensity", fontsize=9)
        ax_prevalent.set_xlabel(xlabel)
        ax_prevalent.grid(True, alpha=0.2, axis="x")
        ax_prevalent.set_yticks([])

        # Add statistics text
        stats_text = (
            f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
            f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}"
        )
        ax_prevalent.text(
            0.02,
            0.95,
            stats_text,
            transform=ax_prevalent.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )
    else:
        # If no prevalent subplot, add xlabel to last subplot
        if has_incident:
            axes[-1].set_xlabel(xlabel)
        else:
            ax_main.set_xlabel(xlabel)

    # Apply metadata and adjust layout
    bottom_margin = apply_plot_metadata(fig, meta_lines) if meta_lines else 0.1
    plt.subplots_adjust(left=0.12, right=0.70, top=0.92, bottom=bottom_margin, hspace=0.3)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
