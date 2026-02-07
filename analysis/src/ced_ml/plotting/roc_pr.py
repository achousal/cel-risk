"""
ROC and Precision-Recall curve plotting.

Provides functions to generate ROC and PR curves with confidence intervals
and threshold annotations.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .dca import apply_plot_metadata
from .style import (
    ALPHA_CI,
    ALPHA_SD,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    DPI,
    FIGSIZE_ROC,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_PRIMARY,
    LW_REFERENCE,
    PAD_INCHES,
    configure_backend,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib  # noqa: F401

    configure_backend()
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    split_ids: np.ndarray | None = None,
    meta_lines: Sequence[str] | None = None,
    threshold_bundle: dict | None = None,
    skip_ci_band: bool = False,
) -> None:
    """
    Plot ROC curve with optional split-wise confidence bands and threshold markers.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Output file path
        title: Plot title
        subtitle: Optional subtitle
        split_ids: Array indicating split membership for each sample
        meta_lines: Optional metadata lines to display at bottom
        threshold_bundle: ThresholdBundle from compute_threshold_bundle() containing
            threshold values and metrics. See ced_ml.metrics.thresholds.compute_threshold_bundle().
        skip_ci_band: If True, skip rendering 95% CI band (only show ±1 SD).
            Useful for ensemble models where CI and SD are redundant.

    Returns:
        None. Saves plot to out_path.

    Example:
        >>> from ced_ml.metrics.thresholds import compute_threshold_bundle
        >>> bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.95)
        >>> plot_roc_curve(y_true, y_pred, "roc.png", "My Model", threshold_bundle=bundle)
    """
    # Extract threshold information from bundle if provided
    youden_threshold = None
    spec_target_threshold = None
    metrics_at_thresholds = None

    if threshold_bundle is not None:
        youden_threshold = threshold_bundle.get("youden_threshold")
        spec_target_threshold = threshold_bundle.get("spec_target_threshold")
        metrics_at_thresholds = {
            "youden": threshold_bundle.get("youden", {}),
            "spec_target": threshold_bundle.get("spec_target", {}),
        }
        if "dca" in threshold_bundle:
            metrics_at_thresholds["dca"] = threshold_bundle["dca"]
    if not _HAS_PLOTTING:
        logger.warning("Matplotlib not available, skipping ROC curve plot")
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
        logger.warning("No valid data for ROC curve plot after filtering")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_ROC)
    ax.plot([0, 1], [0, 1], "k--", linewidth=LW_REFERENCE, alpha=0.6)

    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    if len(unique_splits) > 1:
        base_fpr = np.linspace(0, 1, 120)
        tprs = []
        aucs = []
        for sid in unique_splits:
            m = split_ids == sid
            y_s = y[m]
            p_s = p[m]
            if len(np.unique(y_s)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_s, p_s)
            tpr_i = np.interp(base_fpr, fpr, tpr)
            tpr_i[0] = 0.0
            tprs.append(tpr_i)
            aucs.append(roc_auc_score(y_s, p_s))

        if tprs:
            tprs = np.vstack(tprs)
            tpr_mean = np.mean(tprs, axis=0)
            tpr_sd = np.std(tprs, axis=0)
            tpr_lo = np.nanpercentile(tprs, 2.5, axis=0)
            tpr_hi = np.nanpercentile(tprs, 97.5, axis=0)
            auc_mean = float(np.mean(aucs))
            auc_sd = float(np.std(aucs))

            if not skip_ci_band:
                ax.fill_between(
                    base_fpr,
                    tpr_lo,
                    tpr_hi,
                    color=COLOR_PRIMARY,
                    alpha=ALPHA_CI,
                    label="95% CI",
                )
            ax.fill_between(
                base_fpr,
                np.maximum(0, tpr_mean - tpr_sd),
                np.minimum(1, tpr_mean + tpr_sd),
                color=COLOR_PRIMARY,
                alpha=ALPHA_SD,
                label="±1 SD",
            )
            ax.plot(
                base_fpr,
                tpr_mean,
                color=COLOR_PRIMARY,
                linewidth=LW_PRIMARY,
                label=f"AUC = {auc_mean:.3f} ± {auc_sd:.3f}",
            )
        else:
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(
                fpr,
                tpr,
                color=COLOR_PRIMARY,
                linewidth=LW_PRIMARY,
                label=f"AUC = {auc:.3f}",
            )
    else:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(
            fpr,
            tpr,
            color=COLOR_PRIMARY,
            linewidth=LW_PRIMARY,
            label=f"AUC = {auc:.3f}",
        )

    if metrics_at_thresholds is not None:
        # Collect marker coordinates for overlap detection
        fpr_youden = None
        tpr_youden = None
        fpr_alpha = None
        tpr_alpha = None

        # Extract Youden coordinates
        if youden_threshold is not None and "youden" in metrics_at_thresholds:
            m = metrics_at_thresholds["youden"]
            fpr_youden = m.get("fpr", None)
            tpr_youden = m.get("tpr", None)

        # Extract spec_target coordinates
        if spec_target_threshold is not None and "spec_target" in metrics_at_thresholds:
            m = metrics_at_thresholds["spec_target"]
            fpr_alpha = m.get("fpr", None)
            tpr_alpha = m.get("tpr", None)

        # Check for overlap (within 2% of plot range)
        markers_overlap = False
        if (
            fpr_youden is not None
            and tpr_youden is not None
            and fpr_alpha is not None
            and tpr_alpha is not None
        ):
            dist = np.sqrt((fpr_youden - fpr_alpha) ** 2 + (tpr_youden - tpr_alpha) ** 2)
            markers_overlap = dist < 0.02  # 2% threshold for overlap

        # Apply offset if overlapping (shift Youden left, spec_target right)
        offset = 0.015 if markers_overlap else 0.0

        # Plot Youden marker
        if (
            fpr_youden is not None
            and tpr_youden is not None
            and 0 <= fpr_youden <= 1
            and 0 <= tpr_youden <= 1
        ):
            ax.scatter(
                [fpr_youden - offset],
                [tpr_youden],
                s=100,
                color="green",
                marker="o",
                edgecolors="darkgreen",
                linewidths=2,
                label="Youden",
                zorder=5,
            )

        # Plot spec_target marker
        if (
            fpr_alpha is not None
            and tpr_alpha is not None
            and 0 <= fpr_alpha <= 1
            and 0 <= tpr_alpha <= 1
        ):
            ax.scatter(
                [fpr_alpha + offset],
                [tpr_alpha],
                s=100,
                color="orange",
                marker="D",
                edgecolors=COLOR_SECONDARY,
                linewidths=2,
                label="Alpha threshold",
                zorder=5,
            )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE)
    else:
        ax.set_title(title, fontsize=FONT_TITLE)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, pad_inches=PAD_INCHES)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    split_ids: np.ndarray | None = None,
    meta_lines: Sequence[str] | None = None,
    skip_ci_band: bool = False,
) -> None:
    """
    Plot Precision-Recall curve with optional split-wise confidence bands.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Output file path
        title: Plot title
        subtitle: Optional subtitle
        split_ids: Array indicating split membership for each sample
        meta_lines: Optional metadata lines to display at bottom
        skip_ci_band: If True, skip rendering 95% CI band (only show ±1 SD).
            Useful for ensemble models where CI and SD are redundant.

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        logger.warning("Matplotlib not available, skipping PR curve plot")
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
        logger.warning("No valid data for PR curve plot after filtering")
        return

    baseline = np.mean(y)
    fig, ax = plt.subplots(figsize=FIGSIZE_ROC)
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=LW_REFERENCE,
        alpha=0.6,
        label=f"Prevalence = {baseline:.4f}",
    )

    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    if len(unique_splits) > 1:
        base_recall = np.linspace(0, 1, 120)
        precisions = []
        aps = []
        for sid in unique_splits:
            m = split_ids == sid
            y_s = y[m]
            p_s = p[m]
            if len(np.unique(y_s)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_s, p_s)
            precision_i = np.interp(base_recall, recall[::-1], precision[::-1])
            precisions.append(precision_i)
            aps.append(average_precision_score(y_s, p_s))

        if precisions:
            precisions = np.vstack(precisions)
            prec_mean = np.mean(precisions, axis=0)
            prec_sd = np.std(precisions, axis=0)
            prec_lo = np.nanpercentile(precisions, 2.5, axis=0)
            prec_hi = np.nanpercentile(precisions, 97.5, axis=0)
            ap_mean = float(np.mean(aps))
            ap_sd = float(np.std(aps))

            if not skip_ci_band:
                ax.fill_between(
                    base_recall,
                    np.clip(prec_lo, 0, 1),
                    np.clip(prec_hi, 0, 1),
                    color=COLOR_PRIMARY,
                    alpha=0.15,
                    label="95% CI",
                )
            ax.fill_between(
                base_recall,
                np.clip(prec_mean - prec_sd, 0, 1),
                np.clip(prec_mean + prec_sd, 0, 1),
                color=COLOR_PRIMARY,
                alpha=ALPHA_SD,
                label="±1 SD",
            )
            ax.plot(
                base_recall,
                prec_mean,
                color=COLOR_PRIMARY,
                linewidth=LW_PRIMARY,
                label=f"AP = {ap_mean:.3f} ± {ap_sd:.3f}",
            )
        else:
            precision, recall, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax.plot(
                recall,
                precision,
                color=COLOR_PRIMARY,
                linewidth=LW_PRIMARY,
                label=f"AP = {ap:.3f}",
            )
    else:
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(
            recall,
            precision,
            color=COLOR_PRIMARY,
            linewidth=LW_PRIMARY,
            label=f"AP = {ap:.3f}",
        )

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE)
    else:
        ax.set_title(title, fontsize=FONT_TITLE)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, pad_inches=PAD_INCHES)
    plt.close()
