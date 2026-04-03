"""
Multi-model comparison plots.

Overlays all trained models on a single figure for side-by-side visual
comparison of ROC, PR, calibration, and DCA curves. Each model gets its
own color, SD band, and Youden threshold marker.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from .dca import apply_plot_metadata
from .style import (
    ALPHA_CI_COMPARISON,
    DPI,
    FIGSIZE_COMPARISON,
    FIGSIZE_COMPARISON_DCA,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_COMPARISON,
    LW_COMPARISON_ENSEMBLE,
    LW_REFERENCE,
    LW_SECONDARY,
    MARKER_SIZE_COMPARISON,
    PAD_INCHES,
    configure_backend,
    get_model_color,
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


class ModelCurveData(TypedDict):
    """Data for a single model's curve in comparison plots."""

    y_true: np.ndarray
    y_pred: np.ndarray
    split_ids: np.ndarray | None
    threshold_bundle: dict | None


def _resolve_model_order(
    models: dict[str, ModelCurveData],
    model_order: list[str] | None = None,
) -> list[str]:
    """Resolve draw order: custom order, or alphabetical with ENSEMBLE last."""
    if model_order is not None:
        return [m for m in model_order if m in models]
    names = sorted(models.keys())
    if "ENSEMBLE" in names:
        names.remove("ENSEMBLE")
        names.append("ENSEMBLE")
    return names


def _model_lw(model_name: str) -> float:
    """Line width for a model: thicker for ENSEMBLE."""
    return LW_COMPARISON_ENSEMBLE if model_name == "ENSEMBLE" else LW_COMPARISON


def _clean_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate and filter arrays, returning None if no valid data."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None
    return y, p


def _filter_split_ids(split_ids: np.ndarray | None, mask: np.ndarray) -> list | None:
    """Filter and extract unique split IDs, returning None if < 2 splits."""
    if split_ids is None:
        return None
    split_ids = np.asarray(split_ids)[mask]
    unique = pd.Series(split_ids).dropna().unique().tolist()
    return unique if len(unique) > 1 else None


# ---------------------------------------------------------------------------
# ROC Comparison
# ---------------------------------------------------------------------------


def plot_roc_comparison(
    models: dict[str, ModelCurveData],
    out_path: Path,
    title: str = "ROC Curve Comparison",
    subtitle: str = "",
    meta_lines: Sequence[str] | None = None,
    model_order: list[str] | None = None,
) -> None:
    """
    Overlay ROC curves for all models on a single figure.

    Each model gets its own color, ±1 SD band (when split_ids available),
    and Youden threshold marker (when threshold_bundle provided).
    """
    if not _HAS_PLOTTING:
        logger.warning("Matplotlib not available, skipping ROC comparison plot")
        return

    order = _resolve_model_order(models, model_order)
    if len(order) < 2:
        logger.info("Fewer than 2 models, skipping ROC comparison plot")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_COMPARISON)
    ax.plot([0, 1], [0, 1], "k--", linewidth=LW_REFERENCE, alpha=0.6, label="Chance")

    base_fpr = np.linspace(0, 1, 120)

    for model_name in order:
        data = models[model_name]
        cleaned = _clean_arrays(data["y_true"], data["y_pred"])
        if cleaned is None:
            continue
        y, p = cleaned
        color = get_model_color(model_name)
        lw = _model_lw(model_name)

        # Determine split-based CI
        mask = np.isfinite(np.asarray(data["y_pred"]).astype(float)) & np.isfinite(
            np.asarray(data["y_true"]).astype(int)
        )
        split_ids = data.get("split_ids")
        unique_splits = _filter_split_ids(split_ids, mask)

        if unique_splits is not None:
            filtered_splits = np.asarray(split_ids)[mask]
            tprs = []
            aucs = []
            for sid in unique_splits:
                m = filtered_splits == sid
                y_s, p_s = y[m], p[m]
                if len(np.unique(y_s)) < 2:
                    continue
                fpr_s, tpr_s, _ = roc_curve(y_s, p_s)
                tpr_i = np.interp(base_fpr, fpr_s, tpr_s)
                tpr_i[0] = 0.0
                tprs.append(tpr_i)
                aucs.append(roc_auc_score(y_s, p_s))

            if tprs:
                tprs_arr = np.vstack(tprs)
                tpr_mean = np.mean(tprs_arr, axis=0)
                tpr_sd = np.std(tprs_arr, axis=0)
                auc_mean = float(np.mean(aucs))
                auc_sd = float(np.std(aucs))

                ax.fill_between(
                    base_fpr,
                    np.maximum(0, tpr_mean - tpr_sd),
                    np.minimum(1, tpr_mean + tpr_sd),
                    color=color,
                    alpha=ALPHA_CI_COMPARISON,
                )
                ax.plot(
                    base_fpr,
                    tpr_mean,
                    color=color,
                    linewidth=lw,
                    label=f"{model_name} (AUC={auc_mean:.3f}±{auc_sd:.3f})",
                )
            else:
                _plot_single_roc(ax, y, p, color, lw, model_name)
        else:
            _plot_single_roc(ax, y, p, color, lw, model_name)

        # Youden threshold marker
        _plot_youden_marker(ax, data, color, model_name, "roc")

    ax.set_xlabel("False Positive Rate", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate", fontsize=FONT_LABEL)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    _set_title(ax, title, subtitle)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    _save_figure(fig, out_path, meta_lines)


def _plot_single_roc(
    ax: "plt.Axes",
    y: np.ndarray,
    p: np.ndarray,
    color: str,
    lw: float,
    model_name: str,
) -> None:
    """Plot a single ROC curve without CI bands."""
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, color=color, linewidth=lw, label=f"{model_name} (AUC={auc:.3f})")


# ---------------------------------------------------------------------------
# PR Comparison
# ---------------------------------------------------------------------------


def plot_pr_comparison(
    models: dict[str, ModelCurveData],
    out_path: Path,
    title: str = "PR Curve Comparison",
    subtitle: str = "",
    meta_lines: Sequence[str] | None = None,
    model_order: list[str] | None = None,
) -> None:
    """
    Overlay Precision-Recall curves for all models on a single figure.

    Each model gets its own color, ±1 SD band, and legend entry with AP score.
    """
    if not _HAS_PLOTTING:
        logger.warning("Matplotlib not available, skipping PR comparison plot")
        return

    order = _resolve_model_order(models, model_order)
    if len(order) < 2:
        logger.info("Fewer than 2 models, skipping PR comparison plot")
        return

    # Compute baseline prevalence from first model (shared labels)
    first_data = models[order[0]]
    cleaned_first = _clean_arrays(first_data["y_true"], first_data["y_pred"])
    baseline = float(np.mean(cleaned_first[0])) if cleaned_first is not None else 0.5

    fig, ax = plt.subplots(figsize=FIGSIZE_COMPARISON)
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=LW_REFERENCE,
        alpha=0.6,
        label=f"Prevalence = {baseline:.4f}",
    )

    base_recall = np.linspace(0, 1, 120)

    for model_name in order:
        data = models[model_name]
        cleaned = _clean_arrays(data["y_true"], data["y_pred"])
        if cleaned is None:
            continue
        y, p = cleaned
        color = get_model_color(model_name)
        lw = _model_lw(model_name)

        mask = np.isfinite(np.asarray(data["y_pred"]).astype(float)) & np.isfinite(
            np.asarray(data["y_true"]).astype(int)
        )
        split_ids = data.get("split_ids")
        unique_splits = _filter_split_ids(split_ids, mask)

        if unique_splits is not None:
            filtered_splits = np.asarray(split_ids)[mask]
            precisions = []
            aps = []
            for sid in unique_splits:
                m = filtered_splits == sid
                y_s, p_s = y[m], p[m]
                if len(np.unique(y_s)) < 2:
                    continue
                prec_s, rec_s, _ = precision_recall_curve(y_s, p_s)
                prec_i = np.interp(base_recall, rec_s[::-1], prec_s[::-1])
                precisions.append(prec_i)
                aps.append(average_precision_score(y_s, p_s))

            if precisions:
                prec_arr = np.vstack(precisions)
                prec_mean = np.mean(prec_arr, axis=0)
                prec_sd = np.std(prec_arr, axis=0)
                ap_mean = float(np.mean(aps))
                ap_sd = float(np.std(aps))

                ax.fill_between(
                    base_recall,
                    np.clip(prec_mean - prec_sd, 0, 1),
                    np.clip(prec_mean + prec_sd, 0, 1),
                    color=color,
                    alpha=ALPHA_CI_COMPARISON,
                )
                ax.plot(
                    base_recall,
                    prec_mean,
                    color=color,
                    linewidth=lw,
                    label=f"{model_name} (AP={ap_mean:.3f}±{ap_sd:.3f})",
                )
            else:
                _plot_single_pr(ax, y, p, color, lw, model_name)
        else:
            _plot_single_pr(ax, y, p, color, lw, model_name)

    ax.set_xlabel("Recall (Sensitivity)", fontsize=FONT_LABEL)
    ax.set_ylabel("Precision (PPV)", fontsize=FONT_LABEL)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    _set_title(ax, title, subtitle)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    _save_figure(fig, out_path, meta_lines)


def _plot_single_pr(
    ax: "plt.Axes",
    y: np.ndarray,
    p: np.ndarray,
    color: str,
    lw: float,
    model_name: str,
) -> None:
    """Plot a single PR curve without CI bands."""
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    ax.plot(rec, prec, color=color, linewidth=lw, label=f"{model_name} (AP={ap:.3f})")


# ---------------------------------------------------------------------------
# Calibration Comparison
# ---------------------------------------------------------------------------


def plot_calibration_comparison(
    models: dict[str, ModelCurveData],
    out_path: Path,
    title: str = "Calibration Comparison",
    subtitle: str = "",
    n_bins: int = 10,
    meta_lines: Sequence[str] | None = None,
    model_order: list[str] | None = None,
) -> None:
    """
    Overlay calibration curves for all models on a single panel.

    Uses probability-space with quantile binning. Each model gets its own
    color and LOWESS-smoothed curve overlay when sufficient data is available.
    """
    if not _HAS_PLOTTING:
        logger.warning("Matplotlib not available, skipping calibration comparison plot")
        return

    order = _resolve_model_order(models, model_order)
    if len(order) < 2:
        logger.info("Fewer than 2 models, skipping calibration comparison plot")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_COMPARISON)
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=LW_REFERENCE,
        alpha=0.6,
        label="Perfect calibration",
    )

    for model_name in order:
        data = models[model_name]
        cleaned = _clean_arrays(data["y_true"], data["y_pred"])
        if cleaned is None:
            continue
        y, p = cleaned
        color = get_model_color(model_name)
        lw = _model_lw(model_name)

        # Quantile binning
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, n_bins + 1)
        actual_n_bins = len(bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mask = np.isfinite(np.asarray(data["y_pred"]).astype(float)) & np.isfinite(
            np.asarray(data["y_true"]).astype(int)
        )
        split_ids = data.get("split_ids")
        unique_splits = _filter_split_ids(split_ids, mask)

        if unique_splits is not None:
            filtered_splits = np.asarray(split_ids)[mask]
            curves = []
            for sid in unique_splits:
                m = filtered_splits == sid
                y_s, p_s = y[m], p[m]
                bin_idx = np.digitize(p_s, bins) - 1
                bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)
                obs = []
                for i in range(actual_n_bins):
                    bm = bin_idx == i
                    obs.append(np.nan if bm.sum() == 0 else y_s[bm].mean())
                curves.append(obs)

            curves_arr = np.array(curves, dtype=float)
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                obs_mean = np.nanmean(curves_arr, axis=0)
                obs_sd = np.nanstd(curves_arr, axis=0)

            valid = ~np.isnan(obs_mean)
            ax.fill_between(
                bin_centers,
                np.clip(obs_mean - obs_sd, 0, 1),
                np.clip(obs_mean + obs_sd, 0, 1),
                color=color,
                alpha=ALPHA_CI_COMPARISON,
            )
            ax.plot(
                bin_centers[valid],
                obs_mean[valid],
                color=color,
                linewidth=lw,
                marker="o",
                markersize=4,
                label=model_name,
            )
        else:
            bin_idx = np.digitize(p, bins) - 1
            bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)
            obs = []
            pred_means = []
            for i in range(actual_n_bins):
                bm = bin_idx == i
                if bm.sum() == 0:
                    obs.append(np.nan)
                    pred_means.append(np.nan)
                else:
                    obs.append(y[bm].mean())
                    pred_means.append(p[bm].mean())
            obs = np.array(obs)
            pred_means = np.array(pred_means)
            valid = ~np.isnan(obs)
            ax.plot(
                pred_means[valid],
                obs[valid],
                color=color,
                linewidth=lw,
                marker="o",
                markersize=4,
                label=model_name,
            )

        # LOWESS overlay per model
        _add_lowess_comparison(ax, y, p, color, lw)

    ax.set_xlabel("Predicted probability", fontsize=FONT_LABEL)
    ax.set_ylabel("Observed frequency", fontsize=FONT_LABEL)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")
    _set_title(ax, title, subtitle)
    ax.legend(loc="upper left", fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    _save_figure(fig, out_path, meta_lines)


def _add_lowess_comparison(
    ax: "plt.Axes",
    y: np.ndarray,
    p: np.ndarray,
    color: str,
    lw: float,
) -> None:
    """Add LOWESS-smoothed calibration overlay for one model."""
    try:
        from scipy.interpolate import UnivariateSpline

        if len(y) < 50 or len(np.unique(p)) < 10:
            return
        order = np.argsort(p)
        p_sorted = p[order]
        y_sorted = y[order].astype(float)
        spline = UnivariateSpline(p_sorted, y_sorted, s=len(p_sorted) * 0.1, k=3)
        x_smooth = np.linspace(p_sorted.min(), p_sorted.max(), 200)
        y_smooth = np.clip(spline(x_smooth), 0, 1)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=lw * 0.6, alpha=0.5, linestyle="--")
    except Exception:
        pass  # Silently skip if LOWESS fitting fails


# ---------------------------------------------------------------------------
# DCA Comparison
# ---------------------------------------------------------------------------


def plot_dca_comparison(
    models: dict[str, ModelCurveData],
    out_path: Path,
    title: str = "DCA Comparison",
    subtitle: str = "",
    max_pt: float = 0.20,
    step: float = 0.005,
    meta_lines: Sequence[str] | None = None,
    model_order: list[str] | None = None,
) -> None:
    """
    Overlay Decision Curve Analysis for all models on a single figure.

    Plots net benefit vs threshold for each model, plus shared Treat All
    and Treat None reference lines. Each model gets ±1 SD band when
    split_ids are available.
    """
    if not _HAS_PLOTTING:
        logger.warning("Matplotlib not available, skipping DCA comparison plot")
        return

    from ced_ml.metrics.dca import decision_curve_analysis

    order = _resolve_model_order(models, model_order)
    if len(order) < 2:
        logger.info("Fewer than 2 models, skipping DCA comparison plot")
        return

    thresholds = np.arange(0.0005, max_pt + step, step)
    thresholds_pct = thresholds * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_COMPARISON_DCA)

    # Plot Treat All / Treat None from first model (shared labels)
    treat_all_plotted = False
    y_min, y_max = 0.0, 0.0

    for model_name in order:
        data = models[model_name]
        cleaned = _clean_arrays(data["y_true"], data["y_pred"])
        if cleaned is None:
            continue
        y, p = cleaned
        color = get_model_color(model_name)
        lw = _model_lw(model_name)

        mask = np.isfinite(np.asarray(data["y_pred"]).astype(float)) & np.isfinite(
            np.asarray(data["y_true"]).astype(int)
        )
        split_ids = data.get("split_ids")
        unique_splits = _filter_split_ids(split_ids, mask)

        if unique_splits is not None:
            filtered_splits = np.asarray(split_ids)[mask]
            nb_model_curves = []
            nb_all_curves = []
            nb_none_curves = []

            for sid in unique_splits:
                m = filtered_splits == sid
                y_s, p_s = y[m], p[m]
                if len(np.unique(y_s)) < 2 or len(y_s) < 2:
                    continue
                dca_df = decision_curve_analysis(y_s, p_s, thresholds=thresholds)
                if not dca_df.empty:
                    nb_model_curves.append(dca_df["net_benefit_model"].values)
                    nb_all_curves.append(dca_df["net_benefit_all"].values)
                    nb_none_curves.append(dca_df["net_benefit_none"].values)

            if nb_model_curves:
                nb_model_arr = np.vstack(nb_model_curves)
                nb_model_mean = np.mean(nb_model_arr, axis=0)
                nb_model_sd = np.std(nb_model_arr, axis=0)
                thr_pct = thresholds_pct[: len(nb_model_mean)]

                ax.fill_between(
                    thr_pct,
                    np.maximum(0, nb_model_mean - nb_model_sd),
                    nb_model_mean + nb_model_sd,
                    color=color,
                    alpha=ALPHA_CI_COMPARISON,
                )
                ax.plot(
                    thr_pct,
                    nb_model_mean,
                    color=color,
                    linewidth=lw,
                    label=model_name,
                )

                # Track y-range
                y_min = min(y_min, float(np.nanmin(nb_model_mean - nb_model_sd)))
                y_max = max(y_max, float(np.nanmax(nb_model_mean + nb_model_sd)))

                # Plot treat all/none once
                if not treat_all_plotted:
                    nb_all_mean = np.mean(np.vstack(nb_all_curves), axis=0)
                    nb_none_mean = np.mean(np.vstack(nb_none_curves), axis=0)
                    ax.plot(thr_pct, nb_all_mean, "r--", linewidth=LW_SECONDARY, label="Treat All")
                    ax.plot(thr_pct, nb_none_mean, "k:", linewidth=LW_SECONDARY, label="Treat None")
                    y_min = min(y_min, float(np.nanmin(nb_all_mean)))
                    y_max = max(y_max, float(np.nanmax(nb_all_mean)))
                    treat_all_plotted = True
            else:
                _plot_single_dca(
                    ax,
                    y,
                    p,
                    thresholds,
                    thresholds_pct,
                    color,
                    lw,
                    model_name,
                    not treat_all_plotted,
                )
                treat_all_plotted = True
        else:
            nb_mean, nb_all, nb_none, thr_pct = _plot_single_dca(
                ax,
                y,
                p,
                thresholds,
                thresholds_pct,
                color,
                lw,
                model_name,
                not treat_all_plotted,
            )
            if nb_mean is not None:
                y_min = min(y_min, float(np.nanmin(nb_mean)))
                y_max = max(y_max, float(np.nanmax(nb_mean)))
                if nb_all is not None:
                    y_min = min(y_min, float(np.nanmin(nb_all)))
                    y_max = max(y_max, float(np.nanmax(nb_all)))
            treat_all_plotted = True

    # Y-axis scaling
    y_range = y_max - y_min
    if y_range > 0:
        ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])
    else:
        ax.set_ylim([min(y_min, -0.05), max(y_max, 0.05)])

    x_max_display = min(100, max_pt * 100 * 1.02)
    ax.set_xlim(0, x_max_display)
    ax.set_xlabel("Threshold Probability (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Net Benefit", fontsize=FONT_LABEL)
    _set_title(ax, title, subtitle)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND, ncol=2)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    _save_figure(fig, out_path, meta_lines)


def _plot_single_dca(
    ax: "plt.Axes",
    y: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray,
    thresholds_pct: np.ndarray,
    color: str,
    lw: float,
    model_name: str,
    plot_references: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Plot a single DCA curve, optionally with treat-all/treat-none references."""
    from ced_ml.metrics.dca import decision_curve_analysis

    dca_df = decision_curve_analysis(y, p, thresholds=thresholds)
    if dca_df.empty:
        return None, None, None, None

    thr_pct = dca_df["threshold"].values * 100
    nb_model = dca_df["net_benefit_model"].values
    nb_all = dca_df["net_benefit_all"].values
    nb_none = dca_df["net_benefit_none"].values

    ax.plot(thr_pct, nb_model, color=color, linewidth=lw, label=model_name)

    if plot_references:
        ax.plot(thr_pct, nb_all, "r--", linewidth=LW_SECONDARY, label="Treat All")
        ax.plot(thr_pct, nb_none, "k:", linewidth=LW_SECONDARY, label="Treat None")

    return nb_model, nb_all, nb_none, thr_pct


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _plot_youden_marker(
    ax: "plt.Axes",
    data: ModelCurveData,
    color: str,
    model_name: str,
    plot_type: str,
) -> None:
    """Plot Youden threshold marker if available in threshold_bundle."""
    bundle = data.get("threshold_bundle")
    if bundle is None:
        return

    youden = bundle.get("youden", {})
    fpr = youden.get("fpr")
    tpr = youden.get("tpr")

    if plot_type != "roc" or fpr is None or tpr is None:
        return
    if not (0 <= fpr <= 1 and 0 <= tpr <= 1):
        return

    ax.scatter(
        [fpr],
        [tpr],
        s=MARKER_SIZE_COMPARISON,
        color=color,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
    )


def _set_title(ax: "plt.Axes", title: str, subtitle: str) -> None:
    """Set axis title with optional subtitle."""
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE)
    else:
        ax.set_title(title, fontsize=FONT_TITLE)


def _save_figure(
    fig: "plt.Figure",
    out_path: Path,
    meta_lines: Sequence[str] | None,
) -> None:
    """Apply metadata, adjust layout, and save figure."""
    bottom_margin = apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.12, right=0.92, top=0.88, bottom=bottom_margin)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved to {out_path}")
