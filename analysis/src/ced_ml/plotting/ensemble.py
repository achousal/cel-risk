"""Ensemble-specific plotting utilities.

Visualizations for stacking ensemble interpretability:
- Meta-learner weight/coefficient charts
- Model comparison (ENSEMBLE vs base models)
- Aggregated weights across splits (mean +/- SD)
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER, METRIC_PRAUC

from .style import (
    BBOX_INCHES,
    COLOR_BAR_NEGATIVE,
    COLOR_BAR_PALETTE,
    COLOR_BAR_POSITIVE,
    DPI,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    PAD_INCHES,
    configure_backend,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib  # noqa: F401

    configure_backend()
    import matplotlib.pyplot as plt

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


def plot_meta_learner_weights(
    coef: dict[str, float],
    out_path: Path | str,
    title: str = "Meta-Learner Coefficients",
    subtitle: str = "",
    meta_penalty: str = "l2",
    meta_c: float = 1.0,
    meta_lines: Sequence[str] | None = None,
) -> None:
    """Plot horizontal bar chart of meta-learner coefficients.

    Shows how much each base model contributes to the ensemble decision.
    Bars colored by sign (positive = teal, negative = coral).

    Args:
        coef: Dict mapping base model name to coefficient value.
        out_path: Output file path.
        title: Plot title.
        subtitle: Optional subtitle.
        meta_penalty: Regularization type (for annotation).
        meta_c: Regularization strength (for annotation).
        meta_lines: Optional metadata lines to display at bottom.
    """
    if not _HAS_PLOTTING:
        logger.warning("matplotlib not available, skipping meta-learner weights plot")
        return

    if not coef:
        logger.warning("Empty coefficient dict, skipping meta-learner weights plot")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by absolute magnitude (largest at top)
    sorted_items = sorted(coef.items(), key=lambda x: abs(x[1]))
    names = [item[0] for item in sorted_items]
    values = np.array([item[1] for item in sorted_items])

    colors = [COLOR_BAR_POSITIVE if v >= 0 else COLOR_BAR_NEGATIVE for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(names) + 1.5)))

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Coefficient", fontsize=FONT_LABEL)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate values on bars
    for i, (_bar, val) in enumerate(zip(bars, values, strict=True)):
        offset = 0.01 * max(abs(values)) if max(abs(values)) > 0 else 0.01
        ha = "left" if val >= 0 else "right"
        x_pos = val + offset if val >= 0 else val - offset
        ax.text(x_pos, i, f"{val:.3f}", va="center", ha=ha, fontsize=9)

    # Title and annotation
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=FONT_TITLE, fontweight="bold")

    # Add regularization annotation
    ax.text(
        0.98,
        0.02,
        f"penalty={meta_penalty}, C={meta_c}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )

    # Apply metadata lines if available
    if meta_lines:
        from ced_ml.plotting.dca import apply_plot_metadata

        bottom_margin = apply_plot_metadata(fig, meta_lines)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    else:
        plt.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    logger.info(f"Meta-learner weights plot saved: {out_path}")


def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    out_path: Path | str,
    title: str = "Model Comparison",
    subtitle: str = "",
    highlight_model: str = "ENSEMBLE",
    metric_names: list[str] | None = None,
    meta_lines: Sequence[str] | None = None,
) -> None:
    """Plot grouped bar chart comparing models on key metrics.

    Shows AUROC, PR-AUC, and Brier side by side for all models.
    The ensemble model is highlighted with a distinct edge/hatch.
    Includes robust metadata tracking for reproducibility.

    Args:
        metrics: Dict mapping model name to dict of metric values.
            Expected keys per model: "AUROC", "PR_AUC", "Brier".
        out_path: Output file path.
        title: Plot title.
        subtitle: Optional subtitle.
        highlight_model: Model name to visually highlight.
        metric_names: Metrics to include. Default: ["AUROC", "PR_AUC", "Brier"].
        meta_lines: Optional metadata lines.
    """
    if not _HAS_PLOTTING:
        logger.warning("matplotlib not available, skipping model comparison plot")
        return

    if not metrics or len(metrics) < 2:
        logger.warning("Need at least 2 models for comparison, skipping")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if metric_names is None:
        metric_names = [METRIC_AUROC, METRIC_PRAUC, METRIC_BRIER]

    # Filter to models that have at least one requested metric
    valid_models = {name: m for name, m in metrics.items() if any(mn in m for mn in metric_names)}
    if len(valid_models) < 2:
        logger.warning("Fewer than 2 models with valid metrics, skipping comparison plot")
        return

    model_names = list(valid_models.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    # Color palette (avoid purple per user rules)
    base_colors = COLOR_BAR_PALETTE
    colors = [base_colors[i % len(base_colors)] for i in range(n_models)]

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    bar_width = 0.6

    for ax_idx, metric_name in enumerate(metric_names):
        ax = axes[ax_idx]
        values = []
        for model in model_names:
            val = valid_models[model].get(metric_name, 0.0)
            values.append(val if val is not None else 0.0)

        bars = ax.bar(
            x,
            values,
            bar_width,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )

        # Highlight ensemble bar
        for i, model in enumerate(model_names):
            if model == highlight_model:
                bars[i].set_edgecolor(COLOR_BAR_PALETTE[0])
                bars[i].set_linewidth(2.5)
                bars[i].set_hatch("//")

        # Annotate values
        for i, val in enumerate(values):
            ax.text(
                i,
                val + 0.005 * max(max(values), 0.01),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if model_names[i] == highlight_model else "normal",
            )

        # Format axis
        display_name = metric_name.replace("_", "-")
        ax.set_title(display_name, fontsize=FONT_LABEL, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=FONT_LEGEND)

        # Set y-axis limits with padding
        if values:
            y_min = min(values) * 0.9 if min(values) > 0 else 0
            y_max = max(values) * 1.15
            # For Brier (lower is better), use different scaling
            if metric_name == METRIC_BRIER:
                y_min = 0
                y_max = max(values) * 1.3
            ax.set_ylim(y_min, y_max)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Suptitle
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    # Add summary statistics to metadata
    base_meta_lines = meta_lines or []
    summary_lines = [
        f"n_models={n_models}",
        f"highlight={highlight_model}",
    ]
    # Add best model info
    if METRIC_AUROC in metric_names:
        auroc_values = {m: valid_models[m].get(METRIC_AUROC, 0) for m in model_names}
        best_model = max(auroc_values, key=auroc_values.get)
        best_auroc = auroc_values[best_model]
        summary_lines.append(f"best_model={best_model} (AUROC={best_auroc:.4f})")
    all_meta_lines = base_meta_lines + summary_lines

    # Apply metadata lines
    if all_meta_lines:
        from ced_ml.plotting.dca import apply_plot_metadata

        bottom_margin = apply_plot_metadata(fig, all_meta_lines)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.88, bottom=bottom_margin)
    else:
        plt.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    logger.info(f"Model comparison plot saved: {out_path}")


def plot_aggregated_weights(
    coefs_per_split: dict[int, dict[str, float]],
    out_path: Path | str,
    title: str = "Aggregated Meta-Learner Coefficients",
    subtitle: str = "",
    meta_lines: Sequence[str] | None = None,
) -> None:
    """Plot meta-learner coefficients aggregated across splits with error bars.

    Shows mean coefficient +/- 1 SD across multiple split seeds. Includes robust
    metadata tracking for reproducibility.

    Args:
        coefs_per_split: Dict mapping split_seed to coefficient dict.
        out_path: Output file path.
        title: Plot title.
        subtitle: Optional subtitle (e.g., n_splits info).
        meta_lines: Optional metadata lines.
    """
    if not _HAS_PLOTTING:
        logger.warning("matplotlib not available, skipping aggregated weights plot")
        return

    if not coefs_per_split:
        logger.warning("Empty coefs_per_split, skipping aggregated weights plot")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all base model names across splits
    all_names: set[str] = set()
    for coef_dict in coefs_per_split.values():
        all_names.update(coef_dict.keys())

    if not all_names:
        logger.warning("No model names found in coefs, skipping aggregated weights plot")
        return

    # Compute mean and std for each base model
    name_stats: dict[str, tuple[float, float]] = {}
    for name in all_names:
        vals = [coef_dict[name] for coef_dict in coefs_per_split.values() if name in coef_dict]
        if vals:
            name_stats[name] = (float(np.mean(vals)), float(np.std(vals)))

    # Sort by mean absolute magnitude (largest at top)
    sorted_names = sorted(name_stats.keys(), key=lambda n: abs(name_stats[n][0]))
    means = np.array([name_stats[n][0] for n in sorted_names])
    stds = np.array([name_stats[n][1] for n in sorted_names])

    colors = [COLOR_BAR_POSITIVE if m >= 0 else COLOR_BAR_NEGATIVE for m in means]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(sorted_names) + 1.5)))

    ax.barh(
        range(len(sorted_names)),
        means,
        xerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
        error_kw={"linewidth": 1.2, "capthick": 1.2},
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Coefficient (mean +/- SD)", fontsize=FONT_LABEL)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate mean values with count
    max_extent = max(abs(means).max() + stds.max(), 0.01)
    for i, (m, s) in enumerate(zip(means, stds, strict=True)):
        offset = 0.02 * max_extent
        ha = "left" if m >= 0 else "right"
        x_pos = m + s + offset if m >= 0 else m - s - offset
        n_splits_with_model = sum(1 for cd in coefs_per_split.values() if sorted_names[i] in cd)
        ax.text(x_pos, i, f"{m:.3f} (n={n_splits_with_model})", va="center", ha=ha, fontsize=9)

    # Title - use subtitle if provided, otherwise show n_splits
    n_splits = len(coefs_per_split)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=FONT_TITLE, fontweight="bold")
    else:
        ax.set_title(f"{title}\n(n_splits={n_splits})", fontsize=FONT_TITLE, fontweight="bold")

    # Add summary statistics to metadata
    base_meta_lines = meta_lines or []
    summary_lines = [
        f"n_splits={n_splits}",
        f"n_base_models={len(all_names)}",
        f"coef_range=[{means.min():.3f}, {means.max():.3f}]",
    ]
    all_meta_lines = base_meta_lines + summary_lines

    # Apply metadata lines
    from ced_ml.plotting.dca import apply_plot_metadata

    bottom_margin = apply_plot_metadata(fig, all_meta_lines)
    # Add extra breathing room for x-axis labels/ticks to avoid metadata overlap.
    bottom_margin = min(bottom_margin + 0.04, 0.35)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    fig.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES)
    plt.close(fig)
    logger.info(f"Aggregated weights plot saved: {out_path}")


def save_ensemble_aggregation_metadata(
    coefs_per_split: dict[int, dict[str, float]],
    pooled_test_metrics: dict[str, dict[str, float]] | None = None,
    base_models: list[str] | None = None,
    meta_penalty: str = "l2",
    meta_C: float = 1.0,
    out_dir: Path | str | None = None,
) -> dict[str, any]:
    """Generate and save comprehensive ensemble aggregation metadata.

    Creates a JSON file with ensemble coefficient statistics, performance metrics,
    and configuration for reproducibility and interpretation of aggregated figures.

    Args:
        coefs_per_split: Dict mapping split_seed to coefficient dict
        pooled_test_metrics: Optional dict of test metrics by model
        base_models: List of base model names
        meta_penalty: Meta-learner penalty type
        meta_C: Meta-learner C value
        out_dir: Optional output directory for metadata JSON

    Returns:
        Dict with ensemble aggregation metadata
    """
    import json
    from datetime import datetime

    # Compute coefficient statistics
    all_names: set[str] = set()
    for coef_dict in coefs_per_split.values():
        all_names.update(coef_dict.keys())

    coef_stats = {}
    for name in all_names:
        vals = [coef_dict[name] for coef_dict in coefs_per_split.values() if name in coef_dict]
        if vals:
            coef_stats[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
                "n_splits": len(vals),
            }

    # Build metadata dict
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "ensemble_type": "stacking",
        "meta_learner": {
            "type": "logistic_regression",
            "penalty": meta_penalty,
            "C": meta_C,
        },
        "base_models": base_models or [],
        "n_base_models": len(base_models) if base_models else 0,
        "aggregation": {
            "n_splits": len(coefs_per_split),
            "split_seeds": sorted(coefs_per_split.keys()),
        },
        "meta_learner_coefficients": coef_stats,
        "coefficient_summary": {
            "n_coefficients": len(coef_stats),
            "positive_count": sum(1 for s in coef_stats.values() if s["mean"] >= 0),
            "negative_count": sum(1 for s in coef_stats.values() if s["mean"] < 0),
            "mean_range": [
                float(min((s["mean"] for s in coef_stats.values()), default=0)),
                float(max((s["mean"] for s in coef_stats.values()), default=0)),
            ],
        },
    }

    # Add performance comparison if available
    if pooled_test_metrics and "ENSEMBLE" in pooled_test_metrics:
        ensemble_perf = pooled_test_metrics["ENSEMBLE"]
        base_models_perf = {
            m: pooled_test_metrics[m] for m in pooled_test_metrics if m != "ENSEMBLE"
        }

        best_base_auroc = max(
            (m.get(METRIC_AUROC, 0) for m in base_models_perf.values()), default=0
        )
        ensemble_auroc = ensemble_perf.get(METRIC_AUROC, 0)

        metadata["performance_comparison"] = {
            "ensemble_auroc": ensemble_auroc,
            "best_base_model_auroc": best_base_auroc,
            "auroc_improvement": ensemble_auroc - best_base_auroc,
            "auroc_improvement_percent": (
                ((ensemble_auroc - best_base_auroc) / best_base_auroc * 100)
                if best_base_auroc > 0
                else 0
            ),
            "ensemble_prauc": ensemble_perf.get(METRIC_PRAUC),
            "ensemble_brier": ensemble_perf.get(METRIC_BRIER),
        }

    # Save metadata JSON if output dir provided
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / "ensemble_aggregation_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Ensemble aggregation metadata saved: {metadata_path}")

    return metadata
