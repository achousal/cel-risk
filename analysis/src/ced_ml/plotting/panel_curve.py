"""Panel size vs AUROC curve plotting for RFE results.

Visualizes the Pareto frontier between panel size and discrimination
performance, with annotations for knee points and recommended thresholds.
"""

from pathlib import Path

import numpy as np

from ced_ml.utils.constants import Z_CRITICAL_005

from .style import (
    BBOX_INCHES,
    COLOR_PARETO_MAIN,
    COLOR_TERTIARY,
    COLOR_THRESHOLD_AMBER,
    COLOR_THRESHOLD_GREEN,
    COLOR_THRESHOLD_RED,
    DPI,
    FIGSIZE_DCA,
    FIGSIZE_SINGLE,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    LW_PRIMARY,
    LW_REFERENCE,
    LW_SECONDARY,
    configure_backend,
)

try:
    import matplotlib  # noqa: F401

    configure_backend()
    import matplotlib.pyplot as plt

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


def plot_pareto_curve(
    curve: list[dict],
    recommended: dict[str, int],
    out_path: Path | str,
    title: str = "Panel Size vs AUROC",
    model_name: str = "",
    thresholds_to_show: list[float] | None = None,
    show_ci: bool = True,
    ci_alpha: float = 0.2,
    n_splits: int | None = None,
    n_train_samples: int | None = None,
    n_val_samples: int | None = None,
    n_train_cases: int | None = None,
    n_val_cases: int | None = None,
    feature_selection_method: str | None = None,
    run_id: str | None = None,
) -> None:
    """Plot validation AUROC vs panel size curve with bootstrap CI and annotations.

    Shows only the held-out validation curve (no OOF/CV curve) to avoid
    displaying optimistically biased estimates from shared feature ranking.

    Args:
        curve: List of dicts with keys "size", "auroc_val", "auroc_val_std".
        recommended: Dict with "min_size_95pct", "min_size_90pct", "knee_point", etc.
        out_path: Output file path.
        title: Plot title.
        model_name: Model name for subtitle.
        thresholds_to_show: AUROC fraction thresholds to annotate (default: [0.95, 0.90]).
        show_ci: Whether to show confidence intervals (default: True).
        ci_alpha: Transparency for CI shaded region (default: 0.2).
        n_splits: Number of splits aggregated (optional).
        n_train_samples: Total training samples (optional).
        n_val_samples: Total validation samples (optional).
        n_train_cases: Number of positive cases in training (optional).
        n_val_cases: Number of positive cases in validation (optional).
        feature_selection_method: Feature selection method used (optional).
        run_id: Run identifier (optional).

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        return

    if not curve:
        return

    if thresholds_to_show is None:
        thresholds_to_show = [0.95, 0.90]

    # Extract data
    sizes = np.array([p["size"] for p in curve])
    aurocs_val = np.array([p["auroc_val"] for p in curve])
    aurocs_val_std = np.array([p.get("auroc_val_std", 0.0) for p in curve])
    # Percentile bootstrap CI bounds (preferred); fall back to normal approx
    ci_lower = np.array(
        [
            p.get(
                "auroc_val_ci_low",
                p["auroc_val"] - Z_CRITICAL_005 * p.get("auroc_val_std", 0.0),
            )
            for p in curve
        ]
    )
    ci_upper = np.array(
        [
            p.get(
                "auroc_val_ci_high",
                p["auroc_val"] + Z_CRITICAL_005 * p.get("auroc_val_std", 0.0),
            )
            for p in curve
        ]
    )

    max_auroc = np.max(aurocs_val)

    # Sort by size for line plot
    sort_idx = np.argsort(sizes)[::-1]  # Descending
    sizes = sizes[sort_idx]
    aurocs_val = aurocs_val[sort_idx]
    aurocs_val_std = aurocs_val_std[sort_idx]
    ci_lower = ci_lower[sort_idx]
    ci_upper = ci_upper[sort_idx]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Plot validation AUROC
    ax.plot(
        sizes,
        aurocs_val,
        "o-",
        color=COLOR_PARETO_MAIN,
        linewidth=LW_PRIMARY,
        markersize=6,
        label="Validation AUROC",
        zorder=5,
    )

    # Add 95% CI shaded region for validation AUROC (bootstrap)
    has_ci = np.any(ci_lower != ci_upper)
    if show_ci and has_ci:
        ax.fill_between(
            sizes,
            ci_lower,
            ci_upper,
            color=COLOR_PARETO_MAIN,
            alpha=ci_alpha,
            label="95% CI (percentile bootstrap)",
            zorder=1,
        )

    # Threshold lines
    colors = [COLOR_THRESHOLD_GREEN, COLOR_THRESHOLD_AMBER, COLOR_THRESHOLD_RED]
    for i, thresh in enumerate(thresholds_to_show):
        target_auroc = max_auroc * thresh
        ax.axhline(
            y=target_auroc,
            color=colors[i % len(colors)],
            linestyle=":",
            alpha=0.7,
            linewidth=LW_SECONDARY,
            zorder=2,
        )
        ax.text(
            sizes.max() * 0.98,
            target_auroc + 0.005,
            f"{thresh:.0%} of max",
            color=colors[i % len(colors)],
            fontsize=FONT_LEGEND,
            ha="right",
            va="bottom",
        )

        # Mark recommended panel size for this threshold
        key = f"min_size_{int(thresh * 100)}pct"
        if key in recommended:
            rec_size = recommended[key]
            # Find corresponding AUROC and CI
            idx = np.where(sizes == rec_size)[0]
            if len(idx) > 0:
                rec_auroc = aurocs_val[idx[0]]
                rec_ci_lo = ci_lower[idx[0]]
                rec_ci_hi = ci_upper[idx[0]]

                # Build legend label with CI info
                if rec_ci_lo != rec_ci_hi:
                    legend_label = (
                        f"{thresh:.0%} panel (n={rec_size}): "
                        f"AUROC {rec_auroc:.3f} [{rec_ci_lo:.3f}, {rec_ci_hi:.3f}]"
                    )
                else:
                    legend_label = f"{thresh:.0%} panel (n={rec_size}): AUROC {rec_auroc:.3f}"

                ax.scatter(
                    [rec_size],
                    [rec_auroc],
                    s=120,
                    c=colors[i % len(colors)],
                    marker="D",
                    zorder=10,
                    edgecolors="white",
                    linewidths=1.5,
                    label=legend_label,
                )

    # Mark knee point
    if "knee_point" in recommended:
        knee_size = recommended["knee_point"]
        idx = np.where(sizes == knee_size)[0]
        if len(idx) > 0:
            knee_auroc = aurocs_val[idx[0]]
            knee_ci_lo = ci_lower[idx[0]]
            knee_ci_hi = ci_upper[idx[0]]

            # Build legend label with CI info
            if knee_ci_lo != knee_ci_hi:
                knee_label = (
                    f"Knee (n={knee_size}): "
                    f"AUROC {knee_auroc:.3f} [{knee_ci_lo:.3f}, {knee_ci_hi:.3f}]"
                )
            else:
                knee_label = f"Knee (n={knee_size}): AUROC {knee_auroc:.3f}"

            ax.scatter(
                [knee_size],
                [knee_auroc],
                s=150,
                c=COLOR_TERTIARY,
                marker="*",
                zorder=10,
                edgecolors="white",
                linewidths=1,
                label=knee_label,
            )

    # Add statistical comparison annotations for adjacent recommended sizes
    _add_comparison_annotations(
        ax,
        sizes,
        aurocs_val,
        aurocs_val_std,
        ci_lower,
        ci_upper,
        recommended,
        thresholds_to_show,
    )

    # Styling
    ax.set_xlabel("Panel Size (number of proteins)", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC", fontsize=FONT_LABEL)
    ax.set_xlim(0, sizes.max() * 1.05)

    # Y-axis: show reasonable range around the data
    y_min = ci_lower.min() - 0.02 if has_ci else aurocs_val.min() - 0.02
    y_max = ci_upper.max() + 0.02 if has_ci else aurocs_val.max() + 0.02
    y_min = max(0.5, y_min)  # Don't go below 0.5
    y_max = min(1.0, y_max)  # Don't go above 1.0
    ax.set_ylim(y_min, y_max)

    # Place legend outside plot area (to the right)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=FONT_LEGEND,
        frameon=True,
        fancybox=False,
        shadow=False,
    )
    ax.grid(True, alpha=GRID_ALPHA, zorder=0)

    # Title
    if model_name:
        ax.set_title(f"{title}\n{model_name}", fontsize=FONT_TITLE, fontweight="bold")
    else:
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    # Build summary text with metadata
    summary_lines = [
        f"Max AUROC: {max_auroc:.3f}",
        f"Start size: {int(sizes.max())}",
        f"Min size evaluated: {int(sizes.min())}",
    ]
    if "knee_point" in recommended:
        summary_lines.append(f"Knee point: {recommended['knee_point']}")

    # Add metadata section
    if any([n_splits, n_train_samples, n_val_samples, feature_selection_method, run_id]):
        summary_lines.append("")  # Blank line separator

    if n_splits is not None:
        summary_lines.append(f"Splits: {n_splits}")

    if n_train_samples is not None and n_train_cases is not None:
        summary_lines.append(f"Train: {n_train_cases}/{n_train_samples} cases")
    elif n_train_samples is not None:
        summary_lines.append(f"Train: {n_train_samples} samples")

    if n_val_samples is not None and n_val_cases is not None:
        summary_lines.append(f"Val: {n_val_cases}/{n_val_samples} cases")
    elif n_val_samples is not None:
        summary_lines.append(f"Val: {n_val_samples} samples")

    if feature_selection_method:
        summary_lines.append(f"Method: {feature_selection_method}")

    if run_id:
        summary_lines.append(f"Run: {run_id}")

    summary_text = "  |  ".join(summary_lines)

    plt.tight_layout()

    # Place metadata text below the plot
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5,
        0.02,
        summary_text,
        fontsize=7,
        ha="center",
        va="bottom",
        color="gray",
        fontstyle="italic",
    )

    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)


def _add_comparison_annotations(
    ax,
    sizes: np.ndarray,
    aurocs_cv: np.ndarray,
    aurocs_std: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    recommended: dict[str, int],
    thresholds: list[float],
) -> None:
    """Add statistical comparison annotations between recommended panel sizes.

    Draws lines and text indicating whether differences between adjacent
    recommended panel sizes are statistically significant (non-overlapping CIs).

    Args:
        ax: Matplotlib axes object.
        sizes: Array of panel sizes.
        aurocs_cv: Array of AUROC means (validation or CV).
        aurocs_std: Array of AUROC standard deviations (bootstrap).
        ci_lower: Array of percentile bootstrap CI lower bounds.
        ci_upper: Array of percentile bootstrap CI upper bounds.
        recommended: Dict with recommended panel sizes.
        thresholds: List of AUROC thresholds.
    """
    if not _HAS_PLOTTING:
        return

    # Get sorted recommended sizes (excluding knee_point for now)
    rec_sizes = []
    for thresh in sorted(thresholds, reverse=True):
        key = f"min_size_{int(thresh * 100)}pct"
        if key in recommended:
            rec_sizes.append(recommended[key])

    # Remove duplicates and sort
    rec_sizes = sorted(set(rec_sizes), reverse=True)

    # Compare adjacent pairs
    for i in range(len(rec_sizes) - 1):
        size_larger = rec_sizes[i]
        size_smaller = rec_sizes[i + 1]

        # Find indices
        idx_larger = np.where(sizes == size_larger)[0]
        idx_smaller = np.where(sizes == size_smaller)[0]

        if len(idx_larger) == 0 or len(idx_smaller) == 0:
            continue

        # Get metrics using percentile CI bounds
        auroc_larger = aurocs_cv[idx_larger[0]]
        ci_lower_larger = ci_lower[idx_larger[0]]
        ci_upper_larger = ci_upper[idx_larger[0]]

        auroc_smaller = aurocs_cv[idx_smaller[0]]
        ci_lower_smaller = ci_lower[idx_smaller[0]]
        ci_upper_smaller = ci_upper[idx_smaller[0]]

        std_larger = aurocs_std[idx_larger[0]]
        std_smaller = aurocs_std[idx_smaller[0]]

        # Check if CIs overlap (percentile bootstrap)
        cis_overlap = not (ci_upper_smaller < ci_lower_larger or ci_upper_larger < ci_lower_smaller)

        # Compute Z-score for difference (approximate)
        if std_larger > 0 and std_smaller > 0:
            se_diff = np.sqrt(std_larger**2 + std_smaller**2)
            z_score = abs(auroc_larger - auroc_smaller) / se_diff
            is_significant = z_score > Z_CRITICAL_005  # p < 0.05
        else:
            is_significant = False
            z_score = 0.0

        # Only annotate if there's a meaningful comparison
        if std_larger == 0 and std_smaller == 0:
            continue

        # Position for annotation line (halfway between points in x-axis space)
        x_mid = (size_larger + size_smaller) / 2
        y_pos = min(auroc_larger, auroc_smaller) - 0.015  # Slightly below lower point

        # Color code: green if not significant (CIs overlap), red if significant difference
        if cis_overlap or not is_significant:
            color = COLOR_THRESHOLD_GREEN  # not significantly different
            label = "NS"
        else:
            color = COLOR_THRESHOLD_RED  # significantly different
            label = f"p<0.05\n\u0394={abs(auroc_larger - auroc_smaller):.3f}"

        # Draw horizontal line between points
        ax.plot(
            [size_larger, size_smaller],
            [y_pos, y_pos],
            color=color,
            linestyle="--",
            linewidth=LW_REFERENCE,
            alpha=0.6,
            zorder=3,
        )

        # Add text annotation
        ax.text(
            x_mid,
            y_pos - 0.008,
            label,
            fontsize=7,
            ha="center",
            va="top",
            color=color,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": color,
                "pad": 1,
                "linewidth": 0.5,
            },
        )


def plot_feature_ranking(
    feature_ranking: dict[str, int],
    out_path: Path | str,
    top_n: int = 30,
    title: str = "Feature Elimination Order",
    n_splits: int | None = None,
    feature_selection_method: str | None = None,
    run_id: str | None = None,
) -> None:
    """Plot horizontal bar chart of feature elimination order.

    Features eliminated last (highest order) are most important.

    Args:
        feature_ranking: Dict mapping protein -> elimination_order.
        out_path: Output file path.
        top_n: Number of top features to show.
        title: Plot title.
        n_splits: Number of splits aggregated (optional).
        feature_selection_method: Feature selection method used (optional).
        run_id: Run identifier (optional).

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        return

    if not feature_ranking:
        return

    # Sort by elimination order (descending = eliminated last = most important)
    sorted_features = sorted(feature_ranking.items(), key=lambda x: -x[1])[:top_n]

    proteins = [f[0] for f in sorted_features]
    orders = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(8, max(6, len(proteins) * 0.3)))

    # Color gradient: later elimination = more important = darker
    max_order = max(orders) if orders else 1
    colors = plt.cm.Blues(np.array(orders) / max_order * 0.6 + 0.3)

    ax.barh(range(len(proteins)), orders, color=colors)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=9)
    ax.invert_yaxis()  # Highest order at top

    ax.set_xlabel("Elimination Order (higher = eliminated later = more important)", fontsize=10)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    ax.grid(True, axis="x", alpha=GRID_ALPHA)

    # Add metadata summary text
    total_features = len(feature_ranking)
    summary_lines = [f"Showing top {top_n} of {total_features} features"]

    if n_splits is not None:
        summary_lines.append(f"Splits: {n_splits}")

    if feature_selection_method:
        summary_lines.append(f"Method: {feature_selection_method}")

    if run_id:
        summary_lines.append(f"Run: {run_id}")

    if summary_lines:
        summary_text = "\n".join(summary_lines)
        ax.text(
            0.98,
            0.02,
            summary_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 3},
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)


def plot_rfecv_selection_curve(
    cv_scores_curve_path: Path | str,
    out_path: Path | str,
    title: str = "RFECV Feature Selection Curve",
    model_name: str = "",
) -> None:
    """Plot RFECV internal CV scores vs number of features across folds.

    Shows how cross-validation AUROC varies with feature count during RFECV,
    helping visualize the automatic optimal size selection per fold.

    Args:
        cv_scores_curve_path: Path to cv_scores_curve.csv (from nested_rfe).
        out_path: Output file path for plot.
        title: Plot title.
        model_name: Model name for subtitle.

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
        return

    import pandas as pd

    cv_scores_curve_path = Path(cv_scores_curve_path)
    if not cv_scores_curve_path.exists():
        return

    # Load CV scores data
    df = pd.read_csv(cv_scores_curve_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_DCA)

    # Get unique folds
    folds = sorted(df["fold"].unique())
    n_folds = len(folds)

    # Color palette for folds
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

    # Plot each fold
    for i, fold in enumerate(folds):
        fold_data = df[df["fold"] == fold].sort_values("n_features")
        ax.plot(
            fold_data["n_features"],
            fold_data["cv_score"],
            "o-",
            color=colors[i],
            linewidth=LW_SECONDARY,
            markersize=4,
            alpha=0.7,
            label=f"Fold {fold}",
        )

        # Mark optimal point (max CV score)
        optimal_idx = fold_data["cv_score"].idxmax()
        optimal_row = fold_data.loc[optimal_idx]
        ax.scatter(
            [optimal_row["n_features"]],
            [optimal_row["cv_score"]],
            s=100,
            c=[colors[i]],
            marker="*",
            zorder=10,
            edgecolors="white",
            linewidths=1,
        )

    # Aggregate mean curve across folds
    mean_curve = df.groupby("n_features")["cv_score"].agg(["mean", "std"]).reset_index()
    ax.plot(
        mean_curve["n_features"],
        mean_curve["mean"],
        "k--",
        linewidth=2.5,
        alpha=0.8,
        label="Mean across folds",
    )

    # Add shaded error region
    ax.fill_between(
        mean_curve["n_features"],
        mean_curve["mean"] - mean_curve["std"],
        mean_curve["mean"] + mean_curve["std"],
        color="gray",
        alpha=0.2,
    )

    # Styling
    ax.set_xlabel("Number of Features", fontsize=FONT_LABEL)
    ax.set_ylabel("CV AUROC (Internal)", fontsize=FONT_LABEL)
    ax.set_xlim(0, df["n_features"].max() * 1.05)

    # Y-axis: reasonable range
    y_min = df["cv_score"].min() - 0.02
    y_max = df["cv_score"].max() + 0.02
    y_min = max(0.5, y_min)
    y_max = min(1.0, y_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="best", fontsize=FONT_LEGEND, ncol=2)
    ax.grid(True, alpha=GRID_ALPHA)

    # Title
    if model_name:
        ax.set_title(f"{title}\n{model_name}", fontsize=FONT_TITLE, fontweight="bold")
    else:
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    # Summary text
    optimal_sizes = []
    for fold in folds:
        fold_data = df[df["fold"] == fold]
        optimal_n = fold_data.loc[fold_data["cv_score"].idxmax(), "n_features"]
        optimal_sizes.append(int(optimal_n))

    summary_text = (
        f"Folds: {n_folds}\n"
        f"Mean optimal size: {np.mean(optimal_sizes):.1f} ± {np.std(optimal_sizes):.1f}\n"
        f"Range: [{min(optimal_sizes)}, {max(optimal_sizes)}]"
    )
    ax.text(
        0.98,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=FONT_LEGEND,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray", "pad": 4},
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
