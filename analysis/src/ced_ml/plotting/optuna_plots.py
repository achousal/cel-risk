"""
Optuna hyperparameter optimization visualization utilities.

Provides functions to visualize Optuna study results, including:
- Optimization history (value over trials)
- Parameter importances
- Parallel coordinate plots
- Hyperparameter relationships
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .style import (
    BBOX_INCHES,
    DPI,
    FIGSIZE_SINGLE,
    FONT_LABEL,
    FONT_TITLE,
    GRID_ALPHA,
)

logger = logging.getLogger(__name__)

# Attempt optuna import
try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]


def save_optuna_plots(
    study: Any,
    out_dir: Path,
    prefix: str = "",
    plot_format: str = "html",
    fallback_to_html: bool = True,
) -> None:
    """
    Generate and save Optuna study visualization plots.

    Creates:
    - Optimization history plot (objective value over trials)
    - Parameter importance plot (if study has completed trials)
    - Parallel coordinate plot (parameter relationships)
    - Slice plot (parameter vs objective)

    Args:
        study: Optuna Study object
        out_dir: Output directory for plots
        prefix: Filename prefix (e.g., "RF__")
        plot_format: Output format ("html", "png", or "pdf"). Default is "html" for interactive plots.
        fallback_to_html: If True (default), fall back to HTML format when image export fails (e.g., Kaleido/Chrome not available on HPC)

    Returns:
        None. Plots saved to out_dir.

    Note:
        Requires optuna[plotly] for interactive plots.
        Image formats (png/pdf) require kaleido, which needs Chrome.
        On HPC systems, use plot_format="html" or enable fallback_to_html.
    """
    if not _OPTUNA_AVAILABLE or study is None:
        logger.warning("Optuna not available or study is None, skipping optuna plots")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trials = len(study.trials)
    if n_trials == 0:
        logger.warning("No trials in study, skipping optuna plots")
        return

    logger.info(f"Generating Optuna plots for {n_trials} trials")

    # Detect if multi-objective study
    is_multi_objective = len(study.directions) > 1
    target = None
    target_name = None

    if is_multi_objective:
        # For multi-objective, use first objective as primary target
        target = lambda t: t.values[0]  # noqa: E731
        target_name = (
            study.directions[0].name if hasattr(study.directions[0], "name") else "objective_0"
        )
        logger.info(
            f"Multi-objective study detected ({len(study.directions)} objectives). "
            f"Plotting primary objective: {target_name}"
        )

    # Helper to save plot with fallback
    def _save_plot(fig: Any, out_path: Path, plot_name: str) -> bool:
        """Save plot with optional fallback to HTML if image export fails."""
        nonlocal plot_format
        try:
            if plot_format == "html":
                fig.write_html(str(out_path))
            else:
                fig.write_image(str(out_path))
            logger.info(f"Saved {plot_name}: {out_path}")
            return True
        except Exception as e:
            # Check if it's a Kaleido/Chrome error
            if "Kaleido" in str(e) or "Chrome" in str(e):
                if fallback_to_html and plot_format != "html":
                    html_path = out_path.with_suffix(".html")
                    try:
                        fig.write_html(str(html_path))
                        logger.info(
                            f"Image export failed (Kaleido/Chrome not available), "
                            f"saved as HTML instead: {html_path}"
                        )
                        return True
                    except Exception as html_error:
                        logger.warning(
                            f"Failed to save {plot_name} (both {plot_format} and HTML): {html_error}"
                        )
                        return False
                else:
                    logger.warning(
                        f"Failed to save {plot_name}: Kaleido/Chrome not available. "
                        f"Use plot_format='html' or install Chrome on this system."
                    )
                    return False
            else:
                logger.warning(f"Failed to create {plot_name}: {e}")
                return False

    # Try plotly-based plots first (more interactive)
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )

        # 1. Optimization history
        try:
            if is_multi_objective:
                fig = plot_optimization_history(study, target=target, target_name=target_name)
            else:
                fig = plot_optimization_history(study)
            fig.update_layout(title=f"Optimization History ({n_trials} trials)")
            out_path = out_dir / f"{prefix}optuna_history.{plot_format}"
            _save_plot(fig, out_path, "optimization history plot")
        except Exception as e:
            logger.warning(f"Failed to create optimization history plot: {e}")

        # 2. Parameter importances (requires multiple trials)
        if n_trials >= 2:
            try:
                if is_multi_objective:
                    fig = plot_param_importances(study, target=target, target_name=target_name)
                else:
                    fig = plot_param_importances(study)
                fig.update_layout(title="Parameter Importances")
                out_path = out_dir / f"{prefix}optuna_importances.{plot_format}"
                _save_plot(fig, out_path, "parameter importances plot")
            except Exception as e:
                logger.warning(f"Failed to create parameter importances plot: {e}")

        # 3. Parallel coordinate plot
        if n_trials >= 2:
            try:
                if is_multi_objective:
                    fig = plot_parallel_coordinate(study, target=target, target_name=target_name)
                else:
                    fig = plot_parallel_coordinate(study)
                fig.update_layout(title="Parallel Coordinate Plot")
                out_path = out_dir / f"{prefix}optuna_parallel.{plot_format}"
                _save_plot(fig, out_path, "parallel coordinate plot")
            except Exception as e:
                logger.warning(f"Failed to create parallel coordinate plot: {e}")

        # 4. Slice plot (parameter vs objective)
        if n_trials >= 2:
            try:
                if is_multi_objective:
                    fig = plot_slice(study, target=target, target_name=target_name)
                else:
                    fig = plot_slice(study)
                fig.update_layout(title="Slice Plot (Parameter vs Objective)")
                out_path = out_dir / f"{prefix}optuna_slice.{plot_format}"
                _save_plot(fig, out_path, "slice plot")
            except Exception as e:
                logger.warning(f"Failed to create slice plot: {e}")

    except ImportError:
        logger.warning(
            "Optuna visualization not available (install optuna[plotly]). "
            "Saving trials dataframe only."
        )

    # Always save trials dataframe as CSV
    try:
        trials_df = study.trials_dataframe()
        csv_path = out_dir / f"{prefix}optuna_trials.csv"
        trials_df.to_csv(csv_path, index=False)
        logger.info(f"Saved trials dataframe: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save trials dataframe: {e}")


def aggregate_optuna_trials(
    trials_dfs: list[pd.DataFrame],
    out_dir: Path,
    prefix: str = "aggregated_",
) -> pd.DataFrame | None:
    """
    Aggregate Optuna trials across multiple splits.

    Combines trials from multiple runs into a single dataframe
    and computes summary statistics.

    Args:
        trials_dfs: List of trials DataFrames from different splits
        out_dir: Output directory
        prefix: Filename prefix

    Returns:
        Combined trials DataFrame with split_id column added

    Note:
        Adds 'split_id' column to track which split each trial came from.
    """
    if not trials_dfs:
        logger.warning("No trials dataframes to aggregate")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add split_id to each dataframe
    combined_dfs = []
    for split_id, df in enumerate(trials_dfs):
        df = df.copy()
        df["split_id"] = split_id
        combined_dfs.append(df)

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Save combined trials
    csv_path = out_dir / f"{prefix}optuna_trials.csv"
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Saved aggregated trials ({len(combined_df)} trials): {csv_path}")

    # Compute summary statistics
    try:
        summary_stats = {
            "n_splits": len(trials_dfs),
            "total_trials": len(combined_df),
            "mean_best_value": combined_df.groupby("split_id")["value"].min().mean(),
            "std_best_value": combined_df.groupby("split_id")["value"].min().std(),
            "median_n_trials_per_split": len(combined_df) / len(trials_dfs),
        }

        # Save summary
        summary_path = out_dir / f"{prefix}optuna_summary.json"
        import json

        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Saved optuna summary: {summary_path}")

    except Exception as e:
        logger.warning(f"Failed to compute optuna summary stats: {e}")

    return combined_df


def plot_pareto_frontier(
    search_cv: Any,
    outdir: Path,
    plot_format: str = "png",
    dpi: int = 300,
) -> None:
    """Plot Pareto frontier for multi-objective optimization.

    Creates scatter plot showing:
    - All Pareto-optimal trials (gray points)
    - Selected trial (red star)
    - Annotated with metric values

    Args:
        search_cv: Fitted OptunaSearchCV with multi_objective=True
        outdir: Output directory for plot
        plot_format: File format (png, pdf, svg)
        dpi: Plot resolution

    Raises:
        ValueError: If search_cv is single-objective

    Note:
        Requires matplotlib and the OptunaSearchCV object must have been
        fitted with multi_objective=True.
    """
    import matplotlib.pyplot as plt

    if not hasattr(search_cv, "multi_objective") or not search_cv.multi_objective:
        logger.warning("plot_pareto_frontier() called on single-objective study, skipping")
        return

    # Get Pareto frontier data
    try:
        df = search_cv.get_pareto_frontier()
    except Exception as e:
        logger.warning(f"Failed to get Pareto frontier: {e}")
        return

    # Create output directory
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Plot all Pareto-optimal trials
    ax.scatter(
        df["auroc"],
        df["brier_score"],
        c="lightgray",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        label="Pareto frontier",
    )

    # Highlight selected trial
    selected = df[df["is_selected"]]
    if not selected.empty:
        ax.scatter(
            selected["auroc"],
            selected["brier_score"],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            label=f"Selected ({search_cv.pareto_selection})",
            zorder=10,
        )

        # Annotate selected point
        ax.annotate(
            f"AUROC={selected['auroc'].iloc[0]:.3f}\nBrier={selected['brier_score'].iloc[0]:.4f}",
            xy=(selected["auroc"].iloc[0], selected["brier_score"].iloc[0]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.5",
                "fc": "yellow",
                "alpha": 0.7,
                "edgecolor": "black",
            },
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
        )

    ax.set_xlabel("AUROC", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Brier Score", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(
        "Multi-Objective Optimization: Pareto Frontier",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=GRID_ALPHA, linestyle="--")

    # Set axis limits with padding
    if len(df) > 1:
        x_margin = (df["auroc"].max() - df["auroc"].min()) * 0.1
        y_margin = (df["brier_score"].max() - df["brier_score"].min()) * 0.1
        ax.set_xlim(df["auroc"].min() - x_margin, df["auroc"].max() + x_margin)
        ax.set_ylim(df["brier_score"].min() - y_margin, df["brier_score"].max() + y_margin)

    plt.tight_layout()
    outfile = outdir / f"pareto_frontier.{plot_format}"
    plt.savefig(outfile, dpi=dpi if dpi else DPI, bbox_inches=BBOX_INCHES)
    plt.close()

    logger.info(f"Saved Pareto frontier plot: {outfile}")

    # Also save Pareto frontier data as CSV
    csv_path = outdir / "pareto_frontier.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved Pareto frontier data: {csv_path}")
