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
            del fig
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
                del fig
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
                del fig
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
