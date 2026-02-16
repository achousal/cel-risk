"""
CLI implementation for aggregate-splits command.

Aggregates results across multiple split seeds into summary statistics,
pooled predictions, aggregated plots, and consensus panels.

The aggregation is organized into two phases:
1. Collection phase: Gather all data from split directories
2. Report phase: Compute statistics and write outputs
"""

import logging
from pathlib import Path
from typing import Any

from ced_ml.cli.aggregation.collection_phase import run_collection_phase
from ced_ml.cli.aggregation.report_phase import run_report_phase
from ced_ml.cli.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)
from ced_ml.utils.logging import log_section, setup_command_logger


def run_aggregate_splits(
    results_dir: str,
    stability_threshold: float = 0.75,
    plot_formats: list[str] | None = None,
    target_specificity: float = 0.95,
    n_boot: int = 500,
    log_level: int | None = None,
    save_plots: bool = True,
    plot_roc: bool = True,
    plot_pr: bool = True,
    plot_calibration: bool = True,
    plot_risk_distribution: bool = True,
    plot_dca: bool = True,
    plot_oof_combined: bool = True,
    plot_learning_curve: bool = True,
    plot_shap_summary: bool = True,
    plot_shap_dependence: bool = True,
    plot_shap_heatmap: bool = True,
    control_spec_targets: list[float] | None = None,
) -> dict[str, Any]:
    """
    Aggregate results across multiple split seeds.

    Two-phase pipeline:
    1. Collection: Gather all data from split directories
    2. Report: Compute statistics and write outputs

    Args:
        results_dir: Directory containing split_seedX subdirectories
        stability_threshold: Fraction of splits for feature stability (default 0.75)
        plot_formats: List of plot formats (default ["png"])
        target_specificity: Target specificity for alpha threshold (default 0.95)
        n_boot: Number of bootstrap iterations (for future CI computation)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        save_plots: Whether to save plots at all (default True)
        plot_roc: Whether to generate ROC plots (default True)
        plot_pr: Whether to generate PR plots (default True)
        plot_calibration: Whether to generate calibration plots (default True)
        plot_risk_distribution: Whether to generate risk distribution plots (default True)
        plot_dca: Whether to generate DCA plots (default True)
        plot_oof_combined: Whether to generate OOF combined plots (default True)
        plot_learning_curve: Whether to generate learning curve plots (default True)

    Returns:
        Dictionary with aggregation results summary
    """
    if plot_formats is None:
        plot_formats = ["png"]

    if control_spec_targets is None:
        control_spec_targets = [0.90, 0.95, 0.99]

    if log_level is None:
        log_level = logging.INFO

    results_path = Path(results_dir)
    _model_name = results_path.name if results_path.name != "splits" else results_path.parent.name
    _run_id = None
    _run_level_dir = results_path

    for parent in [results_path] + list(results_path.parents):
        if parent.name.startswith("run_"):
            _run_id = parent.name[4:]
            _run_level_dir = parent
            break

    logger = setup_command_logger(
        command="aggregate-splits",
        log_level=log_level,
        outdir=_run_level_dir,
        run_id=_run_id,
        model=_model_name,
    )

    log_section(logger, "CeD-ML Split Aggregation")
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    split_dirs = discover_split_dirs(results_path, logger=logger)
    logger.info(f"Found {len(split_dirs)} split directories")

    run_level_dir = results_path.parent
    ensemble_dirs = discover_ensemble_dirs(run_level_dir, logger=logger)

    if ensemble_dirs:
        logger.info(f"Found {len(ensemble_dirs)} ENSEMBLE split directories")

        # Apply max_plot_splits filtering to ensemble directories
        # Try to load config to respect plotting limits
        try:
            from ced_ml.config.loader import load_training_config

            config_path = run_level_dir / "config.yaml"
            if config_path.exists():
                cfg = load_training_config(str(config_path))
                max_plot_splits = getattr(cfg.output, "max_plot_splits", 0)

                if max_plot_splits > 0:
                    original_count = len(ensemble_dirs)
                    # Keep only the first N directories (by sorted order)
                    ensemble_dirs = sorted(
                        ensemble_dirs,
                        key=lambda ed: int(ed.name.replace("split_seed", "")),
                    )[:max_plot_splits]
                    filtered_count = len(ensemble_dirs)

                    if filtered_count < original_count:
                        logger.info(
                            f"Filtered ENSEMBLE dirs from {original_count} to {filtered_count} "
                            f"(max_plot_splits={max_plot_splits})"
                        )
        except Exception as e:
            logger.debug(f"Could not load output config for ensemble filtering: {e}")
            logger.debug("Proceeding with all ensemble directories")

    if not split_dirs and not ensemble_dirs:
        logger.warning("No split_seedX directories found. Nothing to aggregate.")
        return {"status": "no_splits_found"}

    for sd in split_dirs:
        logger.info(f"  {sd.name}")
    for ed in ensemble_dirs:
        logger.info(f"  ENSEMBLE/{ed.name}")

    collected = run_collection_phase(
        split_dirs=split_dirs,
        ensemble_dirs=ensemble_dirs,
        logger=logger,
    )

    return run_report_phase(
        collected=collected,
        split_dirs=split_dirs,
        ensemble_dirs=ensemble_dirs,
        results_path=results_path,
        stability_threshold=stability_threshold,
        target_specificity=target_specificity,
        control_spec_targets=control_spec_targets,
        n_boot=n_boot,
        save_plots=save_plots,
        plot_formats=plot_formats,
        plot_roc=plot_roc,
        plot_pr=plot_pr,
        plot_calibration=plot_calibration,
        plot_risk_distribution=plot_risk_distribution,
        plot_dca=plot_dca,
        plot_oof_combined=plot_oof_combined,
        plot_learning_curve=plot_learning_curve,
        plot_shap_summary=plot_shap_summary,
        plot_shap_dependence=plot_shap_dependence,
        plot_shap_heatmap=plot_shap_heatmap,
        logger=logger,
    )
