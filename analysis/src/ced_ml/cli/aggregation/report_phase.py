"""
Report phase for aggregate_splits.

Generates aggregated outputs, plots, and writes all results to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ced_ml.cli.aggregation.aggregation import compute_summary_stats
from ced_ml.cli.aggregation.artifacts import generate_additional_artifacts
from ced_ml.cli.aggregation.collection_phase import CollectedData
from ced_ml.cli.aggregation.ensemble_metadata import collect_ensemble_metadata
from ced_ml.cli.aggregation.orchestrator import (
    aggregate_importance,
    aggregate_shap_importance,
    aggregate_shap_metadata,
    aggregate_shap_values,
    build_return_summary,
    collect_sample_categories_metadata,
    compute_and_save_pooled_metrics,
    save_pooled_predictions,
    setup_aggregation_directories,
)
from ced_ml.cli.aggregation.orchestrator import (
    build_aggregation_metadata as build_agg_metadata,
)
from ced_ml.cli.aggregation.plot_generator import (
    generate_aggregated_plots,
    generate_aggregated_shap_plots,
    generate_model_comparison_report,
)
from ced_ml.cli.aggregation.reporting import (
    aggregate_feature_reports,
    aggregate_feature_stability,
    build_consensus_panels,
)
from ced_ml.data import io_helpers
from ced_ml.utils.metadata import build_aggregated_metadata


def aggregate_hyperparams_summary(
    params_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics for hyperparameters across splits.

    For each model and hyperparameter:
    - Numeric params: mean, std, min, max
    - Categorical params: mode, unique values

    Args:
        params_df: DataFrame with hyperparameters from all splits
        logger: Optional logger instance

    Returns:
        DataFrame with aggregated hyperparameter statistics per model
    """
    if params_df.empty:
        if logger:
            logger.debug("No hyperparameters to aggregate (empty DataFrame)")
        return pd.DataFrame()

    metadata_cols = {
        "split_seed",
        "model",
        "repeat",
        "outer_split",
        "best_score_inner",
        "optuna_n_trials",
        "optuna_sampler",
        "optuna_pruner",
    }
    param_cols = [col for col in params_df.columns if col not in metadata_cols]

    if not param_cols:
        if logger:
            logger.warning("No hyperparameter columns found to aggregate")
        return pd.DataFrame()

    summary_rows = []

    for model_name, model_df in params_df.groupby("model"):
        row = {"model": model_name, "n_cv_folds": len(model_df)}

        for param in param_cols:
            if param not in model_df.columns:
                continue

            values = model_df[param].dropna()
            if len(values) == 0:
                continue

            if pd.api.types.is_numeric_dtype(values):
                row[f"{param}_mean"] = values.mean()
                row[f"{param}_std"] = values.std()
                row[f"{param}_min"] = values.min()
                row[f"{param}_max"] = values.max()
            else:
                mode_val = values.mode()
                row[f"{param}_mode"] = mode_val[0] if len(mode_val) > 0 else None
                row[f"{param}_n_unique"] = values.nunique()
                unique_vals = sorted(values.unique())
                row[f"{param}_values"] = ", ".join(str(v) for v in unique_vals)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if logger:
        logger.info(
            f"Aggregated hyperparams for {len(summary_df)} models, {len(param_cols)} parameters"
        )

    return summary_df


def save_metrics(
    test_metrics: pd.DataFrame,
    val_metrics: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    agg_dir: Path,
    metrics_dir: Path,
    cv_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Save aggregated metrics to disk.

    Args:
        test_metrics: Test metrics DataFrame
        val_metrics: Val metrics DataFrame
        cv_metrics: CV metrics DataFrame
        agg_dir: Aggregation root directory
        metrics_dir: Metrics subdirectory
        cv_dir: CV subdirectory
        logger: Logger instance
    """
    if not test_metrics.empty:
        all_test_path = agg_dir / "all_test_metrics.csv"
        io_helpers.save_metrics(test_metrics, all_test_path)
        logger.info(f"All test metrics saved: {all_test_path}")
        logger.info(
            f"  {len(test_metrics)} rows from {test_metrics['split_seed'].nunique()} splits"
        )

        summary = compute_summary_stats(test_metrics, logger=logger)
        if not summary.empty:
            summary_path = metrics_dir / "test_metrics_summary.csv"
            io_helpers.save_metrics(summary, summary_path)
            logger.info(f"Summary stats saved: {summary_path}")

    if not val_metrics.empty:
        all_val_path = agg_dir / "all_val_metrics.csv"
        io_helpers.save_metrics(val_metrics, all_val_path)
        logger.info(f"All val metrics saved: {all_val_path}")

        val_summary = compute_summary_stats(val_metrics, logger=logger)
        if not val_summary.empty:
            val_summary_path = metrics_dir / "val_metrics_summary.csv"
            io_helpers.save_metrics(val_summary, val_summary_path)
            logger.info(f"Val summary saved: {val_summary_path}")

    if not cv_metrics.empty:
        all_cv_path = cv_dir / "all_cv_repeat_metrics.csv"
        io_helpers.save_metrics(cv_metrics, all_cv_path)
        logger.info(f"All CV metrics saved: {all_cv_path}")

        cv_summary = compute_summary_stats(cv_metrics, logger=logger)
        if not cv_summary.empty:
            cv_summary_path = cv_dir / "cv_metrics_summary.csv"
            io_helpers.save_metrics(cv_summary, cv_summary_path)
            logger.info(f"CV summary saved: {cv_summary_path}")


def save_hyperparams(
    best_params: pd.DataFrame,
    ensemble_params: pd.DataFrame,
    cv_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Save hyperparameters to disk.

    Args:
        best_params: Best hyperparameters DataFrame
        ensemble_params: Ensemble hyperparameters DataFrame
        cv_dir: CV subdirectory
        logger: Logger instance
    """
    if not best_params.empty:
        all_params_path = cv_dir / "all_best_params_per_split.csv"
        io_helpers.save_metrics(best_params, all_params_path)
        logger.info(f"All best hyperparameters saved: {all_params_path}")
        logger.info(
            f"  {len(best_params)} hyperparameter sets from {best_params['split_seed'].nunique()} splits"
        )

        params_summary = aggregate_hyperparams_summary(best_params, logger=logger)
        if not params_summary.empty:
            params_summary_path = cv_dir / "hyperparams_summary.csv"
            io_helpers.save_metrics(params_summary, params_summary_path)
            logger.info(f"Hyperparameters summary saved: {params_summary_path}")

    if not ensemble_params.empty:
        ensemble_params_path = cv_dir / "ensemble_config_per_split.csv"
        io_helpers.save_metrics(ensemble_params, ensemble_params_path)
        logger.info(f"Ensemble configurations saved: {ensemble_params_path}")


def save_optuna_trials(
    optuna_trials_combined: pd.DataFrame | None,
    agg_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Save aggregated Optuna trials to disk.

    Args:
        optuna_trials_combined: Combined Optuna trials DataFrame
        agg_dir: Aggregation root directory
        logger: Logger instance
    """
    if optuna_trials_combined is not None:
        cv_agg_dir = agg_dir / "cv"
        cv_agg_dir.mkdir(parents=True, exist_ok=True)

        combined_csv = cv_agg_dir / "optuna_trials.csv"
        io_helpers.save_metrics(optuna_trials_combined, combined_csv)
        logger.info(f"Aggregated Optuna trials saved: {cv_agg_dir}")


def save_feature_analysis(
    feature_stability_df: pd.DataFrame,
    stable_features_df: pd.DataFrame,
    all_feature_reports: pd.DataFrame,
    agg_feature_report: pd.DataFrame,
    panels_dir: Path,
    stability_threshold: float,
    logger: logging.Logger,
) -> None:
    """
    Save feature stability and reports to disk.

    Args:
        feature_stability_df: Feature stability DataFrame
        stable_features_df: Stable features DataFrame
        all_feature_reports: All feature reports DataFrame
        agg_feature_report: Aggregated feature report DataFrame
        panels_dir: Panels subdirectory
        stability_threshold: Stability threshold
        logger: Logger instance
    """
    panels_dir.mkdir(parents=True, exist_ok=True)

    if not feature_stability_df.empty:
        io_helpers.save_feature_report(
            feature_stability_df, panels_dir / "feature_stability_summary.csv"
        )
        logger.info(f"Feature stability: {len(feature_stability_df)} features analyzed")

    if not stable_features_df.empty:
        io_helpers.save_feature_report(
            stable_features_df, panels_dir / "consensus_stable_features.csv"
        )
        logger.info(
            f"Stable features (>={stability_threshold*100:.0f}% splits): "
            f"{len(stable_features_df)} features"
        )
    else:
        logger.info("No stable features found (or no feature selection data)")

    if not all_feature_reports.empty:
        all_feature_reports_path = panels_dir / "all_feature_reports.csv"
        io_helpers.save_feature_report(all_feature_reports, all_feature_reports_path)
        logger.info(
            f"All feature reports: {len(all_feature_reports)} entries from "
            f"{all_feature_reports['split_seed'].nunique()} splits"
        )

        if not agg_feature_report.empty:
            agg_feature_report_path = panels_dir / "feature_report.csv"
            io_helpers.save_feature_report(agg_feature_report, agg_feature_report_path)
            logger.info(f"Aggregated feature report: {len(agg_feature_report)} proteins analyzed")
            logger.info(
                f"Top 5 proteins by selection frequency: "
                f"{', '.join(agg_feature_report.head(5)['protein'].tolist())}"
            )


def save_consensus_panels(
    consensus_panels: dict[int, dict[str, Any]],
    panels_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Save consensus panels to disk.

    Args:
        consensus_panels: Consensus panel manifests
        panels_dir: Panels subdirectory
        logger: Logger instance
    """
    for panel_size, manifest in consensus_panels.items():
        manifest_path = panels_dir / f"consensus_panel_N{panel_size}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            f"Consensus panel N={panel_size}: {manifest['n_consensus_proteins']} proteins "
            f"(from {manifest['n_splits_with_panel']} splits)"
        )


def generate_ensemble_plots(
    ensemble_dirs: list[Path],
    ensemble_metadata_raw: dict[str, Any],
    pooled_test_metrics: dict[str, dict[str, float]],
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Generate ensemble-specific plots.

    Args:
        ensemble_dirs: List of ensemble directories
        ensemble_metadata_raw: Raw ensemble metadata
        pooled_test_metrics: Test metrics by model
        plots_dir: Plots directory
        logger: Logger instance
    """
    if not ensemble_dirs:
        return

    try:
        from ced_ml.plotting.ensemble import (
            plot_aggregated_weights,
            plot_model_comparison,
            save_ensemble_aggregation_metadata,
        )

        plots_dir.mkdir(parents=True, exist_ok=True)

        coefs_per_split = ensemble_metadata_raw.get("coefs_per_split", {})
        base_models = ensemble_metadata_raw.get("base_models", [])
        meta_penalty = ensemble_metadata_raw.get("meta_penalty", "l2")
        meta_C = ensemble_metadata_raw.get("meta_C", 1.0)

        if coefs_per_split:
            plot_aggregated_weights(
                coefs_per_split=coefs_per_split,
                out_path=plots_dir / "ensemble_weights_aggregated.png",
                title="Aggregated Meta-Learner Coefficients",
            )
            logger.info("Aggregated ensemble weights plot saved")

            save_ensemble_aggregation_metadata(
                coefs_per_split=coefs_per_split,
                pooled_test_metrics=pooled_test_metrics,
                base_models=base_models,
                meta_penalty=meta_penalty,
                meta_C=meta_C,
                out_dir=plots_dir,
            )
            logger.info("Ensemble aggregation metadata saved")

        if len(pooled_test_metrics) >= 2:
            plot_model_comparison(
                metrics=pooled_test_metrics,
                out_path=plots_dir / "model_comparison.png",
                title="Model Comparison (Pooled Test Set)",
                highlight_model="ENSEMBLE",
                meta_lines=[f"n_models={len(pooled_test_metrics)}"],
            )
            logger.info("Model comparison plot saved")

    except Exception as e:
        logger.warning(f"Ensemble aggregate plot generation failed (non-fatal): {e}")


def run_report_phase(
    collected: CollectedData,
    split_dirs: list[Path],
    ensemble_dirs: list[Path],
    results_path: Path,
    stability_threshold: float,
    target_specificity: float,
    control_spec_targets: list[float],
    n_boot: int,
    save_plots: bool,
    plot_formats: list[str],
    plot_roc: bool,
    plot_pr: bool,
    plot_calibration: bool,
    plot_risk_distribution: bool,
    plot_dca: bool,
    plot_oof_combined: bool,
    plot_learning_curve: bool,
    plot_shap_summary: bool = True,
    plot_shap_dependence: bool = True,
    plot_shap_heatmap: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Run the complete report phase, writing all outputs to disk.

    Args:
        collected: Collected data from collection phase
        split_dirs: List of split directories
        ensemble_dirs: List of ensemble directories
        results_path: Results directory path
        stability_threshold: Feature stability threshold
        target_specificity: Target specificity for thresholds
        control_spec_targets: List of specificity targets
        n_boot: Number of bootstrap iterations
        save_plots: Whether to save plots
        plot_formats: List of plot formats
        plot_roc: Whether to generate ROC plots
        plot_pr: Whether to generate PR plots
        plot_calibration: Whether to generate calibration plots
        plot_risk_distribution: Whether to generate risk distribution plots
        plot_dca: Whether to generate DCA plots
        plot_oof_combined: Whether to generate OOF combined plots
        plot_learning_curve: Whether to generate learning curve plots
        plot_shap_summary: Whether to generate SHAP bar and beeswarm plots
        plot_shap_dependence: Whether to generate SHAP scatter (dependence) plots
        plot_shap_heatmap: Whether to generate SHAP heatmap plots
        logger: Logger instance

    Returns:
        Summary dictionary
    """
    from ced_ml.utils.logging import log_section

    dirs = setup_aggregation_directories(results_path)
    agg_dir = Path(dirs.root)
    metrics_dir = Path(dirs.metrics)
    panels_dir = Path(dirs.panels)
    plots_dir = Path(dirs.plots)
    cv_dir = Path(dirs.cv)
    preds_dir = Path(dirs.preds)

    logger.info(f"Output: {agg_dir}")

    log_section(logger, "Saving Pooled Predictions")
    save_pooled_predictions(
        collected.pooled_test_df,
        collected.pooled_val_df,
        collected.pooled_train_oof_df,
        preds_dir,
        logger,
    )

    log_section(logger, "Computing Pooled Metrics")
    pooled_test_metrics, pooled_val_metrics, threshold_info = compute_and_save_pooled_metrics(
        pooled_test_df=collected.pooled_test_df,
        pooled_val_df=collected.pooled_val_df,
        target_specificity=target_specificity,
        control_spec_targets=control_spec_targets,
        metrics_dir=metrics_dir,
        agg_dir=agg_dir,
        logger=logger,
    )

    log_section(logger, "Generating Model Comparison Report")
    _ = generate_model_comparison_report(
        pooled_test_metrics=pooled_test_metrics,
        pooled_val_metrics=pooled_val_metrics,
        threshold_info=threshold_info,
        out_dir=agg_dir,
        logger=logger,
    )

    if ensemble_dirs:
        generate_ensemble_plots(
            ensemble_dirs,
            collected.ensemble_metadata_raw,
            pooled_test_metrics,
            plots_dir,
            logger,
        )

    log_section(logger, "Saving Per-Split Metrics")
    save_metrics(
        collected.test_metrics,
        collected.val_metrics,
        collected.cv_metrics,
        agg_dir,
        metrics_dir,
        cv_dir,
        logger,
    )

    log_section(logger, "Saving Hyperparameters")
    save_hyperparams(collected.best_params, collected.ensemble_params, cv_dir, logger)

    save_optuna_trials(collected.optuna_trials_combined, agg_dir, logger)

    log_section(logger, "Feature Stability Analysis")
    feature_stability_df, stable_features_df = aggregate_feature_stability(
        split_dirs, stability_threshold=stability_threshold, logger=logger
    )

    agg_feature_report = pd.DataFrame()
    if not collected.all_feature_reports.empty:
        agg_feature_report = aggregate_feature_reports(collected.all_feature_reports, logger=logger)

    save_feature_analysis(
        feature_stability_df,
        stable_features_df,
        collected.all_feature_reports,
        agg_feature_report,
        panels_dir,
        stability_threshold,
        logger,
    )

    log_section(logger, "Aggregating OOF Importance")
    for model_name in collected.all_models:
        aggregate_importance(
            split_dirs=split_dirs,
            model_name=model_name,
            output_dir=agg_dir,
            logger=logger,
        )

    log_section(logger, "Aggregating OOF SHAP Importance")
    oof_shap_dfs: dict[str, pd.DataFrame | None] = {}
    for model_name in collected.all_models:
        oof_shap_dfs[model_name] = aggregate_shap_importance(
            split_dirs=split_dirs,
            model_name=model_name,
            output_dir=agg_dir,
            logger=logger,
        )

    log_section(logger, "Aggregating SHAP Values and Metadata")
    shap_payloads: dict[str, tuple[pd.DataFrame | None, dict | None]] = {}
    for model_name in collected.all_models:
        pooled_shap = aggregate_shap_values(split_dirs, model_name, agg_dir, logger)
        shap_meta = aggregate_shap_metadata(split_dirs, model_name, agg_dir, logger)
        shap_payloads[model_name] = (pooled_shap, shap_meta)

    log_section(logger, "Building Consensus Panels")
    consensus_panels = build_consensus_panels(
        split_dirs, threshold=stability_threshold, logger=logger
    )
    save_consensus_panels(consensus_panels, panels_dir, logger)

    log_section(logger, "Saving Aggregation Metadata")
    sample_categories_metadata = collect_sample_categories_metadata(
        pooled_test_df=collected.pooled_test_df,
        pooled_val_df=collected.pooled_val_df,
        pooled_train_oof_df=collected.pooled_train_oof_df,
    )

    log_section(logger, "Generating Aggregated Plots")
    n_splits = len(split_dirs)
    split_seeds = [int(sd.name.replace("split_seed", "")) for sd in split_dirs]
    meta_lines = build_aggregated_metadata(
        n_splits=n_splits,
        split_seeds=split_seeds,
        sample_categories=sample_categories_metadata,
        timestamp=True,
    )

    if save_plots:
        generate_aggregated_plots(
            pooled_test_df=collected.pooled_test_df,
            pooled_val_df=collected.pooled_val_df,
            pooled_train_oof_df=collected.pooled_train_oof_df,
            out_dir=agg_dir,
            threshold_info=threshold_info,
            plot_formats=plot_formats,
            meta_lines=meta_lines,
            logger=logger,
            plot_roc=plot_roc,
            plot_pr=plot_pr,
            plot_calibration=plot_calibration,
            plot_risk_distribution=plot_risk_distribution,
            plot_dca=plot_dca,
            plot_oof_combined=plot_oof_combined,
            target_specificity=target_specificity,
        )

        # Aggregated SHAP plots
        for model_name in collected.all_models:
            pooled_shap, shap_meta = shap_payloads.get(model_name, (None, None))
            oof_shap_df = oof_shap_dfs.get(model_name)
            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_shap,
                oof_shap_importance_df=oof_shap_df,
                shap_metadata=shap_meta,
                model_name=model_name,
                out_dir=agg_dir,
                plot_formats=plot_formats,
                plot_shap_summary=plot_shap_summary,
                plot_shap_dependence=plot_shap_dependence,
                plot_shap_heatmap=plot_shap_heatmap,
                logger=logger,
            )

    log_section(logger, "Generating Additional Artifacts")
    generate_additional_artifacts(
        pooled_test_df=collected.pooled_test_df,
        pooled_val_df=collected.pooled_val_df,
        split_dirs=split_dirs,
        out_dir=agg_dir,
        save_plots=save_plots,
        plot_learning_curve=plot_learning_curve,
        plot_formats=plot_formats,
        meta_lines=meta_lines,
        logger=logger,
    )

    ensemble_metadata = collect_ensemble_metadata(
        ensemble_dirs=ensemble_dirs,
        all_models=collected.all_models,
        pooled_test_metrics=pooled_test_metrics,
        logger=logger,
    )

    _ = build_agg_metadata(
        n_splits=n_splits,
        split_seeds=split_seeds,
        all_models=collected.all_models,
        n_boot=n_boot,
        stability_threshold=stability_threshold,
        target_specificity=target_specificity,
        sample_categories_metadata=sample_categories_metadata,
        pooled_test_metrics=pooled_test_metrics,
        pooled_val_metrics=pooled_val_metrics,
        threshold_info=threshold_info,
        feature_stability_df=feature_stability_df,
        stable_features_df=stable_features_df,
        agg_feature_report=agg_feature_report,
        all_feature_reports=collected.all_feature_reports,
        consensus_panels=consensus_panels,
        ensemble_metadata=ensemble_metadata,
        agg_dir=agg_dir,
    )
    logger.info(f"Metadata saved: {agg_dir / 'aggregation_metadata.json'}")

    log_section(logger, "Aggregation Complete")
    logger.info(f"Results saved to: {agg_dir}")

    return build_return_summary(
        all_models=collected.all_models,
        pooled_test_metrics=pooled_test_metrics,
        threshold_info=threshold_info,
        n_splits=n_splits,
        stable_features_df=stable_features_df,
        agg_dir=agg_dir,
    )
