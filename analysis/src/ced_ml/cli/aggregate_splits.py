"""
CLI implementation for aggregate-splits command.

Aggregates results across multiple split seeds into summary statistics,
pooled predictions, aggregated plots, and consensus panels.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ced_ml.cli.aggregation.aggregation import (
    compute_summary_stats,
)
from ced_ml.cli.aggregation.artifacts import (
    generate_additional_artifacts,
)
from ced_ml.cli.aggregation.collection import (
    collect_best_hyperparams,
    collect_ensemble_hyperparams,
    collect_ensemble_predictions,
    collect_feature_reports,
    collect_metrics,
    collect_predictions,
)
from ced_ml.cli.aggregation.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)
from ced_ml.cli.aggregation.ensemble_metadata import (
    collect_ensemble_metadata,
)
from ced_ml.cli.aggregation.orchestrator import (
    build_aggregation_metadata as build_agg_metadata,
)
from ced_ml.cli.aggregation.orchestrator import (
    build_return_summary,
    collect_sample_categories_metadata,
    compute_and_save_pooled_metrics,
    save_pooled_predictions,
    setup_aggregation_directories,
)
from ced_ml.cli.aggregation.plot_generator import (
    generate_aggregated_plots,
    generate_model_comparison_report,
)
from ced_ml.cli.aggregation.reporting import (
    aggregate_feature_reports,
    aggregate_feature_stability,
    build_consensus_panels,
)
from ced_ml.config.loader import load_aggregate_config
from ced_ml.utils.logging import auto_log_path, log_section, setup_logger
from ced_ml.utils.metadata import build_aggregated_metadata


def _find_results_root() -> Path:
    """Locate the results root directory.

    Checks in order:
    1. CED_RESULTS_DIR environment variable
    2. Project root (relative to this file)
    3. Current working directory
    """
    import os

    from ced_ml.utils.paths import get_project_root

    results_dir_env = os.getenv("CED_RESULTS_DIR")
    if results_dir_env:
        return Path(results_dir_env)

    # Use smart traversal to find project root
    results_dir = get_project_root() / "results"

    if not results_dir.exists() and (Path.cwd() / "results").exists():
        results_dir = Path.cwd() / "results"

    return results_dir


def resolve_results_dir_from_run_id(
    run_id: str | None = None,
    model: str | None = None,
    return_all_models: bool = False,
) -> str | dict[str, str]:
    """Auto-detect results directory from run_id.

    Directory layout: results/run_{RUN_ID}/{MODEL}/

    Args:
        run_id: Run ID (e.g., "20260127_115115"). If None, auto-detects latest.
        model: Model name (e.g., "LR_EN"). If None and return_all_models is False,
               requires single model. If specified, only returns that model.
        return_all_models: If True, returns dict of all models. If False, returns
                          single path string (requires exactly one model unless
                          model is specified).

    Returns:
        If return_all_models is True: Dict mapping model names to their paths
        If return_all_models is False: String path to model results directory

    Raises:
        FileNotFoundError: If no matching results found
        ValueError: If configuration is invalid (e.g., multiple models without specification)
    """
    results_dir = _find_results_root()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Auto-detect run_id if not provided: scan results/run_*/
    if not run_id:
        run_ids = []
        for run_dir in results_dir.glob("run_*"):
            if run_dir.is_dir():
                rid = run_dir.name.replace("run_", "")
                run_ids.append(rid)

        if not run_ids:
            raise FileNotFoundError("No runs found in results directory")

        run_ids.sort(reverse=True)
        run_id = run_ids[0]

    run_path = results_dir / f"run_{run_id}"

    # If model is specified, return results/run_{id}/{model}/
    if model:
        model_path = run_path / model
        if model_path.exists():
            if return_all_models:
                return {model: str(model_path)}
            return str(model_path)

        raise FileNotFoundError(
            f"Results directory not found for model {model}, run {run_id}.\n" f"Tried: {model_path}"
        )

    # If model is not specified, find models under results/run_{id}/
    if not run_path.exists():
        raise FileNotFoundError(
            f"No results found for run {run_id}.\n" f"Searched in: {results_dir}"
        )

    matching_models = []
    for model_dir in sorted(run_path.glob("*/")):
        if model_dir.name.startswith(".") or model_dir.name in (
            "investigations",
            "consensus",
        ):
            continue
        matching_models.append((model_dir.name, str(model_dir)))

    if not matching_models:
        raise FileNotFoundError(
            f"No models found for run {run_id}.\n"
            f"Searched in: {run_path}\n"
            f"Tip: Specify --model to target a specific model."
        )

    # If return_all_models is True, return dict of all models
    if return_all_models:
        return dict(matching_models)

    # Otherwise, require single model
    if len(matching_models) > 1:
        model_names = ", ".join([m[0] for m in matching_models])
        raise ValueError(
            f"Multiple models found for run {run_id}: {model_names}\n"
            f"Please specify --model to choose one:\n"
            + "\n".join(
                [
                    f"  ced aggregate-splits --run-id {run_id} --model {m[0]}"
                    for m in matching_models
                ]
            )
        )

    model_name, model_path = matching_models[0]
    return model_path


def run_aggregate_splits_with_config(
    config_file: str | None = None,
    overrides: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Wrapper for run_aggregate_splits that loads config from YAML.

    Args:
        config_file: Path to aggregate_config.yaml (optional)
        overrides: List of CLI overrides in "key=value" format
        **kwargs: Additional keyword arguments override config values

    Returns:
        Dictionary with aggregation results summary
    """
    config = load_aggregate_config(config_file=config_file, overrides=overrides)

    params = {
        "results_dir": str(config.results_dir),
        "stability_threshold": config.min_stability,
        "plot_formats": [config.plot_format] if hasattr(config, "plot_format") else ["png"],
        "target_specificity": 0.95,
        "n_boot": 500,
        "verbose": 0,
        "save_plots": config.save_plots,
        "plot_roc": config.plot_roc,
        "plot_pr": config.plot_pr,
        "plot_calibration": config.plot_calibration,
        "plot_risk_distribution": config.plot_risk_distribution,
        "plot_dca": config.plot_dca,
        "plot_oof_combined": config.plot_oof_combined,
        "plot_learning_curve": config.plot_learning_curve,
    }

    params.update(kwargs)

    return run_aggregate_splits(**params)


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

    # Identify parameter columns (exclude metadata)
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

    # Group by model
    for model_name, model_df in params_df.groupby("model"):
        row = {"model": model_name, "n_cv_folds": len(model_df)}

        for param in param_cols:
            if param not in model_df.columns:
                continue

            values = model_df[param].dropna()
            if len(values) == 0:
                continue

            # Check if numeric
            if pd.api.types.is_numeric_dtype(values):
                row[f"{param}_mean"] = values.mean()
                row[f"{param}_std"] = values.std()
                row[f"{param}_min"] = values.min()
                row[f"{param}_max"] = values.max()
            else:
                # Categorical: get mode and unique count
                mode_val = values.mode()
                row[f"{param}_mode"] = mode_val[0] if len(mode_val) > 0 else None
                row[f"{param}_n_unique"] = values.nunique()
                # Include all unique values as comma-separated string
                unique_vals = sorted(values.unique())
                row[f"{param}_values"] = ", ".join(str(v) for v in unique_vals)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if logger:
        logger.info(
            f"Aggregated hyperparams for {len(summary_df)} models, " f"{len(param_cols)} parameters"
        )

    return summary_df


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
    control_spec_targets: list[float] | None = None,
) -> dict[str, Any]:
    """
    Aggregate results across multiple split seeds.

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

    # Derive run_id and model from results_dir path (pattern: .../run_{ID}/{MODEL}/...)
    results_path = Path(results_dir)
    _model_name = results_path.name if results_path.name != "splits" else results_path.parent.name
    _run_id = None
    for parent in [results_path] + list(results_path.parents):
        if parent.name.startswith("run_"):
            _run_id = parent.name[4:]  # strip "run_" prefix
            break

    # Auto-file-logging
    log_file = auto_log_path(
        command="aggregate-splits",
        outdir=results_path.parent.parent if _run_id else results_path,
        run_id=_run_id,
        model=_model_name,
    )
    logger = setup_logger("ced_ml", level=log_level, log_file=log_file)
    logger.info(f"Logging to file: {log_file}")

    log_section(logger, "CeD-ML Split Aggregation")
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    split_dirs = discover_split_dirs(results_path, logger=logger)
    logger.info(f"Found {len(split_dirs)} split directories")

    # Discover ENSEMBLE directories at the run level
    # Layout: results/run_{id}/{MODEL}/ -> go up 1 level to run dir
    run_level_dir = results_path.parent

    ensemble_dirs = discover_ensemble_dirs(run_level_dir, logger=logger)
    if ensemble_dirs:
        logger.info(f"Found {len(ensemble_dirs)} ENSEMBLE split directories")

    if not split_dirs and not ensemble_dirs:
        logger.warning("No split_seedX directories found. Nothing to aggregate.")
        return {"status": "no_splits_found"}

    for sd in split_dirs:
        logger.info(f"  {sd.name}")
    for ed in ensemble_dirs:
        logger.info(f"  ENSEMBLE/{ed.name}")

    # Setup directory structure: aggregated/metrics/, aggregated/panels/, aggregated/plots/
    dirs = setup_aggregation_directories(results_path)
    agg_dir = dirs["agg"]
    metrics_dir = dirs["metrics"]  # CSV/JSON metrics
    panels_dir = dirs["panels"]  # Feature panels and stability
    plots_dir = dirs["plots"]  # All visualizations
    cv_dir = dirs["cv"]  # CV artifacts (hyperparams, repeat metrics)
    preds_dir = dirs["preds"]  # Pooled predictions

    logger.info(f"Output: {agg_dir}")

    log_section(logger, "Collecting Pooled Predictions")

    pooled_test_df = collect_predictions(split_dirs, "test", logger)
    pooled_val_df = collect_predictions(split_dirs, "val", logger)
    pooled_train_oof_df = collect_predictions(split_dirs, "train_oof", logger)

    # Collect ENSEMBLE predictions if available and merge with other model predictions
    if ensemble_dirs:
        ensemble_test_df = collect_ensemble_predictions(ensemble_dirs, "test", logger)
        ensemble_val_df = collect_ensemble_predictions(ensemble_dirs, "val", logger)
        ensemble_oof_df = collect_ensemble_predictions(ensemble_dirs, "train_oof", logger)

        if not ensemble_test_df.empty:
            pooled_test_df = pd.concat([pooled_test_df, ensemble_test_df], ignore_index=True)
            logger.info(f"Merged ENSEMBLE test predictions: {len(ensemble_test_df)} samples")

        if not ensemble_val_df.empty:
            pooled_val_df = pd.concat([pooled_val_df, ensemble_val_df], ignore_index=True)
            logger.info(f"Merged ENSEMBLE val predictions: {len(ensemble_val_df)} samples")

        if not ensemble_oof_df.empty:
            pooled_train_oof_df = pd.concat(
                [pooled_train_oof_df, ensemble_oof_df], ignore_index=True
            )
            logger.info(f"Merged ENSEMBLE OOF predictions: {len(ensemble_oof_df)} samples")

    # Save pooled predictions
    save_pooled_predictions(pooled_test_df, pooled_val_df, pooled_train_oof_df, preds_dir, logger)

    log_section(logger, "Computing Pooled Metrics")

    # Compute and save pooled metrics
    pooled_test_metrics, pooled_val_metrics, threshold_info = compute_and_save_pooled_metrics(
        pooled_test_df=pooled_test_df,
        pooled_val_df=pooled_val_df,
        target_specificity=target_specificity,
        control_spec_targets=control_spec_targets,
        metrics_dir=metrics_dir,
        agg_dir=agg_dir,
        logger=logger,
    )

    # Detect all models
    test_models = (
        pooled_test_df["model"].unique().tolist()
        if not pooled_test_df.empty and "model" in pooled_test_df.columns
        else []
    )
    val_models = (
        pooled_val_df["model"].unique().tolist()
        if not pooled_val_df.empty and "model" in pooled_val_df.columns
        else []
    )
    all_models = sorted(set(test_models + val_models))

    # Generate model comparison report (includes ENSEMBLE if available)
    log_section(logger, "Generating Model Comparison Report")
    _ = generate_model_comparison_report(
        pooled_test_metrics=pooled_test_metrics,
        pooled_val_metrics=pooled_val_metrics,
        threshold_info=threshold_info,
        out_dir=agg_dir,
        logger=logger,
    )

    # Generate ensemble-specific aggregate plots and metadata
    if ensemble_dirs and pooled_test_metrics:
        try:
            from ced_ml.plotting.ensemble import (
                plot_aggregated_weights,
                plot_model_comparison,
                save_ensemble_aggregation_metadata,
            )

            agg_plots_dir = plots_dir
            agg_plots_dir.mkdir(parents=True, exist_ok=True)

            # Collect meta-learner coefficients from each ensemble split
            coefs_per_split: dict[int, dict[str, float]] = {}
            base_models_list = []
            meta_penalty = "l2"
            meta_C = 1.0

            for ed in ensemble_dirs:
                settings_path = ed / "core" / "run_settings.json"
                config_path = ed / "config.yaml"

                if settings_path.exists():
                    try:
                        with open(settings_path) as f:
                            settings = json.load(f)
                        meta_coef = settings.get("meta_coef", {})
                        if meta_coef:
                            seed = settings.get("split_seed", 0)
                            coefs_per_split[seed] = meta_coef
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Could not read ensemble settings from {ed}: {e}")

                # Extract ensemble config
                if config_path.exists() and not base_models_list:
                    try:
                        import yaml

                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                        if "ensemble" in config:
                            ensemble_cfg = config["ensemble"]
                            base_models_list = ensemble_cfg.get("base_models", [])
                            meta_learner_cfg = ensemble_cfg.get("meta_model", {})
                            meta_penalty = meta_learner_cfg.get("penalty", "l2")
                            meta_C = meta_learner_cfg.get("C", 1.0)
                    except Exception as e:
                        logger.debug(f"Could not read ensemble config from {config_path}: {e}")

            if coefs_per_split:
                # Generate aggregated weights plot
                plot_aggregated_weights(
                    coefs_per_split=coefs_per_split,
                    out_path=agg_plots_dir / "ensemble_weights_aggregated.png",
                    title="Aggregated Meta-Learner Coefficients",
                    meta_lines=[f"n_splits={len(coefs_per_split)}"],
                )
                logger.info("Aggregated ensemble weights plot saved")

                # Generate and save ensemble metadata JSON
                save_ensemble_aggregation_metadata(
                    coefs_per_split=coefs_per_split,
                    pooled_test_metrics=pooled_test_metrics,
                    base_models=base_models_list,
                    meta_penalty=meta_penalty,
                    meta_C=meta_C,
                    out_dir=agg_plots_dir,
                )
                logger.info("Ensemble aggregation metadata saved")

            # Model comparison chart across all discovered models
            if len(pooled_test_metrics) >= 2:
                plot_model_comparison(
                    metrics=pooled_test_metrics,
                    out_path=agg_plots_dir / "model_comparison.png",
                    title="Model Comparison (Pooled Test Set)",
                    highlight_model="ENSEMBLE",
                    meta_lines=[f"n_models={len(pooled_test_metrics)}"],
                )
                logger.info("Model comparison plot saved")

        except Exception as e:
            logger.warning(f"Ensemble aggregate plot generation failed (non-fatal): {e}")

    log_section(logger, "Aggregating Per-Split Metrics")

    test_metrics = collect_metrics(split_dirs, "core/test_metrics.csv", logger=logger)
    if not test_metrics.empty:
        all_test_path = agg_dir / "all_test_metrics.csv"
        test_metrics.to_csv(all_test_path, index=False)
        logger.info(f"All test metrics saved: {all_test_path}")
        logger.info(
            f"  {len(test_metrics)} rows from {test_metrics['split_seed'].nunique()} splits"
        )

        summary = compute_summary_stats(test_metrics, logger=logger)
        if not summary.empty:
            summary_path = metrics_dir / "test_metrics_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"Summary stats saved: {summary_path}")

    val_metrics = collect_metrics(split_dirs, "core/val_metrics.csv", logger=logger)
    if not val_metrics.empty:
        all_val_path = agg_dir / "all_val_metrics.csv"
        val_metrics.to_csv(all_val_path, index=False)
        logger.info(f"All val metrics saved: {all_val_path}")

        val_summary = compute_summary_stats(val_metrics, logger=logger)
        if not val_summary.empty:
            val_summary_path = metrics_dir / "val_metrics_summary.csv"
            val_summary.to_csv(val_summary_path, index=False)
            logger.info(f"Val summary saved: {val_summary_path}")

    cv_metrics = collect_metrics(split_dirs, "cv/cv_repeat_metrics.csv", logger=logger)
    if not cv_metrics.empty:
        all_cv_path = cv_dir / "all_cv_repeat_metrics.csv"
        cv_metrics.to_csv(all_cv_path, index=False)
        logger.info(f"All CV metrics saved: {all_cv_path}")

        cv_summary = compute_summary_stats(cv_metrics, logger=logger)
        if not cv_summary.empty:
            cv_summary_path = cv_dir / "cv_metrics_summary.csv"
            cv_summary.to_csv(cv_summary_path, index=False)
            logger.info(f"CV summary saved: {cv_summary_path}")
    else:
        logger.info("No CV metrics found (optional)")

    # Aggregate best hyperparameters
    log_section(logger, "Aggregating Hyperparameters")

    best_params = collect_best_hyperparams(split_dirs, logger=logger)
    if not best_params.empty:
        all_params_path = cv_dir / "all_best_params_per_split.csv"
        best_params.to_csv(all_params_path, index=False)
        logger.info(f"All best hyperparameters saved: {all_params_path}")
        logger.info(
            f"  {len(best_params)} hyperparameter sets from {best_params['split_seed'].nunique()} splits"
        )

        params_summary = aggregate_hyperparams_summary(best_params, logger=logger)
        if not params_summary.empty:
            params_summary_path = cv_dir / "hyperparams_summary.csv"
            params_summary.to_csv(params_summary_path, index=False)
            logger.info(f"Hyperparameters summary saved: {params_summary_path}")
    else:
        logger.info("No hyperparameters found (Optuna may not be enabled)")

    # Aggregate ensemble hyperparameters if available
    if ensemble_dirs:
        ensemble_params = collect_ensemble_hyperparams(ensemble_dirs, logger=logger)
        if not ensemble_params.empty:
            ensemble_params_path = cv_dir / "ensemble_config_per_split.csv"
            ensemble_params.to_csv(ensemble_params_path, index=False)
            logger.info(f"Ensemble configurations saved: {ensemble_params_path}")

    log_section(logger, "Feature Stability Analysis")

    feature_stability_df, stable_features_df = aggregate_feature_stability(
        split_dirs, stability_threshold=stability_threshold, logger=logger
    )

    panels_dir.mkdir(parents=True, exist_ok=True)

    if not feature_stability_df.empty:
        feature_stability_df.to_csv(panels_dir / "feature_stability_summary.csv", index=False)
        logger.info(f"Feature stability: {len(feature_stability_df)} features analyzed")

    if not stable_features_df.empty:
        stable_features_df.to_csv(panels_dir / "consensus_stable_features.csv", index=False)
        logger.info(
            f"Stable features (>={stability_threshold*100:.0f}% splits): "
            f"{len(stable_features_df)} features"
        )
    else:
        logger.info("No stable features found (or no feature selection data)")

    log_section(logger, "Building Consensus Panels")

    consensus_panels = build_consensus_panels(
        split_dirs, threshold=stability_threshold, logger=logger
    )

    for panel_size, manifest in consensus_panels.items():
        manifest_path = panels_dir / f"consensus_panel_N{panel_size}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            f"Consensus panel N={panel_size}: {manifest['n_consensus_proteins']} proteins "
            f"(from {manifest['n_splits_with_panel']} splits)"
        )

    log_section(logger, "Aggregating Feature Reports")

    all_feature_reports = collect_feature_reports(split_dirs, logger=logger)
    agg_feature_report = pd.DataFrame()

    if not all_feature_reports.empty:
        all_feature_reports_path = panels_dir / "all_feature_reports.csv"
        all_feature_reports.to_csv(all_feature_reports_path, index=False)
        logger.info(
            f"All feature reports: {len(all_feature_reports)} entries from "
            f"{all_feature_reports['split_seed'].nunique()} splits"
        )

        agg_feature_report = aggregate_feature_reports(all_feature_reports, logger=logger)
        if not agg_feature_report.empty:
            agg_feature_report_path = panels_dir / "feature_report.csv"
            agg_feature_report.to_csv(agg_feature_report_path, index=False)
            logger.info(f"Aggregated feature report: {len(agg_feature_report)} proteins analyzed")
            logger.info(
                f"Top 5 proteins by selection frequency: "
                f"{', '.join(agg_feature_report.head(5)['protein'].tolist())}"
            )
    else:
        logger.info("No feature reports found (optional - depends on feature selection)")

    log_section(logger, "Saving Aggregation Metadata")

    # Aggregate sample category breakdowns from pooled predictions
    sample_categories_metadata = collect_sample_categories_metadata(
        pooled_test_df=pooled_test_df,
        pooled_val_df=pooled_val_df,
        pooled_train_oof_df=pooled_train_oof_df,
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
            pooled_test_df=pooled_test_df,
            pooled_val_df=pooled_val_df,
            pooled_train_oof_df=pooled_train_oof_df,
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

    log_section(logger, "Aggregating Optuna Trials")

    # Aggregate Optuna hyperparameter tuning trials across splits
    try:

        # Concat-as-you-go to avoid accumulating all DataFrames in memory
        optuna_trials_combined = None
        n_optuna_trials = 0
        for split_dir in split_dirs:
            # Optuna files are flat at cv level (no optuna subdirectory)
            cv_dir = split_dir / "cv"
            if not cv_dir.exists():
                continue

            # Look for model-prefixed optuna_trials file (e.g., LinSVM_cal__optuna_trials.csv)
            optuna_files = list(cv_dir.glob("*__optuna_trials.csv"))

            if optuna_files:
                optuna_csv = optuna_files[0]  # Use first match
                try:
                    df = pd.read_csv(optuna_csv)
                    if optuna_trials_combined is None:
                        optuna_trials_combined = df
                    else:
                        optuna_trials_combined = pd.concat(
                            [optuna_trials_combined, df], ignore_index=True
                        )
                    n_optuna_trials += 1
                except Exception as e:
                    logger.warning(f"Failed to load optuna trials from {optuna_csv}: {e}")

        if optuna_trials_combined is not None:
            # Save flat at cv level (no optuna subdirectory)
            cv_agg_dir = agg_dir / "cv"
            cv_agg_dir.mkdir(parents=True, exist_ok=True)

            # Save combined trials directly (already concatenated)
            combined_csv = cv_agg_dir / "optuna_trials.csv"
            optuna_trials_combined.to_csv(combined_csv, index=False)
            logger.info(f"Aggregated {n_optuna_trials} Optuna trial sets: {cv_agg_dir}")
        else:
            logger.info("No Optuna trials found (optional - depends on config.optuna.enabled)")

    except Exception as e:
        logger.warning(f"Failed to aggregate Optuna trials: {e}")

    log_section(logger, "Generating Additional Artifacts")

    generate_additional_artifacts(
        pooled_test_df=pooled_test_df,
        pooled_val_df=pooled_val_df,
        split_dirs=split_dirs,
        out_dir=agg_dir,
        save_plots=save_plots,
        plot_learning_curve=plot_learning_curve,
        plot_formats=plot_formats,
        meta_lines=meta_lines,
        logger=logger,
    )

    # Collect ensemble-specific metadata if ENSEMBLE model present
    ensemble_metadata = collect_ensemble_metadata(
        ensemble_dirs=ensemble_dirs,
        all_models=all_models,
        pooled_test_metrics=pooled_test_metrics,
        logger=logger,
    )

    # Build and save aggregation metadata
    _ = build_agg_metadata(
        n_splits=n_splits,
        split_seeds=split_seeds,
        all_models=all_models,
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
        all_feature_reports=all_feature_reports,
        consensus_panels=consensus_panels,
        ensemble_metadata=ensemble_metadata,
        agg_dir=agg_dir,
    )
    logger.info(f"Metadata saved: {agg_dir / 'aggregation_metadata.json'}")

    log_section(logger, "Aggregation Complete")
    logger.info(f"Results saved to: {agg_dir}")

    # Build and return summary
    return build_return_summary(
        all_models=all_models,
        pooled_test_metrics=pooled_test_metrics,
        threshold_info=threshold_info,
        n_splits=n_splits,
        stable_features_df=stable_features_df,
        agg_dir=agg_dir,
    )
