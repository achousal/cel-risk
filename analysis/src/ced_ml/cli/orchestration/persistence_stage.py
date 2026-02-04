"""
Persistence stage for training orchestration.

This module handles all file writes:
- Model bundle saving
- Metrics saving (val/test)
- CV artifacts (best params, selected proteins, RFECV)
- Optuna artifacts
- CV repeat metrics
- Run settings/metadata
- Predictions (test, val, OOF, controls)
- Additional artifacts (calibration, DCA, learning curve, screening, panels, bootstrap)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from ced_ml.cli.train import build_training_pipeline
from ced_ml.data.schema import METRIC_AUROC, METRIC_PRAUC
from ced_ml.evaluation.reports import ResultsWriter
from ced_ml.features.panels import build_multi_size_panels
from ced_ml.features.stability import compute_selection_frequencies, extract_stable_panel
from ced_ml.metrics.bootstrap import stratified_bootstrap_ci
from ced_ml.metrics.dca import save_dca_results
from ced_ml.metrics.threshold_strategy import get_threshold_strategy
from ced_ml.models.calibration_strategy import get_calibration_strategy
from ced_ml.models.registry import build_models
from ced_ml.plotting.learning_curve import save_learning_curve_csv
from ced_ml.utils.logging import log_section
from ced_ml.utils.metadata import build_plot_metadata

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def save_artifacts(ctx: TrainingContext) -> TrainingContext:
    """Save all training artifacts.

    This stage saves:
    1. Model bundle
    2. Metrics (val/test)
    3. CV artifacts (best params, selected proteins, RFECV)
    4. Optuna artifacts
    5. CV repeat metrics
    6. Run settings/metadata
    7. Predictions (test, val, OOF, controls)
    8. Additional artifacts

    Args:
        ctx: TrainingContext with evaluation completed

    Returns:
        Updated TrainingContext (unchanged, for consistency)
    """
    log_section(logger, "Saving Results")

    _save_model_bundle(ctx)
    _save_metrics(ctx)
    _save_cv_artifacts(ctx)
    _save_optuna_artifacts(ctx)
    _save_cv_repeat_metrics(ctx)
    _save_run_metadata(ctx)
    _save_predictions(ctx)

    log_section(logger, "Generating Additional Artifacts")
    _save_calibration_csv(ctx)
    _save_dca_results(ctx)
    _save_learning_curve(ctx)
    _save_screening_results(ctx)
    _save_feature_reports(ctx)
    _save_bootstrap_ci(ctx)

    log_section(logger, "Training Complete")
    logger.info(f"All results saved to: {ctx.config.outdir}")

    return ctx


def _save_model_bundle(ctx: TrainingContext) -> None:
    """Save model bundle with all metadata."""
    config = ctx.config
    outdirs = ctx.outdirs

    # Determine effective calibration strategy using the strategy pattern
    calibration_strategy = get_calibration_strategy(config, model_name=config.model)

    # Determine threshold strategy using the strategy pattern
    threshold_strategy = get_threshold_strategy(config.thresholds)

    model_bundle = {
        "model": ctx.final_pipeline,
        "scenario": ctx.scenario,
        "model_name": config.model,
        "thresholds": {
            "val_threshold": ctx.val_threshold,
            "test_threshold": ctx.test_metrics["threshold"],
            "objective": config.thresholds.objective,
            "strategy_name": threshold_strategy.name,
            "fixed_spec": config.thresholds.fixed_spec,
        },
        "metadata": {
            "train_prevalence": ctx.train_prev,
            "val_prevalence": ctx.val_target_prev,
            "test_prevalence": ctx.test_target_prev,
        },
        "calibration": {
            "enabled": config.calibration.enabled,
            "method": calibration_strategy.method(),
            "strategy": calibration_strategy.name(),
            "oof_calibrator": ctx.oof_calibrator,
        },
        "resolved_columns": {
            "protein_cols": ctx.protein_cols,
            "numeric_metadata": ctx.resolved.numeric_metadata,
            "categorical_metadata": ctx.resolved.categorical_metadata,
        },
        "fixed_panel": {
            "enabled": ctx.fixed_panel_path is not None,
            "path": str(ctx.fixed_panel_path) if ctx.fixed_panel_path else None,
            "source": (
                "cli"
                if ctx.cli_args.get("fixed_panel")
                else "config" if ctx.fixed_panel_path else None
            ),
            "n_proteins": (len(ctx.fixed_panel_proteins) if ctx.fixed_panel_proteins else None),
        },
        "config": (config.model_dump() if hasattr(config, "model_dump") else config.dict()),
        "seed": ctx.seed,
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    model_filename = f"{config.model}__final_model.joblib"
    model_path = Path(outdirs.core) / model_filename
    joblib.dump(model_bundle, model_path)
    logger.info(f"Model bundle saved: {model_path}")


def _save_metrics(ctx: TrainingContext) -> None:
    """Save validation and test metrics."""
    writer = ResultsWriter(ctx.outdirs)
    config = ctx.config

    if ctx.val_metrics is not None:
        writer.save_val_metrics(ctx.val_metrics, ctx.scenario, config.model)
    writer.save_test_metrics(ctx.test_metrics, ctx.scenario, config.model)


def _save_cv_artifacts(ctx: TrainingContext) -> None:
    """Save CV artifacts (best params, selected proteins, RFECV)."""
    config = ctx.config
    outdirs = ctx.outdirs

    # Best params
    best_params_path = Path(outdirs.cv) / "best_params_per_split.csv"
    ctx.best_params_df.to_csv(best_params_path, index=False)
    logger.info(f"Best params saved: {best_params_path}")

    # Selected proteins
    selected_proteins_path = Path(outdirs.cv) / "selected_proteins_per_split.csv"
    ctx.selected_proteins_df.to_csv(selected_proteins_path, index=False)
    logger.info(f"Selected proteins saved: {selected_proteins_path}")

    # Nested RFECV results
    if ctx.nested_rfecv_result is not None:
        from ced_ml.features.nested_rfe import save_nested_rfecv_results

        rfecv_dir = Path(outdirs.cv) / "rfecv"
        save_nested_rfecv_results(
            result=ctx.nested_rfecv_result,
            output_dir=rfecv_dir,
            model_name=config.model,
            split_seed=ctx.seed,
        )
        logger.info(f"Nested RFECV results saved: {rfecv_dir}")
        logger.info(
            f"  Consensus panel: {len(ctx.nested_rfecv_result.consensus_panel)} proteins "
            f"(selected in >= {config.features.rfe_consensus_thresh:.0%} of folds)"
        )
        logger.info(
            f"  Mean optimal size: {ctx.nested_rfecv_result.mean_optimal_size:.1f} proteins"
        )
        logger.info(
            f"  Mean val AUROC: {np.mean(ctx.nested_rfecv_result.fold_val_aurocs):.4f} "
            f"(std: {np.std(ctx.nested_rfecv_result.fold_val_aurocs):.4f})"
        )

    # Final test panel
    if ctx.final_selected_proteins:
        writer = ResultsWriter(ctx.outdirs)
        panel_metadata = {
            "selection_method": config.features.feature_selection_strategy,
            "n_train": len(ctx.y_train),
            "n_train_pos": int(ctx.y_train.sum()),
            "train_prevalence": float(ctx.train_prev),
            "random_state": ctx.seed,
            "timestamp": datetime.now().isoformat(),
        }
        writer.save_final_test_panel(
            panel_proteins=ctx.final_selected_proteins,
            scenario=ctx.scenario,
            model=config.model,
            metadata=panel_metadata,
        )


def _save_optuna_artifacts(ctx: TrainingContext) -> None:
    """Save Optuna artifacts if enabled."""
    config = ctx.config
    best_params_df = ctx.best_params_df

    if not config.optuna.enabled:
        return

    cv_dir = Path(ctx.outdirs.cv)

    if "optuna_n_trials" not in best_params_df.columns:
        logger.warning(
            "[optuna] Optuna was enabled but no trial metadata found. "
            "Check if optuna is installed: pip install optuna"
        )
        return

    # Save Optuna summary
    optuna_summary = {
        "enabled": True,
        "sampler": config.optuna.sampler,
        "pruner": config.optuna.pruner,
        "n_trials_per_fold": int(config.optuna.n_trials),
        "timeout": config.optuna.timeout,
        "direction": config.optuna.direction or "maximize",
        "total_folds": len(best_params_df),
    }
    optuna_summary_path = cv_dir / "optuna_config.json"
    with open(optuna_summary_path, "w") as f:
        json.dump(optuna_summary, f, indent=2)
    logger.info(f"Optuna config saved: {optuna_summary_path}")

    # Save best params with Optuna metadata
    optuna_params_path = cv_dir / "best_params_optuna.csv"
    best_params_df.to_csv(optuna_params_path, index=False)
    logger.info(f"Optuna best params saved: {optuna_params_path}")

    # Generate Optuna plots (simplified - just try to load from storage)
    if config.output.plot_optuna:
        _generate_optuna_plots(ctx, cv_dir)


def _generate_optuna_plots(ctx: TrainingContext, cv_dir: Path) -> None:
    """Generate Optuna hyperparameter tuning plots."""
    config = ctx.config

    try:
        from ced_ml.plotting.optuna_plots import save_optuna_plots

        logger.info("Generating Optuna hyperparameter tuning plots...")

        study = None
        study_loaded = False

        # Try to load study from persistent storage
        if config.optuna.storage and config.optuna.study_name:
            try:
                import optuna

                logger.info(
                    f"Loading existing Optuna study from storage: {config.optuna.study_name}"
                )
                study = optuna.load_study(
                    study_name=config.optuna.study_name,
                    storage=config.optuna.storage,
                )
                study_loaded = True
                logger.info(
                    f"Successfully loaded study with {len(study.trials)} trials "
                    "(reusing from CV, no refitting needed)"
                )
            except Exception as e:
                logger.warning(f"Could not load study from storage: {e}")

        # Fallback: refit if study not available
        if not study_loaded:
            study = _refit_for_optuna_plots(ctx)

        # Generate plots if we have a study
        if study is not None:
            optuna_fmt = getattr(config.output, "optuna_plot_format", config.output.plot_format)
            save_optuna_plots(
                study=study,
                out_dir=cv_dir,
                prefix=f"{config.model}__",
                plot_format=optuna_fmt,
                fallback_to_html=True,
            )
            logger.info(f"Optuna plots saved to: {cv_dir}")
        else:
            logger.warning("No Optuna study available for plotting")

    except Exception as e:
        logger.warning(f"Failed to generate Optuna plots: {e}")


def _refit_for_optuna_plots(ctx: TrainingContext) -> Any:
    """Refit model to get Optuna study for plots."""
    config = ctx.config
    seed = ctx.seed

    logger.info(
        "No persistent study storage configured, refitting for plots "
        "(consider setting optuna.storage and optuna.study_name to avoid this)"
    )

    try:
        from ced_ml.data.schema import ModelName
        from ced_ml.models.training import _build_hyperparameter_search

        classifier = build_models(config.model, config, seed, config.n_jobs)
        optuna_pipeline = build_training_pipeline(
            config,
            classifier,
            ctx.protein_cols,
            ctx.resolved.categorical_metadata,
            model_name=config.model,
        )

        xgb_spw = None
        if config.model == ModelName.XGBoost:
            from ced_ml.models.registry import compute_scale_pos_weight_from_y

            spw_grid = getattr(config.xgboost, "scale_pos_weight_grid", None)
            if spw_grid and len(spw_grid) == 1 and spw_grid[0] > 0:
                xgb_spw = float(spw_grid[0])
            else:
                xgb_spw = compute_scale_pos_weight_from_y(ctx.y_train)

        optuna_search = _build_hyperparameter_search(
            optuna_pipeline, config.model, config, seed, xgb_spw, grid_rng=None
        )

        if optuna_search is not None:
            optuna_search.fit(ctx.X_train, ctx.y_train)
            if hasattr(optuna_search, "study_") and optuna_search.study_ is not None:
                return optuna_search.study_

    except Exception as e:
        logger.warning(f"Failed to refit for Optuna plots: {e}")

    return None


def _save_cv_repeat_metrics(ctx: TrainingContext) -> None:
    """Save per-repeat OOF metrics."""
    config = ctx.config

    cv_repeat_rows = []
    for repeat in range(ctx.oof_preds.shape[0]):
        repeat_preds = ctx.oof_preds[repeat, :]
        valid_mask = ~np.isnan(repeat_preds)
        if valid_mask.sum() > 0:
            y_repeat = ctx.y_train[valid_mask]
            p_repeat = repeat_preds[valid_mask]
            auroc = roc_auc_score(y_repeat, p_repeat) if len(np.unique(y_repeat)) > 1 else np.nan
            prauc = (
                average_precision_score(y_repeat, p_repeat)
                if len(np.unique(y_repeat)) > 1
                else np.nan
            )
            brier = float(np.mean((y_repeat - p_repeat) ** 2))
            cv_repeat_rows.append(
                {
                    "scenario": ctx.scenario,
                    "model": config.model,
                    "repeat": repeat,
                    "folds": config.cv.folds,
                    "repeats": config.cv.repeats,
                    "n_train": len(ctx.y_train),
                    "n_train_pos": int(ctx.y_train.sum()),
                    "AUROC_oof": auroc,
                    "PR_AUC_oof": prauc,
                    "Brier_oof": brier,
                    "cv_seconds": ctx.cv_elapsed_sec,
                    "feature_selection_strategy": config.features.feature_selection_strategy,
                    "random_state": ctx.seed,
                }
            )

    if cv_repeat_rows:
        writer = ResultsWriter(ctx.outdirs)
        writer.save_cv_repeat_metrics(cv_repeat_rows, ctx.scenario, config.model)


def _save_run_metadata(ctx: TrainingContext) -> None:
    """Save run settings and metadata."""
    config = ctx.config
    outdirs = ctx.outdirs

    # Run settings
    run_settings = {
        "model": config.model,
        "scenario": ctx.scenario,
        "seed": ctx.seed,
        "train_prevalence": float(ctx.train_prev),
        "target_prevalence": float(ctx.test_target_prev),
        "n_train": len(ctx.train_idx),
        "n_val": len(ctx.val_idx),
        "n_test": len(ctx.test_idx),
        "cv_elapsed_sec": ctx.cv_elapsed_sec,
        "columns": {
            "mode": config.columns.mode,
            "n_proteins": len(ctx.resolved.protein_cols),
            "numeric_metadata": ctx.resolved.numeric_metadata,
            "categorical_metadata": ctx.resolved.categorical_metadata,
        },
    }
    run_settings_path = Path(outdirs.core) / "run_settings.json"
    with open(run_settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)
    logger.info(f"Run settings saved: {run_settings_path}")

    # Config metadata
    config_metadata = {
        "pipeline_version": "ced_ml_v2",
        "scenario": ctx.scenario,
        "model": config.model,
        "folds": config.cv.folds,
        "repeats": config.cv.repeats,
        "val_size": getattr(config, "val_size", 0.25),
        "test_size": getattr(config, "test_size", 0.25),
        "random_state": ctx.seed,
        "scoring": config.cv.scoring,
        "inner_folds": getattr(config.cv, "inner_folds", 5),
        "n_iter": getattr(config.cv, "n_iter", 50),
        "feature_selection_strategy": config.features.feature_selection_strategy,
        "kbest_max": getattr(config.features, "kbest_max", 500),
        "screen_method": getattr(config.features, "screen_method", "none"),
        "screen_top_n": getattr(config.features, "screen_top_n", 1000),
        "calibrate_final_models": int(getattr(config.calibration, "enabled", False)),
        "threshold_source": getattr(config.thresholds, "threshold_source", "val"),
        "target_prevalence_source": getattr(config.thresholds, "target_prevalence_source", "train"),
        "n_train": len(ctx.train_idx),
        "n_val": len(ctx.val_idx),
        "n_test": len(ctx.test_idx),
        "train_prevalence": float(ctx.train_prev),
        "target_prevalence": float(ctx.test_target_prev),
        "cv_elapsed_sec": ctx.cv_elapsed_sec,
        "n_proteins": len(ctx.resolved.protein_cols),
        "bootstrap_seed": ctx.seed,
        "timestamp": datetime.now().isoformat(),
    }
    config_metadata_path = Path(outdirs.root) / "config_metadata.json"
    with open(config_metadata_path, "w") as f:
        json.dump(config_metadata, f, indent=2, sort_keys=True)
    logger.info(f"Config metadata saved: {config_metadata_path}")

    # Shared run_metadata.json
    _save_shared_run_metadata(ctx)


def _save_shared_run_metadata(ctx: TrainingContext) -> None:
    """Save shared run metadata (HPC-safe read-merge-write)."""
    config = ctx.config
    run_level_dir = ctx.get_run_level_dir()

    # Infer seed_start and n_splits from available split files
    seed_start = None
    n_splits = None
    if config.split_dir:
        split_dir_path = Path(config.split_dir)
        if split_dir_path.exists():
            meta_pattern = f"split_meta_{ctx.scenario}_seed*.json"
            meta_files = list(split_dir_path.glob(meta_pattern))
            if not meta_files:
                meta_files = list(split_dir_path.glob("split_meta_seed*.json"))

            if meta_files:
                seeds = []
                for meta_file in meta_files:
                    seed_match = meta_file.stem.split("seed")[-1]
                    try:
                        seeds.append(int(seed_match))
                    except ValueError:
                        continue

                if seeds:
                    seed_start = min(seeds)
                    n_splits = len(seeds)
                    logger.info(
                        f"Inferred split config: seed_start={seed_start}, n_splits={n_splits}"
                    )

    # Read-merge-write for HPC safety
    run_metadata_path = run_level_dir / "run_metadata.json"
    run_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    existing_metadata: dict = {}
    if run_metadata_path.exists():
        try:
            with open(run_metadata_path) as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_metadata = {}

    model_entry = {
        "scenario": ctx.scenario,
        "infile": str(config.infile),
        "split_dir": str(config.split_dir),
        "split_seed": ctx.seed,
        "seed_start": seed_start,
        "n_splits": n_splits,
        "timestamp": datetime.now().isoformat(),
    }

    existing_metadata.setdefault("run_id", ctx.run_id)
    existing_metadata.setdefault("infile", str(config.infile))
    existing_metadata.setdefault("split_dir", str(config.split_dir))
    existing_metadata.setdefault("models", {})
    existing_metadata["models"][config.model] = model_entry

    # Atomic write
    tmp_path = run_metadata_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)
    tmp_path.rename(run_metadata_path)
    logger.info(f"Run metadata saved: {run_metadata_path}")


def _save_predictions(ctx: TrainingContext) -> None:
    """Save all prediction files."""
    config = ctx.config
    outdirs = ctx.outdirs

    # Test predictions
    test_preds_path = Path(outdirs.preds_test) / f"test_preds__{config.model}.csv"
    ctx.test_preds_df.to_csv(test_preds_path, index=False)
    logger.info(f"Test predictions saved: {test_preds_path}")

    # Val predictions
    if ctx.has_validation_set:
        val_preds_path = Path(outdirs.preds_val) / f"val_preds__{config.model}.csv"
        ctx.val_preds_df.to_csv(val_preds_path, index=False)
        logger.info(f"Val predictions saved: {val_preds_path}")
    else:
        logger.warning("Skipping validation predictions (no validation set)")

    # OOF predictions
    oof_preds_path = Path(outdirs.preds_train_oof) / f"train_oof__{config.model}.csv"
    ctx.oof_preds_df.to_csv(oof_preds_path, index=False)
    logger.info(f"OOF predictions saved: {oof_preds_path}")

    # Controls OOF predictions
    controls_mask = ctx.y_train == 0
    if controls_mask.sum() > 0:
        controls_idx = ctx.train_idx[controls_mask]
        controls_oof_mean = ctx.oof_preds[:, controls_mask].mean(axis=0)
        controls_oof_df = pd.DataFrame(
            {
                "idx": controls_idx,
                "y_true": ctx.y_train[controls_mask],
                "y_prob_oof_mean": controls_oof_mean,
            }
        )
        controls_oof_path = (
            Path(outdirs.preds_controls) / f"controls_risk__{config.model}__oof_mean.csv"
        )
        controls_oof_df.to_csv(controls_oof_path, index=False)
        logger.info(f"Controls OOF predictions saved: {controls_oof_path}")


def _save_calibration_csv(ctx: TrainingContext) -> None:
    """Save calibration data as CSV."""
    config = ctx.config
    outdirs = ctx.outdirs

    try:
        # Test set calibration data
        test_y_prob = ctx.test_preds_df["y_prob"].values
        prob_true_test, prob_pred_test = calibration_curve(
            ctx.y_test, test_y_prob, n_bins=config.output.calib_bins, strategy="uniform"
        )
        calib_df_test = pd.DataFrame(
            {
                "bin_center": prob_pred_test,
                "observed_freq": prob_true_test,
                "split": "test",
                "scenario": ctx.scenario,
                "model": config.model,
            }
        )

        # Val set calibration data
        val_y_prob = ctx.val_preds_df["y_prob"].values if ctx.has_validation_set else []
        if len(val_y_prob) > 0:
            prob_true_val, prob_pred_val = calibration_curve(
                ctx.y_val,
                val_y_prob,
                n_bins=config.output.calib_bins,
                strategy="uniform",
            )
            calib_df_val = pd.DataFrame(
                {
                    "bin_center": prob_pred_val,
                    "observed_freq": prob_true_val,
                    "split": "val",
                    "scenario": ctx.scenario,
                    "model": config.model,
                }
            )
            calib_df = pd.concat([calib_df_test, calib_df_val], ignore_index=True)
        else:
            calib_df = calib_df_test

        calib_csv_path = Path(outdirs.diag_calibration) / f"{config.model}__calibration.csv"
        calib_df.to_csv(calib_csv_path, index=False)
        logger.info(f"Calibration data saved: {calib_csv_path}")

    except Exception as e:
        logger.warning(f"Failed to save calibration CSV: {e}")


def _save_dca_results(ctx: TrainingContext) -> None:
    """Save DCA results."""
    config = ctx.config
    outdirs = ctx.outdirs

    try:
        dca_summary = save_dca_results(
            y_true=ctx.y_test,
            y_pred_prob=ctx.test_preds_df["y_prob"].values,
            out_dir=str(outdirs.diag_dca),
            prefix=f"{config.model}__test__",
            thresholds=None,
            report_points=None,
            prevalence_adjustment=ctx.test_target_prev,
        )
        logger.info(f"DCA results saved: {dca_summary.get('dca_csv_path', 'N/A')}")

        # Validation set DCA
        if ctx.has_validation_set:
            dca_summary_val = save_dca_results(
                y_true=ctx.y_val,
                y_pred_prob=ctx.val_preds_df["y_prob"].values,
                out_dir=str(outdirs.diag_dca),
                prefix=f"{config.model}__val__",
                thresholds=None,
                report_points=None,
                prevalence_adjustment=ctx.test_target_prev,
            )
            logger.info(f"DCA (val) results saved: {dca_summary_val.get('dca_csv_path', 'N/A')}")

    except Exception as e:
        logger.warning(f"Failed to save DCA results: {e}")


def _save_learning_curve(ctx: TrainingContext) -> None:
    """Save learning curve data and plot."""
    config = ctx.config
    outdirs = ctx.outdirs
    seed = ctx.seed

    try:
        lc_enabled = getattr(config.evaluation, "learning_curve", False)
        plot_lc = getattr(config.output, "plot_learning_curve", True)

        if not (lc_enabled and plot_lc):
            return

        should_plot = config.output.save_plots and (
            config.output.max_plot_splits == 0 or seed < config.output.max_plot_splits
        )

        lc_csv_path = Path(outdirs.diag_learning) / f"{config.model}__learning_curve.csv"
        lc_plot_path = (
            Path(outdirs.plots) / f"{config.model}__learning_curve.{config.output.plot_format}"
            if should_plot
            else None
        )

        lc_meta = build_plot_metadata(
            model=config.model,
            scenario=ctx.scenario,
            seed=seed,
            train_prev=ctx.train_prev,
            cv_folds=min(config.cv.folds, 5),
            cv_repeats=1,
            cv_scoring=config.cv.scoring,
            n_features=(len(ctx.final_selected_proteins) if ctx.final_selected_proteins else None),
            feature_method=config.features.feature_selection_strategy,
            n_train=len(ctx.y_train),
            n_train_pos=int(ctx.y_train.sum()),
            n_train_controls=ctx.train_breakdown.get("controls"),
            n_train_incident=ctx.train_breakdown.get("incident"),
            n_train_prevalent=ctx.train_breakdown.get("prevalent"),
            split_mode="development",
        )

        # Precompute screening
        lc_precomputed_screen = None
        screen_method = getattr(config.features, "screen_method", "none")
        screen_top_n = getattr(config.features, "screen_top_n", 0)
        if screen_method and screen_method != "none" and screen_top_n > 0:
            from ced_ml.features.screening import screen_proteins

            lc_precomputed_screen, _, _ = screen_proteins(
                X_train=ctx.X_train,
                y_train=ctx.y_train,
                protein_cols=ctx.protein_cols,
                method=screen_method,
                top_n=screen_top_n,
            )
            logger.info(
                f"Precomputed screening for learning curve: "
                f"{len(lc_precomputed_screen)} proteins"
            )

        # Build fresh pipeline
        classifier = build_models(config.model, config, seed, config.n_jobs)
        lc_pipeline = build_training_pipeline(
            config,
            classifier,
            ctx.protein_cols,
            ctx.resolved.categorical_metadata,
            model_name=config.model,
        )

        # Inject precomputed screening
        if lc_precomputed_screen is not None and "screen" in dict(lc_pipeline.steps):
            lc_pipeline.named_steps["screen"].precomputed_features = lc_precomputed_screen

        save_learning_curve_csv(
            estimator=lc_pipeline,
            X=ctx.X_train,
            y=ctx.y_train,
            out_csv=lc_csv_path,
            scoring=config.cv.scoring,
            cv=min(config.cv.folds, 5),
            min_frac=0.3,
            n_points=5,
            seed=seed,
            out_plot=lc_plot_path,
            meta_lines=lc_meta,
        )
        logger.info(f"Learning curve saved: {lc_csv_path}")

    except Exception as e:
        logger.warning(f"Failed to save learning curve: {e}")


def _save_screening_results(ctx: TrainingContext) -> None:
    """Save screening results."""
    config = ctx.config
    outdirs = ctx.outdirs

    try:
        screen_method = getattr(config.features, "screen_method", "none")
        if not screen_method or screen_method == "none":
            return

        from ced_ml.features.screening import screen_proteins

        _, screening_stats, _ = screen_proteins(
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            protein_cols=ctx.protein_cols,
            method=screen_method,
            top_n=0,
        )

        if screening_stats.empty:
            return

        screening_stats_export = screening_stats.copy()
        screening_stats_export["scenario"] = ctx.scenario
        screening_stats_export["model"] = config.model
        screening_path = Path(outdirs.diag_screening) / f"{config.model}__screening_results.csv"
        screening_stats_export.to_csv(screening_path, index=False)
        logger.info(f"Screening results saved: {screening_path}")

        # Store for feature reports
        ctx._screening_stats = screening_stats

    except Exception as e:
        logger.warning(f"Failed to save screening results: {e}")


def _save_feature_reports(ctx: TrainingContext) -> None:
    """Save feature reports and stable panels."""
    config = ctx.config

    try:
        # Compute selection frequencies
        selection_freq = compute_selection_frequencies(
            ctx.selected_proteins_df,
            selection_col="selected_proteins",
        )

        if not selection_freq:
            return

        # Build feature report
        feature_report = pd.DataFrame(
            [{"protein": p, "selection_freq": f} for p, f in selection_freq.items()]
        )

        # Merge with screening statistics
        screening_stats = getattr(ctx, "_screening_stats", pd.DataFrame())
        if not screening_stats.empty:
            feature_report = feature_report.merge(
                screening_stats[["protein", "effect_size", "p_value"]],
                on="protein",
                how="left",
            )

        # Add NaN columns if not present
        if "effect_size" not in feature_report.columns:
            feature_report["effect_size"] = np.nan
        if "p_value" not in feature_report.columns:
            feature_report["p_value"] = np.nan

        # Sort and add rank
        feature_report = feature_report.sort_values("selection_freq", ascending=False).reset_index(
            drop=True
        )
        feature_report["rank"] = range(1, len(feature_report) + 1)
        feature_report["scenario"] = ctx.scenario
        feature_report["model"] = config.model

        col_order = [
            "rank",
            "protein",
            "selection_freq",
            "effect_size",
            "p_value",
            "scenario",
            "model",
        ]
        feature_report = feature_report[col_order]

        writer = ResultsWriter(ctx.outdirs)
        writer.save_feature_report(feature_report, config.model)

        # Stable panel extraction
        stable_panel_df, stable_proteins, _ = extract_stable_panel(
            selection_log=ctx.selected_proteins_df,
            n_repeats=config.cv.repeats,
            stability_threshold=0.75,
            selection_col="selected_proteins",
            fallback_top_n=20,
        )
        if not stable_panel_df.empty:
            stable_panel_df["scenario"] = ctx.scenario
            writer.save_stable_panel_report(stable_panel_df, panel_type="KBest")

        # Panel manifests
        _save_panel_manifests(ctx, selection_freq, writer)

    except Exception as e:
        logger.warning(f"Failed to save feature reports/panels: {e}")


def _save_panel_manifests(
    ctx: TrainingContext,
    selection_freq: dict,
    writer: ResultsWriter,
) -> None:
    """Save panel manifests for multiple sizes."""
    config = ctx.config

    panels_config = getattr(config, "panels", None)
    panel_sizes = (
        getattr(panels_config, "panel_sizes", [10, 25, 50]) if panels_config else [10, 25, 50]
    )

    if not panel_sizes or len(selection_freq) < min(panel_sizes):
        return

    corr_threshold = getattr(panels_config, "panel_corr_thresh", 0.80) if panels_config else 0.80
    corr_method = (
        getattr(panels_config, "panel_corr_method", "spearman") if panels_config else "spearman"
    )

    panels = build_multi_size_panels(
        df=ctx.X_train,
        y=ctx.y_train,
        selection_freq=selection_freq,
        panel_sizes=panel_sizes,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        pool_limit=1000,
    )

    for size, (_comp_map, panel_proteins) in panels.items():
        manifest = {
            "scenario": ctx.scenario,
            "model": config.model,
            "panel_size": size,
            "actual_size": len(panel_proteins),
            "corr_threshold": corr_threshold,
            "proteins": panel_proteins,
        }
        writer.save_panel_manifest(manifest, config.model, size)


def _save_bootstrap_ci(ctx: TrainingContext) -> None:
    """Save bootstrap confidence intervals for small test sets."""
    config = ctx.config
    outdirs = ctx.outdirs
    seed = ctx.seed

    try:
        min_bootstrap_threshold = getattr(config.evaluation, "bootstrap_min_samples", 100)
        if len(ctx.y_test) >= min_bootstrap_threshold:
            return

        logger.info(f"Test set small ({len(ctx.y_test)} samples) - computing bootstrap CI")

        # Bootstrap CI for AUROC
        auroc_lo, auroc_hi = stratified_bootstrap_ci(
            y_true=ctx.y_test,
            y_pred=ctx.test_preds_df["y_prob"].values,
            metric_fn=roc_auc_score,
            n_boot=1000,
            seed=seed,
        )

        # Bootstrap CI for PR-AUC
        prauc_lo, prauc_hi = stratified_bootstrap_ci(
            y_true=ctx.y_test,
            y_pred=ctx.test_preds_df["y_prob"].values,
            metric_fn=average_precision_score,
            n_boot=1000,
            seed=seed,
        )

        bootstrap_ci_df = pd.DataFrame(
            [
                {
                    "scenario": ctx.scenario,
                    "model": config.model,
                    "n_test": len(ctx.y_test),
                    "n_boot": 1000,
                    "bootstrap_seed": seed,
                    METRIC_AUROC: ctx.test_metrics[METRIC_AUROC],
                    "AUROC_ci_lo": auroc_lo,
                    "AUROC_ci_hi": auroc_hi,
                    METRIC_PRAUC: ctx.test_metrics[METRIC_PRAUC],
                    "PR_AUC_ci_lo": prauc_lo,
                    "PR_AUC_ci_hi": prauc_hi,
                }
            ]
        )

        bootstrap_ci_path = Path(outdirs.diag_test_ci) / f"{config.model}__test_bootstrap_ci.csv"
        bootstrap_ci_df.to_csv(bootstrap_ci_path, index=False)
        logger.info(
            f"Bootstrap CI saved: {bootstrap_ci_path} "
            f"(AUROC: {auroc_lo:.3f}-{auroc_hi:.3f}, "
            f"PR-AUC: {prauc_lo:.3f}-{prauc_hi:.3f})"
        )

    except Exception as e:
        logger.warning(f"Failed to compute bootstrap CI: {e}")
