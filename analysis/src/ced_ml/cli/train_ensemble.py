"""CLI for training stacking ensemble from base model outputs.

This module provides the entry point for ensemble training. It collects
OOF predictions from previously trained base models, trains a meta-learner,
and generates ensemble predictions.

Usage:
    ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost
    ced train-ensemble --config configs/training_config.yaml --split-seed 0
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ced_ml.cli.ensemble_helpers import (
    compute_ensemble_metrics,
    determine_output_directory,
    save_ensemble_artifacts,
    save_ensemble_metadata,
    validate_base_models,
    validate_probabilities,
)
from ced_ml.cli.ensemble_plotting import (
    generate_ensemble_specific_plots,
    generate_learning_curve,
    generate_oof_plots,
    generate_test_plots,
    generate_train_oof_risk_distribution,
    generate_validation_plots,
)
from ced_ml.config.loader import load_training_config
from ced_ml.data.schema import METRIC_AUROC, METRIC_PRAUC, ModelName
from ced_ml.models.stacking import (
    StackingEnsemble,
    collect_oof_predictions,
    collect_split_predictions,
    load_calibration_info_for_models,
)
from ced_ml.utils.logging import log_section, setup_command_logger

logger = logging.getLogger(__name__)


def discover_base_models_for_run(
    run_id: str | None = None,
    split_seed: int = 0,
    skip_ensemble: bool = True,
    results_dir: str | None = None,
) -> tuple[str, list[str]]:
    """Auto-detect results directory and base models from run_id.

    NOTE: This is a compatibility wrapper. New code should use functions from
    ced_ml.cli.discovery directly.

    Args:
        run_id: Run ID (e.g., "20260127_104409"). If None, auto-detects latest.
        split_seed: Split seed (default: 0)
        skip_ensemble: If True, exclude ENSEMBLE models (default: True)
        results_dir: Root results directory (optional). If None, uses project root/results.

    Returns:
        Tuple of (results_dir, base_models) where:
            - results_dir: Root results directory
            - base_models: List of base model names with OOF predictions

    Raises:
        FileNotFoundError: If no models found or results directory missing
    """
    from ced_ml.cli.discovery import (
        discover_models_for_run,
        get_results_root,
        resolve_run_id,
    )

    results_root = get_results_root(results_dir)
    resolved_run_id = resolve_run_id(run_id, results_dir)

    base_models = discover_models_for_run(
        run_id=resolved_run_id,
        results_dir=results_dir,
        skip_ensemble=skip_ensemble,
        require_oof=True,
        split_seed=split_seed,
    )

    if not base_models:
        msg = f"No base models found for run {resolved_run_id}, split {split_seed}"
        if skip_ensemble:
            msg += " (ENSEMBLE excluded)"
        msg += f"\nExpected OOF predictions at: results/run_{resolved_run_id}/{{MODEL}}/splits/split_seed{split_seed}/preds/"
        raise FileNotFoundError(msg)

    return str(results_root), base_models


def load_data_and_models(
    config_file: str | None,
    results_dir: str | None,
    base_models: list[str] | None,
    run_id: str | None,
    split_seed: int,
) -> tuple[Any, Path, list[str]]:
    """Load configuration and determine base models and results directory.

    Args:
        config_file: Path to YAML config file (optional)
        results_dir: Directory containing base model results
        base_models: List of base model names
        run_id: Run ID for auto-detection
        split_seed: Split seed for identifying model outputs

    Returns:
        Tuple of (config, results_path, base_models)

    Raises:
        ValueError: If results_dir cannot be determined
        FileNotFoundError: If results directory not found
    """
    config = None
    if config_file:
        logger.info(f"Loading config from: {config_file}")
        config = load_training_config(config_file=config_file)

    if run_id is not None:
        logger.info(f"Auto-detecting models for run_id: {run_id}")
        detected_results_dir, detected_base_models = discover_base_models_for_run(
            run_id=run_id,
            split_seed=split_seed,
            skip_ensemble=True,
            results_dir=results_dir,
        )

        if results_dir is None:
            results_dir = detected_results_dir
            logger.info(f"  Auto-detected results_dir: {results_dir}")

        if base_models is None:
            base_models = detected_base_models
            logger.info(f"  Auto-detected base models: {base_models}")

    if results_dir is None and config is not None:
        results_dir = str(config.outdir)
    if results_dir is None:
        raise ValueError("Must provide --results-dir, --run-id, or config with outdir")

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    # Auto-load saved config from run directory when no explicit config provided
    if config is None:
        saved_config_path = results_path / "training_config.yaml"
        if saved_config_path.exists():
            logger.info(f"Auto-loading config from run directory: {saved_config_path}")
            config = load_training_config(config_file=str(saved_config_path))

    if base_models is None:
        base_models = [
            ModelName.LR_EN,
            ModelName.RF,
            ModelName.XGBoost,
            ModelName.LinSVM_cal,
        ]

    logger.info(f"Results directory: {results_path}")
    logger.info(f"Base models: {base_models}")
    logger.info(f"Split seed: {split_seed}")

    return config, results_path, base_models


def train_meta_learner(
    available_models: list[str],
    results_path: Path,
    split_seed: int,
    meta_penalty: str,
    meta_c: float,
    random_state: int,
    run_id: str | None = None,
    calibrate_meta: bool = False,
    meta_calibration_method: str = "isotonic",
) -> tuple[StackingEnsemble, dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    """Train the stacking ensemble meta-learner.

    Args:
        available_models: List of available base model names
        results_path: Root results directory
        split_seed: Split seed
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        random_state: Random state for reproducibility
        run_id: Optional run_id to scope path resolution to a specific run
        calibrate_meta: Whether to wrap meta-learner in CalibratedClassifierCV.
            Defaults to False (base models are already OOF-calibrated).
        meta_calibration_method: Calibration method when calibrate_meta=True.
            'isotonic' or 'sigmoid'.

    Returns:
        Tuple of (ensemble, oof_dict, y_train, train_idx, cat_train)
    """
    log_section(logger, "Loading Calibration Info")
    calibration_info = load_calibration_info_for_models(
        results_path, available_models, split_seed, run_id
    )

    for model_name, calib_info in calibration_info.items():
        strategy = calib_info.strategy
        has_oof_calib = calib_info.oof_calibrator is not None
        logger.info(
            f"  {model_name}: strategy={strategy}, "
            f"needs_posthoc={calib_info.needs_posthoc_calibration}, "
            f"has_oof_calibrator={has_oof_calib}"
        )

    log_section(logger, "Collecting OOF Predictions")
    oof_dict, y_train, train_idx, cat_train = collect_oof_predictions(
        results_path, available_models, split_seed, run_id
    )

    logger.info(f"Training samples: {len(y_train)}")
    logger.info(f"Training prevalence: {y_train.mean():.4f}")

    log_section(logger, "Training Meta-Learner")

    ensemble = StackingEnsemble(
        base_model_names=available_models,
        meta_penalty=meta_penalty,
        meta_C=meta_c,
        calibrate_meta=calibrate_meta,
        meta_calibration_method=meta_calibration_method,
        random_state=random_state,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    coef = ensemble.get_meta_model_coef()
    if coef:
        logger.info("Meta-learner coefficients:")
        for name, value in coef.items():
            logger.info(f"  {name}: {value:.4f}")

    return ensemble, oof_dict, y_train, train_idx, cat_train, calibration_info


def generate_predictions(
    ensemble: StackingEnsemble,
    results_path: Path,
    available_models: list[str],
    split_seed: int,
    calibration_info: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Generate ensemble predictions on validation and test sets.

    Args:
        ensemble: Trained StackingEnsemble instance
        results_path: Root results directory
        available_models: List of base model names
        split_seed: Split seed
        calibration_info: Calibration info for base models
        run_id: Optional run_id to scope path resolution to a specific run

    Returns:
        Dict with predictions and metrics for val/test sets
    """
    log_section(logger, "Generating Ensemble Predictions")

    predictions = {}

    try:
        val_preds_dict, y_val, val_idx, cat_val = collect_split_predictions(
            results_path,
            available_models,
            split_seed,
            "val",
            run_id=run_id,
            calibration_info=calibration_info,
        )
        val_proba = ensemble.predict_proba_from_base_preds(val_preds_dict)[:, 1]
        validate_probabilities(val_proba, "val", logger)

        predictions["val_proba"] = val_proba
        predictions["y_val"] = y_val
        predictions["val_idx"] = val_idx
        predictions["cat_val"] = cat_val

        val_metrics = compute_ensemble_metrics(y_val, val_proba, "val")
        predictions["val_metrics"] = val_metrics

        # Safe formatting for metrics (may be None in single-class scenarios)
        auroc = val_metrics.get(METRIC_AUROC)
        prauc = val_metrics.get(METRIC_PRAUC)
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        prauc_str = f"{prauc:.4f}" if prauc is not None else "N/A"
        logger.info(f"Validation AUROC: {auroc_str}")
        logger.info(f"Validation PR-AUC: {prauc_str}")
    except FileNotFoundError as e:
        logger.warning(f"Could not load validation predictions: {e}")

    try:
        test_preds_dict, y_test, test_idx, cat_test = collect_split_predictions(
            results_path,
            available_models,
            split_seed,
            "test",
            run_id=run_id,
            calibration_info=calibration_info,
        )
        test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)[:, 1]
        validate_probabilities(test_proba, "test", logger)

        predictions["test_proba"] = test_proba
        predictions["y_test"] = y_test
        predictions["test_idx"] = test_idx
        predictions["cat_test"] = cat_test

        test_metrics = compute_ensemble_metrics(y_test, test_proba, "test")
        predictions["test_metrics"] = test_metrics

        # Safe formatting for metrics (may be None in single-class scenarios)
        test_auroc = test_metrics.get(METRIC_AUROC)
        test_prauc = test_metrics.get(METRIC_PRAUC)
        test_auroc_str = f"{test_auroc:.4f}" if test_auroc is not None else "N/A"
        test_prauc_str = f"{test_prauc:.4f}" if test_prauc is not None else "N/A"
        logger.info(f"Test AUROC: {test_auroc_str}")
        logger.info(f"Test PR-AUC: {test_prauc_str}")
    except FileNotFoundError as e:
        logger.warning(f"Could not load test predictions: {e}")

    return predictions


def generate_all_plots(
    config: Any,
    ensemble: StackingEnsemble,
    results: dict[str, Any],
    results_path: Path,
    available_models: list[str],
    outdir: Path,
    split_seed: int,
    meta_penalty: str,
    meta_c: float,
    run_id: str | None = None,
    split_index: int = 0,
) -> None:
    """Generate all ensemble plots.

    Args:
        config: Training configuration object
        ensemble: Trained StackingEnsemble instance
        results: Dict with predictions and metadata
        results_path: Root results directory
        available_models: List of base model names
        outdir: Output directory
        split_seed: Split seed
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        run_id: Optional run_id to scope path resolution to a specific run
    """
    log_section(logger, "Generating Ensemble Plots")

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = outdir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    plot_format = "png"

    max_plot_splits = getattr(config.output, "max_plot_splits", 0) if config else 0
    should_plot = max_plot_splits == 0 or split_index < max_plot_splits
    if not should_plot:
        logger.info(
            f"Skipping plots for split_seed={split_seed} (split_index={split_index}, max_plot_splits={max_plot_splits})"
        )
        return

    meta_lines_plot = [
        f"Model: ENSEMBLE (base: {', '.join(available_models)})",
        f"Split seed: {split_seed}",
        f"Meta-learner: LR(penalty={meta_penalty}, C={meta_c})",
        f"Train n={len(results.get('y_train', []))}, prevalence={results.get('y_train', np.array([0])).mean():.4f}",
    ]

    try:
        generate_validation_plots(results, plots_dir, meta_lines_plot, plot_format)
        generate_test_plots(results, plots_dir, meta_lines_plot, plot_format)
        generate_oof_plots(
            results, ensemble, plots_dir, available_models, meta_penalty, meta_c, plot_format
        )
        generate_train_oof_risk_distribution(
            results, ensemble, plots_dir, meta_lines_plot, plot_format
        )
        generate_ensemble_specific_plots(
            results,
            results_path,
            available_models,
            diagnostics_dir,
            meta_lines_plot,
            meta_penalty,
            meta_c,
            plot_format,
            run_id=run_id,
        )
    except Exception as e:
        logger.warning(f"Plot generation failed (non-fatal): {e}")

    generate_learning_curve(
        config,
        ensemble,
        results.get("oof_dict", {}),
        results.get("y_train"),
        available_models,
        outdir,
        split_seed,
        meta_penalty,
        meta_c,
        results.get("random_state", 42),
        should_plot,
        plot_format,
    )


def run_train_ensemble(
    config_file: str | None = None,
    results_dir: str | None = None,
    base_models: list[str] | None = None,
    run_id: str | None = None,
    split_seed: int = 0,
    outdir: str | None = None,
    meta_penalty: str | None = None,
    meta_c: float | None = None,
    log_level: int | None = None,
    split_index: int = 0,
) -> dict[str, Any]:
    """Run ensemble training from base model outputs.

    This function:
    1. Loads configuration (from file or defaults)
    2. Auto-detects results_dir and base_models from run_id (if provided)
    3. Collects OOF predictions from base models
    4. Trains the meta-learner
    5. Generates and saves ensemble predictions
    6. Computes and reports metrics

    Args:
        config_file: Path to YAML config file (optional)
        results_dir: Directory containing base model results (auto-detected if run_id provided)
        base_models: List of base model names (auto-detected if run_id provided)
        run_id: Run ID for auto-detection (e.g., "20260127_115115")
        split_seed: Split seed for identifying model outputs
        outdir: Output directory (overrides results_dir/ENSEMBLE/split_{seed})
        meta_penalty: Meta-learner regularization (overrides config)
        meta_c: Meta-learner regularization strength (overrides config)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)

    Returns:
        Dict with ensemble results and metrics
    """
    if log_level is None:
        log_level = logging.INFO

    _log_outdir = outdir or results_dir or "results"
    logger = setup_command_logger(
        command="train-ensemble",
        log_level=log_level,
        outdir=_log_outdir,
        run_id=run_id,
        split_seed=split_seed,
    )

    log_section(logger, "CeD-ML Ensemble Training")

    config, results_path, base_models = load_data_and_models(
        config_file, results_dir, base_models, run_id, split_seed
    )

    # Read ensemble settings from config, with CLI overrides taking precedence
    ensemble_cfg = config.ensemble if config is not None else None
    if meta_penalty is None:
        meta_penalty = ensemble_cfg.meta_penalty if ensemble_cfg is not None else "l2"
    if meta_c is None:
        meta_c = ensemble_cfg.meta_c if ensemble_cfg is not None else 1.0
    calibrate_meta = ensemble_cfg.calibrate_meta if ensemble_cfg is not None else False
    meta_calibration_method = (
        ensemble_cfg.meta_calibration_method if ensemble_cfg is not None else "isotonic"
    )

    logger.info(f"Meta-learner: LogisticRegression(penalty={meta_penalty}, C={meta_c})")

    available_models, missing_models = validate_base_models(
        results_path, base_models, split_seed, run_id
    )
    logger.info(f"Using {len(available_models)} base models: {available_models}")

    random_state = config.cv.random_state if config else 42

    ensemble, oof_dict, y_train, train_idx, cat_train, calibration_info = train_meta_learner(
        available_models,
        results_path,
        split_seed,
        meta_penalty,
        meta_c,
        random_state,
        run_id=run_id,
        calibrate_meta=calibrate_meta,
        meta_calibration_method=meta_calibration_method,
    )

    coef = ensemble.get_meta_model_coef()

    results = {
        "base_models": available_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_c,
        "meta_coef": coef,
        "random_state": random_state,
        "calibration_strategies": {name: info.strategy for name, info in calibration_info.items()},
        "y_train": y_train,
        "train_idx": train_idx,
        "cat_train": cat_train,
        "oof_dict": oof_dict,
    }

    predictions = generate_predictions(
        ensemble,
        results_path,
        available_models,
        split_seed,
        calibration_info,
        run_id=run_id,
    )
    results.update(predictions)

    log_section(logger, "Saving Ensemble Results")

    final_outdir = determine_output_directory(
        outdir, results_path, run_id, available_models, split_seed
    )
    final_outdir.mkdir(parents=True, exist_ok=True)

    save_ensemble_artifacts(
        final_outdir,
        ensemble,
        available_models,
        results,
        split_seed,
        meta_penalty,
        meta_c,
        random_state,
    )

    save_ensemble_metadata(
        final_outdir,
        available_models,
        results,
        split_seed,
        meta_penalty,
        meta_c,
        coef,
    )

    generate_all_plots(
        config,
        ensemble,
        results,
        results_path,
        available_models,
        final_outdir,
        split_seed,
        meta_penalty,
        meta_c,
        run_id=run_id,
        split_index=split_index,
    )

    log_section(logger, "Ensemble Training Complete")
    logger.info(f"All results saved to: {final_outdir}")

    return results
