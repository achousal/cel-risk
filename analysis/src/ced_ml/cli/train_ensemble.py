"""CLI for training stacking ensemble from base model outputs.

This module provides the entry point for ensemble training. It collects
OOF predictions from previously trained base models, trains a meta-learner,
and generates ensemble predictions.

Usage:
    ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost
    ced train-ensemble --config configs/training_config.yaml --split-seed 0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ced_ml.config.loader import load_training_config
from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER, METRIC_PRAUC, ModelName
from ced_ml.models.stacking import (
    StackingEnsemble,
    _find_model_split_dir,
    collect_oof_predictions,
    collect_split_predictions,
    load_calibration_info_for_models,
)
from ced_ml.plotting.learning_curve import save_learning_curve_csv
from ced_ml.utils.logging import auto_log_path, log_section, setup_logger

logger = logging.getLogger(__name__)


def compute_ensemble_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "test",
) -> dict[str, float]:
    """Compute standard metrics for ensemble predictions.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        split_name: Name of the split (for logging)

    Returns:
        Dict with AUROC, PR_AUC, Brier score
    """
    metrics = {}

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Only one class present in {split_name} set, metrics may be undefined")
        return metrics

    metrics[METRIC_AUROC] = float(roc_auc_score(y_true, y_prob))
    metrics[METRIC_PRAUC] = float(average_precision_score(y_true, y_prob))
    metrics[METRIC_BRIER] = float(brier_score_loss(y_true, y_prob))
    metrics["n_samples"] = int(len(y_true))
    metrics["n_pos"] = int(y_true.sum())
    metrics["prevalence"] = float(y_true.mean())

    return metrics


def validate_probabilities(
    y_prob: np.ndarray,
    split_name: str,
    logger: logging.Logger,
) -> None:
    """Validate predicted probabilities are in [0, 1] and contain no NaN/Inf.

    Args:
        y_prob: Array of predicted probabilities
        split_name: Name of the split (for error messages)
        logger: Logger instance

    Raises:
        ValueError: If probabilities are invalid (NaN, Inf, or out of bounds)
    """
    if not isinstance(y_prob, np.ndarray):
        y_prob = np.asarray(y_prob)

    # Check for NaN
    n_nan = np.isnan(y_prob).sum()
    if n_nan > 0:
        raise ValueError(
            f"Ensemble predictions on {split_name} contain {n_nan} NaN values. "
            "This indicates a meta-learner or calibration error."
        )

    # Check for Inf
    n_inf = np.isinf(y_prob).sum()
    if n_inf > 0:
        raise ValueError(
            f"Ensemble predictions on {split_name} contain {n_inf} Inf values. "
            "This indicates a meta-learner or calibration error."
        )

    # Check bounds [0, 1]
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            f"Ensemble predictions on {split_name} are out of bounds [0, 1]. "
            f"min={y_prob.min():.6f}, max={y_prob.max():.6f}. "
            "This indicates a calibration error."
        )

    logger.debug(
        f"Ensemble {split_name} probabilities validated: n={len(y_prob)}, "
        f"min={y_prob.min():.4f}, max={y_prob.max():.4f}, mean={y_prob.mean():.4f}"
    )


def _collect_base_model_test_metrics(
    results_path: Path,
    base_models: list[str],
    split_seed: int,
) -> dict[str, dict[str, float]]:
    """Load test metrics from base model split directories for comparison.

    Reads metrics.json (or test_metrics.csv) from each base model's split
    directory to build a comparison dict for the model comparison chart.

    Args:
        results_path: Root results directory.
        base_models: List of base model names.
        split_seed: Split seed for path resolution.

    Returns:
        Dict mapping model name to dict with AUROC, PR_AUC, Brier keys.
    """
    collected: dict[str, dict[str, float]] = {}

    for model in base_models:
        try:
            model_dir = _find_model_split_dir(results_path, model, split_seed)

            # Try JSON metrics first (consistent with ensemble format)
            metrics_path = model_dir / "core" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)

                # metrics.json may have test metrics nested or flat
                test_data = data.get("test", data)
                entry = {}
                for key in (METRIC_AUROC, METRIC_PRAUC, METRIC_BRIER):
                    if key in test_data:
                        entry[key] = float(test_data[key])
                if entry:
                    collected[model] = entry
                    continue

            # Fallback: try test_metrics.csv
            csv_path = model_dir / "core" / "test_metrics.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    row = df.iloc[0]
                    entry = {}
                    for key in (METRIC_AUROC, METRIC_PRAUC, METRIC_BRIER):
                        if key in row.index:
                            entry[key] = float(row[key])
                    if entry:
                        collected[model] = entry

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Could not load metrics for {model}: {e}")

    return collected


def discover_base_models_for_run(
    run_id: str | None = None,
    split_seed: int = 0,
    skip_ensemble: bool = True,
    results_dir: str | None = None,
) -> tuple[str, list[str]]:
    """Auto-detect results directory and base models from run_id.

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
    from pathlib import Path

    from ced_ml.utils.paths import get_project_root

    # Determine results directory
    if results_dir is None:
        # Default: project root / results
        results_dir_path = get_project_root() / "results"
    else:
        results_dir_path = Path(results_dir)

    if not results_dir_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir_path}")

    # Auto-detect run_id: scan results/run_*/
    if not run_id:
        run_ids = [d.name.replace("run_", "") for d in results_dir_path.glob("run_*") if d.is_dir()]
        if not run_ids:
            raise FileNotFoundError("No runs found in results directory")
        run_ids.sort(reverse=True)
        run_id = run_ids[0]

    # New layout: results/run_{id}/{MODEL}/splits/split_seed{N}/preds/
    run_path = results_dir_path / f"run_{run_id}"
    base_models = []

    if run_path.exists():
        for model_dir in sorted(run_path.glob("*/")):
            model_name = model_dir.name

            if model_name.startswith(".") or model_name in ("investigations", "consensus"):
                continue
            if skip_ensemble and model_name == "ENSEMBLE":
                continue

            split_path = model_dir / "splits" / f"split_seed{split_seed}"
            if not split_path.exists():
                continue

            oof_path = split_path / "preds" / f"train_oof__{model_name}.csv"
            if oof_path.exists():
                base_models.append(model_name)

    if not base_models:
        msg = f"No base models found for run {run_id}, split {split_seed}"
        if skip_ensemble:
            msg += " (ENSEMBLE excluded)"
        msg += f"\nExpected OOF predictions at: results/run_{run_id}/{{MODEL}}/splits/split_seed{split_seed}/preds/"
        raise FileNotFoundError(msg)

    return str(results_dir_path), base_models


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
    # Setup logger with auto-file-logging
    if log_level is None:
        log_level = logging.INFO

    # Determine outdir for log path resolution
    _log_outdir = outdir or results_dir or "results"
    log_file = auto_log_path(
        command="train-ensemble",
        outdir=_log_outdir,
        run_id=run_id,
        split_seed=split_seed,
    )
    logger = setup_logger("ced_ml", level=log_level, log_file=log_file)
    logger.info(f"Logging to file: {log_file}")

    log_section(logger, "CeD-ML Ensemble Training")

    # Load config if provided
    config = None
    if config_file:
        logger.info(f"Loading config from: {config_file}")
        config = load_training_config(config_file=config_file)

    # Auto-detect from run_id if provided
    if run_id is not None:
        logger.info(f"Auto-detecting models for run_id: {run_id}")
        detected_results_dir, detected_base_models = discover_base_models_for_run(
            run_id=run_id,
            split_seed=split_seed,
            skip_ensemble=True,
            results_dir=results_dir,
        )

        # Use detected values if not explicitly provided
        if results_dir is None:
            results_dir = detected_results_dir
            logger.info(f"  Auto-detected results_dir: {results_dir}")

        if base_models is None:
            base_models = detected_base_models
            logger.info(f"  Auto-detected base models: {base_models}")

    # Determine results directory
    if results_dir is None and config is not None:
        results_dir = str(config.outdir)
    if results_dir is None:
        raise ValueError("Must provide --results-dir, --run-id, or config with outdir")

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    # Determine base models
    if base_models is None:
        # Default base models (config.ensemble was removed in recent refactor)
        base_models = [ModelName.LR_EN, ModelName.RF, ModelName.XGBoost, ModelName.LinSVM_cal]

    logger.info(f"Results directory: {results_path}")
    logger.info(f"Base models: {base_models}")
    logger.info(f"Split seed: {split_seed}")

    # Determine meta-learner hyperparameters
    # Note: config.ensemble was removed in recent refactor - use CLI args or defaults
    if meta_penalty is None:
        meta_penalty = "l2"
    if meta_c is None:
        meta_c = 1.0

    logger.info(f"Meta-learner: LogisticRegression(penalty={meta_penalty}, C={meta_c})")

    # Check which base models have results (using flexible path discovery)
    available_models = []
    missing_models = []
    for model in base_models:
        try:
            # Use _find_model_split_dir for flexible path resolution (H1 fix)
            # This handles both legacy (split_{seed}) and new (run_{id}/split_seed{seed}) layouts
            model_dir = _find_model_split_dir(results_path, model, split_seed)

            # Flat preds directory structure
            oof_path = model_dir / "preds" / f"train_oof__{model}.csv"
            if oof_path.exists():
                available_models.append(model)
            else:
                missing_models.append(model)
        except FileNotFoundError:
            missing_models.append(model)

    if missing_models:
        logger.warning(f"Missing OOF predictions for: {missing_models}")

    if len(available_models) < 2:
        error_msg = (
            f"Cannot train ensemble for split_seed {split_seed}: "
            f"need at least 2 base models with OOF predictions. "
            f"Available: {available_models}, missing: {missing_models}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Using {len(available_models)} base models: {available_models}")

    # Load calibration info for base models
    log_section(logger, "Loading Calibration Info")
    calibration_info = load_calibration_info_for_models(results_path, available_models, split_seed)

    # Log calibration strategies
    for model_name, calib_info in calibration_info.items():
        strategy = calib_info.strategy
        has_oof_calib = calib_info.oof_calibrator is not None
        logger.info(
            f"  {model_name}: strategy={strategy}, "
            f"needs_posthoc={calib_info.needs_posthoc_calibration}, "
            f"has_oof_calibrator={has_oof_calib}"
        )

    # Collect OOF predictions
    log_section(logger, "Collecting OOF Predictions")
    oof_dict, y_train, train_idx, cat_train = collect_oof_predictions(
        results_path, available_models, split_seed
    )

    logger.info(f"Training samples: {len(y_train)}")
    logger.info(f"Training prevalence: {y_train.mean():.4f}")

    # Train ensemble
    log_section(logger, "Training Meta-Learner")

    random_state = config.cv.random_state if config else 42
    calibrate_meta = True  # Always calibrate for probability estimates

    ensemble = StackingEnsemble(
        base_model_names=available_models,
        meta_penalty=meta_penalty,
        meta_C=meta_c,
        calibrate_meta=calibrate_meta,
        random_state=random_state,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    # Log meta-model coefficients
    coef = ensemble.get_meta_model_coef()
    if coef:
        logger.info("Meta-learner coefficients:")
        for name, value in coef.items():
            logger.info(f"  {name}: {value:.4f}")

    # Generate predictions on val/test sets
    log_section(logger, "Generating Ensemble Predictions")

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
    }

    # Validation set (apply calibration to base model predictions)
    try:
        val_preds_dict, y_val, val_idx, cat_val = collect_split_predictions(
            results_path,
            available_models,
            split_seed,
            "val",
            calibration_info=calibration_info,
        )
        val_proba = ensemble.predict_proba_from_base_preds(val_preds_dict)[:, 1]
        # Validate ensemble val predictions (H4 fix)
        validate_probabilities(val_proba, "val", logger)
        results["val_proba"] = val_proba
        results["y_val"] = y_val
        results["val_idx"] = val_idx
        results["cat_val"] = cat_val

        val_metrics = compute_ensemble_metrics(y_val, val_proba, "val")
        results["val_metrics"] = val_metrics
        logger.info(f"Validation AUROC: {val_metrics.get(METRIC_AUROC, 'N/A'):.4f}")
        logger.info(f"Validation PR-AUC: {val_metrics.get(METRIC_PRAUC, 'N/A'):.4f}")
    except FileNotFoundError as e:
        logger.warning(f"Could not load validation predictions: {e}")

    # Test set (apply calibration to base model predictions)
    try:
        test_preds_dict, y_test, test_idx, cat_test = collect_split_predictions(
            results_path,
            available_models,
            split_seed,
            "test",
            calibration_info=calibration_info,
        )
        test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)[:, 1]
        # Validate ensemble test predictions (H4 fix)
        validate_probabilities(test_proba, "test", logger)
        results["test_proba"] = test_proba
        results["y_test"] = y_test
        results["test_idx"] = test_idx
        results["cat_test"] = cat_test

        test_metrics = compute_ensemble_metrics(y_test, test_proba, "test")
        results["test_metrics"] = test_metrics
        logger.info(f"Test AUROC: {test_metrics.get(METRIC_AUROC, 'N/A'):.4f}")
        logger.info(f"Test PR-AUC: {test_metrics.get(METRIC_PRAUC, 'N/A'):.4f}")
    except FileNotFoundError as e:
        logger.warning(f"Could not load test predictions: {e}")

    # Save results
    log_section(logger, "Saving Ensemble Results")

    if outdir is None:
        # Auto-detect run_id from base model metadata if available
        run_id_dir = None
        if run_id is not None:
            run_id_dir = f"run_{run_id}"
        else:
            # Try to infer from base model directories
            for model in available_models:
                try:
                    model_dir = _find_model_split_dir(results_path, model, split_seed)
                    # Extract run_id from path (e.g., .../MODEL/run_20260128_214845/splits/split_seed0)
                    parts = model_dir.parts
                    for part in parts:
                        if part.startswith("run_"):
                            run_id_dir = part
                            break
                    if run_id_dir:
                        break
                except Exception:
                    pass

        if run_id_dir:
            # New layout: results/run_{id}/ENSEMBLE/splits/split_seed{N}/
            outdir = results_path / run_id_dir / "ENSEMBLE" / "splits" / f"split_seed{split_seed}"
        else:
            logger.warning(
                "Could not auto-detect run_id for ensemble. Using flat structure. "
                "Aggregation may fail. Consider using --run-id or --outdir."
            )
            outdir = results_path / "ENSEMBLE" / f"split_{split_seed}"
    else:
        outdir = Path(outdir)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories matching standard structure
    core_dir = outdir / "core"
    preds_dir = outdir / "preds"
    core_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)

    # Save ensemble model bundle
    import joblib
    import sklearn

    model_bundle = {
        "model": ensemble,
        "model_name": "ENSEMBLE",
        "base_models": available_models,
        "meta_penalty": meta_penalty,
        "meta_C": meta_c,
        "meta_coef": coef,
        "split_seed": split_seed,
        "random_state": random_state,
        "base_calibration_strategies": results.get("calibration_strategies", {}),
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model_path = core_dir / "ENSEMBLE__final_model.joblib"
    joblib.dump(model_bundle, model_path)
    logger.info(f"Ensemble model saved: {model_path}")

    # Save predictions
    if "val_proba" in results:
        val_df = pd.DataFrame(
            {
                "idx": results["val_idx"],
                "y_true": results["y_val"],
                "y_prob": results["val_proba"],
            }
        )
        # Add category if available
        if "cat_val" in results and results["cat_val"] is not None:
            val_df["category"] = results["cat_val"]
        # Save directly in preds/ (matching training output structure)
        val_path = preds_dir / "val_preds__ENSEMBLE.csv"
        val_df.to_csv(val_path, index=False)
        logger.info(f"Validation predictions saved: {val_path}")

    if "test_proba" in results:
        test_df = pd.DataFrame(
            {
                "idx": results["test_idx"],
                "y_true": results["y_test"],
                "y_prob": results["test_proba"],
            }
        )
        # Add category if available
        if "cat_test" in results and results["cat_test"] is not None:
            test_df["category"] = results["cat_test"]
        # Save directly in preds/ (matching training output structure)
        test_path = preds_dir / "test_preds__ENSEMBLE.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test predictions saved: {test_path}")

    # Save OOF predictions (aggregated meta-features used for training)
    oof_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
    # Get ensemble predictions on OOF meta-features
    oof_proba_for_csv = ensemble.predict_proba(oof_meta)[:, 1]

    oof_df = pd.DataFrame(oof_meta, columns=[f"oof_{m}" for m in available_models])
    oof_df["idx"] = train_idx
    oof_df["y_true"] = y_train
    oof_df["y_prob"] = oof_proba_for_csv  # Add ensemble's own predictions
    # Add category if available
    if cat_train is not None:
        oof_df["category"] = cat_train
    # Save directly in preds/ (matching training output structure)
    oof_path = preds_dir / "train_oof__ENSEMBLE.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved: {oof_path}")

    # Save metrics
    metrics_summary = {
        "model": "ENSEMBLE",
        "base_models": available_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_c,
        "timestamp": datetime.now().isoformat(),
    }
    if "val_metrics" in results:
        metrics_summary["val"] = results["val_metrics"]
    if "test_metrics" in results:
        metrics_summary["test"] = results["test_metrics"]

    metrics_path = core_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    # Save run settings
    run_settings = {
        "model": "ENSEMBLE",
        "base_models": available_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_c,
        "meta_coef": coef,
        "n_train": len(y_train),
        "train_prevalence": float(y_train.mean()),
        "random_state": random_state,
    }
    settings_path = core_dir / "run_settings.json"
    with open(settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)
    logger.info(f"Run settings saved: {settings_path}")

    # ---- Generating Ensemble Plots ----
    log_section(logger, "Generating Ensemble Plots")

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = outdir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    plot_format = "png"

    # Check if plots should be generated (respect max_plot_splits)
    max_plot_splits = getattr(config.output, "max_plot_splits", 0) if config else 0
    should_plot = max_plot_splits == 0 or split_seed < max_plot_splits
    if not should_plot:
        logger.info(
            f"Skipping plots for split_seed={split_seed} (max_plot_splits={max_plot_splits})"
        )

    meta_lines_plot = [
        f"Model: ENSEMBLE (base: {', '.join(available_models)})",
        f"Split seed: {split_seed}",
        f"Meta-learner: LR(penalty={meta_penalty}, C={meta_c})",
        f"Train n={len(y_train)}, prevalence={y_train.mean():.4f}",
    ]

    if should_plot:
        try:
            from ced_ml.metrics.thresholds import compute_threshold_bundle
            from ced_ml.plotting.calibration import plot_calibration_curve
            from ced_ml.plotting.dca import plot_dca_curve
            from ced_ml.plotting.ensemble import plot_meta_learner_weights, plot_model_comparison
            from ced_ml.plotting.oof import plot_oof_combined
            from ced_ml.plotting.risk_dist import plot_risk_distribution
            from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve

            # --- Standard discrimination plots for validation set ---
            if "val_proba" in results and "y_val" in results:
                y_val_arr = np.asarray(results["y_val"])
                val_proba_arr = np.asarray(results["val_proba"])

                val_bundle = compute_threshold_bundle(y_val_arr, val_proba_arr, target_spec=0.95)

                plot_roc_curve(
                    y_true=y_val_arr,
                    y_pred=val_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__val_roc.{plot_format}",
                    title="ENSEMBLE - Validation ROC",
                    subtitle=f"split_seed={split_seed}",
                    meta_lines=meta_lines_plot,
                    threshold_bundle=val_bundle,
                )
                plot_pr_curve(
                    y_true=y_val_arr,
                    y_pred=val_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__val_pr.{plot_format}",
                    title="ENSEMBLE - Validation PR Curve",
                    subtitle=f"split_seed={split_seed}",
                    meta_lines=meta_lines_plot,
                )
                plot_calibration_curve(
                    y_true=y_val_arr,
                    y_pred=val_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__val_calibration.{plot_format}",
                    title="ENSEMBLE - Validation Calibration",
                    meta_lines=meta_lines_plot,
                )
                plot_dca_curve(
                    y_true=y_val_arr,
                    y_pred=val_proba_arr,
                    out_path=str(plots_dir / f"ENSEMBLE__val_dca.{plot_format}"),
                    title="ENSEMBLE - Validation DCA",
                    meta_lines=meta_lines_plot,
                )
                # Get category for validation set if available
                cat_val_arr = (
                    np.asarray(results["cat_val"])
                    if "cat_val" in results and results["cat_val"] is not None
                    else None
                )

                plot_risk_distribution(
                    y_true=y_val_arr,
                    scores=val_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__val_risk_dist.{plot_format}",
                    title="ENSEMBLE - Validation Risk Distribution",
                    category_col=cat_val_arr,
                    threshold_bundle=val_bundle,
                    meta_lines=meta_lines_plot,
                )
                logger.info("Validation plots saved")

            # --- Standard discrimination plots for test set ---
            if "test_proba" in results and "y_test" in results:
                y_test_arr = np.asarray(results["y_test"])
                test_proba_arr = np.asarray(results["test_proba"])

                test_bundle = compute_threshold_bundle(y_test_arr, test_proba_arr, target_spec=0.95)

                plot_roc_curve(
                    y_true=y_test_arr,
                    y_pred=test_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__test_roc.{plot_format}",
                    title="ENSEMBLE - Test ROC",
                    subtitle=f"split_seed={split_seed}",
                    meta_lines=meta_lines_plot,
                    threshold_bundle=test_bundle,
                )
                plot_pr_curve(
                    y_true=y_test_arr,
                    y_pred=test_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__test_pr.{plot_format}",
                    title="ENSEMBLE - Test PR Curve",
                    subtitle=f"split_seed={split_seed}",
                    meta_lines=meta_lines_plot,
                )
                plot_calibration_curve(
                    y_true=y_test_arr,
                    y_pred=test_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__test_calibration.{plot_format}",
                    title="ENSEMBLE - Test Calibration",
                    meta_lines=meta_lines_plot,
                )
                plot_dca_curve(
                    y_true=y_test_arr,
                    y_pred=test_proba_arr,
                    out_path=str(plots_dir / f"ENSEMBLE__test_dca.{plot_format}"),
                    title="ENSEMBLE - Test DCA",
                    meta_lines=meta_lines_plot,
                )
                # Get category for test set if available
                cat_test_arr = (
                    np.asarray(results["cat_test"])
                    if "cat_test" in results and results["cat_test"] is not None
                    else None
                )

                plot_risk_distribution(
                    y_true=y_test_arr,
                    scores=test_proba_arr,
                    out_path=plots_dir / f"ENSEMBLE__test_risk_dist.{plot_format}",
                    title="ENSEMBLE - Test Risk Distribution",
                    category_col=cat_test_arr,
                    threshold_bundle=test_bundle,
                    meta_lines=meta_lines_plot,
                )
                logger.info("Test plots saved")

            # --- OOF combined plots (training set with CV repeats) ---
            # For ensemble, we use single repeat of OOF meta-features (already aggregated)
            # So n_repeats=1 (no cross-repeat variability to show)
            try:

                if "y_train" in results:
                    # Build meta-features used for training (OOF predictions from base models)
                    oof_meta_features = ensemble._build_meta_features(
                        oof_dict, aggregate_repeats=True
                    )
                    y_train_arr = np.asarray(results["y_train"])

                    # Get meta-learner predictions on OOF features (training set)
                    # This is what the meta-learner sees during training
                    oof_proba = ensemble.predict_proba(oof_meta_features)[:, 1]

                    # Reshape as (n_repeats=1, n_samples) for plot_oof_combined
                    oof_preds_ensemble = np.expand_dims(oof_proba, axis=0)

                    oof_meta_lines = [
                        "Model: ENSEMBLE (OOF on meta-features)",
                        f"Base models: {', '.join(available_models)}",
                        f"Split seed: {split_seed}",
                        f"Meta-learner: LR(penalty={meta_penalty}, C={meta_c})",
                        f"Train n={len(y_train_arr)}, prevalence={y_train_arr.mean():.4f}",
                        "Note: OOF computed from base model OOF predictions (meta-features)",
                    ]

                    plot_oof_combined(
                        y_true=y_train_arr,
                        oof_preds=oof_preds_ensemble,
                        out_dir=plots_dir,
                        model_name="ENSEMBLE",
                        plot_format=plot_format,
                        calib_bins=10,
                        meta_lines=oof_meta_lines,
                        target_spec=0.95,
                    )
                    logger.info("OOF combined plots saved (training set meta-features)")
            except Exception as e:
                logger.warning(f"OOF combined plots generation failed (non-fatal): {e}")

            # --- Train OOF risk distribution plot ---
            try:
                from ced_ml.metrics.dca import threshold_dca_zero_crossing

                if "y_train" in results:
                    y_train_arr = np.asarray(results["y_train"])

                    # Get meta-learner predictions on OOF meta-features (training set)
                    oof_meta_features = ensemble._build_meta_features(
                        oof_dict, aggregate_repeats=True
                    )
                    oof_proba = ensemble.predict_proba(oof_meta_features)[:, 1]

                    # Compute threshold bundle for train OOF
                    oof_dca_thr = threshold_dca_zero_crossing(y_train_arr, oof_proba)
                    oof_bundle = compute_threshold_bundle(
                        y_train_arr,
                        oof_proba,
                        target_spec=0.95,
                        dca_threshold=oof_dca_thr,
                    )

                    # Get category for training set if available
                    cat_train_arr = cat_train if cat_train is not None else None

                    plot_risk_distribution(
                        y_true=y_train_arr,
                        scores=oof_proba,
                        out_path=plots_dir / f"ENSEMBLE__TRAIN_OOF_risk_distribution.{plot_format}",
                        title="ENSEMBLE - Train OOF",
                        subtitle="Risk Score Distribution (meta-learner on OOF meta-features)",
                        category_col=cat_train_arr,
                        meta_lines=meta_lines_plot,
                        threshold_bundle=oof_bundle,
                    )
                    logger.info("Train OOF risk distribution plot saved")
            except Exception as e:
                logger.warning(
                    f"Train OOF risk distribution plot generation failed (non-fatal): {e}"
                )

            # --- Ensemble-specific plots ---
            if coef:
                plot_meta_learner_weights(
                    coef=coef,
                    out_path=diagnostics_dir / f"ENSEMBLE__meta_weights.{plot_format}",
                    title="Meta-Learner Coefficients",
                    subtitle=f"split_seed={split_seed}",
                    meta_penalty=meta_penalty,
                    meta_c=meta_c,
                    meta_lines=meta_lines_plot,
                )

            # Model comparison chart (ENSEMBLE vs base models)
            base_metrics = _collect_base_model_test_metrics(
                results_path, available_models, split_seed
            )
            if "test_metrics" in results:
                base_metrics["ENSEMBLE"] = results["test_metrics"]
            if len(base_metrics) >= 2:
                plot_model_comparison(
                    metrics=base_metrics,
                    out_path=diagnostics_dir / f"ENSEMBLE__model_comparison.{plot_format}",
                    title="Model Comparison (Test Set)",
                    subtitle=f"split_seed={split_seed}",
                    highlight_model="ENSEMBLE",
                    meta_lines=meta_lines_plot,
                )

                logger.info(f"Ensemble-specific plots saved to: {diagnostics_dir}")

        except Exception as e:
            logger.warning(f"Plot generation failed (non-fatal): {e}")

    # --- Learning curve for meta-learner (on OOF meta-features) ---
    try:
        lc_enabled = getattr(config.evaluation, "learning_curve", False) if config else False
        plot_lc = getattr(config.output, "plot_learning_curve", True) if config else True
        if lc_enabled and plot_lc:
            logger.info("Generating learning curve for ensemble meta-learner...")
            diagnostics_dir = outdir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = outdir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            lc_csv_path = diagnostics_dir / "ENSEMBLE__learning_curve.csv"
            # Only generate plot if within max_plot_splits limit
            lc_plot_path = (
                plots_dir / f"ENSEMBLE__learning_curve.{plot_format}" if should_plot else None
            )

            # Build meta-features for learning curve computation
            # Use OOF predictions from base models as features for the meta-learner
            X_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
            y_meta = y_train

            # Create a simple wrapper pipeline for learning curve (just the meta-learner LR)
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            meta_pipeline = Pipeline(
                [
                    (
                        "meta_learner",
                        LogisticRegression(
                            l1_ratio={"l1": 1.0, "l2": 0.0, "elasticnet": 0.5, None: 0.0}.get(
                                meta_penalty, 0.0
                            ),
                            C=np.inf if meta_penalty is None else meta_c,
                            solver="saga" if meta_penalty in ("l1", "elasticnet") else "lbfgs",
                            max_iter=1000,
                            random_state=random_state,
                            class_weight="balanced" if y_meta.mean() < 0.1 else None,
                        ),
                    )
                ]
            )

            lc_meta = [
                "Model: ENSEMBLE (meta-learner learning curve)",
                f"Base models: {', '.join(available_models)}",
                f"Split seed: {split_seed}",
                f"Meta-learner: LR(penalty={meta_penalty}, C={meta_c})",
                f"Train n={len(y_meta)}, prevalence={y_meta.mean():.4f}",
                f"Meta-features: {X_meta.shape[1]} (from {len(available_models)} base models)",
            ]

            save_learning_curve_csv(
                estimator=meta_pipeline,
                X=X_meta,
                y=y_meta,
                out_csv=lc_csv_path,
                scoring="roc_auc",
                cv=min(config.cv.folds if config else 5, 5),
                min_frac=0.3,
                n_points=5,
                seed=random_state,
                out_plot=lc_plot_path,
                meta_lines=lc_meta,
            )
            logger.info(f"Learning curve saved: {lc_csv_path}")
        else:
            if not lc_enabled:
                logger.debug("Learning curve disabled in config (evaluation.learning_curve=false)")
    except Exception as e:
        logger.warning(f"Learning curve generation failed (non-fatal): {e}")

    log_section(logger, "Ensemble Training Complete")
    logger.info(f"All results saved to: {outdir}")

    return results
