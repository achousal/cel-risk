"""Helper functions for ensemble training CLI.

This module contains staged helper functions to support train_ensemble.py,
extracted to reduce complexity and improve maintainability.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER, METRIC_PRAUC
from ced_ml.models.stacking import _find_model_split_dir

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

    n_nan = np.isnan(y_prob).sum()
    if n_nan > 0:
        raise ValueError(
            f"Ensemble predictions on {split_name} contain {n_nan} NaN values. "
            "This indicates a meta-learner or calibration error."
        )

    n_inf = np.isinf(y_prob).sum()
    if n_inf > 0:
        raise ValueError(
            f"Ensemble predictions on {split_name} contain {n_inf} Inf values. "
            "This indicates a meta-learner or calibration error."
        )

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


def collect_base_model_test_metrics(
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

            metrics_path = model_dir / "core" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)

                test_data = data.get("test", data)
                entry = {}
                for key in (METRIC_AUROC, METRIC_PRAUC, METRIC_BRIER):
                    if key in test_data:
                        entry[key] = float(test_data[key])
                if entry:
                    collected[model] = entry
                    continue

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


def validate_base_models(
    results_path: Path,
    base_models: list[str],
    split_seed: int,
) -> tuple[list[str], list[str]]:
    """Check which base models have OOF predictions available.

    Args:
        results_path: Root results directory
        base_models: List of base model names to validate
        split_seed: Split seed for path resolution

    Returns:
        Tuple of (available_models, missing_models)

    Raises:
        FileNotFoundError: If fewer than 2 models are available
    """
    available_models = []
    missing_models = []

    for model in base_models:
        try:
            model_dir = _find_model_split_dir(results_path, model, split_seed)
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

    return available_models, missing_models


def infer_run_id_from_models(
    results_path: Path,
    available_models: list[str],
    split_seed: int,
) -> str | None:
    """Infer run_id from base model directory structure.

    Args:
        results_path: Root results directory
        available_models: List of available base model names
        split_seed: Split seed for path resolution

    Returns:
        Run ID string (without 'run_' prefix) or None if not found
    """
    for model in available_models:
        try:
            model_dir = _find_model_split_dir(results_path, model, split_seed)
            parts = model_dir.parts
            for part in parts:
                if part.startswith("run_"):
                    return part
        except Exception:
            pass
    return None


def determine_output_directory(
    outdir: str | None,
    results_path: Path,
    run_id: str | None,
    available_models: list[str],
    split_seed: int,
) -> Path:
    """Determine output directory for ensemble results.

    Args:
        outdir: Explicit output directory (takes precedence)
        results_path: Root results directory
        run_id: Run ID string (e.g., "20260127_115115")
        available_models: List of available base models
        split_seed: Split seed

    Returns:
        Output directory path
    """
    if outdir is not None:
        return Path(outdir)

    run_id_dir = None
    if run_id is not None:
        run_id_dir = f"run_{run_id}"
    else:
        inferred = infer_run_id_from_models(results_path, available_models, split_seed)
        if inferred:
            run_id_dir = inferred

    if run_id_dir:
        return results_path / run_id_dir / "ENSEMBLE" / "splits" / f"split_seed{split_seed}"
    else:
        logger.warning(
            "Could not auto-detect run_id for ensemble. Using flat structure. "
            "Aggregation may fail. Consider using --run-id or --outdir."
        )
        return results_path / "ENSEMBLE" / f"split_{split_seed}"


def save_ensemble_artifacts(
    outdir: Path,
    ensemble: Any,
    available_models: list[str],
    results: dict[str, Any],
    split_seed: int,
    meta_penalty: str,
    meta_c: float,
    random_state: int,
) -> None:
    """Save ensemble model bundle and predictions to disk.

    Args:
        outdir: Output directory
        ensemble: Trained StackingEnsemble instance
        available_models: List of base model names
        results: Dict with predictions and metadata
        split_seed: Split seed
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        random_state: Random state for reproducibility
    """
    import joblib
    import sklearn

    core_dir = outdir / "core"
    preds_dir = outdir / "preds"
    core_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)

    coef = ensemble.get_meta_model_coef()

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

    if "val_proba" in results:
        val_df = pd.DataFrame(
            {
                "idx": results["val_idx"],
                "y_true": results["y_val"],
                "y_prob": results["val_proba"],
            }
        )
        if "cat_val" in results and results["cat_val"] is not None:
            val_df["category"] = results["cat_val"]
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
        if "cat_test" in results and results["cat_test"] is not None:
            test_df["category"] = results["cat_test"]
        test_path = preds_dir / "test_preds__ENSEMBLE.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test predictions saved: {test_path}")

    oof_dict = results.get("oof_dict", {})
    if oof_dict:
        oof_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
        oof_proba_for_csv = ensemble.predict_proba(oof_meta)[:, 1]

        oof_df = pd.DataFrame(oof_meta, columns=[f"oof_{m}" for m in available_models])
        oof_df["idx"] = results["train_idx"]
        oof_df["y_true"] = results["y_train"]
        oof_df["y_prob"] = oof_proba_for_csv

        if results.get("cat_train") is not None:
            oof_df["category"] = results["cat_train"]

        oof_path = preds_dir / "train_oof__ENSEMBLE.csv"
        oof_df.to_csv(oof_path, index=False)
        # 1.7-M4: Warn that these are in-sample predictions, not genuine OOF
        logger.warning(
            "train_oof__ENSEMBLE.csv contains in-sample meta-learner predictions, "
            "not genuine out-of-fold ensemble predictions. These predictions are from "
            "the meta-learner trained on the same data and should NOT be used for "
            "performance evaluation."
        )
        logger.info(f"OOF predictions saved: {oof_path}")


def save_ensemble_metadata(
    outdir: Path,
    available_models: list[str],
    results: dict[str, Any],
    split_seed: int,
    meta_penalty: str,
    meta_c: float,
    meta_coef: dict[str, float] | None,
) -> None:
    """Save ensemble metadata files (metrics, run settings).

    Args:
        outdir: Output directory
        available_models: List of base model names
        results: Dict with predictions and metadata
        split_seed: Split seed
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        meta_coef: Meta-learner coefficients
    """
    from datetime import datetime

    core_dir = outdir / "core"

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

    y_train = results.get("y_train")
    run_settings = {
        "model": "ENSEMBLE",
        "split_seed": split_seed,
        "n_train": len(y_train) if y_train is not None else 0,
        "train_prevalence": float(y_train.mean()) if y_train is not None else 0.0,
        "random_state": results.get("random_state", 42),
        "ensemble": {
            "method": "stacking",
            "base_models": available_models,
            "meta_model": {
                "type": "logistic_regression",
                "penalty": meta_penalty,
                "C": meta_c,
                "coefficients": meta_coef,
            },
        },
    }
    settings_path = core_dir / "run_settings.json"
    with open(settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)
    logger.info(f"Run settings saved: {settings_path}")
