"""Plotting functions for ensemble training.

This module handles all plot generation for ensemble models,
extracted from train_ensemble.py to reduce complexity.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ced_ml.cli.ensemble_helpers import collect_base_model_test_metrics

logger = logging.getLogger(__name__)


def generate_validation_plots(
    results: dict[str, Any],
    plots_dir: Path,
    meta_lines: list[str],
    plot_format: str = "png",
) -> None:
    """Generate standard discrimination plots for validation set.

    Args:
        results: Dict with validation predictions and labels
        plots_dir: Directory for plot outputs
        meta_lines: Metadata lines for plot annotations
        plot_format: Image format (default: png)
    """
    if "val_proba" not in results or "y_val" not in results:
        return

    from ced_ml.metrics.thresholds import compute_threshold_bundle
    from ced_ml.plotting.calibration import plot_calibration_curve
    from ced_ml.plotting.dca import plot_dca_curve
    from ced_ml.plotting.risk_dist import plot_risk_distribution
    from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve

    y_val_arr = np.asarray(results["y_val"])
    val_proba_arr = np.asarray(results["val_proba"])

    val_bundle = compute_threshold_bundle(y_val_arr, val_proba_arr, target_spec=0.95)

    split_seed = results.get("split_seed", 0)

    plot_roc_curve(
        y_true=y_val_arr,
        y_pred=val_proba_arr,
        out_path=plots_dir / f"ENSEMBLE__val_roc.{plot_format}",
        title="ENSEMBLE - Validation ROC",
        subtitle=f"split_seed={split_seed}",
        meta_lines=meta_lines,
        threshold_bundle=val_bundle,
    )
    plot_pr_curve(
        y_true=y_val_arr,
        y_pred=val_proba_arr,
        out_path=plots_dir / f"ENSEMBLE__val_pr.{plot_format}",
        title="ENSEMBLE - Validation PR Curve",
        subtitle=f"split_seed={split_seed}",
        meta_lines=meta_lines,
    )
    plot_calibration_curve(
        y_true=y_val_arr,
        y_pred=val_proba_arr,
        out_path=plots_dir / f"ENSEMBLE__val_calibration.{plot_format}",
        title="ENSEMBLE - Validation Calibration",
        meta_lines=meta_lines,
    )
    plot_dca_curve(
        y_true=y_val_arr,
        y_pred=val_proba_arr,
        out_path=str(plots_dir / f"ENSEMBLE__val_dca.{plot_format}"),
        title="ENSEMBLE - Validation DCA",
        meta_lines=meta_lines,
    )

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
        meta_lines=meta_lines,
    )
    logger.info("Validation plots saved")


def generate_test_plots(
    results: dict[str, Any],
    plots_dir: Path,
    meta_lines: list[str],
    plot_format: str = "png",
) -> None:
    """Generate standard discrimination plots for test set.

    Args:
        results: Dict with test predictions and labels
        plots_dir: Directory for plot outputs
        meta_lines: Metadata lines for plot annotations
        plot_format: Image format (default: png)
    """
    if "test_proba" not in results or "y_test" not in results:
        return

    from ced_ml.metrics.thresholds import compute_threshold_bundle
    from ced_ml.plotting.calibration import plot_calibration_curve
    from ced_ml.plotting.dca import plot_dca_curve
    from ced_ml.plotting.risk_dist import plot_risk_distribution
    from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve

    y_test_arr = np.asarray(results["y_test"])
    test_proba_arr = np.asarray(results["test_proba"])

    test_bundle = compute_threshold_bundle(y_test_arr, test_proba_arr, target_spec=0.95)

    split_seed = results.get("split_seed", 0)

    plot_roc_curve(
        y_true=y_test_arr,
        y_pred=test_proba_arr,
        out_path=plots_dir / f"ENSEMBLE__test_roc.{plot_format}",
        title="ENSEMBLE - Test ROC",
        subtitle=f"split_seed={split_seed}",
        meta_lines=meta_lines,
        threshold_bundle=test_bundle,
    )
    plot_pr_curve(
        y_true=y_test_arr,
        y_pred=test_proba_arr,
        out_path=plots_dir / f"ENSEMBLE__test_pr.{plot_format}",
        title="ENSEMBLE - Test PR Curve",
        subtitle=f"split_seed={split_seed}",
        meta_lines=meta_lines,
    )
    plot_calibration_curve(
        y_true=y_test_arr,
        y_pred=test_proba_arr,
        out_path=plots_dir / f"ENSEMBLE__test_calibration.{plot_format}",
        title="ENSEMBLE - Test Calibration",
        meta_lines=meta_lines,
    )
    plot_dca_curve(
        y_true=y_test_arr,
        y_pred=test_proba_arr,
        out_path=str(plots_dir / f"ENSEMBLE__test_dca.{plot_format}"),
        title="ENSEMBLE - Test DCA",
        meta_lines=meta_lines,
    )

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
        meta_lines=meta_lines,
    )
    logger.info("Test plots saved")


def generate_oof_plots(
    results: dict[str, Any],
    ensemble: Any,
    plots_dir: Path,
    available_models: list[str],
    meta_penalty: str,
    meta_c: float,
    plot_format: str = "png",
) -> None:
    """Generate OOF combined plots for training set.

    Args:
        results: Dict with OOF predictions and labels
        ensemble: Trained StackingEnsemble instance
        plots_dir: Directory for plot outputs
        available_models: List of base model names
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        plot_format: Image format (default: png)
    """
    try:
        from ced_ml.plotting.oof import plot_oof_combined

        if "y_train" not in results:
            return

        oof_dict = results.get("oof_dict", {})
        if not oof_dict:
            return

        oof_meta_features = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
        y_train_arr = np.asarray(results["y_train"])

        oof_proba = ensemble.predict_proba(oof_meta_features)[:, 1]
        oof_preds_ensemble = np.expand_dims(oof_proba, axis=0)

        split_seed = results.get("split_seed", 0)

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


def generate_train_oof_risk_distribution(
    results: dict[str, Any],
    ensemble: Any,
    plots_dir: Path,
    meta_lines: list[str],
    plot_format: str = "png",
) -> None:
    """Generate train OOF risk distribution plot.

    Args:
        results: Dict with OOF predictions and labels
        ensemble: Trained StackingEnsemble instance
        plots_dir: Directory for plot outputs
        meta_lines: Metadata lines for plot annotations
        plot_format: Image format (default: png)
    """
    try:
        from ced_ml.metrics.dca import threshold_dca_zero_crossing
        from ced_ml.metrics.thresholds import compute_threshold_bundle
        from ced_ml.plotting.risk_dist import plot_risk_distribution

        if "y_train" not in results:
            return

        oof_dict = results.get("oof_dict", {})
        if not oof_dict:
            return

        y_train_arr = np.asarray(results["y_train"])

        oof_meta_features = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
        oof_proba = ensemble.predict_proba(oof_meta_features)[:, 1]

        oof_dca_thr = threshold_dca_zero_crossing(y_train_arr, oof_proba)
        oof_bundle = compute_threshold_bundle(
            y_train_arr,
            oof_proba,
            target_spec=0.95,
            dca_threshold=oof_dca_thr,
        )

        cat_train_arr = results.get("cat_train")

        plot_risk_distribution(
            y_true=y_train_arr,
            scores=oof_proba,
            out_path=plots_dir / f"ENSEMBLE__TRAIN_OOF_risk_distribution.{plot_format}",
            title="ENSEMBLE - Train OOF",
            subtitle="Risk Score Distribution (meta-learner on OOF meta-features)",
            category_col=cat_train_arr,
            meta_lines=meta_lines,
            threshold_bundle=oof_bundle,
        )
        logger.info("Train OOF risk distribution plot saved")
    except Exception as e:
        logger.warning(f"Train OOF risk distribution plot generation failed (non-fatal): {e}")


def generate_ensemble_specific_plots(
    results: dict[str, Any],
    results_path: Path,
    available_models: list[str],
    diagnostics_dir: Path,
    meta_lines: list[str],
    meta_penalty: str,
    meta_c: float,
    plot_format: str = "png",
    run_id: str | None = None,
) -> None:
    """Generate ensemble-specific plots (meta-learner weights, model comparison).

    Args:
        results: Dict with predictions and metadata
        results_path: Root results directory
        available_models: List of base model names
        diagnostics_dir: Directory for diagnostic plot outputs
        meta_lines: Metadata lines for plot annotations
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        plot_format: Image format (default: png)
        run_id: Optional run_id to scope path resolution to a specific run
    """
    from ced_ml.plotting.ensemble import plot_meta_learner_weights, plot_model_comparison

    split_seed = results.get("split_seed", 0)
    coef = results.get("meta_coef")

    if coef:
        plot_meta_learner_weights(
            coef=coef,
            out_path=diagnostics_dir / f"ENSEMBLE__meta_weights.{plot_format}",
            title="Meta-Learner Coefficients",
            subtitle=f"split_seed={split_seed}",
            meta_penalty=meta_penalty,
            meta_c=meta_c,
            meta_lines=meta_lines,
        )

    base_metrics = collect_base_model_test_metrics(
        results_path, available_models, split_seed, run_id
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
            meta_lines=meta_lines,
        )

        logger.info(f"Ensemble-specific plots saved to: {diagnostics_dir}")


def generate_learning_curve(
    config: Any,
    ensemble: Any,
    oof_dict: dict[str, Any],
    y_train: np.ndarray,
    available_models: list[str],
    outdir: Path,
    split_seed: int,
    meta_penalty: str,
    meta_c: float,
    random_state: int,
    should_plot: bool,
    plot_format: str = "png",
) -> None:
    """Generate learning curve for ensemble meta-learner.

    Args:
        config: Training configuration object
        ensemble: Trained StackingEnsemble instance
        oof_dict: Dict of OOF predictions from base models
        y_train: Training labels
        available_models: List of base model names
        outdir: Output directory
        split_seed: Split seed
        meta_penalty: Meta-learner penalty type
        meta_c: Meta-learner regularization strength
        random_state: Random state for reproducibility
        should_plot: Whether to generate plot image
        plot_format: Image format (default: png)
    """
    try:
        lc_enabled = getattr(config.evaluation, "learning_curve", False) if config else False
        plot_lc = getattr(config.output, "plot_learning_curve", True) if config else True

        if not (lc_enabled and plot_lc):
            if not lc_enabled:
                logger.debug("Learning curve disabled in config (evaluation.learning_curve=false)")
            return

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from ced_ml.plotting.learning_curve import save_learning_curve_csv

        logger.info("Generating learning curve for ensemble meta-learner...")
        diagnostics_dir = outdir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = outdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        lc_csv_path = diagnostics_dir / "ENSEMBLE__learning_curve.csv"
        lc_plot_path = (
            plots_dir / f"ENSEMBLE__learning_curve.{plot_format}" if should_plot else None
        )

        X_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
        y_meta = y_train

        meta_lr_params = {
            "solver": "saga" if meta_penalty in ("l1", "elasticnet") else "lbfgs",
            "max_iter": 1000,
            "random_state": random_state,
            "class_weight": "balanced" if y_meta.mean() < 0.1 else None,
        }

        if meta_penalty is None:
            meta_lr_params["C"] = np.inf
        else:
            meta_lr_params["C"] = meta_c
            meta_lr_params["l1_ratio"] = {
                "l1": 1.0,
                "l2": 0.0,
                "elasticnet": 0.5,
            }.get(meta_penalty, 0.0)

        meta_pipeline = Pipeline(
            [
                (
                    "meta_learner",
                    LogisticRegression(**meta_lr_params),
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
    except Exception as e:
        logger.warning(f"Learning curve generation failed (non-fatal): {e}")
