"""
Plotting stage for training orchestration.

This module handles:
- Validation set plots (ROC, PR, calibration)
- Test set plots (ROC, PR, calibration, DCA)
- OOF combined plots
- Risk distribution plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ced_ml.metrics.dca import threshold_dca_zero_crossing
from ced_ml.metrics.thresholds import compute_threshold_bundle
from ced_ml.models.training import get_model_n_iter
from ced_ml.plotting import (
    plot_calibration_curve,
    plot_oof_combined,
    plot_pr_curve,
    plot_risk_distribution,
    plot_roc_curve,
)
from ced_ml.plotting.dca import plot_dca_curve
from ced_ml.utils.logging import log_section
from ced_ml.utils.metadata import build_oof_metadata, build_plot_metadata, count_category_breakdown

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def generate_plots(ctx: TrainingContext) -> TrainingContext:
    """Generate all diagnostic plots.

    This stage:
    1. Generates validation set plots (ROC, PR, calibration)
    2. Generates test set plots (ROC, PR, calibration, DCA)
    3. Generates OOF combined plots
    4. Generates risk distribution plots

    Args:
        ctx: TrainingContext with evaluation completed

    Returns:
        Updated TrainingContext with category breakdowns populated
    """
    config = ctx.config
    seed = ctx.seed

    # Extract category breakdowns (needed for plots AND metadata)
    # count_category_breakdown expects a DataFrame with 'category' column
    train_breakdown = count_category_breakdown(pd.DataFrame({"category": ctx.cat_train}))
    val_breakdown = count_category_breakdown(pd.DataFrame({"category": ctx.cat_val}))
    test_breakdown = count_category_breakdown(pd.DataFrame({"category": ctx.cat_test}))

    ctx.train_breakdown = train_breakdown
    ctx.val_breakdown = val_breakdown
    ctx.test_breakdown = test_breakdown

    # Check if we should generate plots
    should_plot = config.output.save_plots and (
        config.output.max_plot_splits == 0 or seed < config.output.max_plot_splits
    )

    if not should_plot:
        logger.info("Skipping plot generation (disabled or beyond max_plot_splits)")
        return ctx

    log_section(logger, "Generating Plots")
    plots_dir = Path(ctx.outdirs.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Common metadata for plots
    meta_lines = build_plot_metadata(
        model=config.model,
        scenario=ctx.scenario,
        seed=seed,
        train_prev=ctx.train_prev,
        target_prev=ctx.test_target_prev,
        cv_folds=config.cv.folds,
        cv_repeats=config.cv.repeats,
        cv_scoring=config.cv.scoring,
        n_features=(len(ctx.final_selected_proteins) if ctx.final_selected_proteins else None),
        feature_method=config.features.feature_selection_strategy,
        n_train=len(ctx.y_train),
        n_val=len(ctx.y_val),
        n_test=len(ctx.y_test),
        n_train_pos=int(ctx.y_train.sum()),
        n_val_pos=int(ctx.y_val.sum()),
        n_test_pos=int(ctx.y_test.sum()),
        n_train_controls=train_breakdown.get("controls"),
        n_train_incident=train_breakdown.get("incident"),
        n_train_prevalent=train_breakdown.get("prevalent"),
        n_val_controls=val_breakdown.get("controls"),
        n_val_incident=val_breakdown.get("incident"),
        n_val_prevalent=val_breakdown.get("prevalent"),
        n_test_controls=test_breakdown.get("controls"),
        n_test_incident=test_breakdown.get("incident"),
        n_test_prevalent=test_breakdown.get("prevalent"),
        split_mode="development",
        optuna_enabled=config.optuna.enabled,
        n_trials=config.optuna.n_trials if config.optuna.enabled else None,
        n_iter=(get_model_n_iter(config.model, config) if not config.optuna.enabled else None),
        threshold_objective=config.thresholds.objective,
        prevalence_adjusted=True,
    )

    # Validation set plots
    val_bundle = None
    if ctx.has_validation_set:
        val_bundle = _generate_validation_plots(ctx, plots_dir, meta_lines)
    else:
        logger.info("Skipping validation plots (no validation set)")

    # Test set plots
    test_bundle = _generate_test_plots(ctx, plots_dir, meta_lines)

    # OOF combined plots
    _generate_oof_plots(ctx, plots_dir, train_breakdown)

    # Risk distribution plots
    _generate_risk_distribution_plots(ctx, plots_dir, meta_lines, val_bundle, test_bundle)

    # SHAP explainability plots
    _generate_shap_plots(ctx, plots_dir)

    logger.info(f"All diagnostic plots saved to: {plots_dir}")

    return ctx


def _generate_validation_plots(
    ctx: TrainingContext,
    plots_dir: Path,
    meta_lines: list[str],
) -> dict:
    """Generate validation set plots."""
    config = ctx.config
    _val_prob_col = "y_prob_adjusted" if "y_prob_adjusted" in ctx.val_preds_df.columns else "y_prob"
    val_y_prob = ctx.val_preds_df[_val_prob_col].values
    val_title = f"{config.model} - Validation Set"

    # Compute validation threshold bundle
    val_dca_thr = threshold_dca_zero_crossing(ctx.y_val, val_y_prob)
    val_bundle = compute_threshold_bundle(
        ctx.y_val,
        val_y_prob,
        target_spec=config.thresholds.fixed_spec,
        dca_threshold=val_dca_thr,
    )

    if config.output.plot_roc:
        plot_roc_curve(
            y_true=ctx.y_val,
            y_pred=val_y_prob,
            out_path=plots_dir / f"{config.model}__val_roc.{config.output.plot_format}",
            title=val_title,
            subtitle="ROC Curve",
            meta_lines=meta_lines,
            threshold_bundle=val_bundle,
        )
        logger.info("Val ROC curve saved")

    if config.output.plot_pr:
        plot_pr_curve(
            y_true=ctx.y_val,
            y_pred=val_y_prob,
            out_path=plots_dir / f"{config.model}__val_pr.{config.output.plot_format}",
            title=val_title,
            subtitle="Precision-Recall Curve",
            meta_lines=meta_lines,
        )
        logger.info("Val PR curve saved")

    if config.output.plot_calibration:
        plot_calibration_curve(
            y_true=ctx.y_val,
            y_pred=val_y_prob,
            out_path=plots_dir / f"{config.model}__val_calibration.{config.output.plot_format}",
            title=val_title,
            subtitle="Calibration",
            n_bins=config.output.calib_bins,
            meta_lines=meta_lines,
        )
        logger.info("Val calibration plot saved")

    # DCA plot for validation
    if config.output.plot_dca:
        plot_dca_curve(
            y_true=ctx.y_val,
            y_pred=val_y_prob,
            out_path=str(plots_dir / f"{config.model}__val_dca.{config.output.plot_format}"),
            title=val_title,
            subtitle="Decision Curve Analysis",
            meta_lines=meta_lines,
        )
        logger.info("Validation DCA plot saved")

    return val_bundle


def _generate_test_plots(
    ctx: TrainingContext,
    plots_dir: Path,
    meta_lines: list[str],
) -> dict:
    """Generate test set plots."""
    config = ctx.config
    _test_prob_col = (
        "y_prob_adjusted" if "y_prob_adjusted" in ctx.test_preds_df.columns else "y_prob"
    )
    test_y_prob = ctx.test_preds_df[_test_prob_col].values
    test_title = f"{config.model} - Test Set"

    # Compute test threshold bundle
    dca_thr = threshold_dca_zero_crossing(ctx.y_test, test_y_prob)
    test_bundle = compute_threshold_bundle(
        ctx.y_test,
        test_y_prob,
        target_spec=config.thresholds.fixed_spec,
        dca_threshold=dca_thr,
    )

    # Store test threshold in context (used as fallback if no validation set)
    ctx.test_threshold = test_bundle["spec_target_threshold"]

    youden_thr = test_bundle["youden_threshold"]
    spec_target_thr = test_bundle["spec_target_threshold"]
    dca_str = f"{dca_thr:.4f}" if dca_thr is not None else "N/A"
    logger.info(
        f"Thresholds: Youden={youden_thr:.4f}, SpecTarget={spec_target_thr:.4f}, DCA={dca_str}"
    )

    if config.output.plot_roc:
        plot_roc_curve(
            y_true=ctx.y_test,
            y_pred=test_y_prob,
            out_path=plots_dir / f"{config.model}__test_roc.{config.output.plot_format}",
            title=test_title,
            subtitle="ROC Curve",
            meta_lines=meta_lines,
            threshold_bundle=test_bundle,
        )
        logger.info("Test ROC curve saved")

    if config.output.plot_pr:
        plot_pr_curve(
            y_true=ctx.y_test,
            y_pred=test_y_prob,
            out_path=plots_dir / f"{config.model}__test_pr.{config.output.plot_format}",
            title=test_title,
            subtitle="Precision-Recall Curve",
            meta_lines=meta_lines,
        )
        logger.info("Test PR curve saved")

    if config.output.plot_calibration:
        plot_calibration_curve(
            y_true=ctx.y_test,
            y_pred=test_y_prob,
            out_path=plots_dir / f"{config.model}__test_calibration.{config.output.plot_format}",
            title=test_title,
            subtitle="Calibration",
            n_bins=config.output.calib_bins,
            meta_lines=meta_lines,
        )
        logger.info("Test calibration plot saved")

    if config.output.plot_dca:
        plot_dca_curve(
            y_true=ctx.y_test,
            y_pred=test_y_prob,
            out_path=str(plots_dir / f"{config.model}__test_dca.{config.output.plot_format}"),
            title=test_title,
            subtitle="Decision Curve Analysis",
            meta_lines=meta_lines,
        )
        logger.info("Test DCA plot saved")

    return test_bundle


def _generate_oof_plots(
    ctx: TrainingContext,
    plots_dir: Path,
    train_breakdown: dict,
) -> None:
    """Generate OOF combined plots."""
    config = ctx.config

    if not config.output.plot_oof_combined:
        return

    oof_meta = build_oof_metadata(
        model=config.model,
        scenario=ctx.scenario,
        seed=ctx.seed,
        cv_folds=config.cv.folds,
        cv_repeats=config.cv.repeats,
        train_prev=ctx.train_prev,
        n_train=len(ctx.y_train),
        n_train_pos=int(ctx.y_train.sum()),
        n_train_controls=train_breakdown.get("controls"),
        n_train_incident=train_breakdown.get("incident"),
        n_train_prevalent=train_breakdown.get("prevalent"),
        n_features=(len(ctx.final_selected_proteins) if ctx.final_selected_proteins else None),
        feature_method=config.features.feature_selection_strategy,
        cv_scoring=config.cv.scoring,
    )

    plot_oof_combined(
        y_true=ctx.y_train,
        oof_preds=ctx.oof_preds,
        out_dir=plots_dir,
        model_name=config.model,
        plot_format=config.output.plot_format,
        calib_bins=config.output.calib_bins,
        meta_lines=oof_meta,
    )
    logger.info("OOF combined plots saved")


def _generate_risk_distribution_plots(
    ctx: TrainingContext,
    plots_dir: Path,
    meta_lines: list[str],
    val_bundle: dict | None,
    test_bundle: dict,
) -> None:
    """Generate risk distribution plots."""
    config = ctx.config

    if not config.output.plot_risk_distribution:
        return

    # Test set risk distribution
    plot_risk_distribution(
        y_true=ctx.y_test,
        scores=ctx.test_preds_df[
            "y_prob_adjusted" if "y_prob_adjusted" in ctx.test_preds_df.columns else "y_prob"
        ].values,
        out_path=plots_dir / f"{config.model}__TEST_risk_distribution.{config.output.plot_format}",
        title=f"{config.model} - Test Set",
        subtitle="Risk Score Distribution",
        meta_lines=meta_lines,
        category_col=ctx.cat_test,
        threshold_bundle=test_bundle,
    )
    logger.info("Test risk distribution plot saved")

    # Val set risk distribution
    if ctx.has_validation_set and val_bundle is not None:
        plot_risk_distribution(
            y_true=ctx.y_val,
            scores=ctx.val_preds_df[
                "y_prob_adjusted" if "y_prob_adjusted" in ctx.val_preds_df.columns else "y_prob"
            ].values,
            out_path=plots_dir
            / f"{config.model}__VAL_risk_distribution.{config.output.plot_format}",
            title=f"{config.model} - Validation Set",
            subtitle="Risk Score Distribution",
            meta_lines=meta_lines,
            category_col=ctx.cat_val,
            threshold_bundle=val_bundle,
        )
        logger.info("Val risk distribution plot saved")

    # Train OOF risk distribution
    oof_mean = ctx.oof_preds.mean(axis=0)
    oof_dca_thr = threshold_dca_zero_crossing(ctx.y_train, oof_mean)
    oof_bundle = compute_threshold_bundle(
        ctx.y_train,
        oof_mean,
        target_spec=config.thresholds.fixed_spec,
        dca_threshold=oof_dca_thr,
    )
    plot_risk_distribution(
        y_true=ctx.y_train,
        scores=oof_mean,
        out_path=plots_dir
        / f"{config.model}__TRAIN_OOF_risk_distribution.{config.output.plot_format}",
        title=f"{config.model} - Train OOF",
        subtitle="Risk Score Distribution (mean across repeats)",
        meta_lines=meta_lines,
        category_col=ctx.cat_train,
        threshold_bundle=oof_bundle,
    )
    logger.info("Train OOF risk distribution plot saved")


def _generate_shap_plots(ctx: TrainingContext, plots_dir: Path) -> None:
    """Generate SHAP explainability plots if SHAP data is available."""
    config = ctx.config
    shap_config = getattr(config.features, "shap", None)

    if not shap_config or not shap_config.enabled:
        return

    if ctx.test_shap_payload is None:
        logger.debug("Skipping SHAP plots: no test SHAP payload available")
        return

    try:
        from ced_ml.plotting.shap_plots import generate_all_shap_plots
    except ImportError:
        logger.warning("SHAP plotting not available: shap package not installed")
        return

    try:
        generate_all_shap_plots(
            test_payload=ctx.test_shap_payload,
            oof_shap_df=ctx.oof_shap_df,
            threshold=ctx.val_threshold if ctx.val_threshold is not None else ctx.test_threshold,
            config=config,
            outdir=plots_dir,
            test_preds_df=ctx.test_preds_df,
        )
        logger.info("SHAP plots generated")
    except Exception as e:
        logger.warning("SHAP plot generation failed: %s", e)
