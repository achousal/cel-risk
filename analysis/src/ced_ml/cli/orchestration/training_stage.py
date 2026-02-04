"""
Training stage for training orchestration.

This module handles:
- Building classifier and pipeline
- Running nested CV for OOF predictions
- Fitting final model
- Applying calibration
- Extracting selected proteins
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ced_ml.cli.train import build_training_pipeline
from ced_ml.models.calibration import OOFCalibratedModel
from ced_ml.models.calibration_strategy import (
    get_calibration_strategy,
    get_strategy_display_name,
)
from ced_ml.models.registry import build_models
from ced_ml.models.training import (
    _apply_per_fold_calibration,
    _extract_selected_proteins_from_fold,
    oof_predictions_with_nested_cv,
)
from ced_ml.utils.logging import log_section

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def train_models(ctx: TrainingContext) -> TrainingContext:
    """Train models with nested CV and fit final model.

    This stage:
    1. Builds classifier from config
    2. Builds training pipeline with preprocessing and feature selection
    3. Runs nested CV for OOF predictions
    4. Fits final model on full training set
    5. Applies calibration (per-fold or OOF-posthoc)
    6. Extracts selected proteins from final model

    Args:
        ctx: TrainingContext with features prepared

    Returns:
        Updated TrainingContext with:
        - final_pipeline: Fitted final model
        - oof_preds: OOF predictions array
        - best_params_df: Best params per fold
        - selected_proteins_df: Selected proteins per fold
        - oof_calibrator: OOF calibrator (if applicable)
        - nested_rfecv_result: RFECV result (if enabled)
        - final_selected_proteins: Final test panel proteins
        - cv_elapsed_sec: CV elapsed time
    """
    config = ctx.config
    seed = ctx.seed
    protein_cols = ctx.protein_cols

    # Step 1: Build classifier
    log_section(logger, "Building Model")
    logger.info(f"Model type: {config.model}")

    classifier = build_models(
        model_name=config.model,
        config=config,
        random_state=seed,
        n_jobs=config.n_jobs,
    )

    # Step 2: Build full pipeline
    pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols,
        ctx.resolved.categorical_metadata,
        model_name=config.model,
    )
    logger.info(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")

    # Step 3: Run nested CV for OOF predictions
    log_section(logger, "Nested Cross-Validation")
    total_folds = ctx.total_cv_folds
    logger.info(
        f"Config: {config.model} | {config.cv.folds}-fold x {config.cv.repeats} repeats "
        f"= {total_folds} outer folds | scoring={config.cv.scoring}"
    )

    strategy = config.features.feature_selection_strategy
    screen_top = getattr(config.features, "screen_top_n", "?")
    logger.info(
        f"Features: {strategy} | screen={getattr(config.features, 'screen_method', 'none')} "
        f"top-{screen_top}"
    )

    optuna_cfg = config.optuna
    # Use calibration strategy pattern for display
    calibration_strategy = get_calibration_strategy(config, model_name=config.model)
    calibration_display = get_strategy_display_name(calibration_strategy)
    if optuna_cfg.enabled:
        logger.info(
            f"Optuna: {optuna_cfg.n_trials} trials ({optuna_cfg.sampler}/{optuna_cfg.pruner}) | "
            f"calibration: {calibration_display}"
        )
    else:
        logger.info(f"Optuna: disabled | calibration: {calibration_display}")

    logger.info(f"Running {total_folds} folds...")

    (
        oof_preds,
        elapsed_sec,
        best_params_df,
        selected_proteins_df,
        oof_calibrator,
        nested_rfecv_result,
    ) = oof_predictions_with_nested_cv(
        pipeline=pipeline,
        model_name=config.model,
        X=ctx.X_train,
        y=ctx.y_train,
        protein_cols=protein_cols,
        config=config,
        random_state=seed,
        grid_rng=ctx.grid_rng,
    )

    logger.info(f"CV completed in {elapsed_sec:.1f}s")

    # Step 4: Fit final model on full train set
    log_section(logger, "Training Final Model")
    logger.info("Fitting on full training set...")

    final_pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols,
        ctx.resolved.categorical_metadata,
        model_name=config.model,
    )

    # Determine best k value for final model (if using hybrid_stability with k_grid)
    if strategy == "hybrid_stability" and "sel__k" in best_params_df.columns:
        k_grid = getattr(config.features, "k_grid", None)

        if k_grid and len(k_grid) > 1:
            # Multiple k values were tuned: use most frequently selected k (mode)
            best_k = int(best_params_df["sel__k"].mode()[0])
            logger.info(f"Multiple k values tuned: using mode k={best_k} for final model")
        elif k_grid and len(k_grid) == 1:
            # Single k value: use that value
            best_k = k_grid[0]
            logger.info(f"Single k value in config: using k={best_k} for final model")
        else:
            # Fallback: use first k from CV results
            best_k = int(best_params_df["sel__k"].iloc[0])
            logger.info(f"Using k={best_k} from CV results for final model")

        final_pipeline.set_params(sel__k=best_k)
        logger.info(f"Final model k-best set to k={best_k}")

    final_pipeline.fit(ctx.X_train, ctx.y_train)
    logger.info("Final model fitted")

    # Step 5: Apply calibration to final model
    final_pipeline = _apply_per_fold_calibration(
        estimator=final_pipeline,
        model_name=config.model,
        config=config,
        X_train=ctx.X_train,
        y_train=ctx.y_train,
    )

    # For oof_posthoc strategy, wrap final model with the OOF calibrator
    # Use calibration strategy pattern for final model logging
    if oof_calibrator is not None:
        final_pipeline = OOFCalibratedModel(
            base_model=final_pipeline,
            calibrator=oof_calibrator,
        )
        logger.info(f"Final model wrapped with OOF calibrator (method={oof_calibrator.method})")
    elif (
        calibration_strategy.requires_per_fold_calibration()
        and not calibration_strategy.should_skip_for_model(config.model)
    ):
        logger.info(f"Final model calibrated using {calibration_strategy.method()}")

    # Step 6: Extract selected proteins from final model
    try:
        final_selected_proteins = _extract_selected_proteins_from_fold(
            fitted_model=final_pipeline,
            model_name=config.model,
            protein_cols=protein_cols,
            config=config,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            random_state=seed,
            nested_rfecv_result=nested_rfecv_result,
        )
        logger.info(f"Final test panel: {len(final_selected_proteins)} proteins selected")
    except Exception as e:
        logger.warning(f"Could not extract final test panel: {e}")
        final_selected_proteins = []

    # Update context
    ctx.final_pipeline = final_pipeline
    ctx.oof_preds = oof_preds
    ctx.best_params_df = best_params_df
    ctx.selected_proteins_df = selected_proteins_df
    ctx.oof_calibrator = oof_calibrator
    ctx.nested_rfecv_result = nested_rfecv_result
    ctx.final_selected_proteins = final_selected_proteins
    ctx.cv_elapsed_sec = elapsed_sec

    return ctx
