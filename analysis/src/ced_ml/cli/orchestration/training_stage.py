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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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

# HP parameter key for k inside ProteinOnlySelector wrapper
_SEL_K_PARAM = "sel__selector__k"


def _extract_k_from_best_params(best_params_df: pd.DataFrame) -> pd.Series | None:
    """Extract per-fold k values from the JSON ``best_params`` column.

    The nested CV loop stores hyperparameters as a JSON string. This helper
    parses that string and extracts the k value for each fold.

    Returns
    -------
    Series of int k values (one per fold), or None if k was not tuned.
    """
    if "best_params" not in best_params_df.columns:
        return None

    k_values = []
    for _, row in best_params_df.iterrows():
        try:
            params = json.loads(row["best_params"])
        except (json.JSONDecodeError, TypeError):
            params = {}
        k = params.get(_SEL_K_PARAM)
        if k is not None:
            k_values.append(int(k))

    if not k_values:
        return None

    return pd.Series(k_values, name="k")


def _save_k_summary(
    best_params_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save k distribution + inner-CV AUROC sensitivity summary (Patch C).

    Produces ``k_summary.csv`` with columns: k, n_folds, mean_auroc, std_auroc.
    Helps reviewers assess whether the chosen k is brittle.
    """
    k_series = _extract_k_from_best_params(best_params_df)
    if k_series is None or k_series.empty:
        return

    # Parse inner-CV score alongside k
    rows = []
    for _, row in best_params_df.iterrows():
        try:
            params = json.loads(row["best_params"])
        except (json.JSONDecodeError, TypeError):
            continue
        k = params.get(_SEL_K_PARAM)
        if k is None:
            continue
        rows.append({"k": int(k), "inner_auroc": float(row.get("best_score_inner", np.nan))})

    if not rows:
        return

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("k")["inner_auroc"]
        .agg(n_folds="count", mean_auroc="mean", std_auroc="std")
        .reset_index()
        .sort_values("n_folds", ascending=False)
    )

    out_path = Path(output_dir) / "k_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info(
        "Saved k-selection summary (%d unique values) to %s",
        len(summary),
        out_path,
    )

    # Log top entries for quick inspection
    for _, r in summary.head(3).iterrows():
        logger.info(
            "  k=%d: used in %d folds, inner AUROC=%.4f +/- %.4f",
            int(r["k"]),
            int(r["n_folds"]),
            r["mean_auroc"],
            r["std_auroc"] if np.isfinite(r["std_auroc"]) else 0.0,
        )


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
        - oof_importance_df: OOF importance dataframe (if computed)
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

    cv_result = oof_predictions_with_nested_cv(
        pipeline=pipeline,
        model_name=config.model,
        X=ctx.X_train,
        y=ctx.y_train,
        protein_cols=protein_cols,
        config=config,
        random_state=seed,
        grid_rng=ctx.grid_rng,
    )

    logger.info(f"CV completed in {cv_result.elapsed_sec:.1f}s")

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

    # Determine best k value for final model (if using multi_stage with k_grid)
    # Parse k from the JSON best_params column (fixes previous bug where
    # "sel__k" was checked as a DataFrame column name but params are stored as JSON)
    if strategy == "multi_stage":
        k_series = _extract_k_from_best_params(cv_result.best_params_df)
        k_grid = getattr(config.features, "k_grid", None)

        if k_series is not None and not k_series.empty:
            if k_grid and len(k_grid) > 1:
                best_k = int(k_series.mode()[0])
                logger.info(f"Multiple k values tuned: using mode k={best_k} for final model")
            elif k_grid and len(k_grid) == 1:
                best_k = k_grid[0]
                logger.info(f"Single k value in config: using k={best_k} for final model")
            else:
                best_k = int(k_series.iloc[0])
                logger.info(f"Using k={best_k} from CV results for final model")

            final_pipeline.set_params(**{_SEL_K_PARAM: best_k})
            logger.info(f"Final model k-best set to k={best_k}")

        # Save k selection summary artifact (Patch C)
        _save_k_summary(cv_result.best_params_df, ctx.outdirs.cv)

    final_pipeline.fit(ctx.X_train, ctx.y_train)
    logger.info("Final model fitted")

    # --- Final-model SHAP (before calibration wrapping) ---
    shap_config = getattr(config.features, "shap", None)
    if shap_config and shap_config.enabled and shap_config.compute_final_shap:
        from ced_ml.features.shap_values import compute_final_shap

        log_section(logger, "Computing SHAP Values")
        try:
            ctx.test_shap_payload = compute_final_shap(
                final_pipeline,
                config.model,
                ctx.X_test,
                ctx.y_test,
                ctx.X_train,
                shap_config,
                y_train=ctx.y_train,
            )
            logger.info(
                "Test SHAP computed: %d features, scale=%s",
                len(ctx.test_shap_payload.feature_names),
                ctx.test_shap_payload.shap_output_scale,
            )
        except Exception as e:
            logger.warning("Test SHAP failed: %s", e)
            ctx.test_shap_payload = None

        if shap_config.save_val_shap and ctx.has_validation_set:
            try:
                ctx.val_shap_payload = compute_final_shap(
                    final_pipeline,
                    config.model,
                    ctx.X_val,
                    ctx.y_val,
                    ctx.X_train,
                    shap_config,
                    split="val",
                    y_train=ctx.y_train,
                )
                logger.info("Validation SHAP computed")
            except Exception as e:
                logger.warning("Validation SHAP failed: %s", e)
                ctx.val_shap_payload = None

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
    if cv_result.oof_calibrator is not None:
        final_pipeline = OOFCalibratedModel(
            base_model=final_pipeline,
            calibrator=cv_result.oof_calibrator,
        )
        logger.info(
            f"Final model wrapped with OOF calibrator (method={cv_result.oof_calibrator.method})"
        )
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
            nested_rfecv_result=cv_result.nested_rfecv_result,
        )
        logger.info(f"Final test panel: {len(final_selected_proteins)} proteins selected")
    except Exception as e:
        logger.warning(f"Could not extract final test panel: {e}")
        final_selected_proteins = []

    # Update context
    ctx.final_pipeline = final_pipeline
    ctx.oof_preds = cv_result.preds
    ctx.best_params_df = cv_result.best_params_df
    ctx.selected_proteins_df = cv_result.selected_proteins_df
    ctx.oof_calibrator = cv_result.oof_calibrator
    ctx.nested_rfecv_result = cv_result.nested_rfecv_result
    ctx.oof_importance_df = cv_result.oof_importance_df
    ctx.oof_shap_df = cv_result.oof_shap_df
    ctx.final_selected_proteins = final_selected_proteins
    ctx.cv_elapsed_sec = cv_result.elapsed_sec

    return ctx
