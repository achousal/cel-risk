"""
Evaluation stage for training orchestration.

This module handles:
- Validation set evaluation (threshold selection)
- Test set evaluation (reuse val threshold)
- Prediction DataFrame creation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ced_ml.cli.train import evaluate_on_split
from ced_ml.data.schema import METRIC_AUROC, METRIC_PRAUC
from ced_ml.models.prevalence import adjust_probabilities_for_prevalence
from ced_ml.utils.logging import log_section

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def evaluate_splits(ctx: TrainingContext) -> TrainingContext:
    """Evaluate model on validation and test sets.

    This stage:
    1. Evaluates on validation set (computes threshold)
    2. Evaluates on test set (reuses validation threshold)
    3. Creates prediction DataFrames for all splits
    4. Validates predictions (NaN, Inf, bounds)

    Args:
        ctx: TrainingContext with trained model

    Returns:
        Updated TrainingContext with:
        - val_metrics, test_metrics: Metric dictionaries
        - val_threshold: Validation threshold (for reuse on test)
        - val_target_prev, test_target_prev: Target prevalences
        - test_preds_df, val_preds_df, oof_preds_df: Prediction DataFrames
    """
    config = ctx.config
    final_pipeline = ctx.final_pipeline
    train_prev = ctx.train_prev

    # Step 1: Evaluate on validation set
    log_section(logger, "Validation Set Evaluation")

    if not ctx.has_validation_set:
        if not config.allow_test_thresholding:
            raise ValueError(
                "No validation set available (val_size=0) and threshold-on-test not explicitly allowed. "
                "Set allow_test_thresholding=True in config to proceed (not recommended for production)."
            )
        logger.warning("No validation set available (val_size=0). Skipping validation evaluation.")
        logger.warning("Threshold will be computed on test set (allow_test_thresholding=True).")
        val_metrics = None
        val_threshold = None
        val_target_prev = train_prev
    else:
        # Determine target prevalence for validation
        val_target_prev = _get_target_prevalence(
            config.thresholds.target_prevalence_source,
            config.thresholds.target_prevalence_fixed,
            train_prev,
            ctx.y_val,
            ctx.y_test,
        )

        val_metrics = evaluate_on_split(
            final_pipeline,
            ctx.X_val,
            ctx.y_val,
            train_prev,
            val_target_prev,
            config,
        )

        logger.info(f"Val AUROC: {val_metrics[METRIC_AUROC]:.3f}")
        logger.info(f"Val PRAUC: {val_metrics[METRIC_PRAUC]:.3f}")
        logger.info(f"Selected threshold: {val_metrics['threshold']:.3f}")
        if ctx.final_selected_proteins:
            logger.info(
                f"Val evaluation using {len(ctx.final_selected_proteins)} selected proteins"
            )

        val_threshold = val_metrics["threshold"]

    # Step 2: Evaluate on test set
    log_section(logger, "Test Set Evaluation")

    # Determine target prevalence for test
    test_target_prev = _get_target_prevalence(
        config.thresholds.target_prevalence_source,
        config.thresholds.target_prevalence_fixed,
        train_prev,
        ctx.y_val,
        ctx.y_test,
    )

    # Reuse validation threshold on test set
    test_metrics = evaluate_on_split(
        final_pipeline,
        ctx.X_test,
        ctx.y_test,
        train_prev,
        test_target_prev,
        config,
        precomputed_threshold=val_threshold,
    )

    logger.info(f"Test AUROC: {test_metrics[METRIC_AUROC]:.3f}")
    logger.info(f"Test PRAUC: {test_metrics[METRIC_PRAUC]:.3f}")
    if val_threshold is not None:
        logger.info(f"Test threshold (from val): {test_metrics['threshold']:.3f}")
    else:
        logger.info(f"Test threshold (computed on test): {test_metrics['threshold']:.3f}")
    if ctx.final_selected_proteins:
        logger.info(f"Test evaluation using {len(ctx.final_selected_proteins)} selected proteins")

    # Step 3: Create prediction DataFrames
    # Test predictions
    test_probs_raw = final_pipeline.predict_proba(ctx.X_test)[:, 1]
    test_probs_adj = adjust_probabilities_for_prevalence(
        test_probs_raw, sample_prev=train_prev, target_prev=test_target_prev
    )
    test_preds_df = pd.DataFrame(
        {
            "idx": ctx.test_idx,
            "y_true": ctx.y_test,
            "y_prob": test_probs_raw,
            "y_prob_adjusted": test_probs_adj,
            "category": ctx.cat_test,
        }
    )

    # Validate test predictions
    _validate_predictions(test_preds_df["y_prob"].values, "Test")

    # Val predictions (if validation set exists)
    if ctx.has_validation_set:
        val_probs_raw = final_pipeline.predict_proba(ctx.X_val)[:, 1]
        val_probs_adj = adjust_probabilities_for_prevalence(
            val_probs_raw, sample_prev=train_prev, target_prev=val_target_prev
        )
        val_preds_df = pd.DataFrame(
            {
                "idx": ctx.val_idx,
                "y_true": ctx.y_val,
                "y_prob": val_probs_raw,
                "y_prob_adjusted": val_probs_adj,
                "category": ctx.cat_val,
            }
        )
        _validate_predictions(val_preds_df["y_prob"].values, "Val")
    else:
        val_preds_df = pd.DataFrame(
            columns=["idx", "y_true", "y_prob", "y_prob_adjusted", "category"]
        )

    # OOF predictions
    oof_preds_df = pd.DataFrame(
        {
            "idx": ctx.train_idx,
            "y_true": ctx.y_train,
            "category": ctx.cat_train,
        }
    )
    for repeat in range(ctx.oof_preds.shape[0]):
        oof_preds_df[f"y_prob_repeat{repeat}"] = ctx.oof_preds[repeat, :]

    _validate_predictions(ctx.oof_preds, "OOF")

    # Update context
    ctx.val_metrics = val_metrics
    ctx.test_metrics = test_metrics
    ctx.val_threshold = val_threshold
    ctx.val_target_prev = val_target_prev
    ctx.test_target_prev = test_target_prev

    ctx.test_preds_df = test_preds_df
    ctx.val_preds_df = val_preds_df
    ctx.oof_preds_df = oof_preds_df

    return ctx


def _get_target_prevalence(
    source: str,
    fixed_value: float,
    train_prev: float,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Get target prevalence based on config source."""
    if source == "fixed":
        return fixed_value
    elif source == "train":
        return train_prev
    elif source == "val":
        return float(np.asarray(y_val).mean()) if len(y_val) > 0 else train_prev
    elif source == "test":
        return float(np.asarray(y_test).mean())
    else:
        return train_prev


def _validate_predictions(probs: np.ndarray, split_name: str) -> None:
    """Validate predictions for NaN, Inf, and bounds."""
    if not np.isfinite(probs).all():
        raise ValueError(f"{split_name} predictions contain NaN or Inf values")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(
            f"{split_name} predictions out of [0,1] bounds: "
            f"min={probs.min():.4f}, max={probs.max():.4f}"
        )
