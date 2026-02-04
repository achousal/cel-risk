"""Utility functions for stacking ensemble operations.

This module provides helper functions for:
- Loading base model predictions and calibration information
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationInfo:
    """Container for calibration information from a base model.

    Attributes:
        strategy: Calibration strategy ('none', 'per_fold', 'oof_posthoc')
        method: Calibration method ('isotonic', 'sigmoid', or None)
        oof_calibrator: Optional fitted OOF calibrator object
    """

    strategy: str
    method: str | None = None
    oof_calibrator: Any = None

    @property
    def needs_posthoc_calibration(self) -> bool:
        """Return True if this model needs posthoc calibration."""
        return self.strategy == "oof_posthoc" and self.oof_calibrator is not None


def _find_model_split_dir(
    results_dir: Path,
    model_name: str,
    split_seed: int,
    run_id: str | None = None,
) -> Path:
    """Find the split directory for a model.

    Primary layout: results_dir/run_{run_id}/{model}/splits/split_seed{N}/

    Searches in order of preference:
    1. results_dir/run_{run_id}/{model}/splits/split_seed{N} (explicit run_id)
    2. results_dir/run_*/{model}/splits/split_seed{N} (auto-discover run_id)

    Args:
        results_dir: Root results directory
        model_name: Model name (e.g., 'LR_EN')
        split_seed: Split seed number
        run_id: Optional run_id to target specific run

    Returns:
        Path to the split directory containing model outputs

    Raises:
        FileNotFoundError: If no matching directory found
    """
    # Pattern 1: Explicit run_id
    if run_id is not None:
        candidate = (
            results_dir / f"run_{run_id}" / model_name / "splits" / f"split_seed{split_seed}"
        )
        if candidate.exists():
            return candidate

    # Pattern 2: Auto-discover run directories (prefer most recent)
    run_dirs = sorted(results_dir.glob("run_*"), reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / model_name / "splits" / f"split_seed{split_seed}"
        if candidate.exists():
            logger.debug(f"Auto-discovered run directory: {run_dir.name}")
            return candidate

    searched = [
        f"{results_dir}/run_{run_id or '*'}/{model_name}/splits/split_seed{split_seed}",
    ]
    raise FileNotFoundError(
        f"Could not find split directory for {model_name} seed {split_seed}. Searched: {searched}"
    )


def load_base_model_calibration_info(
    results_dir: Path,
    model_name: str,
    split_seed: int,
    run_id: str | None = None,
) -> CalibrationInfo:
    """Load calibration info from a saved model bundle.

    Args:
        results_dir: Root results directory
        model_name: Model name
        split_seed: Split seed
        run_id: Optional run_id

    Returns:
        CalibrationInfo with strategy and optional OOF calibrator
    """
    model_dir = _find_model_split_dir(results_dir, model_name, split_seed, run_id)
    model_path = model_dir / "core" / f"{model_name}__final_model.joblib"

    if not model_path.exists():
        logger.warning(f"Model bundle not found: {model_path}, assuming no calibration")
        return CalibrationInfo(strategy="none")

    bundle = joblib.load(model_path)
    calib_info = bundle.get("calibration", {})

    strategy = calib_info.get("strategy", "none")
    method = calib_info.get("method")
    oof_calibrator = calib_info.get("oof_calibrator")

    logger.debug(
        f"Loaded calibration info for {model_name}: strategy={strategy}, "
        f"method={method}, has_oof_calibrator={oof_calibrator is not None}"
    )

    return CalibrationInfo(
        strategy=strategy,
        method=method,
        oof_calibrator=oof_calibrator,
    )


def apply_calibration_to_predictions(
    predictions: np.ndarray,
    calib_info: CalibrationInfo,
    model_name: str,
) -> np.ndarray:
    """Apply calibration to predictions based on calibration strategy.

    For 'per_fold': predictions are already calibrated, return as-is
    For 'oof_posthoc': apply the OOF calibrator
    For 'none': return as-is

    Args:
        predictions: Raw predictions array
        calib_info: CalibrationInfo from the base model
        model_name: Model name (for logging)

    Returns:
        Calibrated predictions
    """
    if calib_info.needs_posthoc_calibration:
        logger.debug(f"Applying OOF calibrator to {model_name} predictions")
        return calib_info.oof_calibrator.transform(predictions)

    # per_fold or none: predictions are already in correct form
    return predictions


def load_calibration_info_for_models(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    run_id: str | None = None,
) -> dict[str, CalibrationInfo]:
    """Load calibration info for multiple base models.

    Args:
        results_dir: Root results directory
        base_models: List of model names
        split_seed: Split seed
        run_id: Optional run_id

    Returns:
        Dict mapping model name to CalibrationInfo
    """
    calib_dict = {}
    for model_name in base_models:
        calib_dict[model_name] = load_base_model_calibration_info(
            results_dir, model_name, split_seed, run_id
        )
    return calib_dict


def _validate_indices_match(
    reference_idx: np.ndarray,
    reference_model: str,
    current_idx: np.ndarray,
    current_model: str,
    context: str,
) -> None:
    """Validate that indices from two models match exactly.

    Args:
        reference_idx: Indices from the reference (first) model
        reference_model: Name of the reference model
        current_idx: Indices from the current model being checked
        current_model: Name of the current model
        context: Description of the context (e.g., "OOF predictions", "test predictions")

    Raises:
        ValueError: If indices do not match (different length or different values)
    """
    if len(reference_idx) != len(current_idx):
        raise ValueError(
            f"Index length mismatch in {context}: "
            f"{reference_model} has {len(reference_idx)} samples, "
            f"{current_model} has {len(current_idx)} samples. "
            f"Base models must be trained on the same data split."
        )

    if not np.array_equal(reference_idx, current_idx):
        # Find first mismatching position for helpful error message
        mismatch_mask = reference_idx != current_idx
        first_mismatch_pos = np.argmax(mismatch_mask)
        raise ValueError(
            f"Index mismatch in {context}: "
            f"{reference_model} and {current_model} have different sample indices. "
            f"First mismatch at position {first_mismatch_pos}: "
            f"{reference_model}={reference_idx[first_mismatch_pos]}, "
            f"{current_model}={current_idx[first_mismatch_pos]}. "
            f"Base models must be trained on the same data split."
        )


def collect_oof_predictions(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    run_id: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
    """Collect OOF predictions from trained base models.

    Args:
        results_dir: Root results directory
        base_models: List of base model names to collect
        split_seed: Split seed to identify correct subdirectory
        run_id: Optional run_id to target specific run

    Returns:
        oof_dict: Dict mapping model name to OOF predictions
        y_train: Training labels (from first model)
        train_idx: Training indices (from first model)
        category: Category labels (Controls/Incident/Prevalent), or None if not available

    Raises:
        FileNotFoundError: If OOF predictions file not found for any model
        ValueError: If base models have mismatched indices (trained on different splits)
    """
    oof_dict = {}
    y_train = None
    train_idx = None
    category = None
    reference_model = None

    for model_name in base_models:
        # Look for OOF predictions file using flexible path discovery
        model_dir = _find_model_split_dir(results_dir, model_name, split_seed, run_id)

        # Flat preds directory structure
        oof_path = model_dir / "preds" / f"train_oof__{model_name}.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

        # Load OOF predictions
        oof_df = pd.read_csv(oof_path)

        # Extract predictions (may have multiple repeat columns)
        prob_cols = [c for c in oof_df.columns if c.startswith("y_prob")]
        if not prob_cols:
            raise ValueError(f"No probability columns found in {oof_path}")

        # Stack all repeat predictions
        preds = oof_df[prob_cols].values.T  # (n_repeats x n_samples)
        oof_dict[model_name] = preds

        current_idx = oof_df["idx"].values

        # Get labels, indices, and category from first model, validate subsequent models match
        if y_train is None:
            y_train = oof_df["y_true"].values
            train_idx = current_idx
            # Load category if available (Controls/Incident/Prevalent)
            if "category" in oof_df.columns:
                category = oof_df["category"].values
            reference_model = model_name
        else:
            # Validate that this model's indices match the reference model
            _validate_indices_match(
                train_idx, reference_model, current_idx, model_name, "OOF predictions"
            )

        logger.info(f"Loaded OOF predictions for {model_name}: shape {preds.shape}")

    return oof_dict, y_train, train_idx, category


def collect_split_predictions(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    split_name: str = "test",
    run_id: str | None = None,
    calibration_info: dict[str, CalibrationInfo] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
    """Collect validation or test predictions from trained base models.

    Args:
        results_dir: Root results directory
        base_models: List of base model names
        split_seed: Split seed to identify correct subdirectory
        split_name: 'val' or 'test'
        run_id: Optional run_id to target specific run
        calibration_info: Optional dict mapping model name to CalibrationInfo.
                          If provided, applies calibration based on each model's strategy.

    Returns:
        preds_dict: Dict mapping model name to predictions (calibrated if applicable)
        y_true: True labels
        indices: Sample indices
        category: Category labels (Controls/Incident/Prevalent), or None if not available

    Raises:
        FileNotFoundError: If predictions file not found for any model
        ValueError: If base models have mismatched indices (trained on different splits)
    """
    preds_dict = {}
    y_true = None
    indices = None
    category = None
    reference_model = None

    for model_name in base_models:
        model_dir = _find_model_split_dir(results_dir, model_name, split_seed, run_id)

        # Predictions are stored directly in preds/ directory, not in subdirectories
        if split_name == "val":
            pred_path = model_dir / "preds" / f"val_preds__{model_name}.csv"
        else:
            pred_path = model_dir / "preds" / f"test_preds__{model_name}.csv"

        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_path}")

        pred_df = pd.read_csv(pred_path)
        raw_preds = pred_df["y_prob"].values
        current_idx = pred_df["idx"].values

        # Apply calibration if info provided and model has oof_posthoc strategy
        if calibration_info and model_name in calibration_info:
            calib_info = calibration_info[model_name]
            preds_dict[model_name] = apply_calibration_to_predictions(
                raw_preds, calib_info, model_name
            )
        else:
            preds_dict[model_name] = raw_preds

        # Get labels, indices, and category from first model, validate subsequent models match
        if y_true is None:
            y_true = pred_df["y_true"].values
            indices = current_idx
            # Load category if available (Controls/Incident/Prevalent)
            if "category" in pred_df.columns:
                category = pred_df["category"].values
            reference_model = model_name
        else:
            # Validate that this model's indices match the reference model
            _validate_indices_match(
                indices,
                reference_model,
                current_idx,
                model_name,
                f"{split_name} predictions",
            )

        logger.info(f"Loaded {split_name} predictions for {model_name}")

    return preds_dict, y_true, indices, category
