"""
TrainingContext dataclass for sharing state across training stages.

This module defines the central state container that is passed through
each stage of the training pipeline. It provides a clean interface for
stage functions to access and update shared state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from ced_ml.config.schema import TrainingConfig
    from ced_ml.data.columns import ResolvedColumns
    from ced_ml.evaluation.reports import OutputDirectories
    from ced_ml.features.nested_rfe import NestedRFECVResult
    from ced_ml.models.calibration import IsotonicCalibrator, SigmoidCalibrator


@dataclass
class TrainingContext:
    """Central state container for training pipeline stages.

    This dataclass holds all shared state that flows through the training
    pipeline stages. Each stage function takes a context, reads what it needs,
    and returns an updated context with new state.

    Attributes:
        config: Training configuration object
        cli_args: CLI arguments dictionary
        log_level: Logging level constant

        # Data state (set by data_stage)
        df_filtered: Filtered DataFrame after row filters applied
        resolved: Resolved column configuration
        protein_cols: List of protein column names
        filter_stats: Dictionary of filter statistics

        # Split state (set by split_stage)
        seed: Random seed for reproducibility
        run_id: Run identifier (timestamp-based or user-provided)
        outdirs: OutputDirectories instance
        train_idx: Training set indices
        val_idx: Validation set indices
        test_idx: Test set indices
        scenario: Scenario name (e.g., "IncidentOnly")

        # Feature state (set by feature_stage)
        feature_cols: Final feature columns for training
        fixed_panel_proteins: Fixed panel proteins (if using fixed panel mode)
        fixed_panel_path: Path to fixed panel CSV (if applicable)

        # Training data (set by feature_stage)
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        X_test, y_test: Test features and labels
        cat_train, cat_val, cat_test: Category labels for each split
        train_prev: Training prevalence

        # Model state (set by training_stage)
        final_pipeline: Fitted final model pipeline
        oof_preds: Out-of-fold predictions array
        best_params_df: DataFrame of best params per fold
        selected_proteins_df: DataFrame of selected proteins per fold
        oof_calibrator: OOF calibrator (if using oof_posthoc strategy)
        nested_rfecv_result: Nested RFECV result (if rfe enabled)
        oof_importance_df: OOF importance dataframe (if computed)
        final_selected_proteins: Proteins selected by final model
        cv_elapsed_sec: CV elapsed time in seconds

        # Evaluation state (set by evaluation_stage)
        val_metrics: Validation set metrics dictionary
        test_metrics: Test set metrics dictionary
        val_threshold: Threshold computed on validation set
        val_target_prev: Target prevalence for validation
        test_target_prev: Target prevalence for test

        # Prediction state (set by evaluation_stage)
        test_preds_df: Test predictions DataFrame
        val_preds_df: Validation predictions DataFrame
        oof_preds_df: OOF predictions DataFrame

        # Metadata (accumulated throughout)
        train_breakdown: Category breakdown for training set
        val_breakdown: Category breakdown for validation set
        test_breakdown: Category breakdown for test set
    """

    # Configuration
    config: TrainingConfig
    cli_args: dict[str, Any] = field(default_factory=dict)
    log_level: int = logging.INFO

    # Data state
    df_filtered: pd.DataFrame | None = None
    resolved: ResolvedColumns | None = None
    protein_cols: list[str] = field(default_factory=list)
    filter_stats: dict[str, Any] = field(default_factory=dict)

    # Split state
    seed: int = 0
    run_id: str = ""
    outdirs: OutputDirectories | None = None
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    scenario: str = ""

    # Feature state
    feature_cols: list[str] = field(default_factory=list)
    fixed_panel_proteins: list[str] | None = None
    fixed_panel_path: Path | None = None

    # Training data
    X_train: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    X_val: pd.DataFrame | None = None
    y_val: np.ndarray | None = None
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | None = None
    cat_train: np.ndarray | None = None
    cat_val: np.ndarray | None = None
    cat_test: np.ndarray | None = None
    train_prev: float = 0.0

    # Model state
    final_pipeline: Pipeline | None = None
    oof_preds: np.ndarray | None = None
    best_params_df: pd.DataFrame | None = None
    selected_proteins_df: pd.DataFrame | None = None
    oof_calibrator: IsotonicCalibrator | SigmoidCalibrator | None = None
    nested_rfecv_result: NestedRFECVResult | None = None
    oof_importance_df: pd.DataFrame | None = None
    oof_importance_clustered_df: pd.DataFrame | None = None
    final_selected_proteins: list[str] = field(default_factory=list)
    cv_elapsed_sec: float = 0.0
    grid_rng: np.random.Generator | None = None

    # SHAP state
    oof_shap_df: pd.DataFrame | None = None
    test_shap_payload: Any = None
    val_shap_payload: Any = None

    # Evaluation state
    val_metrics: dict[str, float] | None = None
    test_metrics: dict[str, float] | None = None
    val_threshold: float | None = None
    test_threshold: float | None = (
        None  # Threshold computed on test set (fallback if no validation set)
    )
    val_target_prev: float = 0.0
    test_target_prev: float = 0.0

    # Prediction state
    test_preds_df: pd.DataFrame | None = None
    val_preds_df: pd.DataFrame | None = None
    oof_preds_df: pd.DataFrame | None = None

    # Metadata
    train_breakdown: dict[str, int] = field(default_factory=dict)
    val_breakdown: dict[str, int] = field(default_factory=dict)
    test_breakdown: dict[str, int] = field(default_factory=dict)

    # Scenario-filtered DataFrame (after applying scenario labels)
    df_scenario: pd.DataFrame | None = None

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        cli_args: dict[str, Any] | None = None,
        log_level: int = logging.INFO,
    ) -> TrainingContext:
        """Create a TrainingContext from a config and CLI arguments.

        Args:
            config: Validated TrainingConfig object
            cli_args: Optional dictionary of CLI arguments
            log_level: Logging level constant

        Returns:
            Initialized TrainingContext
        """
        seed = getattr(config, "split_seed", getattr(config, "seed", 0))
        run_id = getattr(config, "run_id", None)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create grid RNG if grid randomization is enabled
        grid_rng = np.random.default_rng(seed) if config.cv.grid_randomize else None

        return cls(
            config=config,
            cli_args=cli_args or {},
            log_level=log_level,
            seed=seed,
            run_id=run_id,
            grid_rng=grid_rng,
        )

    @property
    def has_validation_set(self) -> bool:
        """Check if validation set exists."""
        return len(self.val_idx) > 0

    @property
    def total_cv_folds(self) -> int:
        """Get total number of CV folds."""
        return self.config.cv.folds * self.config.cv.repeats

    def get_run_level_dir(self) -> Path:
        """Extract run-level directory from output directory structure.

        Layout with seed: .../run_{id}/{model}/splits/split_seed{N}/ -> go up 3 levels
        Layout without seed: .../run_{id}/{model}/ -> go up 1 level
        """
        if self.outdirs is None:
            raise ValueError("outdirs not set - call load_splits first")
        outdirs_root = Path(self.outdirs.root)
        if self.seed is not None:
            return outdirs_root.parent.parent.parent
        return outdirs_root.parent
