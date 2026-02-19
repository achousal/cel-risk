"""Configuration schema for CeD-ML pipeline.

This module is a facade that re-exports all configuration dataclasses
from their themed sub-modules for backward compatibility.

All imports like `from ced_ml.config.schema import TrainingConfig` continue to work.
"""

# Data and splits
# Aggregation, panel optimization, holdout
from ced_ml.config.aggregate_schema import (
    AggregateConfig,
    HoldoutEvalConfig,
    PanelOptimizeConfig,
)

# Calibration and thresholds
from ced_ml.config.calibration_schema import CalibrationConfig, ThresholdConfig

# Compute and HPC
from ced_ml.config.compute_schema import (
    ComputeConfig,
    HPCConfig,
    HPCResourceConfig,
    OrchestratorConfig,
)

# Cross-validation and Optuna
from ced_ml.config.cv_schema import CVConfig, OptunaConfig
from ced_ml.config.data_schema import ColumnsConfig, SplitsConfig

# Ensemble
from ced_ml.config.ensemble_schema import EnsembleConfig

# Evaluation and DCA
from ced_ml.config.evaluation_schema import DCAConfig, EvaluationConfig

# Feature selection
from ced_ml.config.features_schema import FeatureConfig

# Model hyperparameters
from ced_ml.config.model_schema import LRConfig, RFConfig, SVMConfig, XGBoostConfig

# Output and strictness
from ced_ml.config.output_schema import OutputConfig, StrictnessConfig

# Permutation testing
from ced_ml.config.permutation_schema import (
    PermutationAggregationConfig,
    PermutationTestConfig,
)

# Root config
from ced_ml.config.root_schema import RootConfig

# Training (main config)
from ced_ml.config.training_schema import TrainingConfig

__all__ = [
    # Data and splits
    "ColumnsConfig",
    "SplitsConfig",
    # Cross-validation and Optuna
    "CVConfig",
    "OptunaConfig",
    # Feature selection
    "FeatureConfig",
    # Model hyperparameters
    "LRConfig",
    "SVMConfig",
    "RFConfig",
    "XGBoostConfig",
    # Calibration and thresholds
    "CalibrationConfig",
    "ThresholdConfig",
    # Ensemble
    "EnsembleConfig",
    # Evaluation and DCA
    "EvaluationConfig",
    "DCAConfig",
    # Output and strictness
    "OutputConfig",
    "StrictnessConfig",
    # Permutation testing
    "PermutationTestConfig",
    "PermutationAggregationConfig",
    # Compute and HPC
    "ComputeConfig",
    "HPCResourceConfig",
    "OrchestratorConfig",
    "HPCConfig",
    # Training (main config)
    "TrainingConfig",
    # Aggregation, panel optimization, holdout
    "AggregateConfig",
    "PanelOptimizeConfig",
    "HoldoutEvalConfig",
    # Root config
    "RootConfig",
]
