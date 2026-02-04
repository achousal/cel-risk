"""
Training orchestration package.

This package provides a staged approach to model training, breaking down
the monolithic run_train() function into discrete, testable stages.

Stages:
1. data_stage: Load data, resolve columns, apply filters
2. split_stage: Load/validate split indices
3. feature_stage: Feature selection strategy dispatch
4. training_stage: Nested CV, OOF calibration
5. evaluation_stage: Evaluate on train/val/test, bootstrap CIs
6. plotting_stage: All plot generation calls
7. persistence_stage: All file writes, metadata JSON
"""

from ced_ml.cli.orchestration.context import TrainingContext
from ced_ml.cli.orchestration.data_stage import load_data
from ced_ml.cli.orchestration.evaluation_stage import evaluate_splits
from ced_ml.cli.orchestration.feature_stage import prepare_features
from ced_ml.cli.orchestration.persistence_stage import save_artifacts
from ced_ml.cli.orchestration.plotting_stage import generate_plots
from ced_ml.cli.orchestration.split_stage import load_splits
from ced_ml.cli.orchestration.training_stage import train_models

__all__ = [
    "TrainingContext",
    "load_data",
    "load_splits",
    "prepare_features",
    "train_models",
    "evaluate_splits",
    "save_artifacts",
    "generate_plots",
]
