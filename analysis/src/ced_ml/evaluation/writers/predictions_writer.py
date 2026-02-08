"""Predictions writer for saving test, validation, train OOF, and controls predictions."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ced_ml.evaluation.reports import OutputDirectories

logger = logging.getLogger(__name__)


class PredictionsWriter:
    """Write predictions to disk (test, val, train OOF, controls)."""

    def __init__(self, output_dirs: "OutputDirectories"):
        """
        Initialize PredictionsWriter.

        Args:
            output_dirs: OutputDirectories instance
        """
        self.dirs = output_dirs

    def save_test_predictions(self, predictions_df: pd.DataFrame, model: str) -> str:
        """
        Save test predictions to preds/test_preds__{model}.csv.

        Args:
            predictions_df: DataFrame with ID, y_true, predictions
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"test_preds__{model}.csv"
        path = self.dirs.get_path("preds_test", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved test predictions: {path}")
        return str(path)

    def save_val_predictions(self, predictions_df: pd.DataFrame, model: str) -> str:
        """
        Save validation predictions to preds/val_preds__{model}.csv.

        Args:
            predictions_df: DataFrame with ID, y_true, predictions
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"val_preds__{model}.csv"
        path = self.dirs.get_path("preds_val", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved validation predictions: {path}")
        return str(path)

    def save_train_oof_predictions(self, predictions_df: pd.DataFrame, model: str) -> str:
        """
        Save train out-of-fold predictions to preds/train_oof__{model}.csv.

        Args:
            predictions_df: DataFrame with ID, y_true, OOF predictions
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"train_oof__{model}.csv"
        path = self.dirs.get_path("preds_train_oof", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved train OOF predictions: {path}")
        return str(path)

    def save_controls_predictions(self, predictions_df: pd.DataFrame, model: str) -> str:
        """
        Save control subjects predictions to preds/controls/controls_risk__{model}__oof_mean.csv.

        Args:
            predictions_df: DataFrame with control subject predictions
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"controls_risk__{model}__oof_mean.csv"
        path = self.dirs.get_path("preds_controls", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        predictions_df.to_csv(path, index=False)
        logger.info(f"Saved controls predictions: {path}")
        return str(path)
