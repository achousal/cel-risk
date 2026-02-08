"""Diagnostics writer for saving calibration curves, learning curves, and split traces."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ced_ml.evaluation.reports import OutputDirectories

logger = logging.getLogger(__name__)


class DiagnosticsWriter:
    """Write diagnostic artifacts to disk (calibration, learning curves, split traces)."""

    def __init__(self, output_dirs: "OutputDirectories"):
        """
        Initialize DiagnosticsWriter.

        Args:
            output_dirs: OutputDirectories instance
        """
        self.dirs = output_dirs

    def save_calibration_curve(self, calibration_df: pd.DataFrame, model: str) -> str:
        """
        Save calibration curve data to diagnostics/calibration/{model}__calibration.csv.

        Args:
            calibration_df: DataFrame with prob_pred, prob_true columns
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__calibration.csv"
        path = self.dirs.get_path("diag_calibration", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        calibration_df.to_csv(path, index=False)
        logger.debug(f"Saved calibration curve: {path}")
        return str(path)

    def save_learning_curve(self, learning_curve_df: pd.DataFrame, model: str) -> str:
        """
        Save learning curve data to diagnostics/learning/{model}__learning_curve.csv.

        Args:
            learning_curve_df: DataFrame with train_size, metric, train/test scores
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__learning_curve.csv"
        path = self.dirs.get_path("diag_learning", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        learning_curve_df.to_csv(path, index=False)
        logger.debug(f"Saved learning curve: {path}")
        return str(path)

    def save_split_trace(self, split_trace_df: pd.DataFrame) -> str:
        """
        Save split trace to diagnostics/splits/train_test_split_trace.csv.

        Args:
            split_trace_df: DataFrame with split assignments and labels

        Returns:
            Path to saved file
        """
        filename = "train_test_split_trace.csv"
        path = self.dirs.get_path("diag_splits", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        split_trace_df.to_csv(path, index=False)
        logger.debug(f"Saved split trace: {path}")
        return str(path)
