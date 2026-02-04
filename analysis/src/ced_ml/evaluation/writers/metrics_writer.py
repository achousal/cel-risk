"""Metrics writer for saving validation, test, CV, and bootstrap CI metrics."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ced_ml.evaluation.reports import OutputDirectories

logger = logging.getLogger(__name__)


class MetricsWriter:
    """Write metrics to disk (val, test, CV repeat, bootstrap CI)."""

    def __init__(self, output_dirs: "OutputDirectories"):
        """
        Initialize MetricsWriter.

        Args:
            output_dirs: OutputDirectories instance
        """
        self.dirs = output_dirs

    def save_val_metrics(self, metrics: dict[str, Any], scenario: str, model: str) -> str:
        """
        Save validation metrics to core/val_metrics.csv (append mode).

        Args:
            metrics: Validation metrics dictionary
            scenario: Scenario name (e.g., "IncidentPlusPrevalent")
            model: Model name (e.g., "LR_EN")

        Returns:
            Path to saved file
        """
        df = pd.DataFrame([metrics])
        df.insert(0, "scenario", scenario)
        df.insert(1, "model", model)

        path = self.dirs.get_path("core", "val_metrics.csv")

        if Path(path).exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved validation metrics: {path}")
        return str(path)

    def save_test_metrics(self, metrics: dict[str, Any], scenario: str, model: str) -> str:
        """
        Save test metrics to core/test_metrics.csv (append mode).

        Args:
            metrics: Test metrics dictionary
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame([metrics])
        df.insert(0, "scenario", scenario)
        df.insert(1, "model", model)

        path = self.dirs.get_path("core", "test_metrics.csv")

        if Path(path).exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved test metrics: {path}")
        return str(path)

    def save_cv_repeat_metrics(
        self, cv_results: list[dict[str, Any]], scenario: str, model: str
    ) -> str:
        """
        Save cross-validation repeat metrics to cv/cv_repeat_metrics.csv (append mode).

        Args:
            cv_results: List of metrics dictionaries (one per repeat)
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame(cv_results)
        if "scenario" not in df.columns:
            df.insert(0, "scenario", scenario)
        if "model" not in df.columns:
            df.insert(1, "model", model)

        path = self.dirs.get_path("cv", "cv_repeat_metrics.csv")

        if Path(path).exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"Saved CV repeat metrics: {path}")
        return str(path)

    def save_bootstrap_ci_metrics(self, metrics: dict[str, Any], scenario: str, model: str) -> str:
        """
        Save bootstrap CI metrics to core/test_bootstrap_ci.csv.

        Args:
            metrics: Bootstrap CI metrics dictionary
            scenario: Scenario name
            model: Model name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame([metrics])
        df.insert(0, "scenario", scenario)
        df.insert(1, "model", model)

        path = self.dirs.get_path("core", "test_bootstrap_ci.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved bootstrap CI metrics: {path}")
        return str(path)
