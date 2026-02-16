"""Feature writer for saving feature reports, panels, and subgroup metrics."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ced_ml.evaluation.reports import OutputDirectories

logger = logging.getLogger(__name__)


class FeatureWriter:
    """Write feature reports, panels, and subgroup metrics to disk."""

    def __init__(self, output_dirs: "OutputDirectories"):
        """
        Initialize FeatureWriter.

        Args:
            output_dirs: OutputDirectories instance
        """
        self.dirs = output_dirs

    def save_feature_report(self, report_df: pd.DataFrame, model: str) -> str:
        """
        Save feature importance report to panels/{model}__feature_report_train.csv.

        Args:
            report_df: DataFrame with protein, effect_size, p_value, selection_freq
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__feature_report_train.csv"
        path = self.dirs.get_path("panels_features", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        report_df.to_csv(path, index=False)
        logger.info(f"Saved feature report: {path}")
        return str(path)

    def save_stable_panel_report(self, panel_df: pd.DataFrame, panel_type: str = "KBest") -> str:
        """
        Save stable panel report to panels/stable_panel__{panel_type}.csv.

        Args:
            panel_df: DataFrame with stable panel proteins and statistics
            panel_type: Panel type (e.g., "KBest", "screening")

        Returns:
            Path to saved file
        """
        filename = f"stable_panel__{panel_type}.csv"
        path = self.dirs.get_path("panels_stable", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        panel_df.to_csv(path, index=False)
        logger.info(f"Saved stable panel report: {path}")
        return str(path)

    def save_final_test_panel(
        self,
        panel_proteins: list[str],
        scenario: str,
        model: str,
        metadata: dict[str, Any] = None,
    ) -> str:
        """
        Save final test panel (proteins used in final model) to panels/{model}__final_test_panel.json.

        Args:
            panel_proteins: List of protein names selected in final model
            scenario: Scenario name
            model: Model name
            metadata: Optional metadata dict (e.g., selection_method, n_train, timestamp)

        Returns:
            Path to saved file
        """
        manifest = {
            "scenario": scenario,
            "model": model,
            "panel_size": len(panel_proteins),
            "proteins": sorted(panel_proteins),
            "metadata": metadata or {},
        }
        filename = f"{model}__final_test_panel.json"
        path = self.dirs.get_path("panels_sizes", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info(f"Saved final test panel: {path}")
        return str(path)

    def save_subgroup_metrics(self, subgroup_df: pd.DataFrame, model: str) -> str:
        """
        Save subgroup analysis to panels/{model}__test_subgroup_metrics.csv.

        Args:
            subgroup_df: DataFrame with per-subgroup metrics
            model: Model name

        Returns:
            Path to saved file
        """
        filename = f"{model}__test_subgroup_metrics.csv"
        path = self.dirs.get_path("panels_subgroups", filename)
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        subgroup_df.to_csv(path, index=False)
        logger.info(f"Saved subgroup metrics: {path}")
        return str(path)
