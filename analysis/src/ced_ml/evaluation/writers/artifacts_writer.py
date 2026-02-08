"""Artifacts writer for saving model artifacts, best params, selected proteins, and settings."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

if TYPE_CHECKING:
    from ced_ml.evaluation.reports import OutputDirectories

logger = logging.getLogger(__name__)


class ArtifactsWriter:
    """Write model artifacts, hyperparameters, selected features, and run settings."""

    def __init__(self, output_dirs: "OutputDirectories"):
        """
        Initialize ArtifactsWriter.

        Args:
            output_dirs: OutputDirectories instance
        """
        self.dirs = output_dirs

    def save_run_settings(self, settings: dict[str, Any]) -> str:
        """
        Save run configuration to core/run_settings.json.

        Args:
            settings: Configuration dictionary (typically from argparse.Namespace)

        Returns:
            Path to saved file
        """
        path = self.dirs.get_path("core", "run_settings.json")
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        with open(path, "w") as f:
            json.dump(settings, f, indent=2, sort_keys=True)
        logger.info(f"Saved run settings: {path}")
        return str(path)

    def save_best_params_per_split(self, best_params: list[dict[str, Any]], scenario: str) -> str:
        """
        Save best hyperparameters per outer split to cv/best_params_per_split.csv.

        Args:
            best_params: List of best param dicts (one per outer fold)
            scenario: Scenario name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame(best_params)
        df.insert(0, "scenario", scenario)

        path = self.dirs.get_path("cv", "best_params_per_split.csv")
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        df.to_csv(path, index=False)
        logger.info(f"Saved best params per split: {path}")
        return str(path)

    def save_selected_proteins_per_split(
        self, selected_proteins: list[dict[str, Any]], scenario: str
    ) -> str:
        """
        Save selected proteins per outer split to cv/selected_proteins_per_outer_split.csv.

        Args:
            selected_proteins: List of dicts with outer_split and selected_proteins_split
            scenario: Scenario name

        Returns:
            Path to saved file
        """
        df = pd.DataFrame(selected_proteins)
        df.insert(0, "scenario", scenario)

        path = self.dirs.get_path("cv", "selected_proteins_per_outer_split.csv")
        if Path(path).exists():
            logger.warning(f"Overwriting existing file: {path}")
        df.to_csv(path, index=False)
        logger.info(f"Saved selected proteins per split: {path}")
        return str(path)

    def save_model_artifact(
        self, model: Any, metadata: dict[str, Any], scenario: str, model_name: str
    ) -> str:
        """
        Save trained model artifact to core/{model}__final_model.joblib.

        Args:
            model: Trained model object (sklearn pipeline or estimator)
            metadata: Model metadata (hyperparams, thresholds, etc.)
            scenario: Scenario name
            model_name: Model name

        Returns:
            Path to saved file

        Raises:
            IOError: If serialization fails
        """
        bundle = {
            "model": model,
            "metadata": metadata,
            "scenario": scenario,
            "model_name": model_name,
        }

        filename = f"{model_name}__final_model.joblib"
        path = self.dirs.get_path("core", filename)
        final_path = Path(path)

        if final_path.exists():
            logger.warning(f"Overwriting existing model artifact: {path}")

        # Atomic write: write to temp file, then rename
        try:
            parent_dir = final_path.parent
            with tempfile.NamedTemporaryFile(
                dir=parent_dir, delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = tmp_file.name

            joblib.dump(bundle, tmp_path, compress=3)
            os.replace(tmp_path, final_path)
            logger.info(f"Saved model artifact: {path}")
            return str(path)
        except Exception as e:
            # Clean up temp file if it exists
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink()
            logger.error(f"Failed to save model artifact: {e}")
            raise OSError(f"Model serialization failed: {e}") from e

    def load_model_artifact(self, model_name: str) -> dict[str, Any]:
        """
        Load trained model artifact from core/{model}__final_model.joblib.

        Args:
            model_name: Model name

        Returns:
            Bundle dictionary with keys: model, metadata, scenario, model_name

        Raises:
            FileNotFoundError: If artifact does not exist
            IOError: If deserialization fails
        """
        filename = f"{model_name}__final_model.joblib"
        path = self.dirs.get_path("core", filename)

        if not Path(path).exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        try:
            bundle = joblib.load(path)
            logger.info(f"Loaded model artifact: {path}")
            return bundle
        except Exception as e:
            logger.error(f"Failed to load model artifact: {e}")
            raise OSError(f"Model deserialization failed: {e}") from e
