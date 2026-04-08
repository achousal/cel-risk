"""Config overlay generator for sweep iterations.

The single modification point: takes a base cell config + sweep parameters
and produces a merged config. No core pipeline code is ever modified.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from ced_ml.config.loader import _deep_merge, load_yaml

logger = logging.getLogger(__name__)


def resolve_base_config(
    recipes_dir: Path,
    recipe_id: str,
    cell_name: str,
) -> tuple[Path, Path]:
    """Resolve base training + pipeline config paths from recipe/cell names.

    Parameters
    ----------
    recipes_dir
        Root of derived recipes (e.g. analysis/configs/recipes/).
    recipe_id
        Recipe identifier (e.g. R1_plateau).
    cell_name
        Cell directory name (e.g. LinSVM_cal_logistic_intercept_log_ds1).

    Returns
    -------
    (training_config_path, pipeline_config_path)
    """
    cell_dir = recipes_dir / recipe_id / cell_name
    training = cell_dir / "training_config.yaml"
    pipeline = cell_dir / "pipeline_hpc.yaml"
    if not training.exists():
        raise FileNotFoundError(
            f"Base training config not found: {training}. "
            f"Run 'ced derive-recipes' first."
        )
    if not pipeline.exists():
        raise FileNotFoundError(f"Base pipeline config not found: {pipeline}")
    return training, pipeline


def generate_overlay(
    base_training_path: Path,
    base_pipeline_path: Path,
    params: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate merged config files for one sweep iteration.

    Parameters
    ----------
    base_training_path
        Path to base training_config.yaml.
    base_pipeline_path
        Path to base pipeline_hpc.yaml.
    params
        Sweep parameter dict to overlay. Keys map to nested YAML paths
        using dot notation (e.g. 'data.downsample_majority_ratio' -> {'data': {'downsample_majority_ratio': value}}).
    output_dir
        Directory to write merged configs into.

    Returns
    -------
    (training_config_path, pipeline_config_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_training = load_yaml(base_training_path)
    base_pipeline = load_yaml(base_pipeline_path)

    # Convert dot-notation params to nested dict
    training_overlay = _params_to_nested(params)

    merged_training = _deep_merge(base_training, training_overlay)

    training_out = output_dir / "training_config.yaml"
    pipeline_out = output_dir / "pipeline_hpc.yaml"

    _write_yaml(merged_training, training_out)
    _write_yaml(base_pipeline, pipeline_out)

    logger.info("Generated overlay configs in %s (params: %s)", output_dir, params)
    return training_out, pipeline_out


def _params_to_nested(params: dict[str, Any]) -> dict[str, Any]:
    """Convert dot-notation keys to nested dicts.

    Example: {'data.downsample_majority_ratio': 5.0}
    becomes: {'data': {'downsample_majority_ratio': 5.0}}
    """
    nested: dict[str, Any] = {}
    for key, value in params.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write YAML with consistent formatting."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
