"""Unified discovery utilities for CLI commands.

This module provides a consistent API for discovering runs, models, and splits
across all CLI commands. It consolidates functions previously scattered across
train_ensemble.py, aggregate_splits.py, optimize_panel.py, and consensus_panel.py.

Directory layout:
    results/
      run_{RUN_ID}/
        {MODEL}/
          splits/
            split_seed{N}/
              preds/
              core/
              ...
          aggregated/
            panels/
            optimize_panel/
            ...
        consensus/
        run_metadata.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Directories to skip when scanning for models
SKIP_DIRS = frozenset({"investigations", "consensus", ".DS_Store"})


def get_results_root(results_dir: str | Path | None = None) -> Path:
    """Get the results root directory.

    Args:
        results_dir: Explicit path, or None to auto-detect.

    Auto-detection order:
        1. CED_RESULTS_DIR environment variable
        2. get_project_root() / "results"

    Returns:
        Path to results root directory.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    import os

    if results_dir is not None:
        path = Path(results_dir)
    elif "CED_RESULTS_DIR" in os.environ:
        path = Path(os.environ["CED_RESULTS_DIR"])
    else:
        from ced_ml.utils.paths import get_project_root

        path = get_project_root() / "results"

    if not path.exists():
        raise FileNotFoundError(f"Results directory not found: {path}")

    return path


def get_run_path(
    run_id: str,
    results_dir: str | Path | None = None,
) -> Path:
    """Get the path to a specific run directory.

    Args:
        run_id: Run ID (e.g., "20260127_115115").
        results_dir: Results root directory (optional).

    Returns:
        Path to run directory.

    Raises:
        FileNotFoundError: If run directory does not exist.
    """
    results_root = get_results_root(results_dir)
    run_path = results_root / f"run_{run_id}"

    if not run_path.exists():
        raise FileNotFoundError(
            f"No results found for run {run_id}.\n" f"Searched in: {results_root}"
        )

    return run_path


# -----------------------------------------------------------------------------
# Run discovery
# -----------------------------------------------------------------------------


def discover_run_ids(
    results_dir: str | Path | None = None,
    sort_descending: bool = True,
) -> list[str]:
    """Discover all run IDs in the results directory.

    Args:
        results_dir: Results root directory (optional).
        sort_descending: If True, sort newest first (default: True).

    Returns:
        List of run IDs (e.g., ["20260202_151900", "20260201_120000"]).
    """
    results_root = get_results_root(results_dir)

    run_ids = []
    for run_dir in results_root.glob("run_*"):
        if run_dir.is_dir():
            rid = run_dir.name.replace("run_", "")
            run_ids.append(rid)

    run_ids.sort(reverse=sort_descending)
    return run_ids


def get_latest_run_id(results_dir: str | Path | None = None) -> str:
    """Get the most recent run ID.

    Args:
        results_dir: Results root directory (optional).

    Returns:
        Latest run ID.

    Raises:
        FileNotFoundError: If no runs found.
    """
    run_ids = discover_run_ids(results_dir, sort_descending=True)
    if not run_ids:
        raise FileNotFoundError("No runs found in results directory")
    return run_ids[0]


# -----------------------------------------------------------------------------
# Model discovery
# -----------------------------------------------------------------------------


def discover_models_for_run(
    run_id: str,
    results_dir: str | Path | None = None,
    skip_ensemble: bool = False,
    require_splits: bool = False,
    require_aggregated: bool = False,
    require_oof: bool = False,
    split_seed: int | None = None,
) -> list[str]:
    """Discover models available for a run.

    Args:
        run_id: Run ID to search.
        results_dir: Results root directory (optional).
        skip_ensemble: If True, exclude ENSEMBLE model (default: False).
        require_splits: If True, only return models with splits/ directory.
        require_aggregated: If True, only return models with aggregated/ directory.
        require_oof: If True, only return models with OOF predictions for split_seed.
        split_seed: Split seed to check for OOF predictions (required if require_oof=True).

    Returns:
        List of model names, sorted alphabetically.
    """
    run_path = get_run_path(run_id, results_dir)
    models = []

    for model_dir in sorted(run_path.glob("*/")):
        model_name = model_dir.name

        # Skip hidden and special directories
        if model_name.startswith(".") or model_name in SKIP_DIRS:
            continue

        if skip_ensemble and model_name == "ENSEMBLE":
            continue

        # Apply filters
        if require_splits:
            splits_dir = model_dir / "splits"
            if not splits_dir.exists():
                continue

        if require_aggregated:
            aggregated_dir = model_dir / "aggregated"
            if not aggregated_dir.exists():
                continue

        if require_oof:
            if split_seed is None:
                raise ValueError("split_seed required when require_oof=True")
            oof_path = (
                model_dir
                / "splits"
                / f"split_seed{split_seed}"
                / "preds"
                / f"train_oof__{model_name}.csv"
            )
            if not oof_path.exists():
                continue

        models.append(model_name)

    return models


def discover_models_with_aggregated_results(
    run_id: str,
    results_dir: str | Path | None = None,
    model_filter: str | None = None,
    skip_ensemble: bool = True,
    require_stability: bool = True,
) -> dict[str, Path]:
    """Discover models with aggregated results (for consensus/optimize-panel).

    Args:
        run_id: Run ID to search.
        results_dir: Results root directory (optional).
        model_filter: Optional model name to filter by.
        skip_ensemble: If True, skip ENSEMBLE models (default: True).
        require_stability: If True, require feature_stability_summary.csv (default: True).

    Returns:
        Dict mapping model_name -> Path to aggregated directory.

    Raises:
        FileNotFoundError: If no valid models found.
    """
    run_path = get_run_path(run_id, results_dir)
    model_dirs = {}

    for model_dir in sorted(run_path.glob("*/")):
        model_name = model_dir.name

        if model_name.startswith(".") or model_name in SKIP_DIRS:
            continue

        if skip_ensemble and model_name == "ENSEMBLE":
            continue

        if model_filter and model_name != model_filter:
            continue

        aggregated_dir = model_dir / "aggregated"
        if not aggregated_dir.exists():
            continue

        if require_stability:
            stability_file = aggregated_dir / "panels" / "feature_stability_summary.csv"
            if not stability_file.exists():
                continue

        model_dirs[model_name] = aggregated_dir

    if not model_dirs:
        raise FileNotFoundError(
            f"No models with aggregated results found for run {run_id}.\n"
            f"Run 'ced aggregate-splits --run-id {run_id}' first."
        )

    return model_dirs


# -----------------------------------------------------------------------------
# Split discovery
# -----------------------------------------------------------------------------


def discover_split_seeds_for_run(
    run_id: str,
    results_dir: str | Path | None = None,
    model: str | None = None,
    skip_ensemble: bool = True,
    require_oof: bool = True,
) -> list[int]:
    """Discover available split seeds for a run.

    Scans model directories to find all split_seedN directories.

    Args:
        run_id: Run ID (e.g., "20260127_104409").
        results_dir: Results root directory (optional).
        model: Specific model to scan (optional). If None, uses first available model.
        skip_ensemble: If True, exclude ENSEMBLE when scanning (default: True).
        require_oof: If True, only return seeds with OOF predictions (default: True).

    Returns:
        List of split seed integers, sorted ascending.

    Raises:
        FileNotFoundError: If no splits found.
    """
    run_path = get_run_path(run_id, results_dir)
    split_seeds: set[int] = set()

    # Determine which models to scan
    if model:
        model_dirs = [run_path / model]
    else:
        model_dirs = sorted(run_path.glob("*/"))

    for model_dir in model_dirs:
        model_name = model_dir.name

        if model_name.startswith(".") or model_name in SKIP_DIRS:
            continue
        if skip_ensemble and model_name == "ENSEMBLE":
            continue

        splits_dir = model_dir / "splits"
        if not splits_dir.exists():
            continue

        for split_dir in splits_dir.glob("split_seed*"):
            if not split_dir.is_dir():
                continue
            try:
                seed = int(split_dir.name.replace("split_seed", ""))

                if require_oof:
                    oof_path = split_dir / "preds" / f"train_oof__{model_name}.csv"
                    if not oof_path.exists():
                        continue

                split_seeds.add(seed)
            except ValueError:
                continue

        # Use first model with splits as reference (unless specific model requested)
        if split_seeds and model is None:
            break

    if not split_seeds:
        raise FileNotFoundError(
            f"No split seeds found for run {run_id}\n"
            f"Expected structure: results/run_{run_id}/{{MODEL}}/splits/split_seed*/"
        )

    return sorted(split_seeds)


def discover_split_dirs(
    model_path: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Discover all split_seedX subdirectories for a model.

    Args:
        model_path: Path to model directory (e.g., results/run_X/LR_EN/).
        logger: Optional logger instance.

    Returns:
        List of split subdirectory paths, sorted by seed number.
    """
    splits_subdir = model_path / "splits"
    if not splits_subdir.exists() or not splits_subdir.is_dir():
        if logger:
            logger.warning(f"Splits directory not found: {splits_subdir}")
        return []

    split_dirs = [
        split_dir for split_dir in splits_subdir.glob("split_seed*") if split_dir.is_dir()
    ]

    if logger:
        logger.debug(f"Found {len(split_dirs)} splits in {splits_subdir}")

    split_dirs = sorted(
        split_dirs,
        key=lambda p: int(p.name.replace("split_seed", "")),
    )

    return split_dirs


def discover_ensemble_dirs(
    run_path: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Discover ENSEMBLE model split directories.

    Args:
        run_path: Run-level directory (e.g., results/run_{RUN_ID}/).
        logger: Optional logger instance.

    Returns:
        List of ensemble split subdirectory paths, sorted by seed number.
    """
    ensemble_base = run_path / "ENSEMBLE"
    if not ensemble_base.exists():
        if logger:
            logger.debug(f"No ENSEMBLE directory found at {ensemble_base}")
        return []

    # Use discover_split_dirs but suppress its logging to provide ensemble-specific message
    dirs = discover_split_dirs(ensemble_base, logger=None)

    if logger:
        logger.debug(f"Discovered {len(dirs)} ENSEMBLE split directories")

    return dirs


# -----------------------------------------------------------------------------
# Data path auto-detection
# -----------------------------------------------------------------------------


def auto_detect_data_paths(
    run_id: str,
    results_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str | None, str | None]:
    """Auto-detect infile and split_dir from run metadata.

    Args:
        run_id: Run ID to search for.
        results_dir: Results root directory (optional).
        logger: Optional logger instance.

    Returns:
        Tuple of (infile, split_dir) or (None, None) if not found.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        run_path = get_run_path(run_id, results_dir)
    except FileNotFoundError:
        return None, None

    metadata_file = run_path / "run_metadata.json"
    logger.debug(f"Checking for metadata: {metadata_file}")

    if not metadata_file.exists():
        logger.debug("No run_metadata.json found")
        return None, None

    try:
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Prefer run-level fields, then fallback to model-level fields for back-compat.
        infile = metadata.get("infile")
        split_dir = metadata.get("split_dir") or metadata.get("splits_dir")
        if not infile or not split_dir:
            models_meta = metadata.get("models", {})
            if models_meta:
                first_model_meta = next(iter(models_meta.values()))
                infile = infile or first_model_meta.get("infile")
                split_dir = split_dir or first_model_meta.get("split_dir")

        logger.debug(f"Metadata infile: {infile}, split_dir: {split_dir}")

        if infile and split_dir:
            return infile, split_dir
    except Exception as e:
        logger.debug(f"Error reading metadata: {e}")

    return None, None


# -----------------------------------------------------------------------------
# Convenience: resolve with auto-detection
# -----------------------------------------------------------------------------


def resolve_run_id(
    run_id: str | None,
    results_dir: str | Path | None = None,
) -> str:
    """Resolve run_id, auto-detecting latest if None.

    Args:
        run_id: Explicit run ID, or None to auto-detect latest.
        results_dir: Results root directory (optional).

    Returns:
        Resolved run ID.

    Raises:
        FileNotFoundError: If no runs found.
    """
    if run_id:
        # Validate it exists
        _ = get_run_path(run_id, results_dir)
        return run_id
    return get_latest_run_id(results_dir)
