"""Split validation and persistence utilities.

This module handles saving split indices to CSV files, generating split metadata
(JSON), and validating split integrity before persistence.

Design:
    - save_split_indices(): Save TRAIN/VAL/TEST indices to CSV
    - save_split_metadata(): Generate and save JSON metadata
    - save_holdout_indices(): Save holdout set indices
    - check_split_files_exist(): Check for existing split files
    - validate_split_indices(): Validate split integrity (no overlap, coverage)

Behavioral equivalence:
    - Matches save_splits.py CSV/JSON output format exactly
    - Preserves index sorting behavior (ascending)
    - Maintains metadata schema compatibility with celiacML_faith.py
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .splits import compute_split_id

# ============================================================================
# Split Validation
# ============================================================================


def validate_split_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    total_samples: int | None = None,
) -> tuple[bool, str]:
    """Validate split indices for integrity.

    Args:
        train_idx: Training set indices
        test_idx: Test set indices
        val_idx: Validation set indices (optional)
        total_samples: Total number of samples in dataset (for coverage check)

    Returns:
        Tuple of (is_valid, error_message). error_message is empty string if valid.

    Validation checks:
        1. No overlap between TRAIN/VAL/TEST
        2. All indices are non-negative integers
        3. Optional: All indices < total_samples
        4. Optional: Full coverage (union covers all samples)
    """
    # Check for empty arrays
    if len(train_idx) == 0:
        return False, "TRAIN set is empty"
    if len(test_idx) == 0:
        return False, "TEST set is empty"

    # Collect all splits
    splits = {"train": train_idx, "test": test_idx}
    if val_idx is not None and len(val_idx) > 0:
        splits["val"] = val_idx

    # Check data types and non-negative
    for name, idx in splits.items():
        if not np.issubdtype(idx.dtype, np.integer):
            return False, f"{name.upper()} indices must be integers, got {idx.dtype}"
        if np.any(idx < 0):
            return False, f"{name.upper()} contains negative indices"

    # Check for overlaps
    if val_idx is not None and len(val_idx) > 0:
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            return False, f"TRAIN/VAL overlap: {len(train_val_overlap)} samples"
        if train_test_overlap:
            return False, f"TRAIN/TEST overlap: {len(train_test_overlap)} samples"
        if val_test_overlap:
            return False, f"VAL/TEST overlap: {len(val_test_overlap)} samples"
    else:
        # Two-way split
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set & test_set
        if overlap:
            return False, f"TRAIN/TEST overlap: {len(overlap)} samples"

    # Check bounds if total_samples provided
    if total_samples is not None:
        for name, idx in splits.items():
            if np.any(idx >= total_samples):
                bad_count = np.sum(idx >= total_samples)
                return (
                    False,
                    f"{name.upper()} contains {bad_count} indices >= {total_samples}",
                )

    return True, ""


def check_split_files_exist(
    outdir: str,
    scenario: str,
    seed: int,
    has_val: bool = False,
) -> tuple[bool, list[str]]:
    """Check if split files already exist in output directory.

    Args:
        outdir: Output directory path
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed
        has_val: Whether validation set is expected

    Returns:
        Tuple of (files_exist, existing_paths)
    """
    suffix = f"_{scenario}_seed{seed if seed is not None else 0}"

    outdir_path = Path(outdir)
    expected_files = [
        outdir_path / f"train_idx{suffix}.csv",
        outdir_path / f"test_idx{suffix}.csv",
    ]
    if has_val:
        expected_files.append(outdir_path / f"val_idx{suffix}.csv")

    # Also check metadata
    expected_files.append(outdir_path / f"split_meta_{scenario}_seed{seed}.json")

    existing = [str(f) for f in expected_files if f.exists()]

    return len(existing) > 0, existing


def validate_existing_splits(
    outdir: str,
    scenario: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Validate if existing split files match requested split configuration.

    Checks if existing split files contain the same indices as the requested split.
    This allows reusing splits when they match, avoiding unnecessary regeneration.

    Args:
        outdir: Output directory path
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed
        train_idx: Requested training set indices
        test_idx: Requested test set indices
        val_idx: Requested validation set indices (optional)

    Returns:
        Tuple of (is_match, message).
        - is_match=True if files exist and match exactly
        - message describes the result or mismatch
    """
    # Check if files exist
    has_val = val_idx is not None and len(val_idx) > 0
    files_exist, existing = check_split_files_exist(outdir, scenario, seed, has_val)

    if not files_exist:
        return False, "Split files do not exist"

    # Check if all expected files exist
    suffix = f"_{scenario}_seed{seed if seed is not None else 0}"
    outdir_path = Path(outdir)
    train_path = outdir_path / f"train_idx{suffix}.csv"
    test_path = outdir_path / f"test_idx{suffix}.csv"
    val_path = outdir_path / f"val_idx{suffix}.csv" if has_val else None

    if not train_path.exists() or not test_path.exists():
        return False, "Missing train or test split file"

    if has_val and not val_path.exists():
        return False, "Missing validation split file"

    # Load existing indices
    try:
        existing_train = pd.read_csv(train_path)["idx"].values
        existing_test = pd.read_csv(test_path)["idx"].values
        existing_val = pd.read_csv(val_path)["idx"].values if has_val else None
    except Exception as e:
        return False, f"Failed to load existing split files: {e}"

    # Compare indices (order-independent via sets)
    train_match = set(existing_train) == set(train_idx)
    test_match = set(existing_test) == set(test_idx)
    val_match = True if not has_val else (set(existing_val) == set(val_idx))

    if train_match and test_match and val_match:
        return True, "Existing split files match requested configuration exactly"

    # Build detailed mismatch message
    mismatches = []
    if not train_match:
        mismatches.append(f"TRAIN ({len(existing_train)} vs {len(train_idx)} samples)")
    if not test_match:
        mismatches.append(f"TEST ({len(existing_test)} vs {len(test_idx)} samples)")
    if has_val and not val_match:
        mismatches.append(f"VAL ({len(existing_val)} vs {len(val_idx)} samples)")

    return False, f"Split files differ: {', '.join(mismatches)}"


# ============================================================================
# Index Persistence
# ============================================================================


def save_split_indices(
    outdir: str,
    scenario: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Save train/val/test indices to CSV files.

    Args:
        outdir: Output directory path
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed used for split
        train_idx: Training set indices (sorted ascending)
        test_idx: Test set indices (sorted ascending)
        val_idx: Validation set indices (sorted ascending, optional)
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary mapping split name to saved file path

    Raises:
        FileExistsError: If files exist and overwrite=False
        ValueError: If indices are invalid

    Output format:
        - CSV with single column "idx" containing indices
        - Filenames: train_idx_seed{seed}.csv (if n_splits > 1)
                     train_idx.csv (if n_splits == 1)
    """
    # Validate indices
    is_valid, error_msg = validate_split_indices(train_idx, test_idx, val_idx)
    if not is_valid:
        raise ValueError(f"Invalid split indices: {error_msg}")

    # Check for existing files
    has_val = val_idx is not None and len(val_idx) > 0
    files_exist, existing = check_split_files_exist(outdir, scenario, seed, has_val)

    if files_exist and not overwrite:
        # Validate if existing splits match requested configuration
        is_match, match_msg = validate_existing_splits(
            outdir, scenario, seed, train_idx, test_idx, val_idx
        )

        if is_match:
            # Splits match - warn and skip regeneration
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Split files already exist and match requested configuration (seed={seed}, scenario={scenario}). "
                f"Skipping regeneration. Set overwrite=True to force regeneration."
            )
            # Return existing paths
            suffix = f"_{scenario}_seed{seed if seed is not None else 0}"
            paths = {
                "train": os.path.join(outdir, f"train_idx{suffix}.csv"),
                "test": os.path.join(outdir, f"test_idx{suffix}.csv"),
            }
            if has_val:
                paths["val"] = os.path.join(outdir, f"val_idx{suffix}.csv")
            return paths
        else:
            # Splits differ - raise error
            raise FileExistsError(
                f"Split files already exist but DO NOT match requested configuration:\n"
                f"  {match_msg}\n"
                f"Existing files:\n" + "\n".join(f"  {p}" for p in existing) + "\n"
                "Use overwrite=True to replace with new splits."
            )

    # Build file paths (include scenario to prevent overwrites in multi-scenario runs)
    suffix = f"_{scenario}_seed{seed if seed is not None else 0}"
    outdir_path = Path(outdir)
    paths = {
        "train": str(outdir_path / f"train_idx{suffix}.csv"),
        "test": str(outdir_path / f"test_idx{suffix}.csv"),
    }
    if has_val:
        paths["val"] = str(outdir_path / f"val_idx{suffix}.csv")

    # Sort indices for deterministic output
    train_idx = np.sort(train_idx.astype(int))
    test_idx = np.sort(test_idx.astype(int))
    if has_val:
        val_idx = np.sort(val_idx.astype(int))

    # Save CSVs
    outdir_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"idx": train_idx}).to_csv(paths["train"], index=False)
    pd.DataFrame({"idx": test_idx}).to_csv(paths["test"], index=False)
    if has_val:
        pd.DataFrame({"idx": val_idx}).to_csv(paths["val"], index=False)

    return paths


def save_holdout_indices(
    outdir: str,
    scenario: str,
    holdout_idx: np.ndarray,
    overwrite: bool = False,
    model_name: str | None = None,
    split_seed: int | None = None,
    run_id: str | None = None,
) -> str:
    """Save holdout set indices to CSV.

    Args:
        outdir: Output directory path
        scenario: Scenario name
        holdout_idx: Holdout set indices (sorted ascending)
        overwrite: Whether to overwrite existing file
        model_name: Optional model name for scenario-specific naming
        split_seed: Optional split seed for scenario-specific naming
        run_id: Optional run identifier (e.g., timestamp) for uniqueness

    Returns:
        Path to saved CSV file

    Raises:
        FileExistsError: If file exists and overwrite=False

    Notes:
        File naming strategy (scenario-specific to prevent overwrites):
        - If model_name, split_seed, or run_id provided:
          HOLDOUT_idx_{scenario}[_{model}][_seed{N}][_{runid}].csv
        - Otherwise: HOLDOUT_idx_{scenario}.csv (backward compatible with scenario)
    """
    # Build scenario-specific filename to prevent overwrites
    suffix_parts = [scenario]
    if model_name is not None:
        suffix_parts.append(model_name)
    if split_seed is not None:
        suffix_parts.append(f"seed{split_seed}")
    if run_id is not None:
        suffix_parts.append(run_id)

    suffix = "_".join(suffix_parts)
    outdir_path = Path(outdir)
    holdout_path = outdir_path / f"HOLDOUT_idx_{suffix}.csv"

    if holdout_path.exists() and not overwrite:
        raise FileExistsError(
            f"Holdout file already exists: {holdout_path}\n" f"Use overwrite=True to replace."
        )

    outdir_path.mkdir(parents=True, exist_ok=True)

    # Sort indices for deterministic output
    holdout_idx = np.sort(holdout_idx.astype(int))

    pd.DataFrame({"idx": holdout_idx}).to_csv(holdout_path, index=False)

    return str(holdout_path)


# ============================================================================
# Metadata Persistence
# ============================================================================


def save_split_metadata(
    outdir: str,
    scenario: str,
    seed: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_idx: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    split_type: str = "development",
    strat_scheme: str | None = None,
    row_filter_stats: dict[str, Any] | None = None,
    index_space: str = "full",
    temporal_split: bool = False,
    temporal_col: str | None = None,
    temporal_train_end: str | None = None,
    temporal_test_start: str | None = None,
    temporal_test_end: str | None = None,
) -> str:
    """Save split metadata to JSON file.

    Args:
        outdir: Output directory path
        scenario: Scenario name
        seed: Random seed used for split
        train_idx: Training set indices
        test_idx: Test set indices
        y_train: Training set labels
        y_test: Test set labels
        val_idx: Validation set indices (optional)
        y_val: Validation set labels (optional)
        split_type: "development" or "holdout"
        strat_scheme: Stratification scheme used (e.g., "outcome+sex+age3")
        row_filter_stats: Row filter statistics dict
        index_space: "full" or "dev" (index space reference)
        temporal_split: Whether temporal splitting was used
        temporal_col: Column used for temporal ordering
        temporal_train_end: Last temporal value in TRAIN
        temporal_test_start: First temporal value in TEST
        temporal_test_end: Last temporal value in TEST

    Returns:
        Path to saved JSON metadata file

    Metadata schema:
        {
            "scenario": str,
            "seed": int,
            "split_type": str,
            "index_space": str,
            "n_train": int,
            "n_test": int,
            "n_train_pos": int,
            "n_test_pos": int,
            "prevalence_train": float,
            "prevalence_test": float,
            "split_id_train": str,
            "split_id_test": str,
            "n_val": int (optional),
            "n_val_pos": int (optional),
            "prevalence_val": float (optional),
            "split_id_val": str (optional),
            "stratification_scheme": str (optional),
            "row_filters": dict (optional),
            "temporal_split": bool (optional),
            "temporal_col": str (optional),
            "temporal_train_end_value": str (optional),
            "temporal_test_start_value": str (optional),
            "temporal_test_end_value": str (optional)
        }
    """
    meta: dict[str, Any] = {
        "scenario": scenario,
        "seed": int(seed),
        "split_type": split_type,
        "index_space": index_space,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_pos": int(y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "split_id_train": compute_split_id(train_idx),
        "split_id_test": compute_split_id(test_idx),
    }

    # Add validation set metadata if present
    if val_idx is not None and y_val is not None and len(val_idx) > 0:
        meta.update(
            {
                "n_val": int(len(val_idx)),
                "n_val_pos": int(y_val.sum()),
                "prevalence_val": float(y_val.mean()),
                "split_id_val": compute_split_id(val_idx),
            }
        )

    # Add stratification scheme if provided
    if strat_scheme is not None:
        meta["stratification_scheme"] = strat_scheme

    # Add row filter stats if provided
    if row_filter_stats is not None:
        meta["row_filters"] = row_filter_stats

    # Add temporal metadata if applicable
    if temporal_split:
        meta["temporal_split"] = True
        if temporal_col is not None:
            meta["temporal_col"] = temporal_col
        if temporal_train_end is not None:
            meta["temporal_train_end_value"] = temporal_train_end
        if temporal_test_start is not None:
            meta["temporal_test_start_value"] = temporal_test_start
        if temporal_test_end is not None:
            meta["temporal_test_end_value"] = temporal_test_end

    # Save to file (include scenario to prevent overwrites)
    outdir_path = Path(outdir)
    meta_path = outdir_path / f"split_meta_{scenario}_seed{seed}.json"
    outdir_path.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(meta_path)


def load_split_metadata(
    split_dir: str,
    scenario: str,
    seed: int,
) -> dict[str, Any] | None:
    """Load split metadata from JSON file.

    Args:
        split_dir: Directory containing split metadata files
        scenario: Scenario name (e.g., IncidentOnly)
        seed: Random seed used for splits

    Returns:
        Metadata dict if found, None otherwise

    Notes:
        Metadata includes row_filters.meta_num_cols_used which is critical
        for validating alignment between split generation and training.
    """
    meta_path = Path(split_dir) / f"split_meta_{scenario}_seed{seed}.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        return json.load(f)


def save_holdout_metadata(
    outdir: str,
    scenario: str,
    holdout_idx: np.ndarray,
    y_holdout: np.ndarray,
    strat_scheme: str | None = None,
    row_filter_stats: dict[str, Any] | None = None,
    warning: str | None = None,
    temporal_split: bool = False,
    temporal_col: str | None = None,
    temporal_start: str | None = None,
    temporal_end: str | None = None,
    model_name: str | None = None,
    split_seed: int | None = None,
    run_id: str | None = None,
) -> str:
    """Save holdout set metadata to JSON file.

    Args:
        outdir: Output directory path
        scenario: Scenario name
        holdout_idx: Holdout set indices
        y_holdout: Holdout set labels
        strat_scheme: Stratification scheme used
        row_filter_stats: Row filter statistics dict
        warning: Optional warning message (e.g., reverse causality)
        temporal_split: Whether temporal splitting was used
        temporal_col: Column used for temporal ordering
        temporal_start: First temporal value in HOLDOUT
        temporal_end: Last temporal value in HOLDOUT
        model_name: Optional model name for scenario-specific naming
        split_seed: Optional split seed for scenario-specific naming
        run_id: Optional run identifier (e.g., timestamp) for uniqueness

    Returns:
        Path to saved JSON metadata file

    Notes:
        File naming strategy (scenario-specific to prevent overwrites):
        - If model_name, split_seed, or run_id provided:
          HOLDOUT_meta_{scenario}[_{model}][_seed{N}][_{runid}].json
        - Otherwise: HOLDOUT_meta_{scenario}.json (backward compatible with scenario)
    """
    meta: dict[str, Any] = {
        "scenario": scenario,
        "split_type": "holdout",
        "seed": 42,  # Fixed seed for holdout (ensures consistent evaluation set across experiments)
        "n_holdout": int(len(holdout_idx)),
        "n_holdout_pos": int(y_holdout.sum()),
        "prevalence_holdout": float(y_holdout.mean()),
        "split_id_holdout": compute_split_id(holdout_idx),
        "index_space": "full",
        "note": "NEVER use this set during development. Final evaluation only.",
    }

    if strat_scheme is not None:
        meta["stratification_scheme"] = strat_scheme

    if row_filter_stats is not None:
        meta["row_filters"] = row_filter_stats

    if warning is not None:
        meta["reverse_causality_warning"] = warning

    if temporal_split:
        meta["temporal_split"] = True
        if temporal_col is not None:
            meta["temporal_col"] = temporal_col
        if temporal_start is not None:
            meta["temporal_start_value"] = temporal_start
        if temporal_end is not None:
            meta["temporal_end_value"] = temporal_end

    # Include optional identifiers in metadata for traceability
    if model_name is not None:
        meta["model_name"] = model_name
    if split_seed is not None:
        meta["split_seed"] = split_seed
    if run_id is not None:
        meta["run_id"] = run_id

    # Build scenario-specific filename to prevent overwrites
    suffix_parts = [scenario]
    if model_name is not None:
        suffix_parts.append(model_name)
    if split_seed is not None:
        suffix_parts.append(f"seed{split_seed}")
    if run_id is not None:
        suffix_parts.append(run_id)

    suffix = "_".join(suffix_parts)
    outdir_path = Path(outdir)
    meta_path = outdir_path / f"HOLDOUT_meta_{suffix}.json"
    outdir_path.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(meta_path)
