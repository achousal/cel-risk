"""
Split loading stage for training orchestration.

This module handles:
- Output directory creation
- Split index loading
- Split validation and alignment checks
- Scenario-based data filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ced_ml.cli.train import load_split_indices
from ced_ml.config.loader import save_config
from ced_ml.data.persistence import load_split_metadata, validate_split_indices
from ced_ml.data.schema import CONTROL_LABEL, SCENARIO_DEFINITIONS, TARGET_COL
from ced_ml.evaluation.reports import OutputDirectories
from ced_ml.utils.logging import log_hpc_context, log_section, setup_command_logger

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def load_splits(ctx: TrainingContext) -> TrainingContext:
    """Load and validate split indices.

    This stage:
    1. Creates output directory structure
    2. Sets up file logging
    3. Loads train/val/test split indices
    4. Validates split alignment with row filters
    5. Filters data to scenario-specific labels
    6. Validates split indices are within bounds

    Args:
        ctx: TrainingContext with data loaded

    Returns:
        Updated TrainingContext with:
        - outdirs: OutputDirectories instance
        - train_idx, val_idx, test_idx: Split indices
        - scenario: Detected/configured scenario name
        - df_scenario: Scenario-filtered DataFrame with 'y' column
    """
    config = ctx.config
    seed = ctx.seed
    run_id = ctx.run_id

    # Step 1: Create output directories
    log_section(logger, "Setting Up Output Structure")
    outdirs = OutputDirectories.create(
        config.outdir,
        exist_ok=True,
        split_seed=seed,
        run_id=run_id,
        model=config.model,
    )
    logger.info(f"Output root: {outdirs.root}")
    logger.info(f"Split seed: {seed}")
    logger.info(f"Run ID: {run_id}")

    # Step 2: Auto-file-logging
    setup_command_logger(
        command="train",
        log_level=ctx.log_level,
        outdir=config.outdir,
        run_id=run_id,
        model=config.model,
        split_seed=seed,
    )
    log_hpc_context(logger)

    # Step 3: Save resolved config
    config_path = Path(outdirs.root) / "training_config.yaml"
    save_config(config, config_path)
    logger.info(f"Saved resolved config to: {config_path}")

    # Log config summary
    logger.info(f"Model: {config.model}")
    logger.info(f"CV: {config.cv.folds} folds x {config.cv.repeats} repeats")
    logger.info(f"Scoring: {config.cv.scoring}")

    # Step 4: Load split indices
    log_section(logger, "Loading Splits")
    try:
        scenario = getattr(config, "scenario", None)
        train_idx, val_idx, test_idx, detected_scenario = load_split_indices(
            str(config.split_dir), scenario, seed
        )
        scenario = detected_scenario

        logger.info(f"Scenario: {scenario}")
        logger.info(f"Loaded splits for seed {seed}:")
        logger.info(f"  Train: {len(train_idx):,} samples")
        logger.info(f"  Val:   {len(val_idx):,} samples")
        logger.info(f"  Test:  {len(test_idx):,} samples")

        # Save split trace
        split_trace_df = pd.DataFrame(
            {
                "idx": np.concatenate([train_idx, val_idx, test_idx]),
                "split": (
                    ["train"] * len(train_idx) + ["val"] * len(val_idx) + ["test"] * len(test_idx)
                ),
                "scenario": scenario,
                "seed": seed,
            }
        )
        split_trace_path = Path(outdirs.diag_splits) / "train_test_split_trace.csv"
        split_trace_df.to_csv(split_trace_path, index=False)
        logger.info(f"Split trace saved: {split_trace_path}")

        # Step 5: Validate row filter alignment
        split_meta = load_split_metadata(str(config.split_dir), scenario, seed)
        if split_meta is not None:
            row_filters = split_meta.get("row_filters", {})
            split_meta_num_cols = set(row_filters.get("meta_num_cols_used", []))
            current_meta_num_cols = set(ctx.resolved.numeric_metadata)

            if split_meta_num_cols and split_meta_num_cols != current_meta_num_cols:
                logger.error("Row filter column mismatch detected!")
                logger.error(f"  Splits used:   {sorted(split_meta_num_cols)}")
                logger.error(f"  Training uses: {sorted(current_meta_num_cols)}")
                logger.error("This can cause train/val/test contamination.")
                raise ValueError(
                    f"Row filter alignment error: splits used {sorted(split_meta_num_cols)}, "
                    f"but training uses {sorted(current_meta_num_cols)}. "
                    "Regenerate splits or update config to match."
                )
            elif split_meta_num_cols:
                logger.info(f"Row filter alignment verified: {sorted(split_meta_num_cols)}")
        else:
            logger.warning("Split metadata not found - cannot verify row filter alignment")

    except FileNotFoundError as e:
        logger.error(f"Split files not found: {e}")
        logger.error("Please run 'ced save-splits' first to generate splits")
        raise

    # Step 6: Filter data to scenario-specific labels
    scenario_def = SCENARIO_DEFINITIONS[scenario]
    target_labels = scenario_def["labels"]

    # Validate expected labels are present
    unique_labels = set(ctx.df_filtered[TARGET_COL].unique())
    missing_labels = set(target_labels) - unique_labels
    if missing_labels:
        logger.warning(
            f"Expected labels {missing_labels} not found in filtered data. "
            f"Available: {unique_labels}"
        )

    # Check for unknown labels
    unknown_labels = unique_labels - set(target_labels)
    if unknown_labels:
        logger.warning(
            f"Found labels not in scenario definition: {unknown_labels}. " "Will be filtered out."
        )

    mask_scenario = ctx.df_filtered[TARGET_COL].isin(target_labels)
    n_filtered = (~mask_scenario).sum()
    if n_filtered > 0:
        logger.info(f"Filtered out {n_filtered:,} samples with labels not in scenario {scenario}")

    df_scenario = ctx.df_filtered[mask_scenario].copy()
    df_scenario["y"] = (df_scenario[TARGET_COL] != CONTROL_LABEL).astype(int)

    # Step 7: Validate split indices are within bounds
    is_valid, error_msg = validate_split_indices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        total_samples=len(df_scenario),
    )

    if not is_valid:
        logger.error(f"Split index bounds validation failed: {error_msg}")
        logger.error(f"Data shape: {len(df_scenario)} samples after filtering")
        logger.error(f"Max train_idx: {train_idx.max() if len(train_idx) > 0 else 'N/A'}")
        logger.error(f"Max val_idx: {val_idx.max() if len(val_idx) > 0 else 'N/A'}")
        logger.error(f"Max test_idx: {test_idx.max() if len(test_idx) > 0 else 'N/A'}")
        raise ValueError(f"Split index bounds validation failed: {error_msg}")

    # Update context
    ctx.outdirs = outdirs
    ctx.train_idx = train_idx
    ctx.val_idx = val_idx
    ctx.test_idx = test_idx
    ctx.scenario = scenario
    ctx.df_scenario = df_scenario

    return ctx
