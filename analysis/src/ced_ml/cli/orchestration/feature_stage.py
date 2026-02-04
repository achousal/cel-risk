"""
Feature preparation stage for training orchestration.

This module handles:
- Fixed panel loading (if specified)
- Feature column preparation
- X/y split preparation for train/val/test
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ced_ml.data.schema import TARGET_COL
from ced_ml.utils.logging import log_section
from ced_ml.utils.paths import get_project_root

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def prepare_features(ctx: TrainingContext) -> TrainingContext:
    """Prepare features for training.

    This stage:
    1. Handles fixed panel loading (if specified via CLI or config)
    2. Prepares feature columns
    3. Creates X/y splits for train/val/test
    4. Extracts category labels for plots
    5. Computes training prevalence

    Args:
        ctx: TrainingContext with splits loaded

    Returns:
        Updated TrainingContext with:
        - feature_cols: Final feature columns
        - fixed_panel_proteins: Fixed panel proteins (if applicable)
        - fixed_panel_path: Path to fixed panel (if applicable)
        - X_train, y_train, X_val, y_val, X_test, y_test
        - cat_train, cat_val, cat_test
        - train_prev: Training prevalence
    """
    config = ctx.config
    protein_cols = ctx.protein_cols
    feature_cols = ctx.resolved.all_feature_cols

    # Step 1: Handle fixed panel (from CLI or config)
    fixed_panel_path = ctx.cli_args.get("fixed_panel") if ctx.cli_args else None

    # Check if fixed panel is specified in config
    if not fixed_panel_path and config.features.feature_selection_strategy == "fixed_panel":
        if config.features.fixed_panel_csv:
            # Resolve path relative to data/ directory if not absolute
            fixed_panel_path = Path(config.features.fixed_panel_csv)
            if not fixed_panel_path.is_absolute():
                data_dir = get_project_root() / "data"
                fixed_panel_path = data_dir / fixed_panel_path
        else:
            raise ValueError(
                "feature_selection_strategy='fixed_panel' but fixed_panel_csv not specified "
                "in config. Set features.fixed_panel_csv in training_config.yaml"
            )

    fixed_panel_proteins = None

    if fixed_panel_path:
        log_section(logger, "Fixed Panel Mode")
        logger.info(f"Loading fixed panel from: {fixed_panel_path}")

        # Load fixed panel CSV
        fixed_panel_df = pd.read_csv(fixed_panel_path)

        # Expect a column named 'protein' or use first column
        if "protein" in fixed_panel_df.columns:
            fixed_panel_proteins = fixed_panel_df["protein"].tolist()
        else:
            fixed_panel_proteins = fixed_panel_df.iloc[:, 0].tolist()

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for p in fixed_panel_proteins:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        if len(deduped) < len(fixed_panel_proteins):
            n_dup = len(fixed_panel_proteins) - len(deduped)
            dupes = [p for p in fixed_panel_proteins if fixed_panel_proteins.count(p) > 1]
            logger.warning(
                f"Fixed panel CSV contains {n_dup} duplicate(s): {sorted(set(dupes))}. "
                "Duplicates removed."
            )
            fixed_panel_proteins = deduped

        # Validate that all fixed panel proteins exist in data
        missing_proteins = set(fixed_panel_proteins) - set(protein_cols)
        if missing_proteins:
            logger.error(f"Fixed panel contains {len(missing_proteins)} proteins not in dataset")
            logger.error(f"Missing proteins (first 10): {list(missing_proteins)[:10]}")
            raise ValueError(
                f"Fixed panel contains {len(missing_proteins)} proteins not found in dataset. "
                "Check that protein names match exactly (e.g., 'PROT_123_resid')."
            )

        # Override protein_cols to use only fixed panel
        protein_cols = fixed_panel_proteins
        logger.info(f"Fixed panel loaded: {len(fixed_panel_proteins)} proteins")
        logger.info("Feature selection will be BYPASSED (using pre-specified panel)")

        # Override config to disable feature selection
        config.features.feature_selection_strategy = "none"
        config.features.screen_top_n = 0
        logger.info("Feature selection strategy set to 'none'")

        # Update feature_cols to use fixed panel proteins + metadata
        feature_cols = (
            list(protein_cols) + ctx.resolved.numeric_metadata + ctx.resolved.categorical_metadata
        )
        logger.info(f"Feature columns updated: {len(feature_cols)} total features")

    # Step 2: Create X/y splits
    df_scenario = ctx.df_scenario
    train_idx = ctx.train_idx
    val_idx = ctx.val_idx
    test_idx = ctx.test_idx

    X_train = df_scenario.iloc[train_idx][feature_cols]
    y_train = df_scenario.iloc[train_idx]["y"].values

    X_val = df_scenario.iloc[val_idx][feature_cols]
    y_val = df_scenario.iloc[val_idx]["y"].values

    X_test = df_scenario.iloc[test_idx][feature_cols]
    y_test = df_scenario.iloc[test_idx]["y"].values

    # Step 3: Extract original category labels for plots
    cat_train = df_scenario.iloc[train_idx][TARGET_COL].values
    cat_val = df_scenario.iloc[val_idx][TARGET_COL].values
    cat_test = df_scenario.iloc[test_idx][TARGET_COL].values

    # Step 4: Compute training prevalence
    train_prev = float(y_train.mean())
    logger.info(f"Training prevalence: {train_prev:.3f}")

    # Update context
    ctx.protein_cols = protein_cols
    ctx.feature_cols = feature_cols
    ctx.fixed_panel_proteins = fixed_panel_proteins
    ctx.fixed_panel_path = Path(fixed_panel_path) if fixed_panel_path else None

    ctx.X_train = X_train
    ctx.y_train = y_train
    ctx.X_val = X_val
    ctx.y_val = y_val
    ctx.X_test = X_test
    ctx.y_test = y_test

    ctx.cat_train = cat_train
    ctx.cat_val = cat_val
    ctx.cat_test = cat_test

    ctx.train_prev = train_prev

    return ctx
