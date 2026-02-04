"""
Data loading stage for training orchestration.

This module handles:
- Column resolution (auto-detect or explicit)
- Data loading with resolved columns
- Row filter application
- Protein column validation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ced_ml.cli.train import validate_protein_columns
from ced_ml.data.columns import get_available_columns_from_file, resolve_columns
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file, usecols_for_proteomics
from ced_ml.utils.logging import log_section

if TYPE_CHECKING:
    from ced_ml.cli.orchestration.context import TrainingContext

logger = logging.getLogger(__name__)


def load_data(ctx: TrainingContext) -> TrainingContext:
    """Load and prepare data for training.

    This stage:
    1. Resolves columns (auto-detect or explicit mode)
    2. Loads data with resolved columns
    3. Applies row filters (uncertain controls, missing metadata)
    4. Validates protein columns (dtype and NaN check)

    Args:
        ctx: TrainingContext with config set

    Returns:
        Updated TrainingContext with:
        - df_filtered: Filtered DataFrame
        - resolved: ResolvedColumns object
        - protein_cols: List of protein column names
        - filter_stats: Filter statistics dictionary
    """
    config = ctx.config

    # Step 1: Resolve columns (auto-detect or explicit)
    log_section(logger, "Resolving Columns")
    logger.info(f"Column mode: {config.columns.mode}")
    available_columns = get_available_columns_from_file(str(config.infile))
    resolved = resolve_columns(available_columns, config.columns)

    logger.info("Resolved columns:")
    logger.info(f"  Proteins: {len(resolved.protein_cols)}")
    logger.info(f"  Numeric metadata: {resolved.numeric_metadata}")
    logger.info(f"  Categorical metadata: {resolved.categorical_metadata}")

    # Step 2: Load data with resolved columns
    log_section(logger, "Loading Data")
    logger.info(f"Reading: {config.infile}")
    usecols_fn = usecols_for_proteomics(
        numeric_metadata=resolved.numeric_metadata,
        categorical_metadata=resolved.categorical_metadata,
    )
    df_raw = read_proteomics_file(config.infile, usecols=usecols_fn)

    # Step 3: Apply row filters
    logger.info("Applying row filters...")
    df_filtered, filter_stats = apply_row_filters(df_raw, meta_num_cols=resolved.numeric_metadata)
    logger.info(f"Filtered: {filter_stats['n_in']:,} -> {filter_stats['n_out']:,} rows")
    logger.info(f"  Removed {filter_stats['n_removed_uncertain_controls']} uncertain controls")
    logger.info(f"  Removed {filter_stats['n_removed_dropna_meta_num']} rows with missing metadata")

    # Step 4: Use resolved columns
    protein_cols = resolved.protein_cols
    logger.info(f"Using {len(protein_cols)} protein columns")

    # Step 5: Validate protein columns (dtype and NaN check)
    validate_protein_columns(df_filtered, protein_cols, logger)

    # Update context
    ctx.df_filtered = df_filtered
    ctx.resolved = resolved
    ctx.protein_cols = protein_cols
    ctx.filter_stats = filter_stats

    return ctx
