"""CLI implementation for save-splits command.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ced_ml.config.loader import load_splits_config, save_config
from ced_ml.config.validation import check_prevalent_in_eval, validate_splits_config

# Import row filtering logic from data layer
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.persistence import (
    save_holdout_indices,
    save_holdout_metadata,
    save_split_indices,
    save_split_metadata,
)
from ced_ml.data.schema import (
    CONTROL_LABEL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    SCENARIO_DEFINITIONS,
    TARGET_COL,
)
from ced_ml.data.splits import (
    add_prevalent_to_train,
    build_working_strata,
    downsample_controls,
    stratified_train_val_test_split,
    temporal_order_indices,
    temporal_train_val_test_split,
)
from ced_ml.utils.logging import log_section, setup_logger


# Helper function to extract positive labels from schema SCENARIO_DEFINITIONS
def get_positives_from_scenario(scenario: str) -> list[str]:
    """Extract positive labels from scenario definition."""
    if scenario not in SCENARIO_DEFINITIONS:
        raise ValueError(
            f"Unknown scenario: {scenario}. Valid: {list(SCENARIO_DEFINITIONS.keys())}"
        )
    # SCENARIO_DEFINITIONS has 'labels' which includes CONTROL_LABEL
    # We need only the positive labels (exclude CONTROL_LABEL)
    all_labels = SCENARIO_DEFINITIONS[scenario]["labels"]
    return [label for label in all_labels if label != CONTROL_LABEL]


def run_save_splits(
    config_file: str | None = None,
    cli_args: dict[str, Any] | None = None,
    overrides: list[str] | None = None,
    log_level: int | None = None,
):
    """
    Run split generation with new config system.

    Args:
        config_file: Path to YAML config file (optional)
        cli_args: Dictionary of CLI arguments (optional)
        overrides: List of config overrides (optional)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
    """
    # Setup logger
    if log_level is None:
        log_level = logging.INFO
    # Use parent logger "ced_ml" so all child modules inherit the level
    logger = setup_logger("ced_ml", level=log_level)

    log_section(logger, "CeD-ML Split Generation")

    # Extract infile from cli_args (not part of SplitsConfig)
    infile = None
    if cli_args and "infile" in cli_args:
        infile = cli_args["infile"]

    # Build overrides list from CLI args
    all_overrides = list(overrides) if overrides else []
    if cli_args:
        for key, value in cli_args.items():
            if value is not None and key != "infile":  # Skip infile - not part of config
                # Skip scenarios - handle separately
                if key == "scenarios":
                    continue
                # Skip train_controls_incident_only - not part of SplitsConfig
                if key in ("train_controls_incident_only",):
                    continue
                # Map temporal_column CLI flag to temporal_col config field
                if key == "temporal_column":
                    all_overrides.append(f"temporal_col={value}")
                    continue
                all_overrides.append(f"{key}={value}")

        # Handle scenarios specially (convert tuple to comma-separated string for parsing)
        if "scenarios" in cli_args and cli_args["scenarios"]:
            scenarios_str = ",".join(cli_args["scenarios"])
            all_overrides.append(f"scenarios={scenarios_str}")

    # Load and validate config
    logger.info("Loading configuration...")
    config = load_splits_config(config_file=config_file, overrides=all_overrides)

    logger.info("Validating configuration...")
    validate_splits_config(config, strictness="warn")

    # Save resolved config
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "splits_config.yaml"
    save_config(config, config_path)
    logger.info(f"Saved resolved config to: {config_path}")

    # Log config summary
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Scenarios: {', '.join(config.scenarios)}")
    logger.info(
        f"Splits: {config.n_splits} (seeds {config.seed_start} to {config.seed_start + config.n_splits - 1})"
    )
    logger.info(f"Val size: {config.val_size:.2%}")
    logger.info(f"Test size: {config.test_size:.2%}")
    if config.mode == "holdout":
        logger.info(f"Holdout size: {config.holdout_size:.2%}")

    if config.prevalent_train_only:
        logger.info(f"Prevalent: TRAIN only ({config.prevalent_train_frac:.0%} sampled)")
    else:
        logger.info("Prevalent: All splits")

    if config.train_control_per_case:
        logger.info(f"TRAIN controls: {config.train_control_per_case:.0f} per case")
    if config.eval_control_per_case:
        logger.info(f"VAL/TEST controls: {config.eval_control_per_case:.0f} per case")

    # Load data
    if infile is None:
        raise ValueError("infile must be provided via cli_args")

    logger.info(f"Loading data from: {infile}")
    df = read_proteomics_file(infile, validate=True)
    # Loaded message logged by read_proteomics_file()

    # Overwrite flag: config is source of truth, CLI --overwrite overrides via config
    overwrite = config.overwrite
    train_controls_incident_only = (
        cli_args.get("train_controls_incident_only", False) if cli_args else False
    )

    # Generate splits for each scenario
    for scenario in config.scenarios:
        _generate_scenario_splits(
            df=df,
            scenario=scenario,
            outdir=outdir,
            config=config,
            overwrite=overwrite,
            train_controls_incident_only=train_controls_incident_only,
            logger=logger,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Split generation complete!")
    logger.info(f"Output directory: {outdir}")
    logger.info("=" * 60)


def _generate_scenario_splits(
    df: pd.DataFrame,
    scenario: str,
    outdir: Path,
    config,
    overwrite: bool,
    train_controls_incident_only: bool,
    logger,
) -> None:
    """Generate splits for a single scenario."""
    positives = get_positives_from_scenario(scenario)
    scenario_description = SCENARIO_DEFINITIONS[scenario]["description"]

    eval_case_labels = positives
    if config.prevalent_train_only and PREVALENT_LABEL in positives:
        eval_case_labels = [INCIDENT_LABEL]
    train_case_labels = positives
    if train_controls_incident_only and PREVALENT_LABEL in positives:
        train_case_labels = [INCIDENT_LABEL]

    logger.info(f"\n{'='*60}")
    logger.info(f"=== Preparing {scenario} scenario ({scenario_description}) ===")
    logger.info(f"{'='*60}")

    if config.prevalent_train_only and PREVALENT_LABEL in positives:
        logger.info(f"  Prevalent handling: TRAIN only (frac={config.prevalent_train_frac:.2f})")

    # Filter to scenario
    keep_labels = [CONTROL_LABEL] + positives

    # Validate that all expected labels are present in data
    unique_labels = set(df[TARGET_COL].unique())
    missing_labels = set(keep_labels) - unique_labels
    if missing_labels:
        logger.warning(
            f"Expected labels {missing_labels} not found in data. Available: {unique_labels}"
        )

    # Check for unknown labels (not in any valid scenario)
    all_valid_labels = {CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL}
    unknown_labels = unique_labels - all_valid_labels
    if unknown_labels:
        logger.warning(f"Found unknown labels in data: {unknown_labels}. Will be filtered out.")

    mask = df[TARGET_COL].isin(keep_labels)
    n_filtered = (~mask).sum()
    if n_filtered > 0:
        logger.info(f"  Filtered out {n_filtered:,} samples with labels not in {keep_labels}")

    df_scenario_raw = df[mask].copy()
    logger.info(f"  {scenario} (raw, pre row-filters): {len(df_scenario_raw):,} samples")

    # Apply training row filters
    df_scenario, rf_stats = apply_row_filters(df_scenario_raw)
    logger.info("  Row-filter alignment (to match training dataset):")
    logger.info("    - drop_uncertain_controls=True")
    logger.info("    - dropna_meta_num=True")
    logger.info(f"    - removed_uncertain_controls={rf_stats['n_removed_uncertain_controls']:,}")
    logger.info(f"    - removed_dropna_meta_num={rf_stats['n_removed_dropna_meta_num']:,}")
    logger.info(f"  {scenario} (post row-filters): {len(df_scenario):,} samples")

    if config.temporal_split:
        logger.info(f"  Temporal ordering enabled (column={config.temporal_col})")
        order_idx = temporal_order_indices(df_scenario, config.temporal_col)
        df_scenario = df_scenario.iloc[order_idx].reset_index(drop=True)
        if len(df_scenario) > 0 and config.temporal_col in df_scenario.columns:
            logger.info(f"    Earliest value: {df_scenario[config.temporal_col].iloc[0]}")
            logger.info(f"    Latest value:   {df_scenario[config.temporal_col].iloc[-1]}")

    # Create outcome variable
    y_full = df_scenario[TARGET_COL].isin(positives).astype(int).to_numpy()
    n_controls = (y_full == 0).sum()
    n_cases = (y_full == 1).sum()
    logger.info(f"  Controls: {n_controls:,}")
    logger.info(f"  Cases: {n_cases:,} ({y_full.mean()*100:.3f}%)")

    full_idx = np.arange(len(df_scenario))

    # Handle holdout mode
    if config.mode == "holdout":
        df_work, y_work, index_space, dev_to_global_map = _create_holdout(
            df_scenario,
            y_full,
            full_idx,
            scenario,
            outdir,
            config,
            overwrite,
            rf_stats,
            logger,
        )
    else:
        logger.info("\n=== Development mode (no holdout) ===")
        df_work = df_scenario.copy()
        y_work = y_full
        index_space = "full"
        dev_to_global_map = None

    # Generate repeated splits
    _generate_repeated_splits(
        df_work,
        y_work,
        scenario,
        positives,
        train_case_labels,
        eval_case_labels,
        outdir,
        config,
        overwrite,
        rf_stats,
        index_space,
        dev_to_global_map,
        logger,
    )


def _create_holdout(
    df_scenario,
    y_full,
    full_idx,
    scenario,
    outdir,
    config,
    overwrite,
    rf_stats,
    logger,
):
    """Create holdout set and return development set."""
    from sklearn.model_selection import train_test_split

    logger.info(
        f"\n=== Creating holdout set ({config.holdout_size*100:.0f}% of post-filter scenario) ==="
    )

    if config.temporal_split:
        n_holdout = int(round(config.holdout_size * len(full_idx)))
        n_holdout = min(max(1, n_holdout), max(1, len(full_idx) - 1))
        holdout_idx_global = full_idx[-n_holdout:]
        dev_idx_global = full_idx[:-n_holdout]
        sch_full = "temporal"
        y_holdout = y_full[holdout_idx_global]
    else:
        strata_full, sch_full = build_working_strata(df_scenario, min_count=2)
        logger.info(f"  Holdout stratification: {sch_full}")
        # Fixed seed 42 for holdout ensures consistent final evaluation set across experiments
        # This is intentional - holdout should be stable for valid cross-experiment comparison
        holdout_seed = 42
        dev_pos_unsorted, holdout_pos, _, y_holdout = train_test_split(
            full_idx,
            y_full,
            test_size=config.holdout_size,
            random_state=holdout_seed,
            stratify=strata_full,
        )
        holdout_idx_global = np.array(holdout_pos, dtype=int)
        dev_mask = np.ones(len(full_idx), dtype=bool)
        dev_mask[holdout_idx_global] = False
        dev_idx_global = full_idx[dev_mask]

    df_dev = df_scenario.iloc[dev_idx_global].copy().reset_index(drop=True)
    y_dev = y_full[dev_idx_global]

    logger.info(f"  Full dataset: {len(full_idx):,}")
    logger.info(f"  Holdout set: {len(holdout_idx_global):,} (global idx space)")
    logger.info(f"  Development set: {len(df_dev):,} (dev-local idx space)")
    logger.info(f"  Holdout prevalence: {y_holdout.mean()*100:.3f}%")

    # Save HOLDOUT using persistence module
    holdout_path = save_holdout_indices(
        outdir=str(outdir),
        scenario=scenario,
        holdout_idx=holdout_idx_global,
        overwrite=overwrite,
    )
    logger.info(f"  [OK] Saved holdout indices: {holdout_path}")

    holdout_meta_path = save_holdout_metadata(
        outdir=str(outdir),
        scenario=scenario,
        holdout_idx=holdout_idx_global,
        y_holdout=y_holdout,
        strat_scheme=sch_full,
        row_filter_stats=rf_stats,
        temporal_split=config.temporal_split,
        temporal_col=config.temporal_col if config.temporal_split else None,
    )
    logger.info(f"  [OK] Saved holdout metadata: {holdout_meta_path}")

    return df_dev, y_dev, "dev", dev_idx_global


def _generate_repeated_splits(
    df_work,
    y_work,
    scenario,
    positives,
    train_case_labels,
    eval_case_labels,
    outdir,
    config,
    overwrite,
    rf_stats,
    index_space,
    dev_to_global_map,
    logger,
):
    """Generate repeated train/val/test splits."""
    # Base set for splitting
    if config.prevalent_train_only and PREVALENT_LABEL in positives:
        base_mask = df_work[TARGET_COL].isin([CONTROL_LABEL, INCIDENT_LABEL])
        df_base = df_work[base_mask].copy()
        base_idx = df_base.index.to_numpy(dtype=int)
        y_base = (df_base[TARGET_COL] == INCIDENT_LABEL).astype(int).to_numpy()
        logger.info(f"  Prevalent excluded from VAL/TEST. Base split set: {len(df_base):,}")
    else:
        df_base = df_work
        base_idx = np.arange(len(df_work), dtype=int)
        y_base = y_work

    # Build strata
    if config.temporal_split:
        strata_base, sch_work = None, "temporal"
    else:
        strata_base, sch_work = build_working_strata(df_base, min_count=2)

    logger.info(f"\n=== Stratification for train/val/test: {sch_work} ===")
    logger.info(f"[INFO] Split index space: {index_space}")
    logger.info(
        f"\n=== Generating {config.n_splits} split(s) (test_size={config.test_size}, val_size={config.val_size}) ==="
    )

    for i in range(config.n_splits):
        seed = config.seed_start + i
        logger.info(f"\n--- Split {i+1}/{config.n_splits} (seed={seed}) ---")

        # Generate base split
        if config.temporal_split:
            idx_train, idx_val, idx_test, y_train, y_val, y_test = temporal_train_val_test_split(
                base_idx, y_base, config.val_size, config.test_size
            )
        else:
            idx_train, idx_val, idx_test, y_train, y_val, y_test = stratified_train_val_test_split(
                base_idx,
                y_base,
                strata_base,
                config.val_size,
                config.test_size,
                seed,
            )

        idx_train = np.sort(idx_train.astype(int))
        idx_val = np.sort(idx_val.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        logger.info(
            f"  Train: {len(idx_train):,} samples ({int(y_train.sum())} cases, {y_train.mean()*100:.3f}%)"
        )
        if len(idx_val) > 0:
            logger.info(
                f"  Val:   {len(idx_val):,} samples ({int(y_val.sum())} cases, {y_val.mean()*100:.3f}%)"
            )
        logger.info(
            f"  Test:  {len(idx_test):,} samples ({int(y_test.sum())} cases, {y_test.mean()*100:.3f}%)"
        )

        rng = np.random.RandomState(seed + 1337)

        # Add prevalent to TRAIN only
        if config.prevalent_train_only and PREVALENT_LABEL in positives:
            idx_train = add_prevalent_to_train(idx_train, df_work, config.prevalent_train_frac, rng)
            # Logging handled by add_prevalent_to_train()

        # Downsample controls
        idx_train = downsample_controls(
            idx_train, df_work, train_case_labels, config.train_control_per_case, rng
        )

        if config.eval_control_per_case is not None and config.eval_control_per_case > 0:
            if len(idx_val) > 0:
                idx_val = downsample_controls(
                    idx_val,
                    df_work,
                    eval_case_labels,
                    config.eval_control_per_case,
                    rng,
                )
            idx_test = downsample_controls(
                idx_test, df_work, eval_case_labels, config.eval_control_per_case, rng
            )

        # Recompute y arrays after modifications
        y_train = (df_work.loc[idx_train, TARGET_COL].isin(positives)).astype(int).to_numpy()
        y_val = (
            (df_work.loc[idx_val, TARGET_COL].isin(eval_case_labels)).astype(int).to_numpy()
            if len(idx_val) > 0
            else np.array([], dtype=int)
        )
        y_test = (df_work.loc[idx_test, TARGET_COL].isin(eval_case_labels)).astype(int).to_numpy()

        logger.info(
            f"  Final Train: {len(idx_train):,} samples ({int(y_train.sum())} cases, {y_train.mean()*100:.3f}%)"
        )
        if len(idx_val) > 0:
            logger.info(
                f"  Final Val:   {len(idx_val):,} samples ({int(y_val.sum())} cases, {y_val.mean()*100:.3f}%)"
            )
        logger.info(
            f"  Final Test:  {len(idx_test):,} samples ({int(y_test.sum())} cases, {y_test.mean()*100:.3f}%)"
        )

        # Validate prevalent cases didn't leak into evaluation sets
        if config.prevalent_train_only and PREVALENT_LABEL in positives:
            prevalent_mask = df_work[TARGET_COL] == PREVALENT_LABEL
            prevalent_idx = df_work.index[prevalent_mask].tolist()
            if len(idx_val) > 0:
                check_prevalent_in_eval(
                    eval_idx=idx_val.tolist(),
                    prevalent_idx=prevalent_idx,
                    split_name="validation",
                    strictness="error",
                )
            check_prevalent_in_eval(
                eval_idx=idx_test.tolist(),
                prevalent_idx=prevalent_idx,
                split_name="test",
                strictness="error",
            )

        # Convert dev-local indices to global indices if in holdout mode
        if dev_to_global_map is not None:
            idx_train = dev_to_global_map[idx_train]
            idx_test = dev_to_global_map[idx_test]
            if len(idx_val) > 0:
                idx_val = dev_to_global_map[idx_val]

        # Save indices using persistence module
        saved_paths = save_split_indices(
            outdir=str(outdir),
            scenario=scenario,
            seed=seed,
            train_idx=idx_train,
            test_idx=idx_test,
            val_idx=idx_val if len(idx_val) > 0 else None,
            overwrite=overwrite,
        )

        logger.info(f"  [OK] Saved: {saved_paths['train']}")
        if "val" in saved_paths:
            logger.info(f"  [OK] Saved: {saved_paths['val']}")
        logger.info(f"  [OK] Saved: {saved_paths['test']}")

        # Save metadata using persistence module
        meta_path = save_split_metadata(
            outdir=str(outdir),
            scenario=scenario,
            seed=seed,
            train_idx=idx_train,
            test_idx=idx_test,
            y_train=y_train,
            y_test=y_test,
            val_idx=idx_val if len(idx_val) > 0 else None,
            y_val=y_val if len(idx_val) > 0 else None,
            split_type="development",
            strat_scheme=sch_work,
            row_filter_stats=rf_stats,
            index_space=index_space,
            temporal_split=config.temporal_split,
            temporal_col=config.temporal_col if config.temporal_split else None,
        )
        logger.info(f"  [OK] Saved metadata: {meta_path}")
