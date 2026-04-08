#!/usr/bin/env python3
"""Compile results from factorial cells into a single results table.

Supports two compilation modes:
  1. Filesystem-based (default): reads per-cell aggregated_results.csv files
  2. Optuna storage-based: reads directly from JournalStorage files

Usage:
    # Filesystem mode
    python compile_factorial.py \
        --manifest configs/recipes/cell_manifest.csv \
        --results-dir results/ \
        --output results/factorial_compiled.csv

    # Optuna storage mode
    python compile_factorial.py \
        --optuna-storage-dir /path/to/optuna/ \
        --output results/factorial_compiled_optuna.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compile_factorial(
    manifest_csv: Path,
    results_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Compile factorial cell results from filesystem into a single table.

    Parameters
    ----------
    manifest_csv : Path
        Global cell_manifest.csv from config generation.
    results_dir : Path
        Root results directory.
    output_path : Path
        Where to write compiled CSV.

    Returns
    -------
    pd.DataFrame
        Compiled results with factorial factor columns.
    """
    manifest = pd.read_csv(manifest_csv)
    logger.info("Cell manifest: %d cells", len(manifest))

    compiled_rows = []
    missing = 0

    for _, cell in manifest.iterrows():
        cell_id = cell["cell_id"]
        recipe_id = cell["recipe_id"]
        model = cell["model"]

        # Locate per-cell aggregated results
        # Convention: results/<run_id>/<model>/aggregated_results.csv
        # The run_id encodes recipe + cell info
        cell_name = cell["cell_name"]

        # Search for aggregated results
        candidates = list(results_dir.glob(f"**/{model}/aggregated_results.csv"))
        # Filter to those matching this cell
        cell_results = [c for c in candidates if cell_name in str(c)]

        if not cell_results:
            logger.warning("No results found for cell %d (%s/%s)", cell_id, recipe_id, cell_name)
            missing += 1
            continue

        # Read the first matching result
        result_df = pd.read_csv(cell_results[0])

        # Add factorial factor columns
        result_df["cell_id"] = cell_id
        result_df["recipe_id"] = recipe_id
        result_df["factorial_model"] = model
        result_df["factorial_calibration"] = cell["calibration"]
        result_df["factorial_weighting"] = cell["weighting"]
        result_df["factorial_downsampling"] = cell["downsampling"]
        result_df["cell_name"] = cell_name

        compiled_rows.append(result_df)

    if not compiled_rows:
        logger.error("No results found for any cell")
        return pd.DataFrame()

    compiled = pd.concat(compiled_rows, ignore_index=True)

    # Sort by recipe, model, calibration, weighting, downsampling
    compiled = compiled.sort_values(
        ["recipe_id", "factorial_model", "factorial_calibration",
         "factorial_weighting", "factorial_downsampling"],
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.to_csv(output_path, index=False)
    logger.info(
        "Compiled %d rows from %d cells (%d missing) -> %s",
        len(compiled), len(manifest), missing, output_path,
    )
    return compiled


def compile_from_storage(
    storage_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Compile factorial results directly from Optuna shared storage.

    Loads each study by name, extracts best trial metrics and factorial
    metadata from study user_attrs. One-liner per study instead of
    filesystem glob.

    Parameters
    ----------
    storage_dir : Path
        Directory containing .optuna.journal files.
    output_path : Path
        Where to write compiled CSV.

    Returns
    -------
    pd.DataFrame
        Compiled results with factorial factors and Optuna metrics.
    """
    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    journal_files = sorted(storage_dir.glob("*.optuna.journal"))
    if not journal_files:
        logger.error("No .optuna.journal files found in %s", storage_dir)
        return pd.DataFrame()

    rows = []
    for journal_path in journal_files:
        try:
            lock_obj = optuna.storages.JournalFileOpenLock(str(journal_path))
            storage = JournalStorage(
                JournalFileStorage(str(journal_path), lock_obj=lock_obj)
            )
        except Exception as e:
            logger.warning("Failed to open %s: %s", journal_path, e)
            continue

        summaries = optuna.study.get_all_study_summaries(storage)
        logger.info(
            "%s: %d studies", journal_path.name, len(summaries)
        )

        for summary in summaries:
            try:
                study = optuna.load_study(
                    study_name=summary.study_name, storage=storage
                )
            except Exception as e:
                logger.warning(
                    "Failed to load study '%s': %s", summary.study_name, e
                )
                continue

            # Extract factorial factors from user_attrs
            row = dict(study.user_attrs)
            row["study_name"] = study.study_name

            completed = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            row["n_completed"] = len(completed)
            row["n_total"] = len(study.trials)

            if completed:
                # Multi-objective: best_trials is Pareto frontier
                try:
                    best = study.best_trials
                    if best:
                        selected = best[0]
                        row["best_auroc"] = selected.values[0]
                        row["best_neg_brier"] = selected.values[1]
                    else:
                        selected = max(completed, key=lambda t: t.value)
                        row["best_score"] = selected.value
                except Exception:
                    # Single-objective fallback
                    selected = max(completed, key=lambda t: t.value)
                    row["best_score"] = selected.value

                row["best_params"] = str(selected.params)

                # Fold-level variance (Enhancement 4)
                if "auroc_std" in selected.user_attrs:
                    row["auroc_std"] = selected.user_attrs["auroc_std"]
                if "brier_std" in selected.user_attrs:
                    row["brier_std"] = selected.user_attrs["brier_std"]

            rows.append(row)

    if not rows:
        logger.error("No studies found")
        return pd.DataFrame()

    compiled = pd.DataFrame(rows)

    # Sort if factorial columns present
    sort_cols = [
        c
        for c in ["recipe_id", "model", "calibration", "weighting", "downsampling"]
        if c in compiled.columns
    ]
    if sort_cols:
        compiled = compiled.sort_values(sort_cols, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.to_csv(output_path, index=False)
    logger.info("Compiled %d study rows -> %s", len(compiled), output_path)
    return compiled


def main():
    parser = argparse.ArgumentParser(description="Compile factorial cell results")

    # Filesystem mode args
    parser.add_argument(
        "--manifest", type=Path,
        help="Path to cell_manifest.csv (filesystem mode)",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        help="Root results directory (filesystem mode)",
    )

    # Optuna storage mode args
    parser.add_argument(
        "--optuna-storage-dir", type=Path,
        help="Directory with .optuna.journal files (storage mode)",
    )

    # Common args
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output CSV path",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.optuna_storage_dir:
        compile_from_storage(args.optuna_storage_dir, args.output)
    elif args.manifest and args.results_dir:
        compile_factorial(args.manifest, args.results_dir, args.output)
    else:
        parser.error(
            "Provide either --optuna-storage-dir (storage mode) "
            "or --manifest + --results-dir (filesystem mode)"
        )


if __name__ == "__main__":
    main()
