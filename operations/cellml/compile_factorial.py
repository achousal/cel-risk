#!/usr/bin/env python3
"""Compile results from factorial cells into a single results table.

Supports two compilation modes:
  1. Filesystem-based (default): reads per-cell test_metrics_summary.csv
     from the deterministic directory layout produced by submit_experiment.sh
  2. Optuna storage-based: reads directly from JournalStorage files

Usage:
    # Filesystem mode — V0 gate
    python compile_factorial.py \
        --manifest analysis/configs/recipes/v0/v0_cell_manifest.csv \
        --results-dir results/v0_gate \
        --output results/v0_gate/v0_compiled.csv

    # Filesystem mode — full factorial
    python compile_factorial.py \
        --manifest analysis/configs/recipes/cell_manifest.csv \
        --results-dir results/factorial \
        --output results/factorial/factorial_compiled.csv

    # Optuna storage mode (unchanged)
    python compile_factorial.py \
        --optuna-storage-dir /path/to/optuna/ \
        --output results/factorial_compiled_optuna.csv

Directory layout convention (produced by submit_experiment.sh):
    {results_dir}/{recipe_id}/{cell_name}/run_{recipe_id}__{cell_name}/{model}/
        aggregated/metrics/test_metrics_summary.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that get the summary_ prefix for validate_tree.R compatibility.
# Only columns ending with these suffixes are metric aggregates.
_METRIC_SUFFIXES = ("_mean", "_std", "_ci95_lo", "_ci95_hi")

# Columns from the pipeline that should NOT be prefixed even if they
# happen to match a suffix (grouping/identity columns).
_NO_PREFIX = {"cell_name", "cell_id", "recipe_id", "scenario", "model"}


def _build_cell_path(
    results_dir: Path,
    recipe_id: str,
    cell_name: str,
    model: str,
) -> Path:
    """Deterministic path to per-cell aggregated test metrics.

    Layout: {results_dir}/{recipe_id}/{cell_name}/
                run_{recipe_id}__{cell_name}/{model}/
                aggregated/metrics/test_metrics_summary.csv
    """
    run_id = f"{recipe_id}__{cell_name}"
    return (
        results_dir
        / recipe_id
        / cell_name
        / f"run_{run_id}"
        / model
        / "aggregated"
        / "metrics"
        / "test_metrics_summary.csv"
    )


def _rename_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix metric aggregate columns with summary_ for validate_tree.R."""
    rename_map = {}
    for col in df.columns:
        if col in _NO_PREFIX:
            continue
        if any(col.endswith(suffix) for suffix in _METRIC_SUFFIXES):
            rename_map[col] = f"summary_{col}"
    return df.rename(columns=rename_map)


def compile_factorial(
    manifest_csv: Path,
    results_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Compile factorial cell results from filesystem into a single table.

    Parameters
    ----------
    manifest_csv : Path
        Cell manifest CSV (v0_cell_manifest.csv or cell_manifest.csv).
    results_dir : Path
        Root results directory (e.g., results/v0_gate or results/factorial).
    output_path : Path
        Where to write compiled CSV.

    Returns
    -------
    pd.DataFrame
        Compiled results with factorial factor columns and summary_ prefixed
        metric columns ready for validate_tree.R.
    """
    manifest = pd.read_csv(manifest_csv)
    logger.info("Cell manifest: %d cells", len(manifest))

    # Detect which factorial columns are available
    has_strategy = "strategy" in manifest.columns
    has_ctrl_ratio = "control_ratio" in manifest.columns
    has_calibration = "calibration" in manifest.columns
    has_weighting = "weighting" in manifest.columns
    has_downsampling = "downsampling" in manifest.columns

    compiled_rows = []
    missing = []

    for _, cell in manifest.iterrows():
        cell_id = int(cell["cell_id"])
        recipe_id = cell["recipe_id"]
        model = cell["model"]
        cell_name = cell["cell_name"]

        metrics_path = _build_cell_path(results_dir, recipe_id, cell_name, model)

        if not metrics_path.exists():
            missing.append((cell_id, recipe_id, cell_name, str(metrics_path)))
            continue

        result_df = pd.read_csv(metrics_path)

        # Rename metric columns: auroc_mean -> summary_auroc_mean
        result_df = _rename_metric_columns(result_df)

        # Add factorial identity columns
        result_df["cell_id"] = cell_id
        result_df["recipe_id"] = recipe_id
        result_df["cell_name"] = cell_name
        result_df["factorial_model"] = model

        if has_calibration:
            result_df["factorial_calibration"] = cell["calibration"]
        if has_weighting:
            result_df["factorial_weighting"] = cell["weighting"]
        if has_downsampling:
            result_df["factorial_downsampling"] = cell["downsampling"]
        if has_strategy:
            result_df["factorial_strategy"] = cell["strategy"]
        if has_ctrl_ratio:
            result_df["factorial_control_ratio"] = cell["control_ratio"]

        compiled_rows.append(result_df)

    # Report missing cells
    if missing:
        logger.warning("%d of %d cells missing results:", len(missing), len(manifest))
        for cell_id, recipe, name, path in missing:
            logger.warning("  cell %d (%s/%s): %s", cell_id, recipe, name, path)

    if not compiled_rows:
        logger.error("No results found for any cell")
        return pd.DataFrame()

    compiled = pd.concat(compiled_rows, ignore_index=True)

    # Sort by available factorial columns
    sort_cols = [
        c for c in [
            "recipe_id", "factorial_model", "factorial_calibration",
            "factorial_weighting", "factorial_downsampling",
            "factorial_strategy", "factorial_control_ratio",
        ]
        if c in compiled.columns
    ]
    if sort_cols:
        compiled = compiled.sort_values(sort_cols, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.to_csv(output_path, index=False)
    logger.info(
        "Compiled %d rows from %d cells (%d missing) -> %s",
        len(compiled),
        len(manifest) - len(missing),
        len(missing),
        output_path,
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

                # Fold-level variance
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
        help="Root results directory, e.g. results/v0_gate (filesystem mode)",
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
