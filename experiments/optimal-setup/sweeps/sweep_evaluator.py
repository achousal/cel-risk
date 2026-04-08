"""Sweep result evaluator.

Reads aggregated results from a completed sweep iteration,
extracts the declared metric, and compares to baseline/previous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one sweep iteration."""

    metric_value: float
    delta_baseline: float | None  # None if no baseline established yet
    delta_previous: float | None  # None if first iteration
    running_best: float
    is_improvement: bool


def evaluate_iteration(
    results_dir: Path,
    metric_column: str,
    metric_direction: str,
    baseline_value: float | None,
    previous_value: float | None,
    running_best: float | None,
    improvement_threshold: float = 0.005,
) -> EvalResult:
    """Evaluate a completed sweep iteration.

    Parameters
    ----------
    results_dir
        Path to iteration results (contains model subdirs with aggregated/).
    metric_column
        Column name in aggregated metrics CSV.
    metric_direction
        'maximize' or 'minimize'.
    baseline_value
        Known baseline (None = this is the first measurement).
    previous_value
        Metric from the previous iteration (None = first iteration).
    running_best
        Best metric seen so far (None = first iteration).
    improvement_threshold
        Minimum delta to count as improvement.

    Returns
    -------
    EvalResult
    """
    metric_value = _extract_metric(results_dir, metric_column)

    # Compute deltas
    delta_baseline = None
    if baseline_value is not None:
        delta_baseline = metric_value - baseline_value

    delta_previous = None
    if previous_value is not None:
        delta_previous = metric_value - previous_value

    # Update running best
    if running_best is None:
        new_best = metric_value
    elif metric_direction == "maximize":
        new_best = max(running_best, metric_value)
    else:
        new_best = min(running_best, metric_value)

    # Determine if this is an improvement
    if running_best is None:
        is_improvement = True  # First iteration always counts
    elif metric_direction == "maximize":
        is_improvement = (metric_value - running_best) >= improvement_threshold
    else:
        is_improvement = (running_best - metric_value) >= improvement_threshold

    return EvalResult(
        metric_value=metric_value,
        delta_baseline=delta_baseline,
        delta_previous=delta_previous,
        running_best=new_best,
        is_improvement=is_improvement,
    )


def _extract_metric(results_dir: Path, metric_column: str) -> float:
    """Extract a scalar metric from aggregated results.

    Scans for aggregated metrics CSVs under results_dir and
    returns the mean value of the specified column across models.
    """
    # Look for aggregated results in model subdirs
    candidates = list(results_dir.glob("*/aggregated/all_test_metrics.csv"))
    if not candidates:
        # Try direct aggregated_results.csv (compile_factorial output)
        candidates = list(results_dir.glob("**/aggregated_results.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No aggregated metrics found in {results_dir}. "
            "Check that the pipeline completed successfully."
        )

    dfs = []
    for csv_path in candidates:
        df = pd.read_csv(csv_path)
        if metric_column not in df.columns:
            available = ", ".join(sorted(df.columns))
            raise KeyError(
                f"Metric '{metric_column}' not found in {csv_path}. "
                f"Available columns: {available}"
            )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    value = combined[metric_column].mean()

    logger.info(
        "Extracted %s = %.6f from %d file(s) in %s",
        metric_column, value, len(candidates), results_dir,
    )
    return float(value)
