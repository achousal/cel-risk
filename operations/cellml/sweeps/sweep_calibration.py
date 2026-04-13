"""Pre-sweep calibration runner.

Executes a small Optuna study to measure where the best-so-far curve
plateaus for a given parameter space + dataset, then emits recommended
n_trials_cap and patience values for the parent sweep. The orchestrator
calls `ensure_calibration()` before the first parent iteration.

This module is sampler-aware but objective-agnostic: the caller supplies
an `objective_fn(trial) -> float`. For cel-risk, the production wiring
uses a cheap surrogate objective (e.g. single-seed cross-validated AUROC
on a subsampled dataset) rather than the full Minerva pipeline. That
wiring lives in sweep_orchestrator.ensure_calibration().

Artifacts per calibration run (Rule 9):
  <calibration_dir>/<calibration_id>.json     -- CalibrationResult
  <calibration_dir>/<calibration_id>.parquet  -- trial-by-trial log
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .calibration_schema import (
    CalibrationConfidence,
    CalibrationConfig,
    CalibrationResult,
    ProposedSweepParams,
    TrialRecord,
    WarmStartPoint,
)
from .sweep_schema import SweepSpec

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[dict], float]
ParamSuggester = Callable[[int], dict]
OnResultFn = Callable[[int, dict, float], None]


def _calibration_id(sweep_id: str, space_hash: str, fingerprint: str) -> str:
    return f"{sweep_id}__{space_hash}__{fingerprint}"


def _n_calib(config: CalibrationConfig, space_dim: int) -> int:
    """Rule 5: n_calib = min(trials_per_dim * dim, max_trials), floor min_trials."""
    raw = config.trials_per_dim * max(space_dim, 1)
    return int(max(config.min_trials, min(raw, config.max_trials)))


def detect_plateau(
    best_so_far: list[float],
    improvement_threshold: float,
    direction: str = "maximize",
) -> int | None:
    """Return the first trial index k where best_so_far[k] is within
    improvement_threshold of the final best_so_far. Returns None if the
    curve never stabilizes (i.e., best improves in the last trial).

    Rule 10: plateau_trial = first k s.t.
        |best_so_far[k] - best_so_far[-1]| <= improvement_threshold.
    """
    if not best_so_far:
        return None
    final = best_so_far[-1]
    for k, value in enumerate(best_so_far):
        if abs(value - final) <= improvement_threshold:
            # Rule 14 plateau-found check: must be stable, not a one-off
            # dip that re-improves later. We required monotone best_so_far
            # (Optuna maintains this for single-objective), so the first
            # time we reach within threshold is the knee.
            return k
    return None


def _noise_sigma(objective_trace: list[float], tail_fraction: float = 0.2) -> float:
    """Rule 11: σ of objective across last 20% of trials."""
    if len(objective_trace) < 5:
        return 0.0
    tail_n = max(2, math.ceil(len(objective_trace) * tail_fraction))
    tail = objective_trace[-tail_n:]
    mean = sum(tail) / len(tail)
    var = sum((x - mean) ** 2 for x in tail) / len(tail)
    return math.sqrt(var)


def _confidence(plateau_trial: int | None, n_calib: int) -> CalibrationConfidence:
    """Rule 14: high/medium/low confidence based on plateau location."""
    if plateau_trial is None:
        return CalibrationConfidence.low
    if plateau_trial <= 0.7 * n_calib:
        return CalibrationConfidence.high
    if plateau_trial < n_calib:
        return CalibrationConfidence.medium
    return CalibrationConfidence.low


def _select_warm_start_points(
    trials: list[TrialRecord],
    direction: str,
    top_k: int,
) -> list[WarmStartPoint]:
    """Return the top-k DISTINCT trials by objective, best first.

    Warm-start points seed the parent sweep's Optuna study via
    enqueue_trial() so calibration's best configs are evaluated by the
    real pipeline instead of being discarded. Points are hints -- the
    parent sweep still explores beyond them.

    De-duplicates on params_json so TPE convergence to a single
    best value does not waste warm-start slots on the same config.
    When fewer than top_k unique params exist, returns what's
    available rather than padding.
    """
    if top_k <= 0 or not trials:
        return []
    reverse = direction == "maximize"
    ordered = sorted(trials, key=lambda t: t.objective, reverse=reverse)
    seen: set[str] = set()
    unique: list[TrialRecord] = []
    for t in ordered:
        if t.params_json in seen:
            continue
        seen.add(t.params_json)
        unique.append(t)
        if len(unique) >= top_k:
            break
    return [
        WarmStartPoint(
            params=json.loads(t.params_json),
            objective=t.objective,
            calibration_trial_idx=t.trial_idx,
        )
        for t in unique
    ]


def _build_proposed(
    plateau_trial: int | None,
    config: CalibrationConfig,
    confidence: CalibrationConfidence,
    calibration_id: str,
    space_dim: int,
    warm_start_points: list[WarmStartPoint],
) -> ProposedSweepParams | None:
    """Rules 12-13: translate plateau trial -> n_trials_cap and patience."""
    if plateau_trial is None:
        return None

    # Rule 12
    raw_cap = math.ceil(config.cap_safety_multiplier * (plateau_trial + 1))
    cap = max(
        max(config.absolute_min_cap, 2 * space_dim),
        min(raw_cap, config.absolute_max_cap),
    )

    # Rule 13
    raw_patience = math.ceil(config.patience_fraction_of_plateau * (plateau_trial + 1))
    patience = max(config.min_patience, min(raw_patience, config.max_patience))

    return ProposedSweepParams(
        n_trials_cap=cap,
        patience=patience,
        calibration_id=calibration_id,
        confidence=confidence,
        warm_start_points=warm_start_points,
    )


def _write_artifacts(
    result: CalibrationResult,
    trials: list[TrialRecord],
    calibration_dir: Path,
) -> None:
    """Persist calibration JSON + trial-by-trial parquet (Rule 9)."""
    calibration_dir.mkdir(parents=True, exist_ok=True)
    json_path = calibration_dir / f"{result.calibration_id}.json"
    parquet_path = calibration_dir / f"{result.calibration_id}.parquet"

    json_path.write_text(result.model_dump_json(indent=2))
    pd.DataFrame([t.model_dump() for t in trials]).to_parquet(parquet_path, index=False)
    logger.info("Calibration artifacts written: %s", json_path.name)


def load_cached_calibration(
    calibration_dir: Path,
    calibration_id: str,
    cache_days: int,
) -> CalibrationResult | None:
    """Rule 3: cache hit if result exists and is younger than cache_days."""
    path = calibration_dir / f"{calibration_id}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        result = CalibrationResult(**raw)
    except Exception as exc:  # malformed or schema-drifted
        logger.warning("Cached calibration %s unreadable: %s", calibration_id, exc)
        return None

    age = datetime.now(timezone.utc) - datetime.fromisoformat(result.created_utc)
    if age > timedelta(days=cache_days):
        logger.info("Cached calibration %s is stale (age=%s)", calibration_id, age)
        return None
    return result


def run_calibration(
    spec: SweepSpec,
    config: CalibrationConfig,
    space_hash: str,
    dataset_fingerprint: str,
    param_suggester: ParamSuggester,
    objective_fn: ObjectiveFn,
    calibration_dir: Path,
    subsample_rows_used: int,
    parent_sweep_version: str | None = None,
    on_result: OnResultFn | None = None,
) -> CalibrationResult:
    """Execute the calibration study and return its result.

    Parameters
    ----------
    spec
        Parent sweep spec (used for id, improvement_threshold, direction).
    config
        CalibrationConfig block from the spec.
    space_hash, dataset_fingerprint
        Cache key components; must already be computed by the caller.
    param_suggester
        Callable (trial_idx) -> dict[str, Any]. In production this wraps
        an optuna.Trial.suggest_* chain; tests may pass a deterministic
        stub.
    objective_fn
        Callable (params) -> float. Maximize or minimize per spec.metric_direction.
    calibration_dir
        Directory to write artifacts.
    subsample_rows_used
        Actual row count after subsampling (for provenance).
    parent_sweep_version
        Optional git SHA or version string for lineage.

    Returns
    -------
    CalibrationResult. If aborted (Rule 15), `aborted=True` and
    `proposed=None`. Callers MUST NOT silently fall back to defaults --
    the orchestrator raises on abort (Rule 19).
    """
    direction = spec.metric_direction
    threshold = spec.constraints.improvement_threshold
    space_dim = len(spec.parameter_space)

    n_calib = _n_calib(config, space_dim)
    calibration_id = _calibration_id(spec.id, space_hash, dataset_fingerprint)

    logger.info(
        "Starting calibration %s: n_calib=%d dim=%d sampler=%s",
        calibration_id, n_calib, space_dim, config.sampler,
    )

    trials: list[TrialRecord] = []
    best_so_far: list[float] = []
    objective_trace: list[float] = []
    running_best: float | None = None
    start = time.time()

    def _is_better(new: float, current: float | None) -> bool:
        if current is None:
            return True
        return new > current if direction == "maximize" else new < current

    for trial_idx in range(n_calib):
        # Rule 4 wall-hour enforcement (hard cap)
        elapsed_h = (time.time() - start) / 3600.0
        if elapsed_h >= config.hard_cap_wall_hours:
            logger.warning(
                "Calibration %s hit hard wall cap at trial %d/%d",
                calibration_id, trial_idx, n_calib,
            )
            break

        params = param_suggester(trial_idx)
        t0 = time.time()
        objective = float(objective_fn(params))
        wall_seconds = time.time() - t0

        if on_result is not None:
            on_result(trial_idx, params, objective)

        objective_trace.append(objective)
        if _is_better(objective, running_best):
            running_best = objective
        assert running_best is not None
        best_so_far.append(running_best)

        trials.append(
            TrialRecord(
                trial_idx=trial_idx,
                objective=objective,
                best_so_far=running_best,
                wall_seconds=wall_seconds,
                params_json=json.dumps(params, sort_keys=True, default=str),
            )
        )

    wall_seconds_total = time.time() - start
    n_executed = len(trials)

    # Rule 10: plateau
    plateau_trial = detect_plateau(best_so_far, threshold, direction)

    # Rule 11: noise floor
    sigma = _noise_sigma(objective_trace)
    noise_warning = sigma > threshold

    # Rule 14: confidence
    confidence = _confidence(plateau_trial, n_calib)

    # Rule 15: abort on unmet min_viable_objective
    aborted = False
    abort_reason: str | None = None
    if config.min_viable_objective is not None and running_best is not None:
        viable = (
            running_best >= config.min_viable_objective
            if direction == "maximize"
            else running_best <= config.min_viable_objective
        )
        if not viable:
            aborted = True
            abort_reason = (
                f"best objective {running_best:.6f} below min_viable_objective "
                f"{config.min_viable_objective} -- parent sweep blocked"
            )

    warm_start_points = _select_warm_start_points(
        trials, direction, config.warm_start_top_k,
    )
    proposed = None if aborted else _build_proposed(
        plateau_trial, config, confidence, calibration_id, space_dim,
        warm_start_points,
    )

    result = CalibrationResult(
        calibration_id=calibration_id,
        sweep_id=spec.id,
        space_hash=space_hash,
        dataset_fingerprint=dataset_fingerprint,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_calib_requested=n_calib,
        n_calib_executed=n_executed,
        sampler=config.sampler,
        calib_seed=config.calib_seed,
        subsample_rows_used=subsample_rows_used,
        wall_seconds_total=round(wall_seconds_total, 2),
        plateau_trial=plateau_trial,
        plateau_value=best_so_far[plateau_trial] if plateau_trial is not None else None,
        noise_sigma=round(sigma, 6),
        noise_warning=noise_warning,
        proposed=proposed,
        aborted=aborted,
        abort_reason=abort_reason,
        parent_sweep_version=parent_sweep_version,
    )

    _write_artifacts(result, trials, calibration_dir)
    return result


class CalibrationError(RuntimeError):
    """Raised by the orchestrator when calibration fails or aborts.

    Rule 19: the parent sweep MUST NOT silently fall back to defaults.
    A CalibrationError is the load-bearing failure signal.
    """
