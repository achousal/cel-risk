"""Sweep orchestrator state machine.

Manages the submit-poll-evaluate loop for one sweep. Enforces budget
constraints and review gates. The orchestrator is the harness an agent
calls -- it is not an LLM agent itself.

States: PROPOSE -> SUBMIT -> POLL -> EVALUATE -> DECIDE -> (PROPOSE | TERMINATE)
"""

from __future__ import annotations

import enum
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ced_ml.data.fingerprint import dataset_fingerprint, space_hash

from .calibration_schema import CalibrationResult
from .optuna_driver import OptunaDriver
from .surrogate_objective import build_surrogate_objective
from .minerva_poller import JobResult, submit_and_wait
from .sweep_calibration import (
    CalibrationError,
    ObjectiveFn,
    ParamSuggester,
    load_cached_calibration,
    run_calibration,
)
from .sweep_config_overlay import generate_overlay, resolve_base_config
from .sweep_evaluator import EvalResult, evaluate_iteration
from .sweep_ledger import SweepLedger
from .sweep_schema import ParameterDef, ParameterType, SweepSpec, SweepType

logger = logging.getLogger(__name__)


class SweepState(str, enum.Enum):
    PROPOSE = "PROPOSE"
    SUBMIT = "SUBMIT"
    POLL = "POLL"
    EVALUATE = "EVALUATE"
    DECIDE = "DECIDE"
    REVIEW_GATE = "REVIEW_GATE"
    TERMINATED = "TERMINATED"


class TerminationReason(str, enum.Enum):
    MAX_ITERATIONS = "max_iterations"
    MAX_WALL_HOURS = "max_wall_hours"
    EARLY_STOP = "early_stop"
    AGENT_DECISION = "agent_decision"
    JOB_FAILURE = "job_failure"
    REVIEW_PENDING = "review_pending"
    CALIBRATION_FAILED = "calibration_failed"
    CALIBRATION_ABORTED = "calibration_aborted"


class SweepOrchestrator:
    """State machine for one adaptive sweep."""

    def __init__(
        self,
        spec: SweepSpec,
        project_root: Path,
        recipes_dir: Path | None = None,
        results_root: Path | None = None,
        ledger_dir: Path | None = None,
        dry_run: bool = False,
    ):
        self.spec = spec
        self.project_root = project_root
        self.recipes_dir = recipes_dir or (project_root / "analysis" / "configs" / "recipes")
        self.results_root = results_root or (project_root / "results" / "sweeps" / spec.id)
        self.dry_run = dry_run

        ledger_dir = ledger_dir or (project_root / "experiments" / "optimal-setup" / "sweeps" / "ledger")
        self.ledger = SweepLedger(ledger_dir, spec.id)

        # Calibration artifact directory mirrors ledger layout (Rule 9).
        self.calibration_dir = ledger_dir.parent / "calibration"

        self.state = SweepState.PROPOSE
        self._total_wall_hours = 0.0
        self._calibration_result: CalibrationResult | None = None

    def ensure_calibration(
        self,
        param_suggester: ParamSuggester | None = None,
        objective_fn: ObjectiveFn | None = None,
        subsample_rows_used: int | None = None,
        parent_sweep_version: str | None = None,
        force_rerun: bool = False,
        on_result=None,
    ) -> CalibrationResult:
        """Run or reuse calibration before the parent sweep starts.

        Rules 1-3 (trigger/skip/cache), 17-19 (provenance, stale, hard fail).

        Parameters
        ----------
        param_suggester
            Callable that takes a trial index and returns a dict of
            parameter values for that trial. Production callers wrap an
            optuna.Trial; tests can pass a stub.
        objective_fn
            Cheap surrogate objective (single-seed CV on a subsample).
            MUST NOT invoke the full Minerva pipeline.
        subsample_rows_used
            Row count after stratified subsample (for provenance).
        parent_sweep_version
            Optional git SHA / version string for lineage.
        force_rerun
            If True, ignore the cache. Use for recalibration after
            dataset or space changes.

        Returns
        -------
        CalibrationResult that passed is_usable(). On failure or abort
        raises CalibrationError -- the parent sweep MUST NOT start with
        defaults (Rule 19).
        """
        config = self.spec.calibration

        if not config.required:
            logger.info("Calibration not required for sweep %s; skipping", self.spec.id)
            return CalibrationResult(
                calibration_id=f"{self.spec.id}__skipped",
                sweep_id=self.spec.id,
                space_hash="skipped",
                dataset_fingerprint="skipped",
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                n_calib_requested=0,
                n_calib_executed=0,
                sampler=config.sampler,
                calib_seed=config.calib_seed,
                subsample_rows_used=0,
                wall_seconds_total=0.0,
                plateau_trial=None,
                plateau_value=None,
                noise_sigma=0.0,
                noise_warning=False,
                proposed=None,
                aborted=False,
                abort_reason=None,
            )

        if not self.spec.data_path:
            raise CalibrationError(
                f"Sweep {self.spec.id} has calibration.required=true but no "
                f"data_path; cannot compute dataset fingerprint."
            )

        data_path = self.project_root / self.spec.data_path
        fingerprint = dataset_fingerprint(
            data_path=data_path,
            label_col=self.spec.label_col,
            seeds=self.spec.constraints.seeds,
            extra={"base_recipe": self.spec.base_recipe, "base_cell": self.spec.base_cell},
        )
        sh = space_hash(self.spec.parameter_space)

        calibration_id = f"{self.spec.id}__{sh}__{fingerprint}"

        if not force_rerun:
            cached = load_cached_calibration(
                self.calibration_dir, calibration_id, config.cache_days,
            )
            if cached is not None and cached.is_usable():
                logger.info("Calibration cache hit: %s", calibration_id)
                self._calibration_result = cached
                return cached

        # Default wiring: Optuna driver + LR surrogate. Callers can
        # override either by passing their own suggester / objective.
        if param_suggester is None or objective_fn is None:
            driver = OptunaDriver.from_spec(
                self.spec, sampler=config.sampler, seed=config.calib_seed,
            )
            built_objective, auto_rows = build_surrogate_objective(
                self.spec, config, self.project_root,
            )
            if param_suggester is None:
                param_suggester = driver.suggest
                if on_result is None:
                    on_result = driver.tell
            if objective_fn is None:
                objective_fn = built_objective
            if subsample_rows_used is None:
                subsample_rows_used = auto_rows

        if subsample_rows_used is None:
            subsample_rows_used = 0

        try:
            result = run_calibration(
                spec=self.spec,
                config=config,
                space_hash=sh,
                dataset_fingerprint=fingerprint,
                param_suggester=param_suggester,
                objective_fn=objective_fn,
                calibration_dir=self.calibration_dir,
                subsample_rows_used=subsample_rows_used,
                parent_sweep_version=parent_sweep_version,
                on_result=on_result,
            )
        except Exception as exc:
            self._log_calibration_failure(str(exc))
            raise CalibrationError(
                f"Calibration crashed for {self.spec.id}: {exc}"
            ) from exc

        if result.aborted:
            self._log_calibration_failure(result.abort_reason or "aborted")
            raise CalibrationError(
                f"Calibration aborted for {self.spec.id}: {result.abort_reason}"
            )

        if not result.is_usable():
            self._log_calibration_failure(
                f"low-confidence calibration (plateau_trial={result.plateau_trial})"
            )
            raise CalibrationError(
                f"Calibration for {self.spec.id} produced no usable proposed "
                f"params (confidence={result.proposed.confidence.value if result.proposed else 'none'}). "
                f"Raise calibration budget and rerun."
            )

        # Rule 16: write proposed block back to spec in-memory only.
        # Promotion to active happens through sweep_config_overlay.
        self.spec.proposed = result.proposed
        self._calibration_result = result

        # Rule 20: roll calibration wall-hours into the parent sweep's
        # budget accounting so can_continue() sees the realized cost.
        self._total_wall_hours += result.wall_seconds_total / 3600.0
        logger.info(
            "Calibration complete for %s: plateau=%s cap=%d patience=%d confidence=%s",
            self.spec.id, result.plateau_trial,
            result.proposed.n_trials_cap, result.proposed.patience,
            result.proposed.confidence.value,
        )
        return result

    def _log_calibration_failure(self, reason: str) -> None:
        entry = self.ledger.make_entry(
            sweep_id=self.spec.id,
            iteration=0,
            params={},
            decision="terminate",
            notes=f"calibration_failed: {reason}",
        )
        self.ledger.append(entry)
        self.state = SweepState.TERMINATED

    def check_prerequisites(self) -> list[str]:
        """Check that all prerequisite files exist. Returns missing paths."""
        missing = []
        for prereq in self.spec.prerequisites:
            if not (self.project_root / prereq).exists():
                missing.append(prereq)
        return missing

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate proposed parameters against the spec's parameter space.

        Returns list of validation errors (empty = valid).
        """
        errors = []
        for name, value in params.items():
            if name not in self.spec.parameter_space:
                errors.append(f"Unknown parameter: {name}")
                continue
            pdef = self.spec.parameter_space[name]
            errors.extend(_validate_param_value(name, value, pdef))

        for name in self.spec.parameter_space:
            if name not in params:
                errors.append(f"Missing required parameter: {name}")

        return errors

    def can_continue(self) -> tuple[bool, TerminationReason | None]:
        """Check if the sweep can continue iterating."""
        iteration = self.ledger.last_iteration()

        if iteration >= self.spec.constraints.max_iterations:
            return False, TerminationReason.MAX_ITERATIONS

        if self._total_wall_hours >= self.spec.constraints.max_wall_hours:
            return False, TerminationReason.MAX_WALL_HOURS

        no_improve = self.ledger.consecutive_no_improve()
        if no_improve >= self.spec.constraints.consecutive_no_improve:
            return False, TerminationReason.EARLY_STOP

        return True, None

    def run_iteration(
        self,
        params: dict[str, Any],
        poll_interval: int = 60,
    ) -> tuple[EvalResult | None, str]:
        """Execute one full iteration: validate -> submit -> poll -> evaluate.

        Parameters
        ----------
        params
            Proposed parameter values for this iteration.
        poll_interval
            Seconds between SLURM polls.

        Returns
        -------
        (eval_result, decision)
            eval_result is None if the job failed or was a dry run.
            decision is one of: keep, discard, terminate, error, dry_run.
        """
        # Rule 19: parent sweep cannot start without a calibration result
        # (or an explicit required=false waiver). No silent fallback.
        if self.spec.calibration.required and self._calibration_result is None:
            raise CalibrationError(
                f"Sweep {self.spec.id} cannot start: calibration is required "
                f"but ensure_calibration() has not been called. Call "
                f"orchestrator.ensure_calibration(...) before run_iteration()."
            )

        # Check budget
        can_go, reason = self.can_continue()
        if not can_go:
            self._log_termination(params, reason)
            self.state = SweepState.TERMINATED
            return None, f"terminate:{reason.value}"

        # Validate params
        errors = self.validate_params(params)
        if errors:
            raise ValueError(f"Invalid parameters: {'; '.join(errors)}")

        iteration = self.ledger.last_iteration() + 1
        iter_dir = self.results_root / f"iter_{iteration:03d}"

        # --- SUBMIT ---
        self.state = SweepState.SUBMIT

        if self.spec.sweep_type == SweepType.new_code:
            self.state = SweepState.REVIEW_GATE
            self._log_review_gate(params, iteration)
            return None, f"terminate:{TerminationReason.REVIEW_PENDING.value}"

        # Resolve base configs
        base_training, base_pipeline = resolve_base_config(
            self.recipes_dir, self.spec.base_recipe, self.spec.base_cell,
        )

        # Generate overlay
        training_cfg, pipeline_cfg = generate_overlay(
            base_training, base_pipeline, params, iter_dir / "configs",
        )

        # Submit and wait
        run_id = f"sweep_{self.spec.id}_iter{iteration:03d}"
        job_name = f"sweep_{self.spec.id}_{iteration:03d}"

        self.state = SweepState.POLL
        job_result = submit_and_wait(
            pipeline_config=pipeline_cfg,
            training_config=training_cfg,
            seeds=self.spec.constraints.seeds,
            results_dir=iter_dir / "results",
            run_id=run_id,
            job_name=job_name,
            poll_interval=poll_interval,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            entry = self.ledger.make_entry(
                sweep_id=self.spec.id, iteration=iteration, params=params,
                job_id="DRY_RUN", status="DRY_RUN", decision="dry_run",
                notes="Dry run -- no job submitted",
            )
            self.ledger.append(entry)
            return None, "dry_run"

        # Track wall time
        if job_result.wall_seconds:
            self._total_wall_hours += job_result.wall_seconds / 3600

        # --- EVALUATE ---
        self.state = SweepState.EVALUATE

        if job_result.status != "COMPLETED":
            entry = self.ledger.make_entry(
                sweep_id=self.spec.id, iteration=iteration, params=params,
                job_id=job_result.job_id, status=job_result.status,
                decision="error", notes=f"Job failed: {job_result.status}",
            )
            self.ledger.append(entry)
            return None, "error"

        # Get baseline and previous values
        baseline = self.spec.baseline_value
        history = self.ledger.read_history()
        previous = None
        if history:
            last_metric = history[-1].get("metric_value")
            if last_metric:
                previous = float(last_metric)

        running_best = self.ledger.running_best()

        eval_result = evaluate_iteration(
            results_dir=iter_dir / "results",
            metric_column=self.spec.metric,
            metric_direction=self.spec.metric_direction,
            baseline_value=baseline,
            previous_value=previous,
            running_best=running_best,
            improvement_threshold=self.spec.constraints.improvement_threshold,
        )

        # --- DECIDE ---
        self.state = SweepState.DECIDE
        decision = "keep" if eval_result.is_improvement else "discard"

        entry = self.ledger.make_entry(
            sweep_id=self.spec.id,
            iteration=iteration,
            params=params,
            job_id=job_result.job_id,
            status=job_result.status,
            metric_value=eval_result.metric_value,
            delta_baseline=eval_result.delta_baseline,
            delta_previous=eval_result.delta_previous,
            running_best=eval_result.running_best,
            decision=decision,
        )
        self.ledger.append(entry)

        self.state = SweepState.PROPOSE
        return eval_result, decision

    def summary(self) -> dict[str, Any]:
        """Return a summary of sweep progress."""
        history = self.ledger.read_history()
        calib = self._calibration_result
        return {
            "sweep_id": self.spec.id,
            "iterations_completed": len(history),
            "max_iterations": self.spec.constraints.max_iterations,
            "running_best": self.ledger.running_best(),
            "baseline": self.spec.baseline_value,
            "consecutive_no_improve": self.ledger.consecutive_no_improve(),
            "total_wall_hours": round(self._total_wall_hours, 2),
            "state": self.state.value,
            "calibration_id": calib.calibration_id if calib else None,
            "calibration_confidence": (
                calib.proposed.confidence.value if calib and calib.proposed else None
            ),
            "proposed_n_trials_cap": (
                calib.proposed.n_trials_cap if calib and calib.proposed else None
            ),
            "proposed_patience": (
                calib.proposed.patience if calib and calib.proposed else None
            ),
        }

    def _log_termination(self, params: dict[str, Any], reason: TerminationReason) -> None:
        entry = self.ledger.make_entry(
            sweep_id=self.spec.id,
            iteration=self.ledger.last_iteration() + 1,
            params=params,
            decision="terminate",
            notes=f"Budget exhausted: {reason.value}",
        )
        self.ledger.append(entry)

    def _log_review_gate(self, params: dict[str, Any], iteration: int) -> None:
        entry = self.ledger.make_entry(
            sweep_id=self.spec.id,
            iteration=iteration,
            params=params,
            decision="terminate",
            notes="Blocked at REVIEW_GATE -- new_code sweep requires human review of staging/",
        )
        self.ledger.append(entry)


def _validate_param_value(name: str, value: Any, pdef: ParameterDef) -> list[str]:
    """Validate a single parameter value against its definition."""
    errors = []
    if pdef.type == ParameterType.choice:
        if value not in pdef.values:
            errors.append(f"{name}={value} not in allowed values {pdef.values}")
    elif pdef.type in (ParameterType.float_range, ParameterType.int_range):
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name}={value} is not numeric")
            return errors
        if pdef.low is not None and num < pdef.low:
            errors.append(f"{name}={value} below minimum {pdef.low}")
        if pdef.high is not None and num > pdef.high:
            errors.append(f"{name}={value} above maximum {pdef.high}")
    return errors


def load_sweep_spec(spec_path: Path) -> SweepSpec:
    """Load and validate a sweep spec from YAML."""
    with open(spec_path) as f:
        raw = yaml.safe_load(f)
    return SweepSpec(**raw)
