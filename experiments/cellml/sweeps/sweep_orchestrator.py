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

from .minerva_poller import JobResult, submit_and_wait
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

        self.state = SweepState.PROPOSE
        self._total_wall_hours = 0.0

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
        return {
            "sweep_id": self.spec.id,
            "iterations_completed": len(history),
            "max_iterations": self.spec.constraints.max_iterations,
            "running_best": self.ledger.running_best(),
            "baseline": self.spec.baseline_value,
            "consecutive_no_improve": self.ledger.consecutive_no_improve(),
            "total_wall_hours": round(self._total_wall_hours, 2),
            "state": self.state.value,
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
