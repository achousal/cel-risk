"""Optuna ask/tell driver for pre-sweep calibration.

`run_calibration()` is objective-agnostic -- it calls a `param_suggester`
to get trial parameters and an `objective_fn` to score them. This module
wires Optuna into that contract using the ask/tell API so calibration
runs share the same sampler as the parent sweep (Rule 6).

Usage
-----
    driver = OptunaDriver.from_spec(spec, sampler="tpe", seed=0)
    result = run_calibration(
        ...,
        param_suggester=driver.suggest,
        on_result=driver.tell,
        ...,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from .calibration_schema import ProposedSweepParams
from .sweep_schema import ParameterDef, ParameterType, SweepSpec

logger = logging.getLogger(__name__)


class OptunaDriver:
    """Holds an Optuna study and exposes suggest/tell callables.

    The driver is a thin adapter that translates `ParameterDef` into
    Optuna's `suggest_*` calls and keeps pending trials alive across the
    ask/tell boundary used by `run_calibration()`.
    """

    def __init__(
        self,
        parameter_space: dict[str, ParameterDef],
        sampler: str = "tpe",
        seed: int = 0,
        direction: str = "maximize",
    ):
        self.parameter_space = parameter_space
        self.direction = direction
        if sampler == "tpe":
            sampler_obj: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler == "random":
            sampler_obj = optuna.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")

        self.study = optuna.create_study(direction=direction, sampler=sampler_obj)
        self._pending: dict[int, optuna.trial.Trial] = {}

    @classmethod
    def from_spec(
        cls,
        spec: SweepSpec,
        sampler: str = "tpe",
        seed: int = 0,
    ) -> "OptunaDriver":
        return cls(
            parameter_space=spec.parameter_space,
            sampler=sampler,
            seed=seed,
            direction=spec.metric_direction,
        )

    def _suggest_for_trial(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, pdef in self.parameter_space.items():
            if pdef.type == ParameterType.choice:
                params[name] = trial.suggest_categorical(name, pdef.values)
            elif pdef.type == ParameterType.float_range:
                step = pdef.step
                params[name] = trial.suggest_float(
                    name, float(pdef.low), float(pdef.high),
                    step=step if step else None,
                )
            elif pdef.type == ParameterType.int_range:
                step = int(pdef.step) if pdef.step else 1
                params[name] = trial.suggest_int(
                    name, int(pdef.low), int(pdef.high), step=step,
                )
            else:  # pragma: no cover - schema validator blocks this
                raise ValueError(f"Unknown parameter type: {pdef.type}")
        return params

    def suggest(self, trial_idx: int) -> dict[str, Any]:
        """Ask Optuna for the next trial's parameters."""
        trial = self.study.ask()
        self._pending[trial_idx] = trial
        return self._suggest_for_trial(trial)

    def tell(self, trial_idx: int, params: dict[str, Any], objective: float) -> None:
        """Report the objective value back to the study."""
        trial = self._pending.pop(trial_idx, None)
        if trial is None:
            logger.warning("OptunaDriver.tell called with no pending trial %d", trial_idx)
            return
        self.study.tell(trial, objective)


def enqueue_warm_start(
    study: "optuna.study.Study",
    proposed: ProposedSweepParams | None,
    label: str = "calibration_warmstart",
) -> int:
    """Enqueue calibration warm-start points into a parent Optuna study.

    Parent sweeps call this on their own study BEFORE `study.optimize()`
    to seed exploration with calibration's best configs. Enqueued trials
    are tagged via `system_attrs` so downstream analysis can exclude or
    highlight them. Returns the number of points actually enqueued.

    Warm-start points are HINTS, not truth -- the surrogate used during
    calibration is a cheap LR proxy on a subsample. The parent sweep is
    free to re-explore beyond these points; enqueue_trial() does not
    constrain Optuna's subsequent search.
    """
    if proposed is None or not proposed.warm_start_points:
        return 0
    for point in proposed.warm_start_points:
        try:
            study.enqueue_trial(
                point.params,
                user_attrs={
                    "source": label,
                    "calibration_id": proposed.calibration_id,
                    "calibration_trial_idx": point.calibration_trial_idx,
                    "calibration_objective": point.objective,
                },
            )
        except Exception as exc:  # e.g., params out of current space
            logger.warning(
                "Skipping warm-start point (trial %d): %s",
                point.calibration_trial_idx, exc,
            )
    n = len(proposed.warm_start_points)
    logger.info("Enqueued %d warm-start points from %s", n, proposed.calibration_id)
    return n
