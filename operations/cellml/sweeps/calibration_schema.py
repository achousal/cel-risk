"""Pydantic v2 models for pre-sweep calibration.

Calibration is a small, cheap Optuna run executed before an expensive
parent sweep. Its job is to measure where the best-so-far curve plateaus
and write back recommended n_trials / patience values as a PROPOSED
block on the parent sweep spec. Proposed values are never auto-promoted
to active -- humans or /next do that.

See ops/methodology (EngramR) for the calibration ruleset these models
encode. Rules 1-20 map one-to-one onto fields below; comments reference
the rule number.
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CalibrationConfidence(str, enum.Enum):
    """Confidence tag emitted by run_calibration().

    high   -- plateau found before 70% of n_calib trials (Rule 14)
    medium -- plateau found between 70% and 100% of n_calib
    low    -- best_so_far still rising at n_calib; raise budget
    """

    high = "high"
    medium = "medium"
    low = "low"


class CalibrationConfig(BaseModel):
    """Per-sweep calibration configuration.

    Lives under the `calibration:` key of a sweep spec YAML. Mirrors the
    cel-risk style: explicit, validated, no silent fallbacks.
    """

    # Rule 1: trigger
    required: bool = Field(
        default=True,
        description=(
            "If true, the parent sweep cannot start without a valid "
            "calibration result. Set false only for exhaustible sweeps."
        ),
    )

    # Rule 4: budget
    budget_fraction: float = Field(
        default=0.15,
        gt=0.0,
        le=0.5,
        description=(
            "Share of parent sweep's wall_hours to spend on calibration. "
            "Hard-capped at 2 wall hours regardless of fraction."
        ),
    )
    hard_cap_wall_hours: float = Field(
        default=2.0,
        gt=0.0,
        description="Absolute wall-hour ceiling for one calibration run.",
    )

    # Rule 5: sample size
    min_trials: int = Field(default=30, ge=5)
    max_trials: int = Field(default=80, ge=10)
    trials_per_dim: int = Field(
        default=20,
        ge=1,
        description="Heuristic: n_calib = min(trials_per_dim * dim, max_trials).",
    )

    # Rule 6: sampler (must mirror parent sweep)
    sampler: Literal["tpe", "random"] = Field(
        default="tpe",
        description="TPE is required for production; 'random' is test-only.",
    )

    # Rule 7: single fixed calibration seed
    calib_seed: int = Field(default=0, ge=0)

    # Rule 8: data subsampling
    subsample_rows: int = Field(
        default=5000,
        ge=100,
        description="Max rows used for calibration (stratified on label).",
    )

    # Rule 3: cache lifetime
    cache_days: int = Field(default=30, ge=1)

    # Rule 15: abort signal
    min_viable_objective: float | None = Field(
        default=None,
        description=(
            "If calibration's best objective < this value, block the "
            "parent sweep and emit a tension note. Null disables."
        ),
    )

    # Rule 12: cap bounds
    absolute_min_cap: int = Field(default=10, ge=1)
    absolute_max_cap: int = Field(default=200, ge=10)
    cap_safety_multiplier: float = Field(
        default=1.5,
        gt=1.0,
        description="n_trials_cap = ceil(multiplier * plateau_trial).",
    )

    # Rule 13: patience bounds
    min_patience: int = Field(default=5, ge=1)
    max_patience: int = Field(default=20, ge=1)
    patience_fraction_of_plateau: float = Field(default=0.3, gt=0.0, le=1.0)

    # Warm-start: top-k calibration trials exported for parent sweep
    # enqueue_trial(). Points are hints, not truth — parent sweep still
    # explores beyond them.
    warm_start_top_k: int = Field(
        default=3,
        ge=0,
        description="Number of best calibration trials to export as warm-start seeds (0 disables)",
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "CalibrationConfig":
        if self.max_trials < self.min_trials:
            raise ValueError("max_trials must be >= min_trials")
        if self.absolute_max_cap < self.absolute_min_cap:
            raise ValueError("absolute_max_cap must be >= absolute_min_cap")
        if self.max_patience < self.min_patience:
            raise ValueError("max_patience must be >= min_patience")
        return self


class WarmStartPoint(BaseModel):
    """One (params, objective) point exported for parent-sweep warm start."""

    params: dict
    objective: float
    calibration_trial_idx: int


class ProposedSweepParams(BaseModel):
    """The write-back block calibration emits onto a sweep spec (Rule 16).

    Never promoted to active parameters automatically. The orchestrator
    reads `active` values from constraints; calibration only mutates
    `proposed`. Promotion is a separate, explicit step.
    """

    n_trials_cap: int = Field(ge=1)
    patience: int = Field(ge=1)
    calibration_id: str = Field(description="Pointer to calibration artifact")
    confidence: CalibrationConfidence
    source: Literal["calibration"] = "calibration"
    warm_start_points: list[WarmStartPoint] = Field(
        default_factory=list,
        description="Top-k (params, objective) from calibration, best first",
    )


class TrialRecord(BaseModel):
    """One Optuna trial row written to the calibration parquet (Rule 9)."""

    trial_idx: int
    objective: float
    best_so_far: float
    wall_seconds: float
    params_json: str  # JSON-serialized params, for warm-start extraction


class CalibrationResult(BaseModel):
    """Artifact returned by run_calibration() and cached on disk.

    One JSON file per calibration: `calibration/<calibration_id>.json`.
    The matching trial-by-trial parquet lives alongside it.
    """

    calibration_id: str
    sweep_id: str
    space_hash: str
    dataset_fingerprint: str
    created_utc: str

    # Inputs
    n_calib_requested: int
    n_calib_executed: int
    sampler: str
    calib_seed: int
    subsample_rows_used: int
    wall_seconds_total: float

    # Measured plateau (Rules 10-11)
    plateau_trial: int | None
    plateau_value: float | None
    noise_sigma: float
    noise_warning: bool

    # Outputs (Rules 12-14)
    proposed: ProposedSweepParams | None
    aborted: bool
    abort_reason: str | None

    # Lineage (Rule 17)
    parent_sweep_version: str | None = None

    def is_usable(self) -> bool:
        """Return True if a parent sweep may start from this calibration."""
        return (not self.aborted) and self.proposed is not None
