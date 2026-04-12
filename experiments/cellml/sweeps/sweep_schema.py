"""Pydantic v2 models for sweep specifications.

A sweep spec is the 'program.md' analog -- it declares the question,
parameter space, constraints, and evaluation criteria for an adaptive
sweep iteration loop.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SweepType(str, enum.Enum):
    """How the sweep executes."""

    config_only = "config_only"  # Agent submits freely via config overlays
    eval_only = "eval_only"  # Runs local R/Python on existing results
    new_code = "new_code"  # Agent proposes code diff, blocks at review gate


class ParameterType(str, enum.Enum):
    """Parameter sampling types."""

    choice = "choice"
    float_range = "float"
    int_range = "int"


class ParameterDef(BaseModel):
    """A single parameter in the sweep search space."""

    type: ParameterType
    values: list[Any] | None = Field(
        default=None, description="Explicit values (for 'choice' type)"
    )
    low: float | None = Field(default=None, description="Lower bound (for float/int)")
    high: float | None = Field(default=None, description="Upper bound (for float/int)")
    step: float | None = Field(
        default=None, description="Step size (for float/int, optional)"
    )

    @model_validator(mode="after")
    def validate_parameter(self) -> "ParameterDef":
        if self.type == ParameterType.choice and not self.values:
            raise ValueError("'choice' parameter requires 'values' list")
        if self.type in (ParameterType.float_range, ParameterType.int_range):
            if self.low is None or self.high is None:
                raise ValueError(f"'{self.type.value}' parameter requires 'low' and 'high'")
        return self


class SweepConstraints(BaseModel):
    """Budget and stopping constraints for a sweep."""

    max_iterations: int = Field(default=15, ge=1)
    max_wall_hours: float = Field(
        default=48.0, gt=0, description="Total Minerva wall hours budget"
    )
    seeds: list[int] = Field(
        default_factory=lambda: [100, 101, 102, 103, 104],
        description="Seeds for screening (fewer than factorial's 30)",
    )
    optuna_trials: int = Field(
        default=50, ge=1, description="Reduced trials for screening"
    )
    improvement_threshold: float = Field(
        default=0.005,
        description="Minimum metric improvement to count as progress",
    )
    consecutive_no_improve: int = Field(
        default=3,
        ge=1,
        description="Stop after this many flat iterations",
    )


class SweepSpec(BaseModel):
    """Top-level sweep specification -- the program.md for one sweep."""

    id: str = Field(description="Unique sweep identifier (e.g. '09_downsampling_ratio')")
    question: str = Field(description="The scientific question this sweep answers")
    sweep_type: SweepType

    # Dataset-agnostic: reference by name, orchestrator resolves paths
    base_recipe: str | None = Field(
        default=None,
        description="Recipe ID to use as base (resolved from manifest + derived dir)",
    )
    base_cell: str | None = Field(
        default=None,
        description="Cell name within recipe (or null = use factorial winner)",
    )

    # Evaluation
    metric: str = Field(
        default="summary_auroc_mean",
        description="Column name in aggregated_results.csv to optimize",
    )
    metric_direction: Literal["maximize", "minimize"] = "maximize"
    baseline_value: float | None = Field(
        default=None,
        description="Known baseline (null = first iteration establishes it)",
    )

    # Search space
    parameter_space: dict[str, ParameterDef] = Field(
        description="Named parameters with types and bounds"
    )

    # Prerequisites: files that must exist before sweep can run
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Relative paths (from project root) that must exist",
    )

    # Budget
    constraints: SweepConstraints = Field(default_factory=SweepConstraints)

    # For eval-only sweeps: script to run instead of ced run-pipeline
    eval_script: str | None = Field(
        default=None,
        description="Path to R/Python script (for eval_only sweep_type)",
    )

    @model_validator(mode="after")
    def validate_eval_script(self) -> "SweepSpec":
        if self.sweep_type == SweepType.eval_only and not self.eval_script:
            raise ValueError("eval_only sweeps require 'eval_script'")
        return self
