"""Permutation testing configuration schema for CeD-ML pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class PermutationAggregationConfig(BaseModel):
    """Configuration for aggregating permutation test results across seeds."""

    method: Literal["pooled_null", "fisher", "stouffer"] = Field(
        default="pooled_null",
        description="Method for aggregating p-values across seeds",
    )
    alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for hypothesis testing",
    )
    min_seeds_required: int = Field(
        default=3,
        ge=1,
        description="Minimum number of seeds required for aggregation",
    )


class PermutationTestConfig(BaseModel):
    """Configuration for permutation testing (label permutation for significance).

    Controls the number of permutations, parallelization, and aggregation
    settings for testing whether models generalize above chance level.
    """

    n_perms: int = Field(
        default=200,
        ge=10,
        description="Number of label permutations (200+ for publication)",
    )
    metric: Literal["auroc"] = Field(
        default="auroc",
        description="Metric to evaluate (only AUROC supported per ADR-011)",
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    split_seed_start: int = Field(
        default=0,
        ge=0,
        description="First split seed to test",
    )
    n_split_seeds: int = Field(
        default=1,
        ge=1,
        description="Number of consecutive seeds to test",
    )
    n_jobs: int = Field(
        default=1,
        ge=-1,
        description="Parallel jobs for local execution (1=sequential, -1=all CPUs)",
    )
    outdir: str | None = Field(
        default=None,
        description="Output directory (None = auto-detect from run_id)",
    )
    save_individual: bool = Field(
        default=True,
        description="Save individual permutation results (for HPC aggregation)",
    )
    aggregation: PermutationAggregationConfig = Field(
        default_factory=PermutationAggregationConfig,
        description="Aggregation settings for combining results across seeds",
    )
