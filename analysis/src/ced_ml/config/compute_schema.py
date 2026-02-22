"""Compute and HPC resource configuration schemas for CeD-ML pipeline."""

import os
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _validate_walltime_format(value: str, field_name: str = "walltime") -> None:
    """Validate LSF-style walltime as HH:MM or HH:MM:SS."""
    if not re.match(r"^\d{1,3}:\d{2}(:\d{2})?$", value):
        raise ValueError(
            f"Invalid {field_name} format: '{value}'\n"
            "Expected HH:MM or HH:MM:SS (e.g., '24:00' or '24:00:00')"
        )


class ComputeConfig(BaseModel):
    """Configuration for compute resources."""

    cpus: int = Field(default_factory=lambda: os.cpu_count() or 1)
    tune_n_jobs: int | None = None


class HPCResourceConfig(BaseModel):
    """HPC resource allocation for a single job type."""

    queue: str = "medium"
    cores: int = Field(default=4, ge=1, le=128, description="Number of CPU cores (max 128)")
    mem_per_core: int = Field(
        default=4000, ge=1, le=65536, description="Memory per core in MB (max 64GB = 65536MB)"
    )
    walltime: str = Field(default="24:00", description="Wall time limit as HH:MM or HH:MM:SS")

    @model_validator(mode="after")
    def validate_walltime_format(self) -> "HPCResourceConfig":
        """Validate walltime format."""
        _validate_walltime_format(self.walltime)
        return self


class OrchestratorConfig(BaseModel):
    """Resources and timeouts for the barrier-orchestrator job."""

    poll_interval: int = Field(
        default=60, ge=10, le=600, description="Seconds between sentinel checks"
    )
    training_timeout: float = Field(
        default=4.0, ge=0.5, description="Hours to wait for training jobs"
    )
    aggregation_timeout: float = Field(
        default=2.0, ge=0.25, description="Hours to wait for per-model aggregation"
    )
    ensemble_timeout: float = Field(
        default=2.0, ge=0.25, description="Hours to wait for ensemble training/aggregation"
    )
    post_timeout: float = Field(
        default=2.0, ge=0.25, description="Hours to wait for post-processing (legacy fallback)"
    )
    perm_timeout: float = Field(
        default=4.0, ge=0.25, description="Hours to wait for permutation jobs"
    )
    panel_timeout: float = Field(
        default=2.0, ge=0.25, description="Hours to wait for panel optimization"
    )
    consensus_timeout: float = Field(
        default=1.0, ge=0.25, description="Hours to wait for consensus"
    )
    max_concurrent_submissions: int = Field(default=20, ge=1, le=100)
    cores: int = Field(default=1, ge=1, le=4)
    mem_per_core: int = Field(default=2000, ge=512, le=8000)
    walltime: str = Field(default="48:00")

    def timeout_seconds(self, stage: str) -> int:
        """Convert a stage timeout from hours to seconds for barrier_wait."""
        field = f"{stage}_timeout"
        hours = getattr(self, field)
        return int(hours * 3600)

    @model_validator(mode="after")
    def validate_walltime_format(self) -> "OrchestratorConfig":
        """Validate walltime format."""
        _validate_walltime_format(self.walltime)
        return self


class HPCConfig(BaseModel):
    """HPC-specific configuration.

    Attributes:
        project: HPC project allocation code (e.g., acc_elahi).
        scheduler: Scheduler type (currently only 'lsf' supported).
        queue: Default queue for job submission.
        cores: Default number of CPU cores per job.
        mem_per_core: Default memory per core in MB.
        walltime: Default wall time limit as HH:MM string.
        training: Optional resource override for training jobs.
        postprocessing: Optional resource override for aggregation/ensemble jobs.
        optimization: Optional resource override for panel optimization jobs.
        orchestrator: Barrier-orchestrator polling settings and resources.
    """

    project: str = Field(
        ...,
        description="HPC project allocation code (e.g., acc_elahi, required)",
    )
    scheduler: Literal["lsf", "slurm"] = Field(
        default="lsf", description="Scheduler type: 'lsf' or 'slurm'"
    )
    queue: str = Field(default="medium", description="Default queue name")
    cores: int = Field(default=4, ge=1, le=128, description="Default number of CPU cores (max 128)")
    mem_per_core: int = Field(
        default=4000,
        ge=1,
        le=65536,
        description="Default memory per core in MB (max 64GB = 65536MB)",
    )
    walltime: str = Field(
        default="24:00", description="Default wall time limit as HH:MM or HH:MM:SS"
    )

    # Optional per-stage resource overrides
    training: HPCResourceConfig | None = None
    postprocessing: HPCResourceConfig | None = None
    optimization: HPCResourceConfig | None = None
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    @model_validator(mode="after")
    def validate_project_and_walltime(self) -> "HPCConfig":
        """Validate that project is not a placeholder and walltime format."""
        # Validate project
        placeholders = {"YOUR_PROJECT_ALLOCATION", "YOUR_ALLOCATION"}
        if self.project in placeholders:
            raise ValueError(
                f"HPC project not configured. Got placeholder '{self.project}'. "
                "Update 'hpc.project' in pipeline_hpc.yaml"
            )

        # Validate walltime format (HH:MM or HH:MM:SS)
        _validate_walltime_format(self.walltime)

        return self

    def get_resources(self, stage: str = "default") -> dict[str, int | str]:
        """Get resource config for a specific pipeline stage.

        Args:
            stage: Pipeline stage ('training', 'postprocessing', 'optimization', 'default').

        Returns:
            Dict with keys: queue, cores, mem_per_core, walltime.
        """
        if stage == "training" and self.training:
            return {
                "queue": self.training.queue,
                "cores": self.training.cores,
                "mem_per_core": self.training.mem_per_core,
                "walltime": self.training.walltime,
            }
        elif stage == "postprocessing" and self.postprocessing:
            return {
                "queue": self.postprocessing.queue,
                "cores": self.postprocessing.cores,
                "mem_per_core": self.postprocessing.mem_per_core,
                "walltime": self.postprocessing.walltime,
            }
        elif stage == "optimization" and self.optimization:
            return {
                "queue": self.optimization.queue,
                "cores": self.optimization.cores,
                "mem_per_core": self.optimization.mem_per_core,
                "walltime": self.optimization.walltime,
            }
        else:
            return {
                "queue": self.queue,
                "cores": self.cores,
                "mem_per_core": self.mem_per_core,
                "walltime": self.walltime,
            }
