"""Compute and HPC resource configuration schemas for CeD-ML pipeline."""

import os

from pydantic import BaseModel, Field, model_validator


class ComputeConfig(BaseModel):
    """Configuration for compute resources."""

    cpus: int = Field(default_factory=lambda: os.cpu_count() or 1)
    tune_n_jobs: int | None = None


class HPCResourceConfig(BaseModel):
    """HPC resource allocation for a single job type."""

    queue: str = "medium"
    cores: int = Field(default=4, ge=1)
    mem_per_core: int = Field(default=4000, ge=1, description="Memory per core in MB")
    walltime: str = Field(default="24:00", description="Wall time limit as HH:MM")


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
    """

    project: str = Field(
        ...,
        description="HPC project allocation code (e.g., acc_elahi, required)",
    )
    scheduler: str = Field(default="lsf", description="Scheduler type (lsf only)")
    queue: str = Field(default="medium", description="Default queue name")
    cores: int = Field(default=4, ge=1, description="Default number of CPU cores")
    mem_per_core: int = Field(default=4000, ge=1, description="Default memory per core in MB")
    walltime: str = Field(default="24:00", description="Default wall time limit as HH:MM")

    # Optional per-stage resource overrides
    training: HPCResourceConfig | None = None
    postprocessing: HPCResourceConfig | None = None
    optimization: HPCResourceConfig | None = None

    @model_validator(mode="after")
    def validate_project(self) -> "HPCConfig":
        """Validate that project is not a placeholder."""
        placeholders = {"YOUR_PROJECT_ALLOCATION", "YOUR_ALLOCATION"}
        if self.project in placeholders:
            raise ValueError(
                f"HPC project not configured. Got placeholder '{self.project}'. "
                "Update 'hpc.project' in pipeline_hpc.yaml"
            )
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
