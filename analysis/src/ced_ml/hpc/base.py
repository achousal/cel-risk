"""Abstract scheduler backend for HPC job submission.

Defines the minimal interface that varies between schedulers (LSF, Slurm).
All scheduler-agnostic logic lives in common.py; only the scheduler-specific
pieces are implemented by concrete backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class SchedulerBackend(ABC):
    """Abstract base for HPC scheduler implementations.

    Concrete subclasses: LSFScheduler, SlurmScheduler.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Scheduler identifier ('lsf' or 'slurm')."""

    @property
    @abstractmethod
    def submit_command(self) -> str:
        """CLI command for job submission ('bsub' or 'sbatch')."""

    @abstractmethod
    def build_directives(
        self,
        *,
        job_name: str,
        project: str,
        queue: str,
        cores: int,
        mem_per_core: int,
        walltime: str,
        stdout_path: str = "/dev/null",
        stderr_path: str = "/dev/null",
        dependency: str | None = None,
    ) -> list[str]:
        """Generate scheduler-specific header directive lines.

        Returns:
            Lines like '#BSUB -P proj' or '#SBATCH --account=proj'.
        """

    @abstractmethod
    def parse_job_id(self, stdout: str) -> str | None:
        """Extract job ID from submission command stdout.

        Args:
            stdout: Raw stdout from the submission command.

        Returns:
            Job ID string, or None if parsing failed.
        """

    @abstractmethod
    def build_orchestrator_submit_func(self) -> str:
        """Generate the submit_and_track() bash function for the orchestrator.

        Must read manifest via manifest_job_tsv(), build an inline job script
        with scheduler directives, submit it, parse the job ID, and append
        the ID to the caller-provided file.
        """

    @abstractmethod
    def build_orchestrator_status_func(self) -> str:
        """Generate the check_upstream_failures() bash function.

        Must query job status by ID and exit non-zero if any upstream job
        has failed or been terminated.
        """

    @abstractmethod
    def build_orchestrator_header(
        self,
        *,
        project: str,
        queue: str,
        job_name: str,
        cores: int,
        mem_per_core: int,
        walltime: str,
        log_path: Path,
    ) -> list[str]:
        """Generate directive lines for the orchestrator job itself.

        Unlike build_directives(), this routes both stdout and stderr
        to ``log_path`` instead of /dev/null.
        """

    @abstractmethod
    def job_array_index_var(self) -> str:
        """Shell variable name for the job array index.

        Returns:
            e.g. 'LSB_JOBINDEX' or 'SLURM_ARRAY_TASK_ID'.
        """

    def monitor_hint(self, job_name_pattern: str) -> str:
        """Human-readable command to monitor jobs matching a pattern.

        Default implementation returns the submit command name; backends
        should override with scheduler-specific monitoring advice.
        """
        return f"{self.submit_command}: monitor jobs matching '{job_name_pattern}'"


_SCHEDULER_REGISTRY: dict[str, type[SchedulerBackend]] = {}


def register_scheduler(name: str):
    """Class decorator to register a scheduler backend."""

    def decorator(cls: type[SchedulerBackend]) -> type[SchedulerBackend]:
        _SCHEDULER_REGISTRY[name] = cls
        return cls

    return decorator


def get_scheduler(scheduler_name: str) -> SchedulerBackend:
    """Instantiate a scheduler backend by name.

    Args:
        scheduler_name: 'lsf' or 'slurm'.

    Returns:
        SchedulerBackend instance.

    Raises:
        ValueError: If scheduler_name is not supported.
    """
    # Lazy-import backends so they register themselves
    if not _SCHEDULER_REGISTRY:
        import ced_ml.hpc.lsf  # noqa: F401
        import ced_ml.hpc.slurm  # noqa: F401

    cls = _SCHEDULER_REGISTRY.get(scheduler_name)
    if cls is None:
        supported = ", ".join(sorted(_SCHEDULER_REGISTRY)) or "(none loaded)"
        raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. Supported: {supported}")
    return cls()
