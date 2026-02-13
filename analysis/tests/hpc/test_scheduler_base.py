"""Tests for the scheduler backend abstraction and factory."""

import pytest

from ced_ml.hpc.base import SchedulerBackend, get_scheduler
from ced_ml.hpc.lsf import LSFScheduler
from ced_ml.hpc.slurm import SlurmScheduler


def test_get_scheduler_lsf():
    scheduler = get_scheduler("lsf")
    assert isinstance(scheduler, LSFScheduler)
    assert scheduler.name == "lsf"


def test_get_scheduler_slurm():
    scheduler = get_scheduler("slurm")
    assert isinstance(scheduler, SlurmScheduler)
    assert scheduler.name == "slurm"


def test_get_scheduler_invalid():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        get_scheduler("pbs")


def test_scheduler_is_abstract():
    """SchedulerBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SchedulerBackend()


@pytest.mark.parametrize("name", ["lsf", "slurm"])
def test_scheduler_contract(name):
    """Every registered backend must implement the full ABC contract."""
    scheduler = get_scheduler(name)

    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.submit_command, str)

    directives = scheduler.build_directives(
        job_name="t",
        project="p",
        queue="q",
        cores=1,
        mem_per_core=1000,
        walltime="01:00",
    )
    assert isinstance(directives, list)
    assert all(isinstance(d, str) for d in directives)

    assert isinstance(scheduler.build_orchestrator_submit_func(), str)
    assert isinstance(scheduler.build_orchestrator_status_func(), str)
    assert isinstance(scheduler.job_array_index_var(), str)
    assert isinstance(scheduler.monitor_hint("test_*"), str)


@pytest.mark.parametrize("name", ["lsf", "slurm"])
def test_parse_job_id_none_on_garbage(name):
    scheduler = get_scheduler(name)
    assert scheduler.parse_job_id("no job id here") is None


@pytest.mark.parametrize(
    "name,stdout,expected",
    [
        ("lsf", "Job <12345> is submitted to queue <medium>.", "12345"),
        ("slurm", "Submitted batch job 67890", "67890"),
    ],
)
def test_parse_job_id_extracts_correctly(name, stdout, expected):
    scheduler = get_scheduler(name)
    assert scheduler.parse_job_id(stdout) == expected


@pytest.mark.parametrize(
    "name,expected_prefix",
    [
        ("lsf", "#BSUB"),
        ("slurm", "#SBATCH"),
    ],
)
def test_directives_use_correct_prefix(name, expected_prefix):
    scheduler = get_scheduler(name)
    directives = scheduler.build_directives(
        job_name="t",
        project="p",
        queue="q",
        cores=1,
        mem_per_core=1000,
        walltime="01:00",
    )
    assert all(d.startswith(expected_prefix) for d in directives)
