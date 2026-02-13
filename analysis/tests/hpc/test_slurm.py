"""Tests for HPC Slurm scheduler backend."""

from pathlib import Path

from ced_ml.hpc.common import (
    _build_orchestrator_bash_functions,
    build_job_script,
)
from ced_ml.hpc.slurm import SlurmScheduler

_SLURM = SlurmScheduler()


def test_slurm_name():
    assert _SLURM.name == "slurm"


def test_slurm_submit_command():
    assert _SLURM.submit_command == "sbatch"


def test_slurm_build_directives_basic():
    """Slurm directives should use #SBATCH syntax."""
    directives = _SLURM.build_directives(
        job_name="test_job",
        project="acc_test",
        queue="short",
        cores=4,
        mem_per_core=4000,
        walltime="02:00",
    )

    joined = "\n".join(directives)
    assert "#SBATCH --account=acc_test" in joined
    assert "#SBATCH --partition=short" in joined
    assert "#SBATCH --job-name=test_job" in joined
    assert "#SBATCH --cpus-per-task=4" in joined
    assert "#SBATCH --time=02:00:00" in joined
    assert "#SBATCH --mem-per-cpu=4000M" in joined
    assert "#SBATCH --output=/dev/null" in joined
    assert "#SBATCH --error=/dev/null" in joined
    assert "--dependency" not in joined


def test_slurm_build_directives_with_dependency():
    """Dependency directive should appear when provided."""
    directives = _SLURM.build_directives(
        job_name="dep_job",
        project="acc_test",
        queue="medium",
        cores=2,
        mem_per_core=2000,
        walltime="01:00",
        dependency="afterok:12345",
    )

    joined = "\n".join(directives)
    assert "#SBATCH --dependency=afterok:12345" in joined


def test_slurm_walltime_normalisation():
    """HH:MM should be normalised to HH:MM:SS."""
    directives = _SLURM.build_directives(
        job_name="t",
        project="p",
        queue="q",
        cores=1,
        mem_per_core=1000,
        walltime="24:00",
    )
    assert any("--time=24:00:00" in d for d in directives)


def test_slurm_walltime_already_hms():
    """HH:MM:SS should pass through unchanged."""
    directives = _SLURM.build_directives(
        job_name="t",
        project="p",
        queue="q",
        cores=1,
        mem_per_core=1000,
        walltime="24:00:00",
    )
    assert any("--time=24:00:00" in d for d in directives)


def test_slurm_parse_job_id():
    """Parse Slurm submission output."""
    assert _SLURM.parse_job_id("Submitted batch job 12345678") == "12345678"
    assert _SLURM.parse_job_id("Some other output") is None


def test_slurm_job_array_index_var():
    assert _SLURM.job_array_index_var() == "SLURM_ARRAY_TASK_ID"


def test_slurm_monitor_hint():
    hint = _SLURM.monitor_hint("CeD_*")
    assert "squeue" in hint


def test_slurm_orchestrator_submit_func():
    """submit_and_track() for Slurm should use sbatch and SBATCH directives."""
    func = _SLURM.build_orchestrator_submit_func()

    assert "sbatch" in func
    assert "sbatch_directive" in func
    assert "Submitted batch job" in func
    assert "manifest_job_tsv" in func
    assert '"$WRAPPER_SCRIPT"' in func
    assert 'CED_JOB_NAME="$job_name"' in func
    assert 'CED_SENTINEL_DIR="$SENTINEL_DIR"' in func


def test_slurm_orchestrator_status_func():
    """check_upstream_failures() for Slurm should use sacct/squeue."""
    func = _SLURM.build_orchestrator_status_func()

    assert "sacct" in func
    assert "squeue" in func
    assert "FAILED" in func
    assert "CANCELLED" in func


def test_slurm_orchestrator_header():
    header = _SLURM.build_orchestrator_header(
        project="acc_test",
        queue="short",
        job_name="orch_test",
        cores=1,
        mem_per_core=2000,
        walltime="48:00",
        log_path=Path("/tmp/orch.log"),
    )

    joined = "\n".join(header)
    assert "#!/bin/bash" in joined
    assert "#SBATCH --account=acc_test" in joined
    assert "#SBATCH --job-name=orch_test" in joined
    assert "#SBATCH --time=48:00:00" in joined


def test_slurm_build_job_script():
    """build_job_script with Slurm backend should produce #SBATCH directives."""
    script = build_job_script(
        scheduler=_SLURM,
        job_name="test_job",
        command="echo hello",
        project="acc_test",
        queue="short",
        cores=2,
        mem_per_core=2000,
        walltime="01:00",
        env_activation="source venv/bin/activate",
        log_dir=Path("/logs"),
    )

    assert "#SBATCH --account=acc_test" in script
    assert "#SBATCH --partition=short" in script
    assert "#BSUB" not in script
    assert "echo hello" in script
    assert "source venv/bin/activate" in script


def test_slurm_orchestrator_bash_functions():
    """Full orchestrator bash for Slurm should include sbatch-based submission."""
    bash = _build_orchestrator_bash_functions(_SLURM)

    assert "sbatch" in bash
    assert "sacct" in bash
    # Shared functions should still be present
    assert "manifest_job_tsv()" in bash
    assert "barrier_wait()" in bash
    assert "submit_batch()" in bash


def test_slurm_submit_func_avoids_literal_sbatch_directives():
    """Embedded child script directives must not be literal #SBATCH lines."""
    func = _SLURM.build_orchestrator_submit_func()

    assert 'local sbatch_directive="#SBATCH"' in func
    assert "#SBATCH --account=" not in func
    assert "${sbatch_directive} --account=$PROJECT" in func
