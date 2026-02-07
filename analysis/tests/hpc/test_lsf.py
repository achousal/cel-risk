"""Tests for HPC LSF job submission utilities."""

from pathlib import Path

from ced_ml.hpc.lsf import (
    _build_consensus_panel_command,
    _build_panel_optimization_command,
    _build_postprocessing_command,
    _build_training_command,
    build_job_script,
)


def test_build_training_command():
    """Test training command builder generates correct ced train call for single model."""
    cmd = _build_training_command(
        config_file=Path("/path/to/config.yaml"),
        infile=Path("/data/input.parquet"),
        split_dir=Path("/splits"),
        outdir=Path("/results"),
        model="LR_EN",
        split_seed=0,
        run_id="20260130_120000",
    )

    assert "ced train" in cmd
    assert "--model LR_EN" in cmd
    assert "--split-seed 0" in cmd
    assert "--run-id 20260130_120000" in cmd
    assert "--config" in cmd
    assert "--infile" in cmd
    assert "--split-dir" in cmd
    assert "--outdir" in cmd


def test_build_postprocessing_command_basic():
    """Test post-processing command builder without ensemble."""
    cmd = _build_postprocessing_command(
        config_file=Path("/path/to/config.yaml"),
        run_id="20260130_120000",
        outdir=Path("/results"),
        infile=Path("/data/input.parquet"),
        split_dir=Path("/splits"),
        models=["LR_EN", "RF"],
        split_seeds=[0, 1],
        enable_ensemble=False,
    )

    assert "ced aggregate-splits --run-id 20260130_120000 --model LR_EN" in cmd
    assert "ced aggregate-splits --run-id 20260130_120000 --model RF" in cmd
    assert "ced train-ensemble" not in cmd
    assert "ced optimize-panel" not in cmd
    assert "ced consensus-panel" not in cmd


def test_build_postprocessing_command_with_ensemble():
    """Test post-processing command builder with ensemble enabled."""
    cmd = _build_postprocessing_command(
        config_file=Path("/path/to/config.yaml"),
        run_id="20260130_120000",
        outdir=Path("/results"),
        infile=Path("/data/input.parquet"),
        split_dir=Path("/splits"),
        models=["LR_EN", "RF", "XGBoost"],
        split_seeds=[0, 1, 2],
        enable_ensemble=True,
    )

    assert "ced train-ensemble --run-id 20260130_120000 --split-seed 0" in cmd
    assert "ced train-ensemble --run-id 20260130_120000 --split-seed 1" in cmd
    assert "ced train-ensemble --run-id 20260130_120000 --split-seed 2" in cmd
    assert "ced aggregate-splits --run-id 20260130_120000 --model ENSEMBLE" in cmd


def test_build_panel_optimization_command():
    """Test panel optimization command builder for a single model."""
    cmd = _build_panel_optimization_command(
        run_id="20260130_120000",
        model="LR_EN",
    )

    assert cmd == "ced optimize-panel --run-id 20260130_120000 --model LR_EN"


def test_build_consensus_panel_command():
    """Test consensus panel command builder."""
    cmd = _build_consensus_panel_command(run_id="20260130_120000")

    assert cmd == "ced consensus-panel --run-id 20260130_120000"


def test_build_job_script_basic():
    """Test LSF job script builder without dependency."""
    script = build_job_script(
        job_name="test_job",
        command="echo 'Hello World'",
        project="test_project",
        queue="medium",
        cores=4,
        mem_per_core=4096,
        walltime="02:00",
        env_activation="source venv/bin/activate",
        log_dir=Path("/logs"),
        dependency=None,
    )

    assert "#BSUB -P test_project" in script
    assert "#BSUB -q medium" in script
    assert "#BSUB -J test_job" in script
    assert "#BSUB -n 4" in script
    assert "#BSUB -W 02:00" in script
    assert "rusage[mem=4096]" in script
    assert "source venv/bin/activate" in script
    assert "echo 'Hello World'" in script
    assert "#BSUB -w" not in script


def test_build_job_script_with_dependency():
    """Test LSF job script builder with job dependency."""
    script = build_job_script(
        job_name="dependent_job",
        command="echo 'Dependent'",
        project="test_project",
        queue="medium",
        cores=2,
        mem_per_core=2048,
        walltime="01:00",
        env_activation="conda activate myenv",
        log_dir=Path("/logs"),
        dependency="done(parent_job*)",
    )

    assert '#BSUB -w "done(parent_job*)"' in script
    assert "conda activate myenv" in script


def test_build_job_script_log_paths():
    """Test that log paths are correctly configured.

    Note: Only stderr (.err) is captured by LSF. Stdout is discarded because
    ced commands create their own log files in logs/training/, logs/ensemble/, etc.
    """
    log_dir = Path("/test/logs")
    script = build_job_script(
        job_name="logging_test",
        command="echo 'test'",
        project="proj",
        queue="short",
        cores=1,
        mem_per_core=1024,
        walltime="00:30",
        env_activation="source venv/bin/activate",
        log_dir=log_dir,
    )

    assert f"#BSUB -eo {log_dir}/logging_test.%J.err" in script
    assert "#BSUB -oo /dev/null" in script
    # No more .live.log files - ced commands create their own logs
    assert ".live.log" not in script
    assert "tee" not in script
    # Verify trap-based cleanup removes .err on success (warnings are not actionable)
    assert "trap cleanup_err EXIT" in script
    assert "rm -f" in script


def test_build_job_script_set_u_safe():
    """Test that job script is safe with set -u (unbound variable checking).

    Regression test for LSB_JOBID usage in cleanup section - must be properly
    quoted to avoid "unbound variable" errors when set -u is enabled.
    """
    script = build_job_script(
        job_name="test_job",
        command="echo 'test'",
        project="test_proj",
        queue="test_queue",
        cores=1,
        mem_per_core=1000,
        walltime="1:00",
        env_activation="source /path/to/venv/bin/activate",
        log_dir=Path("/tmp/logs"),
    )

    # Verify set -u is enabled
    assert "set -euo pipefail" in script

    # Verify LSB_JOBID is properly quoted with ${...} syntax in cleanup section
    # This prevents "unbound variable" errors when LSB_JOBID is not set
    assert '[ -n "${LSB_JOBID:-}"' in script
    assert "${LSB_JOBID}.err" in script

    # Ensure no raw $LSB_JOBID or $LSF_JOBID that could trigger set -u errors
    lines = script.split("\n")
    for i, line in enumerate(lines, 1):
        # After "set -euo pipefail", all variable references must use ${...}
        if i > 10 and ("LSB_JOBID" in line or "LSF_JOBID" in line):
            # Must use ${VAR:-} or ${VAR} syntax, not $VAR
            assert "${" in line, f"Line {i} has unquoted LSB_JOBID: {line}"
