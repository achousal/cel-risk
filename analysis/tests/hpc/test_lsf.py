"""Tests for HPC LSF job submission utilities."""

import logging
from pathlib import Path

from ced_ml.config.schema import HPCConfig
from ced_ml.hpc.lsf import (
    EnvironmentInfo,
    _build_consensus_panel_command,
    _build_panel_optimization_command,
    _build_postprocessing_command,
    _build_training_command,
    build_job_script,
    submit_hpc_pipeline,
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
    """Test that both stdout and stderr are discarded by LSF.

    ced commands create their own structured log files in
    logs/run_{ID}/training/, etc., so LSF-level output is unnecessary.
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

    assert "#BSUB -oo /dev/null" in script
    assert "#BSUB -eo /dev/null" in script
    # No .err files, .live.log files, or cleanup traps
    assert ".err" not in script.split("#BSUB -eo /dev/null")[1]
    assert ".live.log" not in script
    assert "tee" not in script
    assert "cleanup_err" not in script


def test_submit_hpc_pipeline_post_dependency_uses_ids_with_name_fallback(monkeypatch, tmp_path):
    """Post-processing should use ID dependency with model-scoped wildcard fallback."""
    submitted_scripts: list[str] = []

    def fake_submit_job(script: str, dry_run: bool = False) -> str | None:
        submitted_scripts.append(script)
        return str(len(submitted_scripts))

    monkeypatch.setattr(
        "ced_ml.hpc.lsf.detect_environment",
        lambda _: EnvironmentInfo(env_type="venv", activation_cmd="source venv/bin/activate"),
    )
    monkeypatch.setattr("ced_ml.hpc.lsf.submit_job", fake_submit_job)

    hpc_config = HPCConfig(
        project="acc_test",
        queue="short",
        cores=2,
        mem_per_core=2000,
        walltime="01:00",
    )
    logger = logging.getLogger("test_post_dep_job_ids")

    submit_hpc_pipeline(
        config_file=tmp_path / "training_config.yaml",
        infile=tmp_path / "input.parquet",
        split_dir=tmp_path / "splits",
        outdir=tmp_path / "results",
        models=["LR_EN", "RF"],
        split_seeds=[0, 1],
        run_id="20260211_095722",
        enable_ensemble=False,
        enable_consensus=False,
        enable_optimize_panel=False,
        hpc_config=hpc_config,
        logs_dir=tmp_path / "logs",
        dry_run=False,
        pipeline_logger=logger,
    )

    # Training jobs: 2 models x 2 seeds = 4 jobs (IDs "1".."4").
    # Post-processing is the 5th submission (ID "5").
    post_script = submitted_scripts[4]
    # Primary dependency: numeric IDs
    assert "done(1) && done(2) && done(3) && done(4)" in post_script
    # Fallback dependency: model-scoped training wildcards
    assert "done(CeD_20260211_095722_LR_EN_s*)" in post_script
    assert "done(CeD_20260211_095722_RF_s*)" in post_script
    # Combined expression uses OR fallback
    assert "||" in post_script
