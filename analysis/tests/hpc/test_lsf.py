"""Tests for HPC LSF job submission utilities."""

import logging
from pathlib import Path

from ced_ml.config.schema import HPCConfig
from ced_ml.hpc.lsf import (
    EnvironmentInfo,
    _append_sentinel_touch,
    _build_consensus_panel_command,
    _build_orchestrator_bash_functions,
    _build_orchestrator_script,
    _build_panel_optimization_command,
    _build_postprocessing_command,
    _build_training_command,
    _scripts_dir,
    _sentinel_dir,
    _sentinel_path,
    build_job_script,
    submit_hpc_pipeline,
)


def _default_hpc_config(**overrides) -> HPCConfig:
    """Build a valid HPC config for tests."""
    config = {
        "project": "acc_test",
        "queue": "short",
        "cores": 2,
        "mem_per_core": 2000,
        "walltime": "01:00",
    }
    config.update(overrides)
    return HPCConfig(**config)


def _orchestrator_script_for_test(
    tmp_path: Path,
    *,
    hpc_config: HPCConfig | None = None,
    training_job_ids: list[str] | None = None,
    training_sentinels: list[Path] | None = None,
    perm_scripts: list[Path] | None = None,
    perm_sentinels: list[Path] | None = None,
    perm_agg_scripts: list[Path] | None = None,
    perm_agg_sentinels: list[Path] | None = None,
    panel_seed_scripts: list[Path] | None = None,
    panel_seed_sentinels: list[Path] | None = None,
    panel_agg_scripts: list[Path] | None = None,
    panel_agg_sentinels: list[Path] | None = None,
    consensus_script: Path | None = None,
    consensus_sentinel: Path | None = None,
) -> str:
    """Build an orchestrator script with configurable stage inputs."""
    run_id = "20260212_151826"
    sentinel_dir = tmp_path / "sentinels"
    scripts_dir = tmp_path / "scripts"

    if hpc_config is None:
        hpc_config = _default_hpc_config()

    if training_job_ids is None:
        training_job_ids = ["101", "102"]

    if training_sentinels is None:
        training_sentinels = [
            sentinel_dir / "CeD_20260212_151826_LR_EN_s0.done",
            sentinel_dir / "CeD_20260212_151826_RF_s0.done",
        ]

    post_script = scripts_dir / "CeD_20260212_151826_post.sh"
    post_sentinel = sentinel_dir / "CeD_20260212_151826_post.done"

    return _build_orchestrator_script(
        run_id=run_id,
        hpc_config=hpc_config,
        sentinel_dir=sentinel_dir,
        scripts_dir=scripts_dir,
        orchestrator_log=tmp_path / "orchestrator.log",
        orchestrator_sentinel=sentinel_dir / "CeD_20260212_151826_orchestrator.done",
        training_job_ids=training_job_ids,
        training_sentinels=training_sentinels,
        post_script=post_script,
        post_sentinel=post_sentinel,
        perm_scripts=perm_scripts or [],
        perm_sentinels=perm_sentinels or [],
        perm_agg_scripts=perm_agg_scripts or [],
        perm_agg_sentinels=perm_agg_sentinels or [],
        panel_seed_scripts=panel_seed_scripts or [],
        panel_seed_sentinels=panel_seed_sentinels or [],
        panel_agg_scripts=panel_agg_scripts or [],
        panel_agg_sentinels=panel_agg_sentinels or [],
        consensus_script=consensus_script,
        consensus_sentinel=consensus_sentinel,
        expected_training_jobs=2,
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
    """Test LSF job script builder with dependency."""
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


def test_sentinel_path_helpers():
    """Sentinel/script path helpers should construct run-scoped paths correctly."""
    logs_dir = Path("/tmp/logs")
    run_id = "20260212_151826"

    sent_dir = _sentinel_dir(logs_dir, run_id)
    scripts_dir = _scripts_dir(logs_dir, run_id)
    sentinel = _sentinel_path(sent_dir, "CeD_20260212_151826_post")

    assert sent_dir == Path("/tmp/logs/run_20260212_151826/sentinels")
    assert scripts_dir == Path("/tmp/logs/run_20260212_151826/scripts")
    assert sentinel == Path("/tmp/logs/run_20260212_151826/sentinels/CeD_20260212_151826_post.done")


def test_append_sentinel_touch():
    """Sentinel touch should be appended without removing existing commands."""
    script = "#!/bin/bash\nset -euo pipefail\nced train --run-id 1\n"
    sentinel = Path("/tmp/logs/run_1/sentinels/CeD_1_LR_EN_s0.done")

    updated = _append_sentinel_touch(script, sentinel)

    assert "ced train --run-id 1" in updated
    assert f'touch "{sentinel}"' in updated
    assert updated.strip().endswith(f'touch "{sentinel}"')


def test_barrier_bash_uses_bjobs_and_bhist():
    """Failure checks must use bjobs first and bhist as fallback."""
    bash = _build_orchestrator_bash_functions()

    assert 'bjobs -noheader -o "stat"' in bash
    assert "bhist -l" in bash


def test_barrier_bash_no_grep_p():
    """Generated orchestrator bash must avoid grep -P for POSIX compatibility."""
    bash = _build_orchestrator_bash_functions()

    assert "grep -P" not in bash


def test_submit_and_track_uses_sed_for_id():
    """Job ID extraction should use sed parsing."""
    bash = _build_orchestrator_bash_functions()

    assert "sed -n 's/.*Job" in bash


def test_submit_and_track_writes_to_id_file():
    """submit_and_track should append parsed IDs to caller-provided file."""
    bash = _build_orchestrator_bash_functions()

    assert 'echo "$job_id" >> "$id_file"' in bash


def test_submit_batch_writes_ids_to_file():
    """submit_batch should route IDs through id_file to avoid stdout word-splitting."""
    bash = _build_orchestrator_bash_functions()

    assert "submit_batch()" in bash
    assert 'local id_file="$1"' in bash
    assert 'submit_and_track "${scripts[$i]}"' in bash


def test_orchestrator_script_training_only(tmp_path):
    """Training-only orchestrator should include post stage but no optional stages."""
    script = _orchestrator_script_for_test(tmp_path)

    assert 'barrier_wait "training"' in script
    assert 'barrier_wait "post-processing"' in script
    assert "PERM_SCRIPTS=(" not in script
    assert "PANEL_SEED_SCRIPTS=(" not in script
    assert 'barrier_wait "consensus"' not in script


def test_orchestrator_script_full(tmp_path):
    """Full orchestrator script should include permutation, panel, and consensus stages."""
    scripts_dir = tmp_path / "scripts"
    sentinels_dir = tmp_path / "sentinels"

    perm_scripts = [
        scripts_dir / "CeD_20260212_151826_perm_LR_EN_s0.sh",
        scripts_dir / "CeD_20260212_151826_perm_RF_s0.sh",
    ]
    perm_sentinels = [
        sentinels_dir / "CeD_20260212_151826_perm_LR_EN_s0.done",
        sentinels_dir / "CeD_20260212_151826_perm_RF_s0.done",
    ]
    perm_agg_scripts = [
        scripts_dir / "CeD_20260212_151826_perm_LR_EN_agg.sh",
        scripts_dir / "CeD_20260212_151826_perm_RF_agg.sh",
    ]
    perm_agg_sentinels = [
        sentinels_dir / "CeD_20260212_151826_perm_LR_EN_agg.done",
        sentinels_dir / "CeD_20260212_151826_perm_RF_agg.done",
    ]
    panel_seed_scripts = [
        scripts_dir / "CeD_20260212_151826_panel_LR_EN_s0.sh",
        scripts_dir / "CeD_20260212_151826_panel_RF_s0.sh",
    ]
    panel_seed_sentinels = [
        sentinels_dir / "CeD_20260212_151826_panel_LR_EN_s0.done",
        sentinels_dir / "CeD_20260212_151826_panel_RF_s0.done",
    ]
    panel_agg_scripts = [
        scripts_dir / "CeD_20260212_151826_panel_LR_EN_agg.sh",
        scripts_dir / "CeD_20260212_151826_panel_RF_agg.sh",
    ]
    panel_agg_sentinels = [
        sentinels_dir / "CeD_20260212_151826_panel_LR_EN_agg.done",
        sentinels_dir / "CeD_20260212_151826_panel_RF_agg.done",
    ]

    script = _orchestrator_script_for_test(
        tmp_path,
        perm_scripts=perm_scripts,
        perm_sentinels=perm_sentinels,
        perm_agg_scripts=perm_agg_scripts,
        perm_agg_sentinels=perm_agg_sentinels,
        panel_seed_scripts=panel_seed_scripts,
        panel_seed_sentinels=panel_seed_sentinels,
        panel_agg_scripts=panel_agg_scripts,
        panel_agg_sentinels=panel_agg_sentinels,
        consensus_script=scripts_dir / "CeD_20260212_151826_consensus.sh",
        consensus_sentinel=sentinels_dir / "CeD_20260212_151826_consensus.done",
    )

    assert "PERM_SCRIPTS=(" in script
    assert 'barrier_wait "permutation-tests"' in script
    assert "PERM_AGG_SCRIPTS=(" in script
    assert 'barrier_wait "permutation-aggregation"' in script
    assert "PANEL_SEED_SCRIPTS=(" in script
    assert 'barrier_wait "panel-seed"' in script
    assert "PANEL_AGG_SCRIPTS=(" in script
    assert 'barrier_wait "panel-aggregation"' in script
    assert 'barrier_wait "consensus"' in script


def test_orchestrator_per_stage_timeouts(tmp_path):
    """Each stage should use its own timeout from orchestrator config."""
    hpc_config = _default_hpc_config(
        orchestrator={
            "poll_interval": 30,
            "training_timeout": 4000,
            "post_timeout": 2000,
            "perm_timeout": 5000,
            "panel_timeout": 3000,
            "consensus_timeout": 1200,
            "max_concurrent_submissions": 9,
            "cores": 1,
            "mem_per_core": 1024,
            "walltime": "10:00",
        }
    )

    scripts_dir = tmp_path / "scripts"
    sentinels_dir = tmp_path / "sentinels"
    script = _orchestrator_script_for_test(
        tmp_path,
        hpc_config=hpc_config,
        perm_scripts=[scripts_dir / "perm.sh"],
        perm_sentinels=[sentinels_dir / "perm.done"],
        perm_agg_scripts=[scripts_dir / "perm_agg.sh"],
        perm_agg_sentinels=[sentinels_dir / "perm_agg.done"],
        panel_seed_scripts=[scripts_dir / "panel_seed.sh"],
        panel_seed_sentinels=[sentinels_dir / "panel_seed.done"],
        panel_agg_scripts=[scripts_dir / "panel_agg.sh"],
        panel_agg_sentinels=[sentinels_dir / "panel_agg.done"],
        consensus_script=scripts_dir / "consensus.sh",
        consensus_sentinel=sentinels_dir / "consensus.done",
    )

    assert 'barrier_wait "training" 4000 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "post-processing" 2000 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "permutation-tests" 5000 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "permutation-aggregation" 5000 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "panel-seed" 3000 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "panel-aggregation" 3000 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "consensus" 1200 "$POLL_INTERVAL"' in script


def test_orchestrator_batch_chunking(tmp_path):
    """submit_batch calls should be chunked according to max_concurrent_submissions."""
    hpc_config = _default_hpc_config(
        orchestrator={
            "max_concurrent_submissions": 7,
            "poll_interval": 60,
            "training_timeout": 3600,
            "post_timeout": 1800,
            "perm_timeout": 3600,
            "panel_timeout": 2400,
            "consensus_timeout": 900,
            "cores": 1,
            "mem_per_core": 1024,
            "walltime": "12:00",
        }
    )

    scripts_dir = tmp_path / "scripts"
    sentinels_dir = tmp_path / "sentinels"
    script = _orchestrator_script_for_test(
        tmp_path,
        hpc_config=hpc_config,
        perm_scripts=[scripts_dir / "perm.sh"],
        perm_sentinels=[sentinels_dir / "perm.done"],
        perm_agg_scripts=[scripts_dir / "perm_agg.sh"],
        perm_agg_sentinels=[sentinels_dir / "perm_agg.done"],
    )

    assert "MAX_CHUNK=7" in script
    assert 'submit_batch "$MAX_CHUNK"' in script


def test_orchestrator_training_ids_passed(tmp_path):
    """Training job IDs should be embedded in orchestrator script."""
    script = _orchestrator_script_for_test(
        tmp_path,
        training_job_ids=["11", "12", "13", "14"],
    )

    assert "TRAINING_IDS=(" in script
    assert '"11"' in script
    assert '"12"' in script
    assert '"13"' in script
    assert '"14"' in script


def test_orchestrator_state_file(tmp_path):
    """Orchestrator should track stage transitions in state jsonl."""
    script = _orchestrator_script_for_test(tmp_path)

    assert "orchestrator_state.jsonl" in script
    assert '"status":"done"' in script
    assert '"status":"timeout"' in script


def test_submit_orchestrator_dry_run(monkeypatch, tmp_path):
    """Dry run should stage scripts but only dry-submit the orchestrator job."""
    submitted: list[tuple[str, bool]] = []

    def fake_submit_job(script: str, dry_run: bool = False) -> str | None:
        submitted.append((script, dry_run))
        return None

    run_id = "20260212_151826"
    monkeypatch.setattr(
        "ced_ml.hpc.lsf.detect_environment",
        lambda _: EnvironmentInfo(env_type="venv", activation_cmd="source venv/bin/activate"),
    )
    monkeypatch.setattr("ced_ml.hpc.lsf.submit_job", fake_submit_job)

    result = submit_hpc_pipeline(
        config_file=tmp_path / "training_config.yaml",
        infile=tmp_path / "input.parquet",
        split_dir=tmp_path / "splits",
        outdir=tmp_path / "results",
        models=["LR_EN"],
        split_seeds=[0, 1],
        run_id=run_id,
        enable_ensemble=False,
        enable_consensus=False,
        enable_optimize_panel=False,
        enable_permutation_test=False,
        hpc_config=_default_hpc_config(),
        logs_dir=tmp_path / "logs",
        dry_run=True,
        pipeline_logger=logging.getLogger("test_submit_orchestrator_dry_run"),
    )

    scripts_dir = tmp_path / "logs" / f"run_{run_id}" / "scripts"
    assert scripts_dir.exists()
    assert (scripts_dir / f"CeD_{run_id}_orchestrator.sh").exists()
    assert len(list(scripts_dir.glob("*.sh"))) == 4  # 2 training + post + orchestrator

    assert len(submitted) == 1
    assert submitted[0][1] is True
    assert result["orchestrator_job"].startswith("DRYRUN_")


def test_submit_orchestrator_scripts_have_sentinels(monkeypatch, tmp_path):
    """Every generated downstream job script should end with a sentinel touch."""

    def fake_submit_job(script: str, dry_run: bool = False) -> str | None:
        return None

    run_id = "20260212_151827"
    monkeypatch.setattr(
        "ced_ml.hpc.lsf.detect_environment",
        lambda _: EnvironmentInfo(env_type="venv", activation_cmd="source venv/bin/activate"),
    )
    monkeypatch.setattr("ced_ml.hpc.lsf.submit_job", fake_submit_job)

    submit_hpc_pipeline(
        config_file=tmp_path / "training_config.yaml",
        infile=tmp_path / "input.parquet",
        split_dir=tmp_path / "splits",
        outdir=tmp_path / "results",
        models=["LR_EN", "RF"],
        split_seeds=[0],
        run_id=run_id,
        enable_ensemble=False,
        enable_consensus=True,
        enable_optimize_panel=True,
        enable_permutation_test=True,
        permutation_split_seeds=[0],
        hpc_config=_default_hpc_config(),
        logs_dir=tmp_path / "logs",
        dry_run=True,
        pipeline_logger=logging.getLogger("test_submit_orchestrator_scripts_have_sentinels"),
    )

    scripts_dir = tmp_path / "logs" / f"run_{run_id}" / "scripts"
    assert scripts_dir.exists()

    for script_path in scripts_dir.glob("*.sh"):
        # Orchestrator script has its own internal touch logic and terminal echo.
        if script_path.name == f"CeD_{run_id}_orchestrator.sh":
            continue

        content = script_path.read_text()
        lines = [line for line in content.strip().splitlines() if line.strip()]
        assert lines[-1].startswith('touch "')
        assert f"/run_{run_id}/sentinels/" in lines[-1]
