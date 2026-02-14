"""Tests for HPC LSF job submission utilities."""

import base64
import json
import logging
from pathlib import Path

from ced_ml.config.schema import HPCConfig
from ced_ml.hpc.common import (
    EnvironmentInfo,
    _build_consensus_panel_command,
    _build_orchestrator_bash_functions,
    _build_orchestrator_script,
    _build_panel_optimization_command,
    _build_permutation_aggregation_command,
    _build_postprocessing_command,
    _build_training_command,
    _build_wrapper_script,
    _scripts_dir,
    _sentinel_dir,
    _sentinel_log_path,
    build_job_script,
    detect_environment,
    submit_hpc_pipeline,
)
from ced_ml.hpc.lsf import LSFScheduler

_LSF = LSFScheduler()


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
    training_job_names: list[str] | None = None,
    perm_keys: list[str] | None = None,
    perm_job_names: list[str] | None = None,
    perm_agg_keys: list[str] | None = None,
    perm_agg_job_names: list[str] | None = None,
    panel_seed_keys: list[str] | None = None,
    panel_seed_job_names: list[str] | None = None,
    panel_agg_keys: list[str] | None = None,
    panel_agg_job_names: list[str] | None = None,
    consensus_key: str | None = None,
    consensus_job_name: str | None = None,
) -> str:
    """Build an orchestrator script with configurable stage inputs."""
    run_id = "20260212_151826"
    sentinel_dir = tmp_path / "sentinels"
    scripts_dir = tmp_path / "scripts"

    if hpc_config is None:
        hpc_config = _default_hpc_config()

    if training_job_ids is None:
        training_job_ids = ["101", "102"]

    if training_job_names is None:
        training_job_names = [
            "CeD_20260212_151826_LR_EN_s0",
            "CeD_20260212_151826_RF_s0",
        ]

    return _build_orchestrator_script(
        scheduler=_LSF,
        run_id=run_id,
        hpc_config=hpc_config,
        sentinel_dir=sentinel_dir,
        scripts_dir=scripts_dir,
        orchestrator_log=tmp_path / "orchestrator.log",
        orchestrator_job_name=f"CeD_{run_id}_orchestrator",
        manifest_path=scripts_dir / "jobs_manifest.json",
        wrapper_script_path=scripts_dir / "CeD_20260212_151826_job_wrapper.sh",
        training_job_ids=training_job_ids,
        training_job_names=training_job_names,
        post_key="post",
        post_job_name=f"CeD_{run_id}_post",
        perm_keys=perm_keys or [],
        perm_job_names=perm_job_names or [],
        perm_agg_keys=perm_agg_keys or [],
        perm_agg_job_names=perm_agg_job_names or [],
        panel_seed_keys=panel_seed_keys or [],
        panel_seed_job_names=panel_seed_job_names or [],
        panel_agg_keys=panel_agg_keys or [],
        panel_agg_job_names=panel_agg_job_names or [],
        consensus_key=consensus_key,
        consensus_job_name=consensus_job_name,
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


def test_build_permutation_aggregation_command():
    """Test permutation aggregation command builds --aggregate-only CLI call."""
    cmd = _build_permutation_aggregation_command(run_id="20260130_120000", model="LR_EN")

    assert cmd == "ced permutation-test --run-id 20260130_120000 --model LR_EN --aggregate-only"


def test_build_consensus_panel_command():
    """Test consensus panel command builder."""
    cmd = _build_consensus_panel_command(run_id="20260130_120000")

    assert cmd == "ced consensus-panel --run-id 20260130_120000"


def test_build_job_script_basic():
    """Test LSF job script builder without dependency."""
    script = build_job_script(
        scheduler=_LSF,
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
        scheduler=_LSF,
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


def test_sentinel_helpers():
    """Sentinel/script path helpers should construct run-scoped paths correctly."""
    logs_dir = Path("/tmp/logs")
    run_id = "20260212_151826"

    sent_dir = _sentinel_dir(logs_dir, run_id)
    scripts_dir = _scripts_dir(logs_dir, run_id)
    log_path = _sentinel_log_path(sent_dir)

    assert sent_dir == Path("/tmp/logs/run_20260212_151826/sentinels")
    assert scripts_dir == Path("/tmp/logs/run_20260212_151826/scripts")
    assert log_path == Path("/tmp/logs/run_20260212_151826/sentinels/completed.log")


def test_sentinel_log_path():
    """Sentinel log path should return consolidated completed.log in sentinel dir."""
    sentinel_dir = Path("/tmp/logs/run_1/sentinels")

    log_path = _sentinel_log_path(sentinel_dir)

    assert log_path == sentinel_dir / "completed.log"
    assert log_path.name == "completed.log"


def test_detect_environment_prefers_venv(tmp_path, monkeypatch):
    """If both venv and .venv exist, prefer analysis/venv for stability."""
    project_root = tmp_path / "project"
    venv_activate = project_root / "analysis" / "venv" / "bin" / "activate"
    dotvenv_activate = project_root / "analysis" / ".venv" / "bin" / "activate"
    venv_activate.parent.mkdir(parents=True)
    dotvenv_activate.parent.mkdir(parents=True)
    venv_activate.write_text("# venv\n")
    dotvenv_activate.write_text("# dotvenv\n")

    monkeypatch.setattr("ced_ml.utils.paths.get_project_root", lambda: project_root)
    env = detect_environment(project_root)

    assert env.env_type == "venv"
    assert str(venv_activate) in env.activation_cmd


def test_detect_environment_uses_dotvenv(tmp_path, monkeypatch):
    """Fallback to analysis/.venv when analysis/venv is absent."""
    project_root = tmp_path / "project"
    dotvenv_activate = project_root / "analysis" / ".venv" / "bin" / "activate"
    dotvenv_activate.parent.mkdir(parents=True)
    dotvenv_activate.write_text("# dotvenv\n")

    monkeypatch.setattr("ced_ml.utils.paths.get_project_root", lambda: project_root)
    env = detect_environment(project_root)

    assert env.env_type == "venv"
    assert str(dotvenv_activate) in env.activation_cmd


def test_detect_environment_uses_virtual_env_fallback(tmp_path, monkeypatch):
    """Fallback to $VIRTUAL_ENV when no project-local venv path exists."""
    project_root = tmp_path / "project"
    (project_root / "analysis").mkdir(parents=True)
    fallback_activate = tmp_path / "external_venv" / "bin" / "activate"
    fallback_activate.parent.mkdir(parents=True)
    fallback_activate.write_text("# external venv\n")

    monkeypatch.setattr("ced_ml.utils.paths.get_project_root", lambda: project_root)
    monkeypatch.setenv("VIRTUAL_ENV", str(fallback_activate.parent.parent))
    env = detect_environment(project_root)

    assert env.env_type == "venv"
    assert str(fallback_activate) in env.activation_cmd


def test_detect_environment_raises_with_checked_paths(tmp_path, monkeypatch):
    """Error should list checked project-local venv candidates for diagnostics."""
    project_root = tmp_path / "project"
    (project_root / "analysis").mkdir(parents=True)

    monkeypatch.setattr("ced_ml.utils.paths.get_project_root", lambda: project_root)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    import pytest

    with pytest.raises(RuntimeError) as exc_info:
        detect_environment(project_root)

    message = str(exc_info.value)
    assert "analysis/venv/bin/activate" in message
    assert "analysis/.venv/bin/activate" in message


def test_wrapper_script_decodes_base64_and_marks_sentinel():
    """Wrapper script should decode command payload and append to consolidated log."""
    script = _build_wrapper_script('source "/venv/bin/activate"')

    assert "CED_JOB_COMMAND_B64" in script
    assert "CED_JOB_NAME" in script
    assert "CED_SENTINEL_DIR" in script
    assert "base64.b64decode" in script
    # Consolidated sentinel via EXIT trap
    assert '>> "$CED_SENTINEL_DIR/completed.log"' in script
    assert "trap" in script
    # Old per-job touch must be gone
    assert ".done" not in script


def test_barrier_bash_uses_bjobs_and_bhist():
    """Failure checks must use bjobs first and bhist as fallback."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert 'bjobs -noheader -o "stat"' in bash
    assert "bhist -l" in bash


def test_bhist_pipelines_guarded_against_pipefail():
    """bhist pipelines in check_upstream_failures must have ``|| true`` to prevent
    silent abort under ``set -eo pipefail`` when bhist returns non-zero for
    purged jobs (regression for exit-code-255 orchestrator crash)."""
    bash = _build_orchestrator_bash_functions(_LSF)

    # Both bhist pipelines must be guarded
    for line in bash.splitlines():
        if "bhist -l" in line and "$(" in line:
            assert "|| true)" in line, f"bhist pipeline missing '|| true' guard: {line.strip()}"


def test_barrier_bash_no_grep_p():
    """Generated orchestrator bash must avoid grep -P for POSIX compatibility."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert "grep -P" not in bash


def test_barrier_wait_uses_consolidated_log():
    """barrier_wait should check completion via grep in consolidated log."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert 'grep -qx "${name}" "$SENTINEL_DIR/completed.log"' in bash
    # Old per-job file check must be gone
    assert ".done" not in bash


def test_submit_and_track_uses_sed_for_id():
    """Job ID extraction should use sed parsing."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert "sed -n 's/.*Job" in bash


def test_submit_and_track_uses_manifest_and_wrapper():
    """submit_and_track should read manifest entries and run shared wrapper."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert "manifest_job_tsv" in bash
    assert 'python - "$MANIFEST_PATH" "$job_key"' in bash
    assert '"$WRAPPER_SCRIPT"' in bash


def test_submit_and_track_exports_sentinel_env():
    """submit_and_track should export CED_JOB_NAME and CED_SENTINEL_DIR."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert 'CED_JOB_NAME="$job_name"' in bash
    assert 'CED_SENTINEL_DIR="$SENTINEL_DIR"' in bash


def test_submit_and_track_avoids_literal_embedded_bsub_directives():
    """Embedded child script directives must not be literal #BSUB lines in orchestrator source."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert 'local bsub_directive="#BSUB"' in bash
    assert '#BSUB -R "rusage[mem=$mem_per_core] span[hosts=1]"' not in bash
    assert '${bsub_directive} -R "rusage[mem=$mem_per_core] span[hosts=1]"' in bash


def test_submit_and_track_writes_to_id_file():
    """submit_and_track should append parsed IDs to caller-provided file."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert 'echo "$job_id" >> "$id_file"' in bash


def test_submit_batch_writes_ids_to_file():
    """submit_batch should route IDs through id_file to avoid stdout word-splitting."""
    bash = _build_orchestrator_bash_functions(_LSF)

    assert "submit_batch()" in bash
    assert 'local id_file="$1"' in bash
    assert 'local -a job_keys=("$@")' in bash
    assert 'submit_and_track "${job_keys[$i]}"' in bash


def test_orchestrator_script_training_only(tmp_path):
    """Training-only orchestrator should include post stage but no optional stages."""
    script = _orchestrator_script_for_test(tmp_path)

    assert 'barrier_wait "training"' in script
    assert 'barrier_wait "post-processing"' in script
    assert 'submit_and_track "post" "post-processing"' in script
    assert "PERM_KEYS=(" not in script
    assert "PANEL_SEED_KEYS=(" not in script
    assert 'barrier_wait "consensus"' not in script
    assert script.count("#BSUB -R ") == 1
    assert "SENTINEL_DIR=" in script
    # Consolidated log initialized and used for orchestrator completion
    assert 'touch "$SENTINEL_DIR/completed.log"' in script
    assert '>> "$SENTINEL_DIR/completed.log"' in script


def test_orchestrator_script_full(tmp_path):
    """Full orchestrator script should include permutation, panel, and consensus stages."""
    script = _orchestrator_script_for_test(
        tmp_path,
        perm_keys=["perm_LR_EN_s0", "perm_RF_s0"],
        perm_job_names=[
            "CeD_20260212_151826_perm_LR_EN_s0",
            "CeD_20260212_151826_perm_RF_s0",
        ],
        perm_agg_keys=["perm_LR_EN_agg", "perm_RF_agg"],
        perm_agg_job_names=[
            "CeD_20260212_151826_perm_LR_EN_agg",
            "CeD_20260212_151826_perm_RF_agg",
        ],
        panel_seed_keys=["panel_LR_EN_s0", "panel_RF_s0"],
        panel_seed_job_names=[
            "CeD_20260212_151826_panel_LR_EN_s0",
            "CeD_20260212_151826_panel_RF_s0",
        ],
        panel_agg_keys=["panel_LR_EN_agg", "panel_RF_agg"],
        panel_agg_job_names=[
            "CeD_20260212_151826_panel_LR_EN_agg",
            "CeD_20260212_151826_panel_RF_agg",
        ],
        consensus_key="consensus",
        consensus_job_name="CeD_20260212_151826_consensus",
    )

    assert "PERM_KEYS=(" in script
    assert 'barrier_wait "permutation-tests"' in script
    assert "PERM_AGG_KEYS=(" in script
    assert 'barrier_wait "permutation-aggregation"' in script
    assert "PANEL_SEED_KEYS=(" in script
    assert 'barrier_wait "panel-seed"' in script
    assert "PANEL_AGG_KEYS=(" in script
    assert 'barrier_wait "panel-aggregation"' in script
    assert 'barrier_wait "consensus"' in script


def test_orchestrator_per_stage_timeouts(tmp_path):
    """Each stage should use its own timeout from orchestrator config."""
    hpc_config = _default_hpc_config(
        orchestrator={
            "poll_interval": 30,
            "training_timeout": 2.0,
            "post_timeout": 1.0,
            "perm_timeout": 2.5,
            "panel_timeout": 1.5,
            "consensus_timeout": 0.5,
            "max_concurrent_submissions": 9,
            "cores": 1,
            "mem_per_core": 1024,
            "walltime": "10:00",
        }
    )

    script = _orchestrator_script_for_test(
        tmp_path,
        hpc_config=hpc_config,
        perm_keys=["perm"],
        perm_job_names=["CeD_perm"],
        perm_agg_keys=["perm_agg"],
        perm_agg_job_names=["CeD_perm_agg"],
        panel_seed_keys=["panel_seed"],
        panel_seed_job_names=["CeD_panel_seed"],
        panel_agg_keys=["panel_agg"],
        panel_agg_job_names=["CeD_panel_agg"],
        consensus_key="consensus",
        consensus_job_name="CeD_consensus",
    )

    assert 'barrier_wait "training" 7200 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "post-processing" 3600 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "permutation-tests" 9000 "$POLL_INTERVAL"' in script
    # perm aggregation uses post_timeout (1.0h = 3600s)
    assert 'barrier_wait "permutation-aggregation" 3600 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "panel-seed" 5400 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "panel-aggregation" 5400 "$POLL_INTERVAL"' in script
    assert 'barrier_wait "consensus" 1800 "$POLL_INTERVAL"' in script


def test_orchestrator_batch_chunking(tmp_path):
    """submit_batch calls should be chunked according to max_concurrent_submissions."""
    hpc_config = _default_hpc_config(
        orchestrator={
            "max_concurrent_submissions": 7,
            "poll_interval": 60,
            "training_timeout": 1.0,
            "post_timeout": 0.5,
            "perm_timeout": 1.0,
            "panel_timeout": 0.5,
            "consensus_timeout": 0.25,
            "cores": 1,
            "mem_per_core": 1024,
            "walltime": "12:00",
        }
    )

    script = _orchestrator_script_for_test(
        tmp_path,
        hpc_config=hpc_config,
        perm_keys=["perm"],
        perm_job_names=["CeD_perm"],
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
    """Dry run should stage wrapper+orchestrator and dry-submit only orchestrator."""
    submitted: list[tuple[str, bool]] = []

    def fake_submit_job(script: str, *, scheduler=None, dry_run: bool = False) -> str | None:
        submitted.append((script, dry_run))
        return None

    run_id = "20260212_151826"
    monkeypatch.setattr(
        "ced_ml.hpc.common.detect_environment",
        lambda _: EnvironmentInfo(env_type="venv", activation_cmd="source venv/bin/activate"),
    )
    monkeypatch.setattr("ced_ml.hpc.common.submit_job", fake_submit_job)

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
    assert (scripts_dir / f"CeD_{run_id}_job_wrapper.sh").exists()
    assert (scripts_dir / f"CeD_{run_id}_orchestrator.sh").exists()
    assert len(list(scripts_dir.glob("*.sh"))) == 2
    assert (scripts_dir / "jobs_manifest.json").exists()
    run_metadata_path = tmp_path / "results" / f"run_{run_id}" / "run_metadata.json"
    assert run_metadata_path.exists()
    run_metadata = json.loads(run_metadata_path.read_text())
    assert run_metadata["run_id"] == run_id
    assert "LR_EN" in run_metadata["models"]

    assert len(submitted) == 1
    assert submitted[0][1] is True
    assert result["orchestrator_job"].startswith("DRYRUN_")


def test_submit_orchestrator_manifest_format(monkeypatch, tmp_path):
    """Manifest should contain base64 commands and job_name for all staged jobs."""

    def fake_submit_job(script: str, *, scheduler=None, dry_run: bool = False) -> str | None:
        return None

    run_id = "20260212_151827"
    monkeypatch.setattr(
        "ced_ml.hpc.common.detect_environment",
        lambda _: EnvironmentInfo(env_type="venv", activation_cmd="source venv/bin/activate"),
    )
    monkeypatch.setattr("ced_ml.hpc.common.submit_job", fake_submit_job)

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
        pipeline_logger=logging.getLogger("test_submit_orchestrator_manifest_format"),
    )

    scripts_dir = tmp_path / "logs" / f"run_{run_id}" / "scripts"
    manifest_path = scripts_dir / "jobs_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert "post" in manifest
    assert "consensus" in manifest
    assert any(key.startswith("perm_") for key in manifest)
    assert any(key.startswith("panel_") for key in manifest)

    for entry in manifest.values():
        assert "job_name" in entry
        assert "command_b64" in entry
        assert "sentinel" not in entry
        decoded = base64.b64decode(entry["command_b64"]).decode("utf-8")
        assert decoded.startswith("ced ") or decoded.startswith("echo ")

    run_metadata_path = tmp_path / "results" / f"run_{run_id}" / "run_metadata.json"
    assert run_metadata_path.exists()
    run_metadata = json.loads(run_metadata_path.read_text())
    assert "LR_EN" in run_metadata["models"]
    assert "RF" in run_metadata["models"]

    # With manifest+wrapper approach, script files remain bounded.
    assert len(list(scripts_dir.glob("*.sh"))) == 2
