"""Submit dry-run tests — mock subprocess.run, verify bsub command shape."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

from ced_ml.cellml.registry import list_all
from ced_ml.cellml.schema import (
    AxesSpec,
    BaseConfigs,
    ExperimentSpec,
    PanelSpec,
    ResourcesSpec,
    SeedRange,
)
from ced_ml.cellml.submit import submit_experiment


def _make_experiment(tmp_path: Path, name: str = "test_exp") -> tuple[ExperimentSpec, Path]:
    exp_dir = tmp_path / name
    recipes_dir = exp_dir / "recipes"
    recipes_dir.mkdir(parents=True)
    manifest = recipes_dir / "cell_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id",
                "recipe_id",
                "model",
                "cell_name",
                "pipeline_config",
                "calibration",
                "weighting",
                "downsampling",
            ],
        )
        w.writeheader()
        for i in range(1, 4):
            w.writerow(
                {
                    "cell_id": i,
                    "recipe_id": "R1",
                    "model": "LR_EN",
                    "cell_name": f"cell_{i}",
                    "pipeline_config": f"/p/pipeline_{i}.yaml",
                    "calibration": "logistic_intercept",
                    "weighting": "none",
                    "downsampling": "1.0",
                }
            )
    spec = ExperimentSpec(
        name=name,
        base_configs=BaseConfigs(training="t.yaml", pipeline="p.yaml", splits="s.yaml"),
        trunks=[],
        panels=[PanelSpec(id="R1", source="fixed_csv", csv=tmp_path / "x.csv")],
        axes=AxesSpec(
            model=["LR_EN"],
            calibration=["logistic_intercept"],
            weighting=["none"],
            downsampling=[1.0],
        ),
        seeds=SeedRange(start=100, end=102),
        resources=ResourcesSpec(
            wall="24:00", cores=4, mem_mb_per_core=2000, queue="premium", project="acc"
        ),
    )
    (tmp_path / "x.csv").write_text("protein\nX\n")
    return spec, exp_dir


def test_dry_run_writes_runners_no_bsub(tmp_path, monkeypatch):
    spec, exp_dir = _make_experiment(tmp_path)

    # Redirect registry to tmp so we don't touch real state
    monkeypatch.setattr(
        "ced_ml.cellml.registry._default_registry_path",
        lambda: tmp_path / "_registry.csv",
    )

    bsub_mock = MagicMock()
    result = submit_experiment(
        spec,
        exp_dir,
        repo_root=tmp_path,
        dry_run=True,
        bsub_runner=bsub_mock,
    )
    assert result.dry_run is True
    assert result.job_id is None
    bsub_mock.assert_not_called()

    # Runner scripts exist for all 3 cells
    for i in range(1, 4):
        runner = exp_dir / "logs" / f"{spec.name}_runners" / f"cell_{i}.sh"
        assert runner.exists()
        text = runner.read_text()
        assert "ced run-pipeline" in text
        assert f"cell_{i}" in text

    # bsub command shape
    cmd = result.bsub_cmd
    assert "bsub" in cmd[0]
    assert "-J" in cmd and f"{spec.name}[1-3]" in cmd
    assert "-n" in cmd and "4" in cmd
    assert "-W" in cmd and "24:00" in cmd
    assert "-R" in cmd
    assert any("rusage[mem=2000]" in c for c in cmd)

    # Registry NOT updated (dry-run)
    rows = list_all(registry_path=tmp_path / "_registry.csv")
    assert rows == [] or all(r.get("status", "") != "submitted" for r in rows)


def test_live_submit_parses_job_id(tmp_path, monkeypatch):
    spec, exp_dir = _make_experiment(tmp_path, name="live_exp")
    monkeypatch.setattr(
        "ced_ml.cellml.registry._default_registry_path",
        lambda: tmp_path / "_registry.csv",
    )
    # Register first so update_status has a row
    from ced_ml.cellml.registry import register

    register(spec, spec_path=tmp_path / "live_exp.yaml", registry_path=tmp_path / "_registry.csv")

    fake = MagicMock()
    fake.return_value = MagicMock(
        returncode=0,
        stdout="Job <987654> is submitted to queue <premium>.\n",
        stderr="",
    )
    result = submit_experiment(
        spec,
        exp_dir,
        repo_root=tmp_path,
        dry_run=False,
        bsub_runner=fake,
    )
    assert result.job_id == "987654"
    fake.assert_called_once()
