"""Tests for run-level metadata manifest utilities."""

import json

from ced_ml.utils.run_manifest import build_model_manifest_entry, ensure_run_manifest


def test_ensure_run_manifest_creates_new_file(tmp_path):
    """Should create run_metadata.json with top-level and model-level fields."""
    run_dir = tmp_path / "run_20260213_000000"

    manifest_path, changed = ensure_run_manifest(
        run_level_dir=run_dir,
        run_id="20260213_000000",
        infile="/data/input.parquet",
        split_dir="/data/splits",
        model_entries={
            "LR_EN": build_model_manifest_entry(
                scenario="IncidentOnly",
                infile="/data/input.parquet",
                split_dir="/data/splits",
            )
        },
    )

    assert changed is True
    assert manifest_path.exists()

    metadata = json.loads(manifest_path.read_text())
    assert metadata["run_id"] == "20260213_000000"
    assert metadata["infile"] == "/data/input.parquet"
    assert metadata["split_dir"] == "/data/splits"
    assert "LR_EN" in metadata["models"]
    assert metadata["models"]["LR_EN"]["scenario"] == "IncidentOnly"


def test_ensure_run_manifest_is_non_destructive(tmp_path):
    """Should keep existing entries untouched and only add missing models."""
    run_dir = tmp_path / "run_20260213_000001"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_metadata.json"

    existing = {
        "run_id": "20260213_000001",
        "infile": "/data/original.parquet",
        "split_dir": "/data/original_splits",
        "models": {
            "LR_EN": {
                "scenario": "IncidentOnly",
                "infile": "/data/original.parquet",
                "split_dir": "/data/original_splits",
                "split_seed": 42,
                "timestamp": "2026-02-13T10:00:00",
            }
        },
    }
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    _, changed = ensure_run_manifest(
        run_level_dir=run_dir,
        run_id="DIFFERENT_RUN_ID",
        infile="/data/new.parquet",
        split_dir="/data/new_splits",
        model_entries={
            "LR_EN": build_model_manifest_entry(
                scenario="DifferentScenario",
                infile="/data/new.parquet",
                split_dir="/data/new_splits",
            ),
            "RF": build_model_manifest_entry(
                scenario="IncidentOnly",
                infile="/data/original.parquet",
                split_dir="/data/original_splits",
            ),
        },
    )

    assert changed is True  # RF added

    metadata = json.loads(manifest_path.read_text())
    assert metadata["run_id"] == "20260213_000001"
    assert metadata["infile"] == "/data/original.parquet"
    assert metadata["split_dir"] == "/data/original_splits"
    assert metadata["models"]["LR_EN"]["split_seed"] == 42
    assert metadata["models"]["LR_EN"]["timestamp"] == "2026-02-13T10:00:00"
    assert metadata["models"]["LR_EN"]["scenario"] == "IncidentOnly"
    assert "RF" in metadata["models"]

    # Idempotent once all keys exist
    _, changed_again = ensure_run_manifest(
        run_level_dir=run_dir,
        run_id="20260213_000001",
        infile="/data/original.parquet",
        split_dir="/data/original_splits",
        model_entries={"RF": build_model_manifest_entry(scenario="IncidentOnly")},
    )
    assert changed_again is False
