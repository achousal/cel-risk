"""Tests for CLI discovery helpers."""

import json

from ced_ml.cli.discovery import auto_detect_data_paths


def test_auto_detect_data_paths_prefers_top_level_metadata(tmp_path):
    """Top-level infile/split_dir should win over model-level values."""
    results_dir = tmp_path / "results"
    run_id = "20260213_120000"
    run_dir = results_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True)

    metadata = {
        "run_id": run_id,
        "infile": "/top/data.parquet",
        "split_dir": "/top/splits",
        "models": {
            "LR_EN": {
                "infile": "/model/data.parquet",
                "split_dir": "/model/splits",
            }
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    infile, split_dir = auto_detect_data_paths(run_id=run_id, results_dir=results_dir)
    assert infile == "/top/data.parquet"
    assert split_dir == "/top/splits"


def test_auto_detect_data_paths_falls_back_to_model_level(tmp_path):
    """Model-level values should be used when top-level metadata is absent."""
    results_dir = tmp_path / "results"
    run_id = "20260213_120001"
    run_dir = results_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True)

    metadata = {
        "run_id": run_id,
        "models": {
            "LR_EN": {
                "infile": "/model/data.parquet",
                "split_dir": "/model/splits",
            }
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    infile, split_dir = auto_detect_data_paths(run_id=run_id, results_dir=results_dir)
    assert infile == "/model/data.parquet"
    assert split_dir == "/model/splits"
