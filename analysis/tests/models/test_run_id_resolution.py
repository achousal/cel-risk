"""Regression tests for run_id propagation in ensemble path resolution.

Verifies that validate_base_models, collect_base_model_test_metrics, and
_find_model_split_dir correctly scope to the requested run_id when multiple
runs exist under the same results root.

Bug: Without explicit run_id, _find_model_split_dir picks the newest run_*
directory, which may differ from the run that discovery detected models in.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ced_ml.cli.ensemble_helpers import (
    collect_base_model_test_metrics,
    validate_base_models,
)
from ced_ml.models.stacking_utils import _find_model_split_dir


def _create_oof_csv(path: Path, model_name: str, n: int = 20) -> None:
    """Create a minimal OOF CSV at the given path."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "idx": np.arange(n),
            "y_true": rng.randint(0, 2, size=n),
            "y_prob_r0": rng.rand(n),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _create_metrics_json(path: Path) -> None:
    """Create a minimal metrics.json at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"test": {"auroc": 0.85, "prauc": 0.40, "brier_score": 0.05}}))


def _build_complete_run(results_dir: Path, run_id: str, models: list[str], seed: int):
    """Create a complete run directory with OOF files and metrics."""
    for model in models:
        split_dir = results_dir / f"run_{run_id}" / model / "splits" / f"split_seed{seed}"
        _create_oof_csv(split_dir / "preds" / f"train_oof__{model}.csv", model)
        _create_metrics_json(split_dir / "core" / "metrics.json")


def _build_incomplete_run(results_dir: Path, run_id: str, models: list[str], seed: int):
    """Create a run directory with split dirs but NO OOF files."""
    for model in models:
        split_dir = results_dir / f"run_{run_id}" / model / "splits" / f"split_seed{seed}"
        # Only create the directory structure, no OOF files
        (split_dir / "preds").mkdir(parents=True, exist_ok=True)


class TestRunIdResolution:
    """Regression tests for run_id path scoping."""

    def test_find_model_split_dir_without_run_id_picks_newest(self, tmp_path):
        """Without run_id, _find_model_split_dir picks the newest run."""
        models = ["LR_EN", "RF"]
        _build_complete_run(tmp_path, "20260101_000000", models, seed=0)
        _build_incomplete_run(tmp_path, "20260201_000000", models, seed=0)

        # Without run_id: picks newest (20260201), which has the directory
        result = _find_model_split_dir(tmp_path, "LR_EN", 0)
        assert "run_20260201_000000" in str(result)

    def test_find_model_split_dir_with_run_id_picks_correct(self, tmp_path):
        """With run_id, _find_model_split_dir scopes to the requested run."""
        models = ["LR_EN", "RF"]
        _build_complete_run(tmp_path, "20260101_000000", models, seed=0)
        _build_incomplete_run(tmp_path, "20260201_000000", models, seed=0)

        result = _find_model_split_dir(tmp_path, "LR_EN", 0, run_id="20260101_000000")
        assert "run_20260101_000000" in str(result)

    def test_validate_base_models_with_run_id_finds_oof(self, tmp_path):
        """validate_base_models with run_id finds OOF in the correct run."""
        models = ["LR_EN", "RF"]
        _build_complete_run(tmp_path, "20260101_000000", models, seed=0)
        _build_incomplete_run(tmp_path, "20260201_000000", models, seed=0)

        # With run_id pointing to the complete run: should find both models
        available, missing = validate_base_models(
            tmp_path, models, split_seed=0, run_id="20260101_000000"
        )
        assert available == models
        assert missing == []

    def test_validate_base_models_without_run_id_picks_wrong_run(self, tmp_path):
        """Without run_id, validate_base_models resolves to the newest (incomplete) run."""
        models = ["LR_EN", "RF"]
        _build_complete_run(tmp_path, "20260101_000000", models, seed=0)
        _build_incomplete_run(tmp_path, "20260201_000000", models, seed=0)

        # Without run_id: resolves to newest run (20260201) which has no OOF
        with pytest.raises(FileNotFoundError, match="need at least 2"):
            validate_base_models(tmp_path, models, split_seed=0)

    def test_collect_test_metrics_with_run_id(self, tmp_path):
        """collect_base_model_test_metrics with run_id reads from the correct run."""
        models = ["LR_EN", "RF"]
        _build_complete_run(tmp_path, "20260101_000000", models, seed=0)
        _build_incomplete_run(tmp_path, "20260201_000000", models, seed=0)

        metrics = collect_base_model_test_metrics(
            tmp_path, models, split_seed=0, run_id="20260101_000000"
        )
        assert set(metrics.keys()) == set(models)
        assert metrics["LR_EN"]["auroc"] == pytest.approx(0.85)
