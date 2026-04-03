"""
Tests for multi-model comparison plots.
"""

import numpy as np
import pytest

from ced_ml.metrics.thresholds import compute_threshold_bundle
from ced_ml.plotting.comparison import (
    ModelCurveData,
    _resolve_model_order,
    plot_calibration_comparison,
    plot_dca_comparison,
    plot_pr_comparison,
    plot_roc_comparison,
)
from ced_ml.plotting.style import MODEL_COLORS, get_model_color


@pytest.fixture
def three_model_data():
    """Synthetic data for 3 models with split IDs."""
    rng = np.random.default_rng(42)
    n_splits = 3
    n_per_split = 60

    models = {}
    for model_name, auc_shift in [("LR_EN", 0.0), ("RF", 0.3), ("XGBoost", 0.5)]:
        y_true_all = []
        y_pred_all = []
        split_ids_all = []

        for sid in range(n_splits):
            y = np.concatenate([np.zeros(50), np.ones(10)])
            p = np.concatenate(
                [
                    rng.beta(2, 5, 50),
                    rng.beta(3 + auc_shift, 2, 10),
                ]
            )
            p = np.clip(p, 0, 1)
            y_true_all.append(y)
            y_pred_all.append(p)
            split_ids_all.append(np.full(n_per_split, sid))

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        split_ids = np.concatenate(split_ids_all)
        bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.95)

        models[model_name] = ModelCurveData(
            y_true=y_true,
            y_pred=y_pred,
            split_ids=split_ids,
            threshold_bundle=bundle,
        )

    return models


@pytest.fixture
def two_model_data_no_splits():
    """Minimal 2-model data without split IDs."""
    rng = np.random.default_rng(99)
    models = {}
    for name in ["ModelA", "ModelB"]:
        y = np.concatenate([np.zeros(80), np.ones(20)])
        p = np.clip(rng.beta(2, 5, 80).tolist() + rng.beta(5, 2, 20).tolist(), 0, 1)
        models[name] = ModelCurveData(
            y_true=y,
            y_pred=np.array(p),
            split_ids=None,
            threshold_bundle=None,
        )
    return models


class TestModelOrder:
    """Tests for _resolve_model_order helper."""

    def test_alphabetical_default(self):
        models = {"RF": {}, "LR_EN": {}, "XGBoost": {}}
        order = _resolve_model_order(models)
        assert order == ["LR_EN", "RF", "XGBoost"]

    def test_ensemble_drawn_last(self):
        models = {"ENSEMBLE": {}, "RF": {}, "LR_EN": {}}
        order = _resolve_model_order(models)
        assert order[-1] == "ENSEMBLE"
        assert order == ["LR_EN", "RF", "ENSEMBLE"]

    def test_custom_order(self):
        models = {"RF": {}, "LR_EN": {}, "XGBoost": {}}
        order = _resolve_model_order(models, model_order=["XGBoost", "LR_EN", "RF"])
        assert order == ["XGBoost", "LR_EN", "RF"]

    def test_custom_order_filters_missing(self):
        models = {"RF": {}, "LR_EN": {}}
        order = _resolve_model_order(models, model_order=["XGBoost", "LR_EN", "RF"])
        assert order == ["LR_EN", "RF"]


class TestModelColors:
    """Tests for model color assignment."""

    def test_known_models(self):
        for name in ["LR_EN", "RF", "LinSVM_cal", "XGBoost", "ENSEMBLE"]:
            assert get_model_color(name) in MODEL_COLORS.values()

    def test_unknown_model_fallback(self):
        color = get_model_color("UnknownModel")
        assert color == "#6c757d"


class TestROCComparison:
    """Tests for ROC comparison plot."""

    def test_basic_output(self, three_model_data, tmp_path):
        out = tmp_path / "roc_comparison.png"
        plot_roc_comparison(models=three_model_data, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_with_meta_lines(self, three_model_data, tmp_path):
        out = tmp_path / "roc_meta.png"
        plot_roc_comparison(
            models=three_model_data,
            out_path=out,
            meta_lines=["Test run", "3 splits"],
        )
        assert out.exists()

    def test_no_splits(self, two_model_data_no_splits, tmp_path):
        out = tmp_path / "roc_nosplit.png"
        plot_roc_comparison(models=two_model_data_no_splits, out_path=out)
        assert out.exists()

    def test_single_model_skipped(self, tmp_path):
        rng = np.random.default_rng(42)
        models = {
            "OnlyModel": ModelCurveData(
                y_true=np.concatenate([np.zeros(80), np.ones(20)]),
                y_pred=rng.random(100),
                split_ids=None,
                threshold_bundle=None,
            )
        }
        out = tmp_path / "roc_single.png"
        plot_roc_comparison(models=models, out_path=out)
        assert not out.exists()  # Should skip with < 2 models


class TestPRComparison:
    """Tests for PR comparison plot."""

    def test_basic_output(self, three_model_data, tmp_path):
        out = tmp_path / "pr_comparison.png"
        plot_pr_comparison(models=three_model_data, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_splits(self, two_model_data_no_splits, tmp_path):
        out = tmp_path / "pr_nosplit.png"
        plot_pr_comparison(models=two_model_data_no_splits, out_path=out)
        assert out.exists()


class TestCalibrationComparison:
    """Tests for calibration comparison plot."""

    def test_basic_output(self, three_model_data, tmp_path):
        out = tmp_path / "cal_comparison.png"
        plot_calibration_comparison(models=three_model_data, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_bins(self, three_model_data, tmp_path):
        out = tmp_path / "cal_bins.png"
        plot_calibration_comparison(models=three_model_data, out_path=out, n_bins=5)
        assert out.exists()

    def test_no_splits(self, two_model_data_no_splits, tmp_path):
        out = tmp_path / "cal_nosplit.png"
        plot_calibration_comparison(models=two_model_data_no_splits, out_path=out)
        assert out.exists()


class TestDCAComparison:
    """Tests for DCA comparison plot."""

    def test_basic_output(self, three_model_data, tmp_path):
        out = tmp_path / "dca_comparison.png"
        plot_dca_comparison(models=three_model_data, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_threshold_range(self, three_model_data, tmp_path):
        out = tmp_path / "dca_custom.png"
        plot_dca_comparison(models=three_model_data, out_path=out, max_pt=0.30, step=0.01)
        assert out.exists()

    def test_no_splits(self, two_model_data_no_splits, tmp_path):
        out = tmp_path / "dca_nosplit.png"
        plot_dca_comparison(models=two_model_data_no_splits, out_path=out)
        assert out.exists()
