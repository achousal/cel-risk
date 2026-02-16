"""
Tests for evaluation.reports module (OutputDirectories, ResultsWriter).

Coverage:
- Directory structure creation
- Path management
- Metrics saving (val/test/cv/bootstrap)
- Predictions saving (test/val/train_oof/controls)
- CV artifacts (best params, selected proteins)
- Reports (features, panels, subgroups)
- Model artifacts (save/load)
- Diagnostics (calibration, learning curves, split trace)
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ced_ml.evaluation.reports import OutputDirectories, ResultsWriter

# ========== Fixtures ==========


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def output_dirs(temp_output_dir):
    """Create OutputDirectories instance."""
    return OutputDirectories.create(temp_output_dir)


@pytest.fixture
def results_writer(output_dirs):
    """Create ResultsWriter instance."""
    return ResultsWriter(output_dirs)


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary."""
    return {
        "auroc_test": 0.85,
        "prauc_test": 0.72,
        "brier_test": 0.12,
        "sens_at_spec95_test": 0.45,
        "threshold_obj": 0.15,
        "n_features": 100,
    }


@pytest.fixture
def sample_model():
    """Sample trained model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )


# ========== OutputDirectories Tests ==========


def test_output_directories_creation(temp_output_dir):
    """Test directory structure creation."""
    dirs = OutputDirectories.create(temp_output_dir)

    assert dirs.root == temp_output_dir
    assert os.path.isdir(dirs.core)
    assert os.path.isdir(dirs.cv)
    assert os.path.isdir(dirs.preds_test)
    assert os.path.isdir(dirs.preds_val)
    assert os.path.isdir(dirs.panels_features)
    assert os.path.isdir(dirs.diag_calibration)


def test_output_directories_all_categories(output_dirs):
    """Test all directory categories exist."""
    expected_categories = [
        "core",
        "cv",
        "preds_test",
        "preds_val",
        "preds_controls",
        "preds_train_oof",
        "panels_features",
        "panels_stable",
        "panels_sizes",
        "panels_subgroups",
        "diag",
        "diag_splits",
        "diag_calibration",
        "diag_learning",
        "diag_dca",
        "diag_screening",
        "diag_test_ci",
    ]

    for category in expected_categories:
        assert hasattr(output_dirs, category)
        path = getattr(output_dirs, category)
        assert os.path.isdir(path)


def test_output_directories_get_path(output_dirs):
    """Test get_path method."""
    path = output_dirs.get_path("core", "test_metrics.csv")
    expected = os.path.join(output_dirs.core, "test_metrics.csv")
    assert path == expected


def test_output_directories_get_path_invalid_category(output_dirs):
    """Test get_path with invalid category raises ValueError."""
    with pytest.raises(ValueError, match="Unknown output category"):
        output_dirs.get_path("invalid_category", "file.csv")


def test_output_directories_exist_ok(temp_output_dir):
    """Test exist_ok parameter (no error on re-creation)."""
    OutputDirectories.create(temp_output_dir, exist_ok=True)
    dirs2 = OutputDirectories.create(temp_output_dir, exist_ok=True)
    assert os.path.isdir(dirs2.core)


# ========== ResultsWriter: Settings ==========


def test_save_run_settings(results_writer, sample_metrics):
    """Test save_run_settings."""
    settings = {
        "scenario": "IncidentPlusPrevalent",
        "model": "LR_EN",
        "folds": 5,
        "repeats": 10,
        "scoring": "neg_brier_score",
    }

    path = results_writer.save_run_settings(settings)
    assert os.path.exists(path)

    with open(path) as f:
        loaded = json.load(f)
    assert loaded["scenario"] == "IncidentPlusPrevalent"
    assert loaded["folds"] == 5


# ========== ResultsWriter: Metrics ==========


def test_save_val_metrics(results_writer, sample_metrics):
    """Test save_val_metrics."""
    path = results_writer.save_val_metrics(
        sample_metrics, scenario="IncidentPlusPrevalent", model="LR_EN"
    )

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df["scenario"].iloc[0] == "IncidentPlusPrevalent"
    assert df["model"].iloc[0] == "LR_EN"
    assert df["auroc_test"].iloc[0] == pytest.approx(0.85)


def test_save_test_metrics(results_writer, sample_metrics):
    """Test save_test_metrics."""
    path = results_writer.save_test_metrics(sample_metrics, scenario="IncidentOnly", model="RF")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert df["scenario"].iloc[0] == "IncidentOnly"
    assert df["brier_test"].iloc[0] == pytest.approx(0.12)


def test_save_cv_repeat_metrics(results_writer):
    """Test save_cv_repeat_metrics."""
    cv_results = [
        {"repeat": 0, "auroc_oof": 0.82, "brier_oof": 0.14},
        {"repeat": 1, "auroc_oof": 0.84, "brier_oof": 0.13},
        {"repeat": 2, "auroc_oof": 0.83, "brier_oof": 0.135},
    ]

    path = results_writer.save_cv_repeat_metrics(
        cv_results, scenario="IncidentPlusPrevalent", model="XGBoost"
    )

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df["scenario"].iloc[0] == "IncidentPlusPrevalent"
    assert df["repeat"].tolist() == [0, 1, 2]


def test_save_bootstrap_ci_metrics(results_writer):
    """Test save_bootstrap_ci_metrics."""
    metrics = {
        "auroc_test": 0.85,
        "auroc_test_95CI": "[0.78, 0.91]",
        "brier_test": 0.12,
        "brier_test_95CI": "[0.10, 0.15]",
        "n_boot": 500,
    }

    path = results_writer.save_bootstrap_ci_metrics(
        metrics, scenario="IncidentOnly", model="LinSVM_cal"
    )

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert df["n_boot"].iloc[0] == 500
    assert df["auroc_test_95CI"].iloc[0] == "[0.78, 0.91]"


# ========== ResultsWriter: CV Artifacts ==========


def test_save_best_params_per_split(results_writer):
    """Test save_best_params_per_split."""
    best_params = [
        {"outer_split": 0, "C": 0.1, "penalty": "l2", "solver": "lbfgs"},
        {"outer_split": 1, "C": 0.5, "penalty": "l2", "solver": "lbfgs"},
        {"outer_split": 2, "C": 0.01, "penalty": "l1", "solver": "saga"},
    ]

    path = results_writer.save_best_params_per_split(best_params, scenario="IncidentPlusPrevalent")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df["scenario"].iloc[0] == "IncidentPlusPrevalent"
    assert df["outer_split"].tolist() == [0, 1, 2]


def test_save_selected_proteins_per_split(results_writer):
    """Test save_selected_proteins_per_split."""
    selected = [
        {"outer_split": 0, "selected_proteins_split": "TGM2;CXCL9;ITGB7"},
        {"outer_split": 1, "selected_proteins_split": "TGM2;CXCL9;MUC2"},
        {"outer_split": 2, "selected_proteins_split": "TGM2;ITGB7;IL15"},
    ]

    path = results_writer.save_selected_proteins_per_split(
        selected, scenario="IncidentPlusPrevalent"
    )

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert "TGM2" in df["selected_proteins_split"].iloc[0]


# ========== ResultsWriter: Predictions ==========


def test_save_test_predictions(results_writer):
    """Test save_test_predictions."""
    preds_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "y_true": [0, 1, 0],
            "p_raw": [0.05, 0.75, 0.12],
            "p_adjusted": [0.02, 0.60, 0.08],
        }
    )

    path = results_writer.save_test_predictions(preds_df, model="RF")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df["y_true"].tolist() == [0, 1, 0]


def test_save_val_predictions(results_writer):
    """Test save_val_predictions."""
    preds_df = pd.DataFrame(
        {
            "ID": [10, 20, 30],
            "y_true": [1, 0, 1],
            "p_raw": [0.65, 0.15, 0.82],
        }
    )

    path = results_writer.save_val_predictions(preds_df, model="LR_EN")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert df["p_raw"].iloc[2] == pytest.approx(0.82)


def test_save_train_oof_predictions(results_writer):
    """Test save_train_oof_predictions."""
    preds_df = pd.DataFrame(
        {
            "ID": np.arange(100),
            "y_true": np.random.randint(0, 2, 100),
            "p_oof_mean": np.random.rand(100),
            "split_id": [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20,
        }
    )

    path = results_writer.save_train_oof_predictions(preds_df, model="XGBoost")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 100


def test_save_controls_predictions(results_writer):
    """Test save_controls_predictions."""
    preds_df = pd.DataFrame(
        {
            "ID": [100, 101, 102],
            "risk_RF_oof_mean_adjusted": [0.01, 0.02, 0.015],
            "risk_RF_oof_mean_adjusted_pct": [1.0, 2.0, 1.5],
        }
    )

    path = results_writer.save_controls_predictions(preds_df, model="RF")

    assert os.path.exists(path)
    assert "controls_risk" in path


# ========== ResultsWriter: Reports ==========


def test_save_feature_report(results_writer):
    """Test save_feature_report with all expected columns."""
    report_df = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "protein": ["TGM2", "CXCL9", "ITGB7"],
            "selection_freq": [1.0, 0.9, 0.8],
            "effect_size": [1.73, 1.53, 1.50],
            "p_value": [1e-20, 1e-18, 1e-17],
            "scenario": ["IncidentPlusPrevalent"] * 3,
            "model": ["LR_EN"] * 3,
        }
    )

    path = results_writer.save_feature_report(report_df, model="LR_EN")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df["protein"].iloc[0] == "TGM2"
    assert "effect_size" in df.columns
    assert "p_value" in df.columns
    assert "selection_freq" in df.columns
    assert df["rank"].iloc[0] == 1


def test_save_stable_panel_report(results_writer):
    """Test save_stable_panel_report."""
    panel_df = pd.DataFrame(
        {
            "protein": ["TGM2", "CXCL9"],
            "selection_freq": [1.0, 0.95],
            "cohens_d": [1.73, 1.53],
        }
    )

    path = results_writer.save_stable_panel_report(panel_df, panel_type="KBest")

    assert os.path.exists(path)
    assert "stable_panel" in path


def test_save_final_test_panel(results_writer):
    """Test save_final_test_panel."""
    panel_proteins = ["TGM2", "CXCL9", "ITGB7", "IL15", "MUC2"]
    metadata = {
        "selection_method": "hybrid",
        "n_train": 1000,
        "n_train_pos": 50,
        "train_prevalence": 0.05,
        "random_state": 42,
    }

    path = results_writer.save_final_test_panel(
        panel_proteins,
        scenario="IncidentPlusPrevalent",
        model="LR_EN",
        metadata=metadata,
    )

    assert os.path.exists(path)
    assert "final_test_panel" in path
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["scenario"] == "IncidentPlusPrevalent"
    assert loaded["model"] == "LR_EN"
    assert loaded["panel_size"] == 5
    assert len(loaded["proteins"]) == 5
    assert "TGM2" in loaded["proteins"]
    assert loaded["metadata"]["selection_method"] == "hybrid"
    assert loaded["metadata"]["n_train"] == 1000


def test_save_final_test_panel_no_metadata(results_writer):
    """Test save_final_test_panel without metadata."""
    panel_proteins = ["TGM2", "CXCL9"]

    path = results_writer.save_final_test_panel(panel_proteins, scenario="IncidentOnly", model="RF")

    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["panel_size"] == 2
    assert loaded["metadata"] == {}


def test_save_subgroup_metrics(results_writer):
    """Test save_subgroup_metrics."""
    subgroup_df = pd.DataFrame(
        {
            "subgroup": ["European", "African", "Asian"],
            "n": [150, 30, 20],
            "n_cases": [10, 2, 1],
            "auroc": [0.85, 0.78, 0.82],
        }
    )

    path = results_writer.save_subgroup_metrics(subgroup_df, model="RF")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3


# ========== ResultsWriter: Model Artifacts ==========


def test_save_model_artifact(results_writer, sample_model):
    """Test save_model_artifact."""
    metadata = {
        "hyperparameters": {"C": 0.1, "penalty": "l2"},
        "threshold_obj": 0.15,
        "n_features": 100,
        "training_time": 12.5,
    }

    path = results_writer.save_model_artifact(
        sample_model, metadata, scenario="IncidentPlusPrevalent", model_name="LR_EN"
    )

    assert os.path.exists(path)
    assert path.endswith("final_model.joblib")


def test_load_model_artifact(results_writer, sample_model):
    """Test load_model_artifact."""
    metadata = {
        "hyperparameters": {"C": 0.5},
        "threshold_obj": 0.20,
    }

    results_writer.save_model_artifact(
        sample_model, metadata, scenario="IncidentOnly", model_name="RF"
    )

    bundle = results_writer.load_model_artifact(model_name="RF")

    assert bundle["scenario"] == "IncidentOnly"
    assert bundle["model_name"] == "RF"
    assert "model" in bundle
    assert bundle["metadata"]["hyperparameters"]["C"] == 0.5


def test_load_model_artifact_not_found(results_writer):
    """Test load_model_artifact with missing file."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        results_writer.load_model_artifact(model_name="FakeModel")


# ========== ResultsWriter: Diagnostics ==========


def test_save_calibration_curve(results_writer):
    """Test save_calibration_curve."""
    calib_df = pd.DataFrame(
        {
            "prob_pred": [0.1, 0.2, 0.3, 0.4, 0.5],
            "prob_true": [0.08, 0.18, 0.32, 0.42, 0.51],
        }
    )

    path = results_writer.save_calibration_curve(calib_df, model="LR_EN")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 5
    assert df["prob_pred"].iloc[0] == pytest.approx(0.1)


def test_save_learning_curve(results_writer):
    """Test save_learning_curve."""
    lc_df = pd.DataFrame(
        {
            "train_size": [100, 200, 300],
            "metric": ["neg_brier_score"] * 3,
            "train_score_mean": [-0.15, -0.14, -0.13],
            "test_score_mean": [-0.16, -0.145, -0.135],
        }
    )

    path = results_writer.save_learning_curve(lc_df, model="XGBoost")

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 3


def test_save_split_trace(results_writer):
    """Test save_split_trace."""
    trace_df = pd.DataFrame(
        {
            "index": np.arange(1000),
            "Celiac disease status": ["Controls"] * 980 + ["Incident CeD"] * 20,
            "split": ["TRAIN"] * 500 + ["VAL"] * 250 + ["TEST"] * 250,
            "y": [0] * 980 + [1] * 20,
        }
    )

    path = results_writer.save_split_trace(trace_df)

    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 1000


# ========== ResultsWriter: Utility ==========


def test_summarize_outputs(results_writer, sample_metrics):
    """Test summarize_outputs."""
    results_writer.save_test_metrics(sample_metrics, scenario="IncidentOnly", model="RF")
    results_writer.save_run_settings({"scenario": "IncidentOnly"})

    summary = results_writer.summarize_outputs()
    assert len(summary) >= 2
    assert any("test_metrics.csv" in s for s in summary)
    assert any("run_settings.json" in s for s in summary)


def test_summarize_outputs_empty(results_writer):
    """Test summarize_outputs with no files created."""
    summary = results_writer.summarize_outputs()
    assert len(summary) == 0


# ========== Integration Tests ==========


def test_full_workflow(results_writer, sample_model, sample_metrics):
    """Test full workflow: settings → metrics → predictions → model → report."""
    # Settings
    settings = {"scenario": "IncidentPlusPrevalent", "folds": 5}
    results_writer.save_run_settings(settings)

    # Metrics
    results_writer.save_test_metrics(sample_metrics, "IncidentPlusPrevalent", "LR_EN")
    results_writer.save_val_metrics(sample_metrics, "IncidentPlusPrevalent", "LR_EN")

    # Predictions
    preds = pd.DataFrame({"ID": [1, 2], "y_true": [0, 1], "p_raw": [0.1, 0.9]})
    results_writer.save_test_predictions(preds, "LR_EN")

    # Model
    metadata = {"threshold_obj": 0.15}
    results_writer.save_model_artifact(sample_model, metadata, "IncidentPlusPrevalent", "LR_EN")

    # Verify all exist
    summary = results_writer.summarize_outputs()
    assert len(summary) >= 3
    assert any("test_metrics.csv" in s for s in summary)
    assert any("run_settings.json" in s for s in summary)
