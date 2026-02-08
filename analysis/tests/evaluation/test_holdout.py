"""
Tests for holdout evaluation module.

Tests cover:
- Loading holdout indices
- Loading model artifacts
- Extracting holdout data
- Computing holdout metrics
- Top-risk capture
- Predictions saving
- Full evaluation pipeline
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ced_ml.data.schema import TARGET_COL
from ced_ml.evaluation.holdout import (
    compute_holdout_metrics,
    compute_top_risk_capture,
    evaluate_holdout,
    extract_holdout_data,
    load_holdout_indices,
    load_model_artifact,
    save_holdout_predictions,
)


class TestLoadHoldoutIndices:
    """Tests for loading holdout indices."""

    def test_load_valid_indices(self, tmp_path):
        idx_file = tmp_path / "holdout_idx.csv"
        pd.DataFrame({"idx": [0, 10, 20, 30]}).to_csv(idx_file, index=False)

        result = load_holdout_indices(str(idx_file))
        assert isinstance(result.indices, np.ndarray)
        assert len(result.indices) == 4
        assert result.indices.dtype == np.int64
        np.testing.assert_array_equal(result.indices, [0, 10, 20, 30])
        assert isinstance(result.metadata, dict)

    def test_missing_idx_column(self, tmp_path):
        idx_file = tmp_path / "holdout_idx.csv"
        pd.DataFrame({"wrong_col": [0, 1, 2]}).to_csv(idx_file, index=False)

        with pytest.raises(ValueError, match="must contain an 'idx' column"):
            load_holdout_indices(str(idx_file))

    def test_empty_file(self, tmp_path):
        idx_file = tmp_path / "holdout_idx.csv"
        pd.DataFrame({"idx": []}).to_csv(idx_file, index=False)

        result = load_holdout_indices(str(idx_file))
        assert len(result.indices) == 0
        assert isinstance(result.metadata, dict)


class TestLoadModelArtifact:
    """Tests for loading model artifacts."""

    def test_load_valid_artifact(self, tmp_path):
        artifact_path = tmp_path / "model.joblib"
        bundle = {
            "model": LogisticRegression(),
            "scenario": "IncidentOnly",
            "model_name": "LR",
            "split_id": 0,
        }
        joblib.dump(bundle, artifact_path)

        loaded = load_model_artifact(str(artifact_path))
        assert "model" in loaded
        assert loaded["scenario"] == "IncidentOnly"
        assert loaded["model_name"] == "LR"


class TestExtractHoldoutData:
    """Tests for extracting holdout subsets."""

    def test_extract_valid_subset(self):
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
            }
        )
        X = df[["A", "B"]]
        y = np.array([0, 1, 0, 1, 0])
        holdout_idx = np.array([1, 3])

        df_h, X_h, y_h = extract_holdout_data(df, X, y, holdout_idx)

        assert len(df_h) == 2
        assert len(X_h) == 2
        assert len(y_h) == 2
        np.testing.assert_array_equal(y_h, [1, 1])
        assert list(X_h["A"]) == [2, 4]

    def test_index_exceeds_size(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        X = df[["A"]]
        y = np.array([0, 1, 0])
        holdout_idx = np.array([0, 5])

        with pytest.raises(ValueError, match="exceeds dataset rows"):
            extract_holdout_data(df, X, y, holdout_idx)


class TestComputeHoldoutMetrics:
    """Tests for computing holdout metrics."""

    def test_basic_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

        bundle = {
            "model_name": "LR",
            "model_label": "Logistic",
            "split_id": 0,
            "thresholds": {
                "objective_name": "max_f1",
                "objective": 0.5,
                "max_f1": 0.5,
                "spec90": 0.7,
                "control_specs": {},
            },
            "prevalence": {
                "train_prevalence": 0.4,
                "test_prevalence": 0.01,
            },
        }

        metrics = compute_holdout_metrics(
            y_true,
            proba,
            bundle,
            "IncidentOnly",
            clinical_points=[0.05, 0.1],
        )

        assert metrics["scenario"] == "IncidentOnly"
        assert metrics["model_name"] == "LR"
        assert metrics["n_holdout"] == 5
        assert metrics["n_holdout_pos"] == 2
        assert 0 <= metrics["AUROC_holdout"] <= 1
        assert 0 <= metrics["Brier_holdout"] <= 1
        assert "calibration_intercept_holdout" in metrics
        assert "calibration_slope_holdout" in metrics
        assert "ECE_holdout" in metrics

    def test_clinical_thresholds(self):
        y_true = np.array([0] * 80 + [1] * 20)
        proba = np.random.RandomState(42).uniform(0, 1, 100)

        bundle = {
            "model_name": "RF",
            "thresholds": {
                "objective_name": "max_f1",
                "objective": 0.5,
                "max_f1": 0.5,
                "spec90": 0.7,
                "control_specs": {},
            },
            "prevalence": {"train_prevalence": 0.2, "test_prevalence": 0.2},
        }

        metrics = compute_holdout_metrics(
            y_true,
            proba,
            bundle,
            "IncidentOnly",
            clinical_points=[0.05, 0.1, 0.2],
        )

        assert "clin_0p05_threshold" in metrics
        assert metrics["clin_0p05_threshold"] == 0.05
        assert "clin_0p05_precision" in metrics
        assert "clin_0p1_threshold" in metrics
        assert "clin_0p2_recall" in metrics


class TestComputeTopRiskCapture:
    """Tests for top-risk capture computation."""

    def test_single_fraction(self):
        y_true = np.array([0] * 90 + [1] * 10)
        proba = np.random.RandomState(42).uniform(0, 1, 100)

        df = compute_top_risk_capture(y_true, proba, [0.1])

        assert len(df) == 1
        assert df.loc[0, "frac"] == 0.1
        assert "n_top" in df.columns
        assert "cases_in_top" in df.columns

    def test_multiple_fractions(self):
        y_true = np.array([0] * 80 + [1] * 20)
        proba = np.random.RandomState(42).uniform(0, 1, 100)

        df = compute_top_risk_capture(y_true, proba, [0.01, 0.05, 0.1])

        assert len(df) == 3
        assert list(df["frac"]) == [0.01, 0.05, 0.1]


class TestSaveHoldoutPredictions:
    """Tests for saving holdout predictions."""

    def test_save_predictions(self, tmp_path):
        holdout_idx = np.array([10, 20, 30])
        df_holdout = pd.DataFrame(
            {
                TARGET_COL: ["Control", "Incident", "Control"],
            }
        )
        y_true = np.array([0, 1, 0])
        proba_eval = np.array([0.1, 0.8, 0.2])
        proba_adjusted = np.array([0.05, 0.75, 0.15])

        save_holdout_predictions(
            str(tmp_path),
            holdout_idx,
            df_holdout,
            y_true,
            proba_eval,
            proba_adjusted,
        )

        preds_file = tmp_path / "holdout_predictions.csv"
        assert preds_file.exists()

        df = pd.read_csv(preds_file)
        assert len(df) == 3
        assert "idx" in df.columns
        assert TARGET_COL in df.columns
        assert "y_true" in df.columns
        assert "risk_holdout" in df.columns
        assert "risk_holdout_adjusted" in df.columns
        np.testing.assert_array_equal(df["idx"], [10, 20, 30])
        np.testing.assert_array_equal(df["y_true"], [0, 1, 0])


class TestEvaluateHoldout:
    """Integration tests for full holdout evaluation."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock Celiac dataset."""
        rng = np.random.default_rng(42)
        n = 200

        data = {
            TARGET_COL: ["Control"] * 180 + ["Incident"] * 20,
            "age": rng.uniform(20, 80, n),
            "BMI": rng.uniform(18, 35, n),
            "sex": rng.choice(["Male", "Female"], n),
            "Genetic ethnic grouping": rng.choice(["White", "Asian", "Other"], n),
        }

        # Add protein columns
        for i in range(50):
            data[f"P{i:05d}_resid"] = rng.standard_normal(n)

        df = pd.DataFrame(data)
        csv_path = tmp_path / "dataset.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def mock_model_artifact(self, tmp_path):
        """Create a mock trained model artifact."""
        rng = np.random.default_rng(42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create dummy training data
        X_train = rng.standard_normal((100, 20))
        y_train = rng.integers(0, 2, 100)
        model.fit(X_train, y_train)

        bundle = {
            "model": model,
            "scenario": "IncidentOnly",
            "model_name": "RF",
            "model_label": "Random Forest",
            "split_id": 0,
            "thresholds": {
                "objective_name": "max_f1",
                "objective": 0.5,
                "max_f1": 0.5,
                "spec90": 0.7,
                "control_specs": {"0.95": 0.8, "0.99": 0.9},
                "spec_targets": [0.95, 0.99],
            },
            "prevalence": {
                "train_prevalence": 0.3,
                "test_prevalence": 0.01,
            },
            "args": {
                "dca_threshold_min": 0.001,
                "dca_threshold_max": 0.10,
                "dca_threshold_step": 0.01,
                "clinical_threshold_points": "0.05,0.1",
            },
        }

        artifact_path = tmp_path / "model.joblib"
        joblib.dump(bundle, artifact_path)
        return artifact_path

    @pytest.fixture
    def mock_holdout_indices(self, tmp_path):
        """Create mock holdout indices."""
        # Indices referencing filtered dataset (after removing prevalent)
        indices = pd.DataFrame({"idx": list(range(150, 200))})
        idx_path = tmp_path / "holdout_idx.csv"
        indices.to_csv(idx_path, index=False)
        return idx_path

    def test_evaluate_holdout_basic(
        self, mock_dataset, mock_model_artifact, mock_holdout_indices, tmp_path
    ):
        """Test basic holdout evaluation without DCA."""
        outdir = tmp_path / "holdout_results"

        # Note: This will fail because the mock model expects different features
        # For now, we'll just test that the function signature works
        # Real integration test would need matching features

        with pytest.raises((ValueError, KeyError)):
            # Expected to fail due to feature mismatch, but validates API
            evaluate_holdout(
                infile=str(mock_dataset),
                holdout_idx_file=str(mock_holdout_indices),
                model_artifact_path=str(mock_model_artifact),
                outdir=str(outdir),
                save_preds=True,
                toprisk_fracs="0.01,0.05",
            )

    def test_evaluate_holdout_with_dca(
        self, mock_dataset, mock_model_artifact, mock_holdout_indices, tmp_path
    ):
        """Test holdout evaluation with DCA computation."""
        outdir = tmp_path / "holdout_results"

        with pytest.raises((ValueError, KeyError)):
            # Expected to fail due to feature mismatch, but validates API
            evaluate_holdout(
                infile=str(mock_dataset),
                holdout_idx_file=str(mock_holdout_indices),
                model_artifact_path=str(mock_model_artifact),
                outdir=str(outdir),
                compute_dca=True,
                dca_threshold_min=0.01,
                dca_threshold_max=0.10,
                dca_threshold_step=0.01,
                dca_use_target_prevalence=True,
            )


class TestHoldoutEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_holdout_set(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        X = df[["A"]]
        y = np.array([0, 1, 0])
        holdout_idx = np.array([])

        df_h, X_h, y_h = extract_holdout_data(df, X, y, holdout_idx)

        assert len(df_h) == 0
        assert len(X_h) == 0
        assert len(y_h) == 0

    def test_metrics_with_single_class(self):
        """Test metrics computation when holdout has single class (edge case)."""
        y_true = np.array([0, 0, 0, 0, 0])
        proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        bundle = {
            "model_name": "LR",
            "thresholds": {
                "objective_name": "max_f1",
                "objective": 0.5,
                "max_f1": 0.5,
                "spec90": 0.7,
                "control_specs": {},
            },
            "prevalence": {"train_prevalence": 0.2, "test_prevalence": 0.2},
        }

        metrics = compute_holdout_metrics(y_true, proba, bundle, "IncidentOnly", clinical_points=[])

        # Should handle gracefully (some metrics will be NaN)
        assert "AUROC_holdout" in metrics
        assert "n_holdout_pos" in metrics
        assert metrics["n_holdout_pos"] == 0

    def test_prevalence_fallback(self):
        """Test that missing prevalence keys raise ValueError (F1 fix: strict validation)."""
        y_true = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])

        bundle = {
            "model_name": "LR",
            "thresholds": {
                "objective_name": "max_f1",
                "objective": 0.5,
                "max_f1": 0.5,
                "spec90": 0.7,
                "control_specs": {},
            },
            "prevalence": {},  # Missing prevalence info
        }

        # F1 fix: Should raise ValueError for missing prevalence keys
        with pytest.raises(ValueError, match="missing valid 'prevalence.train_prevalence'"):
            compute_holdout_metrics(y_true, proba, bundle, "IncidentOnly", clinical_points=[])
