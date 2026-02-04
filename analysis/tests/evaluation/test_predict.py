"""Tests for evaluation/predict.py (prediction generation)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from ced_ml.evaluation.predict import (
    export_predictions,
    generate_predictions,
    generate_predictions_with_adjustment,
    predict_on_holdout,
    predict_on_test,
    predict_on_validation,
)
from ced_ml.models.prevalence import PrevalenceAdjustedModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def simple_model():
    """Create a simple trained logistic regression pipeline."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 5)), columns=[f"feat_{i}" for i in range(5)])
    y = rng.integers(0, 2, size=100)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def test_data():
    """Create test data for predictions."""
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"feat_{i}" for i in range(5)])
    y = rng.integers(0, 2, size=50)
    return X, y


def test_generate_predictions_basic(simple_model, test_data):
    """Test basic prediction generation."""
    X, _ = test_data
    preds = generate_predictions(simple_model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_generate_predictions_clipping():
    """Test that predictions are clipped to [0, 1]."""

    class MockModel:
        def predict_proba(self, X):
            # Return out-of-range values
            return np.array([[-0.1, 1.1], [0.5, 0.5], [-0.5, 1.5]])

    X = pd.DataFrame(np.zeros((3, 2)))
    preds = generate_predictions(MockModel(), X)

    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert preds[0] == 1.0  # 1.1 clipped to 1.0
    assert preds[1] == 0.5  # 0.5 unchanged
    assert preds[2] == 1.0  # 1.5 clipped to 1.0


def test_generate_predictions_prevalence_adjusted_model(simple_model, test_data):
    """Test prediction generation with PrevalenceAdjustedModel."""
    X, _ = test_data
    adjusted_model = PrevalenceAdjustedModel(
        base_model=simple_model, sample_prevalence=0.5, target_prevalence=0.1
    )

    # Raw predictions (from base model)
    preds_raw = generate_predictions(adjusted_model, X, return_raw=True)
    assert isinstance(preds_raw, np.ndarray)
    assert preds_raw.shape == (len(X),)

    # Adjusted predictions
    preds_adj = generate_predictions(adjusted_model, X, return_raw=False)
    assert isinstance(preds_adj, np.ndarray)
    assert preds_adj.shape == (len(X),)

    # Adjusted should be lower than raw (prevalence 0.5 → 0.1)
    assert preds_adj.mean() < preds_raw.mean()


def test_generate_predictions_with_adjustment_basic(simple_model, test_data):
    """Test generating both raw and adjusted predictions."""
    X, _ = test_data
    preds = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.5, target_prevalence=0.1
    )

    assert "raw" in preds
    assert "adjusted" in preds
    assert preds["raw"].shape == (len(X),)
    assert preds["adjusted"].shape == (len(X),)
    assert np.all((preds["raw"] >= 0.0) & (preds["raw"] <= 1.0))
    assert np.all((preds["adjusted"] >= 0.0) & (preds["adjusted"] <= 1.0))


def test_generate_predictions_with_adjustment_prevalence_effect(simple_model, test_data):
    """Test that prevalence adjustment affects predictions correctly."""
    X, _ = test_data

    # Adjust from 50% to 10% (should decrease)
    preds_down = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.5, target_prevalence=0.1
    )

    # Adjust from 10% to 50% (should increase)
    preds_up = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.1, target_prevalence=0.5
    )

    assert preds_down["adjusted"].mean() < preds_down["raw"].mean()
    assert preds_up["adjusted"].mean() > preds_up["raw"].mean()


def test_export_predictions_basic(simple_model, test_data):
    """Test basic prediction export."""
    X, y = test_data
    preds = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.5, target_prevalence=0.1
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "preds.csv")
        export_predictions(
            predictions=preds,
            y_true=y,
            out_path=out_path,
        )

        assert os.path.exists(out_path)
        df = pd.read_csv(out_path)

        # Check required columns
        assert "id" in df.columns
        assert "y_true" in df.columns
        assert "risk_raw" in df.columns
        assert "risk_adjusted" in df.columns
        assert len(df) == len(y)


def test_export_predictions_with_ids_and_target(simple_model, test_data):
    """Test prediction export with custom IDs and target column."""
    X, y = test_data
    preds = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.5, target_prevalence=0.1
    )

    ids = np.array([f"SUBJ_{i:04d}" for i in range(len(y))])
    target_col = pd.Series(["Control" if yi == 0 else "Incident" for yi in y])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "preds.csv")
        export_predictions(
            predictions=preds,
            y_true=y,
            out_path=out_path,
            ids=ids,
            target_col=target_col,
        )

        df = pd.read_csv(out_path)
        assert "target" in df.columns
        assert df["id"].iloc[0] == "SUBJ_0000"
        assert df["target"].iloc[0] in ["Control", "Incident"]


def test_export_predictions_active_key(simple_model, test_data):
    """Test prediction export with custom active key."""
    X, y = test_data
    preds = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.5, target_prevalence=0.1
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "preds.csv")
        export_predictions(
            predictions=preds,
            y_true=y,
            out_path=out_path,
            active_key="raw",
        )

        df = pd.read_csv(out_path)
        assert "risk" in df.columns
        # Active should match raw
        np.testing.assert_array_almost_equal(df["risk"].values, preds["raw"])


def test_export_predictions_percentile(simple_model, test_data):
    """Test prediction export with percentile columns."""
    X, y = test_data
    preds = generate_predictions_with_adjustment(
        model=simple_model, X=X, train_prevalence=0.5, target_prevalence=0.1
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "preds.csv")
        export_predictions(
            predictions=preds,
            y_true=y,
            out_path=out_path,
            percentile=True,
        )

        df = pd.read_csv(out_path)
        assert "risk_raw_pct" in df.columns
        assert "risk_adjusted_pct" in df.columns
        np.testing.assert_array_almost_equal(df["risk_raw_pct"].values, 100.0 * preds["raw"])


def test_export_predictions_no_percentile(simple_model, test_data):
    """Test prediction export without percentile columns."""
    rng = np.random.default_rng(42)
    X, y = test_data
    preds = {"raw": rng.random(len(y))}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "preds.csv")
        export_predictions(
            predictions=preds,
            y_true=y,
            out_path=out_path,
            percentile=False,
        )

        df = pd.read_csv(out_path)
        assert "risk_raw_pct" not in df.columns


def test_export_predictions_length_mismatch(test_data):
    """Test that export raises error on prediction length mismatch."""
    rng = np.random.default_rng(42)
    _, y = test_data
    preds = {"raw": rng.random(10)}  # Wrong length

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "preds.csv")
        with pytest.raises(ValueError, match="has length 10, expected"):
            export_predictions(
                predictions=preds,
                y_true=y,
                out_path=out_path,
            )


def test_predict_on_validation(simple_model, test_data):
    """Test prediction generation on validation set."""
    X, y = test_data
    result = predict_on_validation(
        model=simple_model,
        X_val=X,
        y_val=y,
        train_prevalence=0.5,
        target_prevalence=0.1,
        use_adjusted=True,
    )

    assert "raw" in result
    assert "adjusted" in result
    assert "active" in result
    assert "n" in result
    assert "n_pos" in result
    assert "prevalence" in result

    assert result["n"] == len(y)
    assert result["n_pos"] == int(y.sum())
    assert result["prevalence"] == pytest.approx(y.mean())

    # Active should be adjusted
    np.testing.assert_array_almost_equal(result["active"], result["adjusted"])


def test_predict_on_validation_raw_active(simple_model, test_data):
    """Test validation predictions with raw as active."""
    X, y = test_data
    result = predict_on_validation(
        model=simple_model,
        X_val=X,
        y_val=y,
        train_prevalence=0.5,
        target_prevalence=0.1,
        use_adjusted=False,
    )

    # Active should be raw
    np.testing.assert_array_almost_equal(result["active"], result["raw"])


def test_predict_on_test(simple_model, test_data):
    """Test prediction generation on test set."""
    X, y = test_data
    result = predict_on_test(
        model=simple_model,
        X_test=X,
        y_test=y,
        train_prevalence=0.5,
        target_prevalence=0.1,
        use_adjusted=True,
    )

    assert "raw" in result
    assert "adjusted" in result
    assert "active" in result
    assert result["n"] == len(y)
    assert result["n_pos"] == int(y.sum())


def test_predict_on_holdout_prevalence_adjusted_model(simple_model, test_data):
    """Test holdout predictions with PrevalenceAdjustedModel."""
    X, y = test_data
    adjusted_model = PrevalenceAdjustedModel(
        base_model=simple_model, sample_prevalence=0.5, target_prevalence=0.1
    )

    result = predict_on_holdout(model=adjusted_model, X_holdout=X, y_holdout=y)

    assert "raw" in result
    assert "adjusted" in result
    assert "active" in result
    assert result["n"] == len(y)

    # Active should be adjusted
    np.testing.assert_array_almost_equal(result["active"], result["adjusted"])


def test_predict_on_holdout_pipeline(simple_model, test_data):
    """Test holdout predictions with Pipeline model."""
    X, y = test_data
    result = predict_on_holdout(
        model=simple_model,
        X_holdout=X,
        y_holdout=y,
        train_prevalence=0.5,
        target_prevalence=0.1,
    )

    assert "raw" in result
    assert "adjusted" in result
    assert result["n"] == len(y)


def test_predict_on_holdout_pipeline_missing_prevalence(simple_model, test_data):
    """Test that holdout predictions raise error when prevalence missing."""
    X, y = test_data
    with pytest.raises(ValueError, match="train_prevalence and target_prevalence required"):
        predict_on_holdout(model=simple_model, X_holdout=X, y_holdout=y)


def test_predict_on_holdout_metadata(simple_model, test_data):
    """Test holdout predictions include correct metadata."""
    X, y = test_data
    y_custom = np.array([0] * 40 + [1] * 10)  # 10 positives

    result = predict_on_holdout(
        model=simple_model,
        X_holdout=X,
        y_holdout=y_custom,
        train_prevalence=0.5,
        target_prevalence=0.1,
    )

    assert result["n"] == 50
    assert result["n_pos"] == 10
    assert result["prevalence"] == pytest.approx(0.2)


def test_end_to_end_prediction_workflow(simple_model):
    """Test complete prediction workflow: train → val → test → export."""
    rng = np.random.default_rng(99)
    X_val = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"feat_{i}" for i in range(5)])
    y_val = rng.integers(0, 2, size=30)

    X_test = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"feat_{i}" for i in range(5)])
    y_test = rng.integers(0, 2, size=40)

    # Generate validation predictions
    val_preds = predict_on_validation(
        model=simple_model,
        X_val=X_val,
        y_val=y_val,
        train_prevalence=0.5,
        target_prevalence=0.003,
        use_adjusted=True,
    )

    # Generate test predictions
    test_preds = predict_on_test(
        model=simple_model,
        X_test=X_test,
        y_test=y_test,
        train_prevalence=0.5,
        target_prevalence=0.003,
        use_adjusted=True,
    )

    # Export both
    with tempfile.TemporaryDirectory() as tmpdir:
        val_path = os.path.join(tmpdir, "val_preds.csv")
        test_path = os.path.join(tmpdir, "test_preds.csv")

        export_predictions(
            predictions={"raw": val_preds["raw"], "adjusted": val_preds["adjusted"]},
            y_true=y_val,
            out_path=val_path,
        )

        export_predictions(
            predictions={"raw": test_preds["raw"], "adjusted": test_preds["adjusted"]},
            y_true=y_test,
            out_path=test_path,
        )

        assert os.path.exists(val_path)
        assert os.path.exists(test_path)

        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        assert len(df_val) == 30
        assert len(df_test) == 40
        assert "risk_adjusted" in df_val.columns
        assert "risk_adjusted" in df_test.columns
