"""
Tests for hyperparameter aggregation functions in aggregate_splits.

Tests cover:
- Collecting best hyperparameters from split directories
- Aggregating hyperparameter summaries
- Ensemble hyperparameter collection
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ced_ml.cli.aggregation.collection import (
    collect_best_hyperparams,
    collect_ensemble_hyperparams,
)
from ced_ml.cli.aggregation.report_phase import aggregate_hyperparams_summary


@pytest.fixture
def temp_results_dir():
    """Create temporary results directory with mock split structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create split_seed directories
        for seed in [0, 1]:
            split_dir = results_dir / f"split_seed{seed}"
            split_dir.mkdir(parents=True)

            # Create cv directory (optuna files are flat at cv level)
            cv_dir = split_dir / "cv"
            cv_dir.mkdir(parents=True)

        yield results_dir


@pytest.fixture
def mock_hyperparams_data():
    """Sample hyperparameter data."""
    return [
        {
            "model": "LR_EN",
            "repeat": 0,
            "outer_split": 0,
            "best_score_inner": 0.85,
            "best_params": json.dumps(
                {
                    "clf__C": 10.0,
                    "clf__l1_ratio": 0.5,
                    "clf__class_weight": "balanced",
                    "sel__selector__k": 50,
                }
            ),
            "optuna_n_trials": 100,
            "optuna_sampler": "tpe",
            "optuna_pruner": "hyperband",
        },
        {
            "model": "LR_EN",
            "repeat": 0,
            "outer_split": 1,
            "best_score_inner": 0.87,
            "best_params": json.dumps(
                {
                    "clf__C": 15.0,
                    "clf__l1_ratio": 0.4,
                    "clf__class_weight": "balanced",
                    "sel__selector__k": 60,
                }
            ),
            "optuna_n_trials": 100,
            "optuna_sampler": "tpe",
            "optuna_pruner": "hyperband",
        },
    ]


def test_collect_best_hyperparams_empty_dir(temp_results_dir):
    """Test collection when no hyperparameter files exist."""
    split_dirs = list(temp_results_dir.glob("split_seed*"))

    result = collect_best_hyperparams(split_dirs)

    assert result.empty
    assert isinstance(result, pd.DataFrame)


def test_collect_best_hyperparams_single_split(temp_results_dir, mock_hyperparams_data):
    """Test collecting hyperparameters from a single split."""
    split_dir = temp_results_dir / "split_seed0"
    params_file = split_dir / "cv" / "best_params_optuna.csv"

    # Write mock data
    df = pd.DataFrame(mock_hyperparams_data)
    df.to_csv(params_file, index=False)

    split_dirs = [split_dir]
    result = collect_best_hyperparams(split_dirs)

    assert not result.empty
    assert "split_seed" in result.columns
    assert result["split_seed"].iloc[0] == 0
    assert "clf__C" in result.columns
    assert "clf__l1_ratio" in result.columns
    assert len(result) == 2  # Two CV folds


def test_collect_best_hyperparams_multiple_splits(temp_results_dir, mock_hyperparams_data):
    """Test collecting hyperparameters from multiple splits."""
    for seed in [0, 1]:
        split_dir = temp_results_dir / f"split_seed{seed}"
        params_file = split_dir / "cv" / "best_params_optuna.csv"

        # Write mock data
        df = pd.DataFrame(mock_hyperparams_data)
        df.to_csv(params_file, index=False)

    split_dirs = list(temp_results_dir.glob("split_seed*"))
    result = collect_best_hyperparams(split_dirs)

    assert not result.empty
    assert result["split_seed"].nunique() == 2
    assert len(result) == 4  # 2 splits x 2 folds each


def test_aggregate_hyperparams_summary_empty():
    """Test aggregation with empty DataFrame."""
    result = aggregate_hyperparams_summary(pd.DataFrame())

    assert result.empty


def test_aggregate_hyperparams_summary_numeric_params():
    """Test aggregation of numeric hyperparameters."""
    data = {
        "model": ["LR_EN", "LR_EN", "LR_EN"],
        "split_seed": [0, 0, 1],
        "clf__C": [10.0, 15.0, 12.0],
        "clf__l1_ratio": [0.5, 0.4, 0.6],
        "sel__selector__k": [50, 60, 55],
    }
    df = pd.DataFrame(data)

    result = aggregate_hyperparams_summary(df)

    assert not result.empty
    assert "model" in result.columns
    assert result["model"].iloc[0] == "LR_EN"
    assert "clf__C_mean" in result.columns
    assert "clf__C_std" in result.columns
    assert "clf__C_min" in result.columns
    assert "clf__C_max" in result.columns

    # Check computed values
    assert result["clf__C_mean"].iloc[0] == pytest.approx(12.333, rel=1e-2)
    assert result["clf__C_min"].iloc[0] == 10.0
    assert result["clf__C_max"].iloc[0] == 15.0


def test_aggregate_hyperparams_summary_categorical_params():
    """Test aggregation of categorical hyperparameters."""
    data = {
        "model": ["LR_EN", "LR_EN", "LR_EN"],
        "split_seed": [0, 0, 1],
        "clf__class_weight": ["balanced", "balanced", "balanced"],
        "clf__penalty": ["l1", "l2", "l1"],
    }
    df = pd.DataFrame(data)

    result = aggregate_hyperparams_summary(df)

    assert not result.empty
    assert "clf__class_weight_mode" in result.columns
    assert result["clf__class_weight_mode"].iloc[0] == "balanced"
    assert "clf__penalty_n_unique" in result.columns
    assert result["clf__penalty_n_unique"].iloc[0] == 2
    assert "clf__penalty_values" in result.columns
    assert "l1" in result["clf__penalty_values"].iloc[0]
    assert "l2" in result["clf__penalty_values"].iloc[0]


def test_aggregate_hyperparams_summary_multiple_models():
    """Test aggregation with multiple models."""
    data = {
        "model": ["LR_EN", "LR_EN", "RF", "RF"],
        "split_seed": [0, 1, 0, 1],
        "clf__C": [10.0, 15.0, np.nan, np.nan],
        "clf__n_estimators": [np.nan, np.nan, 100, 200],
    }
    df = pd.DataFrame(data)

    result = aggregate_hyperparams_summary(df)

    assert len(result) == 2  # Two models
    assert set(result["model"]) == {"LR_EN", "RF"}

    # LR_EN should have C stats, not n_estimators
    lr_row = result[result["model"] == "LR_EN"].iloc[0]
    assert "clf__C_mean" in lr_row.index
    assert pd.notna(lr_row["clf__C_mean"])

    # RF should have n_estimators stats
    rf_row = result[result["model"] == "RF"].iloc[0]
    assert "clf__n_estimators_mean" in rf_row.index
    assert pd.notna(rf_row["clf__n_estimators_mean"])


def test_collect_ensemble_hyperparams_empty_dir(temp_results_dir):
    """Test ensemble collection when no ensemble directories exist."""
    ensemble_dirs = []

    result = collect_ensemble_hyperparams(ensemble_dirs)

    assert result.empty


def test_collect_ensemble_hyperparams_with_config(temp_results_dir):
    """Test collecting ensemble configurations."""
    ensemble_dir = temp_results_dir / "split_seed0"

    # Create mock ensemble config
    config = {
        "ensemble": {
            "enabled": True,
            "method": "stacking",
            "base_models": ["LR_EN", "RF", "XGBoost"],
            "meta_model": {
                "type": "logistic_regression",
                "penalty": "l2",
                "C": 1.0,
            },
        },
    }

    config_file = ensemble_dir / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    result = collect_ensemble_hyperparams([ensemble_dir])

    assert not result.empty
    assert result["model"].iloc[0] == "ENSEMBLE"
    assert result["method"].iloc[0] == "stacking"
    assert "LR_EN" in result["base_models"].iloc[0]
    assert result["meta_model_type"].iloc[0] == "logistic_regression"
    assert result["meta_model_C"].iloc[0] == 1.0


def test_collect_best_hyperparams_invalid_json(temp_results_dir):
    """Test handling of invalid JSON in best_params column."""
    split_dir = temp_results_dir / "split_seed0"
    params_file = split_dir / "cv" / "best_params_optuna.csv"

    # Write data with invalid JSON
    data = {
        "model": ["LR_EN"],
        "repeat": [0],
        "outer_split": [0],
        "best_score_inner": [0.85],
        "best_params": ["not valid json"],
    }
    df = pd.DataFrame(data)
    df.to_csv(params_file, index=False)

    split_dirs = [split_dir]
    result = collect_best_hyperparams(split_dirs)

    # Should handle gracefully (skip invalid rows)
    assert isinstance(result, pd.DataFrame)


def test_aggregate_hyperparams_summary_n_cv_folds():
    """Test that n_cv_folds is correctly computed."""
    data = {
        "model": ["LR_EN"] * 10,  # 10 CV folds
        "split_seed": [0] * 5 + [1] * 5,
        "repeat": [0, 0, 1, 1, 2] * 2,
        "outer_split": [0, 1, 0, 1, 0] * 2,
        "clf__C": np.random.uniform(5, 20, 10),
    }
    df = pd.DataFrame(data)

    result = aggregate_hyperparams_summary(df)

    assert result["n_cv_folds"].iloc[0] == 10


def test_collect_best_hyperparams_preserves_optuna_metadata(temp_results_dir):
    """Test that Optuna metadata columns are parsed correctly."""
    split_dir = temp_results_dir / "split_seed0"
    params_file = split_dir / "cv" / "best_params_optuna.csv"

    data = {
        "model": ["LR_EN"],
        "repeat": [0],
        "outer_split": [0],
        "best_score_inner": [0.85],
        "best_params": [json.dumps({"clf__C": 10.0})],
        "optuna_n_trials": [100],
        "optuna_sampler": ["tpe"],
        "optuna_pruner": ["hyperband"],
    }
    df = pd.DataFrame(data)
    df.to_csv(params_file, index=False)

    split_dirs = [split_dir]
    result = collect_best_hyperparams(split_dirs)

    assert "best_score_inner" in result.columns
    assert result["best_score_inner"].iloc[0] == 0.85


def test_optuna_trials_aggregation_with_model_prefix(temp_results_dir):
    """Test aggregation of model-prefixed optuna_trials.csv files (bug fix regression test)."""
    from ced_ml.cli.aggregate_splits import run_aggregate_splits

    # Create CeliacRisks directory structure for auto_log_path compatibility
    celiac_root = temp_results_dir / "CeliacRisks"
    celiac_root.mkdir(parents=True)
    results_base = celiac_root / "results" / "run_test" / "LinSVM_cal" / "splits"

    # Create mock optuna_trials files with model prefix (real-world format)
    # Directory structure: CeliacRisks/results/run_test/LinSVM_cal/splits/split_seed*/
    splits_base = results_base
    for seed in [0, 1]:
        split_dir = splits_base / f"split_seed{seed}"
        cv_dir = split_dir / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)

        # Use model-prefixed filename (LinSVM_cal__optuna_trials.csv)
        trials_file = cv_dir / "LinSVM_cal__optuna_trials.csv"

        # Create mock trials data
        trials_data = {
            "number": list(range(10)),
            "value": np.random.uniform(0.7, 0.9, 10),
            "params_clf__estimator__C": np.random.uniform(0.001, 100, 10),
            "params_sel__selector__k": [100] * 10,
            "state": ["COMPLETE"] * 10,
        }
        df = pd.DataFrame(trials_data)
        df.to_csv(trials_file, index=False)

        # Also create required files for aggregation to run
        config_metadata = split_dir / "config_metadata.json"
        with open(config_metadata, "w") as f:
            json.dump({"model": "LinSVM_cal", "split_seed": seed}, f)

        # Create minimal prediction files
        for pred_type in ["test", "val", "train_oof"]:
            pred_dir = (
                split_dir / "preds" / f"{pred_type}_preds"
                if pred_type == "test"
                else split_dir / "preds" / pred_type
            )
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_file = pred_dir / f"LinSVM_cal__{pred_type}.csv"
            pd.DataFrame({"y_true": [0, 1], "y_prob": [0.2, 0.8], "y_pred": [0, 1]}).to_csv(
                pred_file, index=False
            )

    # Run aggregation (use parent directory of splits)
    run_aggregate_splits(
        results_dir=str(results_base.parent),
        save_plots=False,
        n_boot=10,
    )

    # Verify optuna trials were aggregated
    agg_optuna_file = results_base.parent / "aggregated" / "cv" / "optuna_trials.csv"
    assert agg_optuna_file.exists(), "Aggregated optuna_trials.csv not found"

    # Verify combined trials
    agg_df = pd.read_csv(agg_optuna_file)
    assert len(agg_df) == 20, f"Expected 20 trials (10 per split), got {len(agg_df)}"
    assert "number" in agg_df.columns
    assert "value" in agg_df.columns
    assert "params_clf__estimator__C" in agg_df.columns


# REMOVED: Backward compatibility test for cv/optuna/ nested structure
# The new flat structure (cv/optuna_trials.csv) does not support backward compatibility
# with the old nested structure (cv/optuna/optuna_trials.csv)
