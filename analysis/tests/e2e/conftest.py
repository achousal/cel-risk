"""Shared fixtures for E2E tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ced_ml.data.schema import (
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)


@pytest.fixture
def minimal_proteomics_data(tmp_path):
    """
    Create minimal proteomics dataset for E2E testing.

    200 samples: 150 controls, 30 incident, 20 prevalent
    15 protein features + demographics
    Small enough for fast tests but realistic structure.
    """
    rng = np.random.default_rng(42)

    n_controls = 150
    n_incident = 30
    n_prevalent = 20
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 15

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    data = {
        ID_COL: [f"SAMPLE_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": rng.integers(25, 75, n_total),
        "BMI": rng.uniform(18, 35, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic ethnic grouping": rng.choice(["White", "Asian"], n_total),
    }

    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        if i < 5:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.0, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(0.8, 0.3, n_prevalent)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "minimal_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def temporal_proteomics_data(tmp_path):
    """
    Create proteomics dataset with temporal component for temporal validation testing.

    200 samples with sample_date spanning 2020-2023
    Temporal split should use chronological ordering.
    """
    rng = np.random.default_rng(42)

    n_controls = 150
    n_incident = 30
    n_prevalent = 20
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 15

    labels = []
    ctrl_idx, inc_idx, prev_idx = 0, 0, 0
    for i in range(n_total):
        if i % 6 == 1 and inc_idx < n_incident:
            labels.append(INCIDENT_LABEL)
            inc_idx += 1
        elif i % 10 == 9 and prev_idx < n_prevalent:
            labels.append(PREVALENT_LABEL)
            prev_idx += 1
        elif ctrl_idx < n_controls:
            labels.append(CONTROL_LABEL)
            ctrl_idx += 1
        elif inc_idx < n_incident:
            labels.append(INCIDENT_LABEL)
            inc_idx += 1
        else:
            labels.append(PREVALENT_LABEL)
            prev_idx += 1

    base_date = pd.Timestamp("2020-01-01")
    days_span = 1460
    dates = [base_date + pd.Timedelta(days=int(d)) for d in np.linspace(0, days_span, n_total)]

    data = {
        ID_COL: [f"SAMPLE_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "sample_date": dates,
        "age": rng.integers(25, 75, n_total),
        "BMI": rng.uniform(18, 35, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic ethnic grouping": rng.choice(["White", "Asian"], n_total),
    }

    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        if i < 5:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.0, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(0.8, 0.3, n_prevalent)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "temporal_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def small_proteomics_data(tmp_path):
    """
    Create minimal proteomics dataset for fast E2E testing.

    180 samples: 120 controls, 48 incident, 12 prevalent
    10 protein features + demographics
    Fast execution while maintaining realistic structure.
    Balanced demographics ensure proper stratification (2x sex x 3x age = 6 groups, 8 per group).
    """
    rng = np.random.default_rng(42)

    n_controls = 120
    n_incident = 48
    n_prevalent = 12
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 10

    sex_values = []
    age_values = []

    for sex in ["M", "F"]:
        for age_bin in [(30, 35), (45, 55), (65, 70)]:
            for _ in range(8):
                sex_values.append(sex)
                age_values.append(rng.integers(age_bin[0], age_bin[1]))

    for sex in ["M", "F"]:
        for age_bin in [(30, 35), (45, 55), (65, 70)]:
            for _ in range(2):
                sex_values.append(sex)
                age_values.append(rng.integers(age_bin[0], age_bin[1]))

    for sex in ["M", "F"]:
        for age_bin in [(30, 35), (45, 55), (65, 70)]:
            for _ in range(20):
                sex_values.append(sex)
                age_values.append(rng.integers(age_bin[0], age_bin[1]))

    labels = (
        [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
        + [CONTROL_LABEL] * n_controls
    )

    data = {
        ID_COL: [f"SAMPLE_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": age_values,
        "BMI": rng.uniform(18, 35, n_total),
        "sex": sex_values,
        "Genetic_ethnic_grouping": rng.choice(["White", "Asian"], n_total),
    }

    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        if i < 3:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.2, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(0.9, 0.3, n_prevalent)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "small_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def minimal_training_config(tmp_path):
    """
    Create minimal training config for fast E2E tests.

    Reduced CV folds and iterations for speed.
    """
    config = {
        "scenario": "IncidentOnly",
        "cv": {
            "folds": 2,
            "repeats": 1,
            "inner_folds": 2,
            "scoring": "roc_auc",
            "n_jobs": 1,
            "random_state": 42,
        },
        "optuna": {
            "enabled": False,
        },
        "features": {
            "feature_select": "hybrid",
            "kbest_scope": "protein",
            "screen_method": "mannwhitney",
            "screen_top_n": 10,
            "k_grid": [3, 5],
            "stability_thresh": 0.7,
            "corr_thresh": 0.85,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {
            "objective": "youden",
            "fixed_spec": 0.95,
        },
        "lr": {
            "C_min": 0.1,
            "C_max": 10.0,
            "C_points": 2,
            "l1_ratio": [0.5],
            "solver": "saga",
            "max_iter": 500,
        },
        "rf": {
            "n_estimators_grid": [50],
            "max_depth_grid": [3],
            "min_samples_split_grid": [2],
            "min_samples_leaf_grid": [1],
            "max_features_grid": [0.5],
        },
        "xgboost": {
            "n_estimators_grid": [50],
            "max_depth_grid": [3],
            "learning_rate_grid": [0.1],
            "subsample_grid": [0.8],
            "colsample_bytree_grid": [0.8],
        },
        "ensemble": {
            "method": "stacking",
            "base_models": ["LR_EN", "RF"],
            "meta_model": {
                "type": "logistic_regression",
                "penalty": "l2",
                "C": 1.0,
            },
        },
    }

    config_path = tmp_path / "minimal_training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def fast_training_config(tmp_path):
    """
    Create minimal training config optimized for speed.

    2-fold CV, single repeat, no Optuna, minimal features.
    """
    config = {
        "scenario": "IncidentOnly",
        "cv": {
            "folds": 2,
            "repeats": 1,
            "inner_folds": 2,
            "scoring": "roc_auc",
            "n_jobs": 1,
            "random_state": 42,
        },
        "optuna": {"enabled": False},
        "features": {
            "feature_select": "hybrid",
            "kbest_scope": "protein",
            "screen_method": "mannwhitney",
            "screen_top_n": 8,
            "k_grid": [3, 5],
            "stability_thresh": 0.6,
            "corr_thresh": 0.85,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {
            "objective": "youden",
            "fixed_spec": 0.95,
        },
        "lr": {
            "C_min": 0.1,
            "C_max": 10.0,
            "C_points": 2,
            "l1_ratio": [0.5],
            "solver": "saga",
            "max_iter": 500,
        },
        "rf": {
            "n_estimators_grid": [30],
            "max_depth_grid": [3],
            "min_samples_split_grid": [2],
            "min_samples_leaf_grid": [1],
            "max_features_grid": [0.5],
        },
        "xgboost": {
            "n_estimators_grid": [30],
            "max_depth_grid": [3],
            "learning_rate_grid": [0.1],
            "subsample_grid": [0.8],
            "colsample_bytree_grid": [0.8],
        },
        "ensemble": {
            "method": "stacking",
            "base_models": ["LR_EN", "RF"],
            "meta_model": {
                "type": "logistic_regression",
                "penalty": "l2",
                "C": 1.0,
            },
        },
    }

    config_path = tmp_path / "fast_training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def minimal_splits_config(tmp_path):
    """Create minimal splits config."""
    config = {
        "mode": "development",
        "scenarios": ["IncidentOnly"],
        "n_splits": 2,
        "val_size": 0.25,
        "test_size": 0.25,
        "seed_start": 42,
        "train_control_per_case": 5.0,
        "prevalent_train_only": False,
    }

    config_path = tmp_path / "minimal_splits_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def fast_splits_config(tmp_path):
    """Create minimal splits config."""
    config = {
        "mode": "development",
        "scenarios": ["IncidentOnly"],
        "n_splits": 2,
        "val_size": 0.25,
        "test_size": 0.25,
        "seed_start": 42,
        "train_control_per_case": 5.0,
        "prevalent_train_only": False,
    }

    config_path = tmp_path / "fast_splits_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def hpc_config(tmp_path):
    """Create HPC pipeline config for dry-run testing."""
    config = {
        "environment": "hpc",
        "paths": {
            "infile": "../data/test.parquet",
            "splits_dir": "../splits",
            "results_dir": "../results",
        },
        "hpc": {
            "project": "TEST_ALLOCATION",
            "queue": "short",
            "cores": 4,
            "memory": "8G",
            "walltime": "02:00",
        },
        "execution": {
            "models": ["LR_EN", "RF"],
            "n_boot": 100,
        },
    }

    config_path = tmp_path / "hpc_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


def extract_run_id_from_dir(results_dir: Path) -> str:
    """Extract run_id from results directory structure."""
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise ValueError("No run directory found")
    return run_dirs[0].name.replace("run_", "")


def verify_run_metadata(run_dir: Path, expected_model: str, expected_split_seed: int):
    """Verify run_metadata.json has correct structure and content."""
    metadata_path = run_dir / "run_metadata.json"
    assert metadata_path.exists(), f"Missing run_metadata.json in {run_dir}"

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "run_id" in metadata
    assert "infile" in metadata
    assert "split_dir" in metadata
    assert "models" in metadata

    assert expected_model in metadata["models"], f"Model {expected_model} not in metadata"
    model_entry = metadata["models"][expected_model]
    assert "scenario" in model_entry
    assert "infile" in model_entry
    assert "split_dir" in model_entry
    assert "split_seed" in model_entry
    assert "timestamp" in model_entry

    assert model_entry["split_seed"] == expected_split_seed
