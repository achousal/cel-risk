"""
Shared pytest fixtures for CeD-ML tests.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ced_ml.data.schema import (
    CED_DATE_COL,
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)
from sklearn.datasets import make_classification


@pytest.fixture
def rng():
    """
    Provide a numpy random Generator for deterministic tests.

    Use this fixture instead of np.random.seed(42) to avoid global state pollution.
    The Generator API provides better statistical properties and isolation.

    Usage:
        def test_something(rng):
            data = rng.standard_normal(100)
            choice = rng.choice([0, 1], size=10)
    """
    return np.random.default_rng(42)


@pytest.fixture
def rng_factory():
    """
    Factory fixture to create RNGs with custom seeds.

    Usage:
        def test_multiple_rngs(rng_factory):
            rng1 = rng_factory(123)
            rng2 = rng_factory(456)
    """

    def _make_rng(seed: int = 42):
        return np.random.default_rng(seed)

    return _make_rng


def make_rng(seed: int = 42) -> np.random.Generator:
    """
    Create a numpy random Generator for deterministic tests.

    Use this instead of np.random.seed(seed) to avoid polluting global state.
    Import this function directly for use in test functions that cannot
    easily use fixtures (e.g., pytest parametrize, module-level helpers).

    Args:
        seed: Random seed for reproducibility (default: 42)

    Returns:
        numpy Generator object

    Usage:
        from conftest import make_rng

        def test_something():
            rng = make_rng(42)
            data = rng.standard_normal(100)
    """
    return np.random.default_rng(seed)


class MockCalibrationConfig:
    """Mock calibration config for tests."""

    def __init__(
        self,
        enabled=True,
        method="isotonic",
        strategy="oof_posthoc",
        per_model=None,
        cv=3,
    ):
        self.enabled = enabled
        self.method = method
        self.strategy = strategy
        self.per_model = per_model or {}
        self.cv = cv

    def get_strategy_for_model(self, model_name: str) -> str:
        """Get the effective calibration strategy for a specific model."""
        if not self.enabled:
            return "none"
        if self.per_model and model_name in self.per_model:
            return self.per_model[model_name]
        return self.strategy


def make_mock_config(**overrides):
    """
    Create a mock config object for testing.

    This bypasses Pydantic validation and creates a simple namespace
    with reasonable defaults. Use this for unit tests of isolated modules.

    Args:
        **overrides: Override default values

    Returns:
        SimpleNamespace with config attributes
    """
    defaults = {
        "cv": SimpleNamespace(
            folds=3,
            repeats=2,
            inner_folds=2,
            n_iter=5,
            scoring="neg_brier_score",
            scoring_target_fpr=0.05,
            random_state=0,
            tune_n_jobs="auto",
            error_score="nan",
            grid_randomize=False,
        ),
        "features": SimpleNamespace(
            feature_select="none",
            feature_selection_strategy="none",
            k_grid=[],
            kbest_scope="protein",
            coef_threshold=1e-12,
            rf_use_permutation=False,
            rf_perm_top_n=50,
            rf_perm_repeats=3,
            rf_perm_min_importance=0.0,
        ),
        "compute": SimpleNamespace(cpus=2, tune_n_jobs=None),
        # Model configs at top level (matching TrainingConfig schema)
        "lr": SimpleNamespace(
            C_min=0.01,
            C_max=100.0,
            C_points=5,
            l1_ratio=[0.1, 0.5, 0.9],
            solver="saga",
            max_iter=1000,
            class_weight_options="None,balanced",
            # Optuna-specific ranges (None = derive from grid)
            optuna_C=None,
            optuna_l1_ratio=None,
        ),
        "svm": SimpleNamespace(
            C_min=0.01,
            C_max=100.0,
            C_points=5,
            class_weight_options="balanced",
            # Optuna-specific ranges
            optuna_C=None,
        ),
        "rf": SimpleNamespace(
            n_estimators_grid=[100, 200],
            max_depth_grid=[5, 10],
            min_samples_split_grid=[2, 5],
            min_samples_leaf_grid=[1, 2],
            max_features_grid=[0.3, 0.5],
            class_weight_options="None,balanced",
            # Optuna-specific ranges
            optuna_n_estimators=None,
            optuna_max_depth=None,
            optuna_min_samples_split=None,
            optuna_min_samples_leaf=None,
            optuna_max_features=None,
        ),
        "xgboost": SimpleNamespace(
            n_estimators_grid=[100, 200],
            max_depth_grid=[3, 5],
            learning_rate_grid=[0.01, 0.1],
            subsample_grid=[0.8, 1.0],
            colsample_bytree_grid=[0.8, 1.0],
            scale_pos_weight=None,
            scale_pos_weight_grid=[1.0, 5.0],
            min_child_weight_grid=[1, 3, 5],
            gamma_grid=[0.0, 0.1, 0.3],
            reg_alpha_grid=[0.0, 0.01, 0.1],
            reg_lambda_grid=[1.0, 2.0, 5.0],
            # Optuna-specific ranges
            optuna_n_estimators=None,
            optuna_max_depth=None,
            optuna_learning_rate=None,
            optuna_min_child_weight=None,
            optuna_gamma=None,
            optuna_subsample=None,
            optuna_colsample_bytree=None,
            optuna_reg_alpha=None,
            optuna_reg_lambda=None,
        ),
        "optuna": SimpleNamespace(enabled=False),
        "calibration": MockCalibrationConfig(
            enabled=True,
            method="isotonic",
            strategy="oof_posthoc",
            per_model={},
        ),
        "thresholds": SimpleNamespace(
            objective="youden",
            fixed_spec=0.95,
            fbeta=1.0,
            fixed_ppv=0.5,
        ),
    }

    # Deep merge overrides
    config_dict = defaults.copy()
    for key, value in overrides.items():
        if key in config_dict and isinstance(value, dict):
            # Merge nested dict
            for k2, v2 in value.items():
                setattr(config_dict[key], k2, v2)
        else:
            config_dict[key] = value

    return SimpleNamespace(**config_dict)


@pytest.fixture(autouse=True)
def set_results_env(tmp_path, monkeypatch):
    """
    Point CED_RESULTS_DIR to tmp_path/results for E2E tests.

    Automatically applied to all tests. Ensures E2E tests write to isolated
    temporary directories and do not interfere with actual results directory.

    Used by:
    - test_e2e_temporal_workflows.py
    - test_e2e_fixed_panel_workflows.py
    - test_e2e_multi_model_workflows.py
    - test_e2e_run_id_workflows.py
    - test_e2e_output_structure.py
    - test_e2e_calibration_workflows.py
    """
    monkeypatch.setenv("CED_RESULTS_DIR", str(tmp_path / "results"))


@pytest.fixture
def toy_data():
    """
    Generate toy classification data for quick training tests.

    Creates a simple binary classification dataset using sklearn's
    make_classification with reproducible random state.

    Returns:
        tuple: (X, y) where X is feature array and y is label array

    Used by:
    - test_models_stacking.py
    - test_models_optuna_multiobjective.py (imports from make_classification directly)
    """
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    return X, y


@pytest.fixture
def toy_data_screening():
    """
    Generate toy data optimized for feature screening tests.

    Small dataset with clear signal in first 5 proteins for screening tests.
    Returns DataFrame with protein column naming convention.

    Returns:
        tuple: (X, y, protein_cols) where:
            - X: DataFrame with n_proteins columns (prot_{i}_resid naming)
            - y: binary labels array
            - protein_cols: list of protein column names

    Used by:
    - features/test_screening_fold_count.py
    """
    rng = np.random.default_rng(42)
    n = 200
    n_proteins = 30
    y = np.array([0] * 170 + [1] * 30)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, n_proteins)),
        columns=[f"prot_{i}_resid" for i in range(n_proteins)],
    )
    # Inject signal in first 5 proteins
    X.iloc[170:, :5] += 2.0
    protein_cols = list(X.columns)
    return X, y, protein_cols


@pytest.fixture
def toy_data_training():
    """
    Generate toy data for training module tests with demographics.

    Small dataset with protein features + demographics (age, sex).
    Imbalanced labels (10% positive) to simulate CeD prevalence.

    Returns:
        tuple: (X, y) where:
            - X: DataFrame with protein columns + age + sex
            - y: binary labels array (imbalanced 10% positive)

    Used by:
    - test_training.py
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    n_proteins = 20

    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_proteins)),
        columns=[f"prot_{i}_resid" for i in range(n_proteins)],
    )
    # Add demographics
    X["age"] = rng.uniform(20, 80, n_samples)
    X["sex"] = rng.choice(["M", "F"], n_samples)

    # Imbalanced labels (10% positive)
    y = rng.binomial(1, 0.1, n_samples)

    return X, y


@pytest.fixture
def sample_data_filters():
    """
    Create sample data with various filtering scenarios.

    Dataset designed to test data filtering logic including:
    - Normal controls
    - Uncertain controls (controls with CeD_date)
    - Missing metadata (age, BMI)
    - Incident cases
    - Prevalent cases

    Returns:
        DataFrame with ID, Target, CeD_date, age, BMI columns

    Used by:
    - test_data_filters.py
    """
    return pd.DataFrame(
        {
            ID_COL: range(10),
            TARGET_COL: [
                CONTROL_LABEL,  # 0: Normal control
                CONTROL_LABEL,  # 1: Uncertain control (has CeD_date)
                CONTROL_LABEL,  # 2: Control with missing age
                CONTROL_LABEL,  # 3: Control with missing BMI
                INCIDENT_LABEL,  # 4: Normal incident
                INCIDENT_LABEL,  # 5: Incident with missing age
                PREVALENT_LABEL,  # 6: Normal prevalent
                PREVALENT_LABEL,  # 7: Prevalent with CeD_date (expected)
                CONTROL_LABEL,  # 8: Normal control
                CONTROL_LABEL,  # 9: Uncertain control with missing metadata
            ],
            CED_DATE_COL: [
                None,  # 0: No date (normal control)
                "2020-01-01",  # 1: Uncertain control
                None,  # 2
                None,  # 3
                "2021-05-15",  # 4: Normal incident
                "2021-08-20",  # 5
                "2019-03-10",  # 6: Normal prevalent
                "2018-11-22",  # 7: Normal prevalent
                None,  # 8: Normal control
                "2022-02-14",  # 9: Uncertain control
            ],
            "age": [45, 50, None, 55, 60, None, 65, 70, 75, None],
            "BMI": [22.5, 25.0, 27.5, None, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5],
        }
    )


@pytest.fixture
def sample_data_screening():
    """
    Generate sample data for screening cache tests.

    Simple dataset with generic protein naming (protein_{i} not _resid suffix).

    Returns:
        tuple: (X, y, protein_cols) where:
            - X: DataFrame with protein columns
            - y: binary labels (80/20 split)
            - protein_cols: list of protein column names

    Used by:
    - features/test_screening_cache.py
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    n_proteins = 50

    # Create DataFrame with protein columns
    X = pd.DataFrame(
        rng.normal(0, 1, (n_samples, n_proteins)),
        columns=[f"protein_{i}" for i in range(n_proteins)],
    )

    # Create binary labels
    y = rng.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    protein_cols = list(X.columns)

    return X, y, protein_cols


@pytest.fixture
def sample_data_nested_rfe():
    """
    Generate sample classification data for nested RFE testing.

    Uses sklearn make_classification with meaningful class separation.

    Returns:
        tuple: (X_df, y, feature_names) where:
            - X_df: DataFrame with protein_{i} columns
            - y: binary labels array
            - feature_names: list of feature names

    Used by:
    - features/test_nested_rfe.py
    """
    X, y = make_classification(
        n_samples=200,
        n_features=50,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        class_sep=1.0,
    )
    feature_names = [f"protein_{i}" for i in range(50)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


@pytest.fixture
def temporal_proteomics_data_e2e(tmp_path):
    """
    Create proteomics dataset with temporal dimension for E2E temporal testing.

    150 samples: 130 controls, 12 incident, 8 prevalent.
    Samples span 4 time points (years) with random date distribution.
    Ensures sufficient samples in each temporal split for validation.

    Returns:
        DataFrame with sample_date, Target, and protein columns

    Used by:
    - test_e2e_temporal_workflows.py

    Notes:
        Uses random date offsets (not sequential) to avoid all cases
        ending up in the test set during temporal splitting.
    """
    rng = np.random.default_rng(42)

    n_controls = 130
    n_incident = 12
    n_prevalent = 8
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 10

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    # Create temporal dimension: distribute cases across time
    # This ensures temporal splits have cases in train/val/test
    base_date = pd.Timestamp("2020-01-01")

    # Generate random dates uniformly across ~4 years
    # CRITICAL: Use random dates (not sequential by index) to avoid all cases
    # ending up in test set
    days_offsets = rng.integers(0, 1460, n_total)  # Random day within 4 years
    sample_dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offsets]

    # Create proteomics data
    protein_data = pd.DataFrame(
        rng.normal(0, 1, (n_total, n_proteins)),
        columns=[f"prot_{i}_resid" for i in range(n_proteins)],
    )

    df = pd.DataFrame(
        {
            "sample_date": sample_dates,
            TARGET_COL: labels,
        }
    )

    # Merge proteomics
    df = pd.concat([df, protein_data], axis=1)

    return df


@pytest.fixture
def temporal_proteomics_data_runner(tmp_path):
    """
    Create proteomics dataset with temporal component for runner testing.

    200 samples with sample_date spanning 2020-2023.
    Interleaved labels so incident/prevalent cases are distributed across timeline.
    Temporal split uses chronological ordering.

    Returns:
        DataFrame with sample_date, Target, and protein columns

    Used by:
    - test_e2e_runner.py

    Notes:
        Larger dataset (200 samples) with explicit label interleaving pattern
        to ensure cases appear in all temporal windows.
    """
    rng = np.random.default_rng(42)

    n_controls = 150
    n_incident = 30
    n_prevalent = 20
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 15

    # Interleave labels so incident/prevalent cases are distributed across timeline
    # (temporal splits need cases in all time windows)
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

    # Generate dates spanning 2020-2023
    base_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2023-12-31")
    date_range_days = (end_date - base_date).days

    # Random dates across range
    days_offsets = rng.integers(0, date_range_days, n_total)
    sample_dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offsets]

    # Create proteomics data
    protein_data = pd.DataFrame(
        rng.normal(0, 1, (n_total, n_proteins)),
        columns=[f"prot_{i}_resid" for i in range(n_proteins)],
    )

    df = pd.DataFrame(
        {
            "sample_date": sample_dates,
            TARGET_COL: labels,
        }
    )

    # Merge proteomics
    df = pd.concat([df, protein_data], axis=1)

    return df
