"""
Shared pytest fixtures for CeD-ML tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest


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
