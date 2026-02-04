"""
Tests for models.hyperparams module (hyperparameter grids).

Tests cover:
- Parameter distributions for all models
- Grid randomization for sensitivity analysis
- Class weight parsing
- Log-space grid generation
- K-best parameter integration
"""

import numpy as np
import pytest
from ced_ml.models.hyperparams import (
    _make_logspace,
    _parse_class_weight_options,
    _randomize_float_list,
    _randomize_int_list,
    get_param_distributions,
)
from conftest import make_mock_config


@pytest.fixture
def minimal_config():
    """Minimal config for testing."""
    from types import SimpleNamespace

    # Model configs at top level (matching TrainingConfig schema)
    return make_mock_config(
        lr=SimpleNamespace(
            C_min=0.01,
            C_max=100.0,
            C_points=10,
            l1_ratio=[0.1, 0.5, 0.9],
            class_weight_options="None,balanced",
        ),
        svm=SimpleNamespace(C_min=0.01, C_max=100.0, C_points=10, class_weight_options="balanced"),
        rf=SimpleNamespace(
            n_estimators_grid=[100, 200, 500],
            max_depth_grid=[5, 10, 20],
            min_samples_split_grid=[2, 5, 10],
            min_samples_leaf_grid=[1, 2, 4],
            max_features_grid=[0.3, 0.5, 0.7],
            class_weight_options="None,balanced",
        ),
        xgboost=SimpleNamespace(
            n_estimators_grid=[100, 200],
            max_depth_grid=[3, 5, 7],
            learning_rate_grid=[0.01, 0.1, 0.3],
            subsample_grid=[0.7, 0.8, 1.0],
            colsample_bytree_grid=[0.7, 0.8, 1.0],
            min_child_weight_grid=[1, 3, 5],
            gamma_grid=[0.0, 0.1, 0.3],
            reg_alpha_grid=[0.0, 0.01, 0.1],
            reg_lambda_grid=[1.0, 2.0, 5.0],
            scale_pos_weight=None,
            scale_pos_weight_grid=[1.0, 5.0, 10.0],
        ),
    )


# ==================== Parameter Distribution Tests ====================


def test_get_param_distributions_lr(minimal_config):
    """Test LR parameter distributions."""
    params = get_param_distributions("LR_EN", minimal_config)

    assert "clf__C" in params
    assert len(params["clf__C"]) == 10

    assert "clf__l1_ratio" in params
    assert params["clf__l1_ratio"] == [0.1, 0.5, 0.9]

    assert "clf__class_weight" in params
    assert None in params["clf__class_weight"]
    assert "balanced" in params["clf__class_weight"]


def test_get_param_distributions_svm(minimal_config):
    """Test LinSVM parameter distributions."""
    params = get_param_distributions("LinSVM_cal", minimal_config)

    # SVM wrapped in CalibratedClassifierCV uses estimator__ prefix
    assert "clf__estimator__C" in params
    assert len(params["clf__estimator__C"]) == 10

    assert "clf__estimator__class_weight" in params
    assert "balanced" in params["clf__estimator__class_weight"]


def test_get_param_distributions_rf(minimal_config):
    """Test RF parameter distributions."""
    params = get_param_distributions("RF", minimal_config)

    assert "clf__n_estimators" in params
    assert params["clf__n_estimators"] == [100, 200, 500]

    assert "clf__max_depth" in params
    assert params["clf__max_depth"] == [5, 10, 20]

    assert "clf__min_samples_split" in params
    assert "clf__min_samples_leaf" in params
    assert "clf__max_features" in params


def test_get_param_distributions_xgboost(minimal_config):
    """Test XGBoost parameter distributions."""
    params = get_param_distributions("XGBoost", minimal_config)

    assert "clf__n_estimators" in params
    assert "clf__max_depth" in params
    assert "clf__learning_rate" in params
    assert "clf__subsample" in params
    assert "clf__colsample_bytree" in params
    assert "clf__scale_pos_weight" in params


def test_get_param_distributions_xgboost_custom_spw(minimal_config):
    """Test XGBoost with custom scale_pos_weight."""
    params = get_param_distributions("XGBoost", minimal_config, xgb_spw=7.5)

    spw_grid = params["clf__scale_pos_weight"]
    # Should use custom spw +/- 20%
    assert 7.5 in spw_grid
    assert 7.5 * 0.8 in spw_grid
    assert 7.5 * 1.2 in spw_grid


def test_get_param_distributions_with_kbest(minimal_config):
    """Test parameter distributions with hybrid_stability strategy (replaces deprecated kbest)."""
    minimal_config.features.feature_selection_strategy = "hybrid_stability"
    minimal_config.features.k_grid = [10, 25, 50, 100]

    params = get_param_distributions("LR_EN", minimal_config)

    # Always uses 'sel__k' regardless of kbest_scope (pipeline step name)
    assert "sel__k" in params
    assert params["sel__k"] == [10, 25, 50, 100]


def test_get_param_distributions_with_kbest_transformed(minimal_config):
    """Test parameter distributions with hybrid_stability in transformed space."""
    minimal_config.features.feature_selection_strategy = "hybrid_stability"
    minimal_config.features.k_grid = [10, 25, 50]

    params = get_param_distributions("LR_EN", minimal_config)

    assert "sel__k" in params
    assert params["sel__k"] == [10, 25, 50]


def test_get_param_distributions_no_k_grid_raises(minimal_config):
    """Test that hybrid_stability without k_grid raises ValueError."""
    minimal_config.features.feature_selection_strategy = "hybrid_stability"
    minimal_config.features.k_grid = []

    with pytest.raises(ValueError, match="k_grid"):
        get_param_distributions("LR_EN", minimal_config)


def test_get_param_distributions_hybrid_stability_with_k_grid(minimal_config):
    """Test that hybrid_stability strategy includes k_grid in hyperparameters."""
    minimal_config.features.feature_selection_strategy = "hybrid_stability"
    minimal_config.features.k_grid = [25, 50, 100, 150, 200]

    params = get_param_distributions("LR_EN", minimal_config)

    assert "sel__k" in params
    assert params["sel__k"] == [25, 50, 100, 150, 200]


def test_get_param_distributions_hybrid_stability_no_k_grid_raises(minimal_config):
    """Test that hybrid_stability without k_grid raises ValueError."""
    minimal_config.features.feature_selection_strategy = "hybrid_stability"
    minimal_config.features.k_grid = []

    with pytest.raises(ValueError, match="k_grid"):
        get_param_distributions("LR_EN", minimal_config)


def test_get_param_distributions_rfecv_no_k_grid(minimal_config):
    """Test that rfecv strategy does not include k_grid (uses RFECV instead)."""
    minimal_config.features.feature_selection_strategy = "rfecv"
    minimal_config.features.k_grid = []

    params = get_param_distributions("LR_EN", minimal_config)

    # RFECV does not tune k_grid (uses CV-based RFE instead)
    assert "sel__k" not in params


def test_get_param_distributions_none_strategy_no_k_grid(minimal_config):
    """Test that 'none' strategy does not include k_grid."""
    minimal_config.features.feature_selection_strategy = "none"
    minimal_config.features.k_grid = [25, 50]

    params = get_param_distributions("LR_EN", minimal_config)

    # No feature selection, so no sel__k
    assert "sel__k" not in params


def test_get_param_distributions_unknown_model(minimal_config):
    """Test that unknown model returns empty dict."""
    params = get_param_distributions("UnknownModel", minimal_config)
    assert params == {}


# ==================== Grid Randomization Tests ====================


def test_randomize_int_list():
    """Test integer list randomization."""
    rng = np.random.default_rng(42)
    values = [10, 50, 100, 200]

    randomized = _randomize_int_list(values, rng, min_val=1)

    # Should have same length
    assert len(randomized) == len(values)

    # Values should be different (with high probability)
    assert randomized != values

    # All values should be >= min_val
    assert all(v >= 1 for v in randomized)


def test_randomize_int_list_unique():
    """Test integer list randomization with uniqueness."""
    rng = np.random.default_rng(42)
    values = [10, 50, 100]

    randomized = _randomize_int_list(values, rng, min_val=1, unique=True)

    # Should have unique values
    assert len(randomized) == len(set(randomized))


def test_randomize_int_list_with_none():
    """Test integer list randomization preserves None values."""
    rng = np.random.default_rng(42)
    values = [None, 10, 20, 30]

    randomized = _randomize_int_list(values, rng, min_val=1)

    # Should preserve None
    assert None in randomized
    assert len(randomized) == len(values)

    # Non-None values should be randomized
    non_none_values = [v for v in randomized if v is not None]
    assert all(v >= 1 for v in non_none_values)


def test_randomize_int_list_with_none_unique():
    """Test integer list randomization with None and uniqueness."""
    rng = np.random.default_rng(42)
    values = [None, 10, 20, 30]

    randomized = _randomize_int_list(values, rng, min_val=1, unique=True)

    # Should preserve None
    assert None in randomized

    # Non-None values should be unique
    non_none_values = [v for v in randomized if v is not None]
    assert len(non_none_values) == len(set(non_none_values))


def test_randomize_float_list():
    """Test float list randomization."""
    rng = np.random.default_rng(42)
    values = [0.1, 0.5, 1.0]

    randomized = _randomize_float_list(values, rng, min_val=0.0, max_val=1.0)

    # Should have same length
    assert len(randomized) == len(values)

    # Values should be in range
    assert all(0.0 <= v <= 1.0 for v in randomized)


def test_randomize_float_list_log_scale():
    """Test float list randomization in log scale."""
    rng = np.random.default_rng(42)
    values = [0.001, 0.01, 0.1, 1.0]

    randomized = _randomize_float_list(values, rng, min_val=1e-6, log_scale=True)

    # Should have same length
    assert len(randomized) == len(values)

    # All values should be positive
    assert all(v > 0 for v in randomized)


def test_randomize_float_list_with_strings():
    """Test float list randomization with mixed types (strings pass through)."""
    rng = np.random.default_rng(42)
    values = ["sqrt", "log2", 0.5]

    randomized = _randomize_float_list(values, rng, min_val=0.1, max_val=1.0)

    # Should have same length
    assert len(randomized) == len(values)

    # Strings should be preserved unchanged
    assert randomized[0] == "sqrt"
    assert randomized[1] == "log2"

    # Float should be perturbed but within bounds
    assert isinstance(randomized[2], float)
    assert 0.1 <= randomized[2] <= 1.0


def test_get_param_distributions_with_randomization(minimal_config):
    """Test that grid randomization works end-to-end."""
    rng = np.random.default_rng(42)

    params1 = get_param_distributions("RF", minimal_config, grid_rng=rng)
    rng2 = np.random.default_rng(42)  # Same seed
    params2 = get_param_distributions("RF", minimal_config, grid_rng=rng2)

    # Same seed should give same results
    assert params1["clf__n_estimators"] == params2["clf__n_estimators"]

    # Different from original config
    assert params1["clf__n_estimators"] != minimal_config.rf.n_estimators_grid


# ==================== Utility Function Tests ====================


def test_make_logspace():
    """Test log-spaced grid generation."""
    grid = _make_logspace(0.01, 100.0, 5)

    assert len(grid) == 5
    assert grid[0] == pytest.approx(0.01, rel=1e-6)
    assert grid[-1] == pytest.approx(100.0, rel=1e-6)

    # Check log spacing
    log_grid = np.log10(grid)
    diffs = np.diff(log_grid)
    assert np.allclose(diffs, diffs[0])


def test_make_logspace_single_point():
    """Test log-spaced grid with n=1."""
    grid = _make_logspace(0.01, 100.0, 1)

    assert len(grid) == 1
    # Should be geometric mean
    assert grid[0] == pytest.approx(1.0, rel=1e-6)


def test_make_logspace_empty():
    """Test log-spaced grid with n=0."""
    grid = _make_logspace(0.01, 100.0, 0)
    assert grid == []


def test_make_logspace_with_randomization():
    """Test log-spaced grid with perturbation."""
    rng = np.random.default_rng(42)
    grid1 = _make_logspace(0.01, 100.0, 5)
    grid2 = _make_logspace(0.01, 100.0, 5, rng=rng)

    # Should be different
    assert grid1 != grid2

    # Should be in same range
    assert all(0.01 <= v <= 100.0 for v in grid2)


def test_parse_class_weight_options_none():
    """Test parsing 'None' class weight."""
    options = _parse_class_weight_options("None")
    assert options == [None]


def test_parse_class_weight_options_balanced():
    """Test parsing 'balanced' class weight."""
    options = _parse_class_weight_options("balanced")
    assert options == ["balanced"]


def test_parse_class_weight_options_dict():
    """Test parsing dictionary class weights."""
    options = _parse_class_weight_options("{0:1,1:5}")

    assert len(options) == 1
    assert options[0] == {0: 1, 1: 5}


def test_parse_class_weight_options_multiple():
    """Test parsing multiple class weight options."""
    options = _parse_class_weight_options("None,balanced,{0:1,1:10}")

    assert len(options) == 3
    assert None in options
    assert "balanced" in options
    assert {0: 1, 1: 10} in options


def test_parse_class_weight_options_empty():
    """Test parsing empty string."""
    options = _parse_class_weight_options("")
    assert options == [None]


def test_parse_class_weight_options_whitespace():
    """Test parsing with extra whitespace."""
    options = _parse_class_weight_options("  None , balanced  ")
    assert len(options) == 2
    assert None in options
    assert "balanced" in options


# ==================== Integration Tests ====================


def test_all_models_have_params(minimal_config):
    """Test that all standard models return parameters."""
    models = ["LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"]

    for model in models:
        params = get_param_distributions(model, minimal_config)
        # All models should have at least one parameter to tune
        assert len(params) > 0, f"Model {model} has no parameters"


def test_params_are_json_serializable(minimal_config):
    """Test that all parameter values are JSON-serializable."""
    import json

    models = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]

    for model in models:
        params = get_param_distributions(model, minimal_config)

        # Convert to JSON-compatible format
        for key, values in params.items():
            # Lists of numbers/strings/dicts should be serializable
            try:
                json.dumps(values)
            except (TypeError, ValueError):
                pytest.fail(f"Parameter {key} for {model} is not JSON-serializable")


# ==================== Per-Model n_iter Tests ====================
# Priority: 1) global cv.n_iter (if set), 2) per-model n_iter, 3) default (30)


def test_get_model_n_iter_global_override():
    """Test that global cv.n_iter overrides all per-model n_iter values."""
    from types import SimpleNamespace

    from ced_ml.models.training import get_model_n_iter

    config = make_mock_config()
    config.cv.n_iter = 5  # Global override

    # Set model-specific n_iter (should be ignored when global is set)
    config.lr = SimpleNamespace(
        n_iter=25,
        C_min=0.01,
        C_max=100.0,
        C_points=5,
        class_weight_options="balanced",
    )
    config.rf = SimpleNamespace(
        n_iter=50,
        n_estimators_grid=[100],
        max_depth_grid=[5],
        min_samples_split_grid=[2],
        min_samples_leaf_grid=[1],
        max_features_grid=[0.5],
        class_weight_options="balanced",
    )

    # Global cv.n_iter takes precedence over per-model
    assert get_model_n_iter("LR_EN", config) == 5
    assert get_model_n_iter("RF", config) == 5


def test_get_model_n_iter_per_model_when_global_none():
    """Test that per-model n_iter is used when global is None."""
    from types import SimpleNamespace

    from ced_ml.models.training import get_model_n_iter

    config = make_mock_config()
    config.cv.n_iter = None  # No global override

    # Set model-specific n_iter
    config.lr = SimpleNamespace(
        n_iter=25,
        C_min=0.01,
        C_max=100.0,
        C_points=5,
        class_weight_options="balanced",
    )
    config.rf = SimpleNamespace(
        n_iter=50,
        n_estimators_grid=[100],
        max_depth_grid=[5],
        min_samples_split_grid=[2],
        min_samples_leaf_grid=[1],
        max_features_grid=[0.5],
        class_weight_options="balanced",
    )
    config.xgboost = SimpleNamespace(
        n_iter=100,
        n_estimators_grid=[100],
        max_depth_grid=[3],
        learning_rate_grid=[0.1],
        subsample_grid=[0.8],
        colsample_bytree_grid=[0.8],
        scale_pos_weight=None,
        scale_pos_weight_grid=[1.0],
    )
    config.svm = SimpleNamespace(
        n_iter=15,
        C_min=0.01,
        C_max=100.0,
        C_points=5,
        class_weight_options="balanced",
    )

    # Per-model values used when global is None
    assert get_model_n_iter("LR_EN", config) == 25
    assert get_model_n_iter("LR_L1", config) == 25
    assert get_model_n_iter("LinSVM_cal", config) == 15
    assert get_model_n_iter("RF", config) == 50
    assert get_model_n_iter("XGBoost", config) == 100


def test_get_model_n_iter_default_fallback():
    """Test fallback to default (30) when both global and per-model are None."""
    from types import SimpleNamespace

    from ced_ml.models.training import _DEFAULT_N_ITER, get_model_n_iter

    config = make_mock_config()
    config.cv.n_iter = None  # No global override

    # LR has no n_iter attribute
    config.lr = SimpleNamespace(
        C_min=0.01,
        C_max=100.0,
        C_points=5,
        class_weight_options="balanced",
    )

    # Falls back to default
    assert get_model_n_iter("LR_EN", config) == _DEFAULT_N_ITER
    assert get_model_n_iter("UnknownModel", config) == _DEFAULT_N_ITER


def test_get_model_n_iter_partial_per_model():
    """Test mixed scenario: some models have n_iter, others fall back to default."""
    from types import SimpleNamespace

    from ced_ml.models.training import _DEFAULT_N_ITER, get_model_n_iter

    config = make_mock_config()
    config.cv.n_iter = None  # No global override

    # Only RF has n_iter set
    config.rf = SimpleNamespace(
        n_iter=75,
        n_estimators_grid=[100],
        max_depth_grid=[5],
        min_samples_split_grid=[2],
        min_samples_leaf_grid=[1],
        max_features_grid=[0.5],
        class_weight_options="balanced",
    )
    # LR has no n_iter attribute
    config.lr = SimpleNamespace(
        C_min=0.01,
        C_max=100.0,
        C_points=5,
        class_weight_options="balanced",
    )

    # RF uses its per-model setting
    assert get_model_n_iter("RF", config) == 75

    # LR falls back to default
    assert get_model_n_iter("LR_EN", config) == _DEFAULT_N_ITER


# ==================== Optuna Parameter Distribution Tests ====================


def test_get_optuna_params_xgboost_default_ranges():
    """Test XGBoost Optuna params use wider default ranges."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    params = get_param_distributions_optuna("XGBoost", config)

    # Check learning_rate uses log scale
    assert params["clf__learning_rate"]["type"] == "float"
    assert params["clf__learning_rate"]["log"] is True
    assert params["clf__learning_rate"]["low"] == 0.001
    assert params["clf__learning_rate"]["high"] == 0.3

    # Check regularization params use log scale
    assert params["clf__reg_alpha"]["log"] is True
    assert params["clf__reg_lambda"]["log"] is True

    # Check min_child_weight uses log scale
    assert params["clf__min_child_weight"]["log"] is True

    # Check subsample does NOT use log scale
    assert params["clf__subsample"]["log"] is False


def test_get_optuna_params_xgboost_custom_ranges():
    """Test XGBoost Optuna params can be customized via config."""
    from types import SimpleNamespace

    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    # Override with custom Optuna ranges
    config.xgboost = SimpleNamespace(
        n_estimators_grid=[100, 200],
        max_depth_grid=[3, 5],
        learning_rate_grid=[0.01, 0.1],
        subsample_grid=[0.8, 1.0],
        colsample_bytree_grid=[0.8, 1.0],
        scale_pos_weight_grid=[1.0, 5.0],
        min_child_weight_grid=[1, 3, 5],
        gamma_grid=[0.0, 0.1, 0.3],
        reg_alpha_grid=[0.0, 0.01, 0.1],
        reg_lambda_grid=[1.0, 2.0, 5.0],
        # Custom Optuna ranges
        optuna_n_estimators=(100, 1000),  # Wider than default
        optuna_max_depth=(3, 15),
        optuna_learning_rate=(0.0001, 0.5),  # Custom range
        optuna_min_child_weight=None,  # Use default
        optuna_gamma=None,
        optuna_subsample=(0.6, 0.95),
        optuna_colsample_bytree=None,
        optuna_reg_alpha=(1e-10, 10.0),  # Custom range
        optuna_reg_lambda=None,
    )

    params = get_param_distributions_optuna("XGBoost", config)

    # Check custom ranges are used
    assert params["clf__n_estimators"]["low"] == 100
    assert params["clf__n_estimators"]["high"] == 1000
    assert params["clf__max_depth"]["low"] == 3
    assert params["clf__max_depth"]["high"] == 15
    assert params["clf__learning_rate"]["low"] == 0.0001
    assert params["clf__learning_rate"]["high"] == 0.5
    assert params["clf__subsample"]["low"] == 0.6
    assert params["clf__subsample"]["high"] == 0.95
    assert params["clf__reg_alpha"]["low"] == 1e-10
    assert params["clf__reg_alpha"]["high"] == 10.0


def test_get_optuna_params_rf_default_ranges():
    """Test RF Optuna params use appropriate ranges."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    params = get_param_distributions_optuna("RF", config)

    # Check n_estimators is int
    assert params["clf__n_estimators"]["type"] == "int"
    assert params["clf__n_estimators"]["log"] is False

    # Check max_depth is int
    assert params["clf__max_depth"]["type"] == "int"

    # Check min_samples params
    assert params["clf__min_samples_split"]["type"] == "int"
    assert params["clf__min_samples_leaf"]["type"] == "int"


def test_get_optuna_params_rf_custom_ranges():
    """Test RF Optuna params can be customized via config."""
    from types import SimpleNamespace

    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    config.rf = SimpleNamespace(
        n_estimators_grid=[100, 200],
        max_depth_grid=[5, 10],
        min_samples_split_grid=[2, 5],
        min_samples_leaf_grid=[1, 2],
        max_features_grid=[0.3, 0.5],
        class_weight_options="balanced",
        # Custom Optuna ranges
        optuna_n_estimators=(50, 1000),
        optuna_max_depth=(5, 30),
        optuna_min_samples_split=(2, 50),
        optuna_min_samples_leaf=(1, 20),
        optuna_max_features=(0.05, 0.9),
    )

    params = get_param_distributions_optuna("RF", config)

    assert params["clf__n_estimators"]["low"] == 50
    assert params["clf__n_estimators"]["high"] == 1000
    assert params["clf__max_depth"]["low"] == 5
    assert params["clf__max_depth"]["high"] == 30
    assert params["clf__min_samples_split"]["low"] == 2
    assert params["clf__min_samples_split"]["high"] == 50
    assert params["clf__max_features"]["low"] == 0.05
    assert params["clf__max_features"]["high"] == 0.9


def test_get_optuna_params_lr_log_scale():
    """Test LR Optuna params use log scale for C."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    params = get_param_distributions_optuna("LR_EN", config)

    # C should use log scale
    assert params["clf__C"]["type"] == "float"
    assert params["clf__C"]["log"] is True

    # l1_ratio should NOT use log scale (it's a proportion)
    assert params["clf__l1_ratio"]["type"] == "float"
    assert params["clf__l1_ratio"]["log"] is False


def test_get_optuna_params_lr_custom_c_range():
    """Test LR Optuna C range can be customized."""
    from types import SimpleNamespace

    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    config.lr = SimpleNamespace(
        C_min=0.01,
        C_max=100.0,
        C_points=5,
        l1_ratio=[0.1, 0.5, 0.9],
        class_weight_options="balanced",
        # Custom Optuna ranges
        optuna_C=(1e-6, 1000.0),
        optuna_l1_ratio=(0.0, 1.0),
    )

    params = get_param_distributions_optuna("LR_EN", config)

    assert params["clf__C"]["low"] == 1e-6
    assert params["clf__C"]["high"] == 1000.0
    assert params["clf__l1_ratio"]["low"] == 0.0
    assert params["clf__l1_ratio"]["high"] == 1.0


def test_get_optuna_params_svm_log_scale():
    """Test SVM Optuna params use log scale for C."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    params = get_param_distributions_optuna("LinSVM_cal", config)

    # C should use log scale
    assert params["clf__estimator__C"]["type"] == "float"
    assert params["clf__estimator__C"]["log"] is True


def test_get_optuna_params_with_kbest():
    """Test Optuna params include k_grid with hybrid_stability (replaces deprecated kbest)."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    config.features.feature_selection_strategy = "hybrid_stability"
    config.features.k_grid = [25, 50, 100, 200]

    params = get_param_distributions_optuna("LR_EN", config)

    assert "sel__k" in params
    assert params["sel__k"]["type"] == "categorical"
    assert params["sel__k"]["choices"] == [25, 50, 100, 200]


def test_get_optuna_params_hybrid_stability_with_k_grid():
    """Test Optuna params include k_grid with hybrid_stability strategy."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    config.features.feature_selection_strategy = "hybrid_stability"
    config.features.k_grid = [25, 50, 100, 150, 200, 300, 400]

    params = get_param_distributions_optuna("LR_EN", config)

    assert "sel__k" in params
    assert params["sel__k"]["type"] == "categorical"
    assert params["sel__k"]["choices"] == [25, 50, 100, 150, 200, 300, 400]


def test_get_optuna_params_rfecv_no_k_grid():
    """Test Optuna params do not include k_grid with rfecv strategy."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    config.features.feature_selection_strategy = "rfecv"
    config.features.k_grid = []

    params = get_param_distributions_optuna("LR_EN", config)

    # RFECV does not tune k_grid
    assert "sel__k" not in params


def test_get_optuna_params_none_strategy_no_k_grid():
    """Test Optuna params do not include k_grid with 'none' strategy."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    config.features.feature_selection_strategy = "none"
    config.features.k_grid = [25, 50]

    params = get_param_distributions_optuna("LR_EN", config)

    # No feature selection
    assert "sel__k" not in params


def test_get_optuna_params_xgboost_with_spw():
    """Test XGBoost Optuna params with custom scale_pos_weight."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    params = get_param_distributions_optuna("XGBoost", config, xgb_spw=10.0)

    # Should use spw +/- 30% as range
    assert params["clf__scale_pos_weight"]["type"] == "float"
    assert params["clf__scale_pos_weight"]["low"] == pytest.approx(7.0, rel=0.01)
    assert params["clf__scale_pos_weight"]["high"] == pytest.approx(13.0, rel=0.01)


def test_get_optuna_params_all_models_have_specs():
    """Test that all standard models return Optuna specs."""
    from ced_ml.models.hyperparams import get_param_distributions_optuna

    config = make_mock_config()
    models = ["LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"]

    for model in models:
        params = get_param_distributions_optuna(model, config)
        assert len(params) > 0, f"Model {model} has no Optuna parameters"

        # Check all specs have required fields
        for name, spec in params.items():
            assert "type" in spec, f"Param {name} missing 'type'"
            if spec["type"] in ("int", "float"):
                assert "low" in spec, f"Param {name} missing 'low'"
                assert "high" in spec, f"Param {name} missing 'high'"
                assert "log" in spec, f"Param {name} missing 'log'"
            elif spec["type"] == "categorical":
                assert "choices" in spec, f"Param {name} missing 'choices'"


def test_optuna_default_ranges_are_wider():
    """Test that Optuna default ranges are wider than grid-derived ranges."""
    from ced_ml.models.hyperparams import (
        DEFAULT_OPTUNA_RANGES,
        get_param_distributions,
        get_param_distributions_optuna,
    )

    config = make_mock_config()

    # XGBoost learning_rate
    sklearn_params = get_param_distributions("XGBoost", config)
    optuna_params = get_param_distributions_optuna("XGBoost", config)

    # Optuna default should be wider
    sklearn_lr_range = max(sklearn_params["clf__learning_rate"]) - min(
        sklearn_params["clf__learning_rate"]
    )
    optuna_lr_range = (
        optuna_params["clf__learning_rate"]["high"] - optuna_params["clf__learning_rate"]["low"]
    )
    assert optuna_lr_range >= sklearn_lr_range

    # Check the default constants exist and have expected values
    assert "XGBoost" in DEFAULT_OPTUNA_RANGES
    assert DEFAULT_OPTUNA_RANGES["XGBoost"]["learning_rate"]["log"] is True
    assert DEFAULT_OPTUNA_RANGES["XGBoost"]["reg_alpha"]["log"] is True
