"""Tests for SHAP explainability module.

Tests cover:
- Config validation (C3)
- Normalization helpers (_normalize_expected_value, _normalize_shap_values)
- _positive_class_index
- _infer_output_scale
- _sample_background
- aggregate_fold_shap (C9)
- select_waterfall_samples (C8)
- Pipeline unwrapping (_unwrap_calibrated_for_shap)
- Additivity tests (REQUIRE shap)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ced_ml.config.shap_schema import SHAPConfig
from ced_ml.data.schema import ModelName
from ced_ml.features.shap_values import (
    SHAPFoldResult,
    SHAPTestPayload,
    _infer_output_scale,
    _normalize_expected_value,
    _normalize_shap_values,
    _positive_class_index,
    _resolve_tree_model_output,
    _sample_background,
    _unwrap_calibrated_for_shap,
    aggregate_fold_shap,
    get_model_matrix_and_feature_names,
    select_waterfall_samples,
)
from ced_ml.models.calibration import OOFCalibratedModel

try:
    import shap
except ImportError:
    shap = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(200),
            "feat_b": np.random.randn(200),
            "feat_c": np.random.randn(200),
        }
    )
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values
    return X, y


@pytest.fixture
def shap_config_default():
    """Default SHAP config."""
    return SHAPConfig(
        enabled=True,
        max_background_samples=50,
        background_strategy="random_train",
        tree_feature_perturbation="tree_path_dependent",
        tree_model_output="raw",
    )


@pytest.fixture
def fitted_lr_model(simple_classification_data):
    """Fitted LogisticRegression model."""
    X, y = simple_classification_data
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_rf_model(simple_classification_data):
    """Fitted RandomForestClassifier model."""
    X, y = simple_classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_linear_svc_cal(simple_classification_data):
    """Fitted CalibratedClassifierCV wrapping LinearSVC."""
    X, y = simple_classification_data
    base = LinearSVC(random_state=42, max_iter=200, dual=False)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_pipeline_lr(simple_classification_data):
    """Fitted Pipeline with preprocessing + LR."""
    X, y = simple_classification_data
    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer([("num", StandardScaler(), ["feat_a", "feat_b", "feat_c"])]),
            ),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def fitted_xgb_model(simple_classification_data):
    """Fitted XGBoost model (skip if xgboost not installed)."""
    if xgb is None:
        pytest.skip("xgboost not installed")
    X, y = simple_classification_data
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


# =============================================================================
# Config validation (C3)
# =============================================================================


def test_config_rejects_tree_path_dependent_probability():
    """SHAPConfig rejects tree_path_dependent + probability (invalid combo)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="tree_model_output='probability'"):
        SHAPConfig(
            tree_feature_perturbation="tree_path_dependent",
            tree_model_output="probability",
        )


def test_config_rejects_invalid_model_output():
    """SHAPConfig rejects model_output not in Literal."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SHAPConfig(tree_model_output="log_loss")  # type: ignore[arg-type]


def test_config_valid_combos():
    """SHAPConfig accepts valid combinations."""
    SHAPConfig(
        tree_feature_perturbation="tree_path_dependent",
        tree_model_output="auto",
    )
    SHAPConfig(
        tree_feature_perturbation="tree_path_dependent",
        tree_model_output="raw",
    )
    SHAPConfig(
        tree_feature_perturbation="interventional",
        tree_model_output="probability",
    )
    SHAPConfig(
        tree_feature_perturbation="interventional",
        tree_model_output="raw",
    )


def test_resolve_tree_model_output_auto_defaults():
    """Auto model_output resolves to RF=probability, XGBoost=raw."""
    config = SHAPConfig(tree_model_output="auto")
    assert _resolve_tree_model_output(ModelName.RF, config) == "probability"
    assert _resolve_tree_model_output(ModelName.XGBoost, config) == "raw"


def test_resolve_tree_model_output_explicit_probability():
    """Explicit probability is preserved for both tree model families."""
    config = SHAPConfig(
        tree_feature_perturbation="interventional",
        tree_model_output="probability",
    )
    assert _resolve_tree_model_output(ModelName.RF, config) == "probability"
    assert _resolve_tree_model_output(ModelName.XGBoost, config) == "probability"


# =============================================================================
# Normalization helpers
# =============================================================================


def test_normalize_expected_value_scalar():
    """_normalize_expected_value passes through scalar."""
    assert _normalize_expected_value(1.23) == 1.23


def test_normalize_expected_value_length2_classes_01():
    """_normalize_expected_value extracts positive class from length-2 array (classes=[0,1])."""
    ev = np.array([0.5, 1.5])
    classes = np.array([0, 1])
    result = _normalize_expected_value(ev, classes, positive_label=1)
    assert result == 1.5


def test_normalize_expected_value_length2_classes_10():
    """_normalize_expected_value extracts positive class from length-2 array (classes=[1,0])."""
    ev = np.array([1.5, 0.5])
    classes = np.array([1, 0])
    result = _normalize_expected_value(ev, classes, positive_label=1)
    assert result == 1.5


def test_normalize_expected_value_list_wrap_single():
    """_normalize_expected_value unwraps single-element list."""
    ev = [1.23]
    result = _normalize_expected_value(ev)
    assert result == 1.23


def test_normalize_expected_value_list_wrap_binary():
    """_normalize_expected_value unwraps binary list."""
    ev = [0.5, 1.5]
    classes = np.array([0, 1])
    result = _normalize_expected_value(ev, classes, positive_label=1)
    assert result == 1.5


def test_normalize_shap_values_2d_passthrough():
    """_normalize_shap_values passes through 2D array."""
    values = np.random.randn(10, 5)
    result = _normalize_shap_values(values)
    assert result.shape == (10, 5)
    np.testing.assert_array_equal(result, values)


def test_normalize_shap_values_3d_binary_classes_01():
    """_normalize_shap_values extracts positive class from 3D (classes=[0,1])."""
    values = np.random.randn(10, 5, 2)
    classes = np.array([0, 1])
    result = _normalize_shap_values(values, classes, positive_label=1)
    assert result.shape == (10, 5)
    np.testing.assert_array_equal(result, values[:, :, 1])


def test_normalize_shap_values_3d_binary_classes_10():
    """_normalize_shap_values extracts positive class from 3D (classes=[1,0])."""
    values = np.random.randn(10, 5, 2)
    classes = np.array([1, 0])
    result = _normalize_shap_values(values, classes, positive_label=1)
    assert result.shape == (10, 5)
    np.testing.assert_array_equal(result, values[:, :, 0])


def test_normalize_shap_values_list_wrap_binary():
    """_normalize_shap_values unwraps binary list."""
    val_neg = np.random.randn(10, 5)
    val_pos = np.random.randn(10, 5)
    values = [val_neg, val_pos]
    classes = np.array([0, 1])
    result = _normalize_shap_values(values, classes, positive_label=1)
    assert result.shape == (10, 5)
    np.testing.assert_array_equal(result, val_pos)


def test_normalize_shap_values_list_wrap_single():
    """_normalize_shap_values unwraps single-element list."""
    val = np.random.randn(10, 5)
    values = [val]
    result = _normalize_shap_values(values)
    assert result.shape == (10, 5)
    np.testing.assert_array_equal(result, val)


# =============================================================================
# _positive_class_index
# =============================================================================


def test_positive_class_index_classes_01_positive1():
    """_positive_class_index returns index 1 for classes=[0,1] positive_label=1."""
    classes = np.array([0, 1])
    assert _positive_class_index(classes, positive_label=1) == 1


def test_positive_class_index_classes_10_positive1():
    """_positive_class_index returns index 0 for classes=[1,0] positive_label=1."""
    classes = np.array([1, 0])
    assert _positive_class_index(classes, positive_label=1) == 0


def test_positive_class_index_none_default():
    """_positive_class_index defaults to 1 when classes is None."""
    assert _positive_class_index(None, positive_label=1) == 1


# =============================================================================
# _infer_output_scale
# =============================================================================


def test_infer_output_scale_xgboost(fitted_xgb_model):
    """_infer_output_scale returns log_odds for XGBoost."""
    assert _infer_output_scale(fitted_xgb_model) == "log_odds"


def test_infer_output_scale_logistic_regression(fitted_lr_model):
    """_infer_output_scale returns log_odds for LogisticRegression."""
    assert _infer_output_scale(fitted_lr_model) == "log_odds"


def test_infer_output_scale_random_forest(fitted_rf_model):
    """_infer_output_scale returns raw for RandomForest."""
    assert _infer_output_scale(fitted_rf_model) == "raw"


def test_infer_output_scale_linear_svc():
    """_infer_output_scale returns margin for LinearSVC."""
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model = LinearSVC(random_state=42, max_iter=200, dual=False)
    model.fit(X, y)
    assert _infer_output_scale(model) == "margin"


# =============================================================================
# _sample_background
# =============================================================================


def test_sample_background_random_train():
    """_sample_background random_train strategy returns correct number of samples."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)
    config = SHAPConfig(
        background_strategy="random_train",
        max_background_samples=30,
    )
    rng = np.random.default_rng(42)
    bg = _sample_background(X, y, config, rng)
    assert bg.shape == (30, 5)


def test_sample_background_controls_only_enough():
    """_sample_background controls_only returns only controls when enough available."""
    X = np.random.randn(100, 5)
    y = np.concatenate([np.zeros(80), np.ones(20)])
    config = SHAPConfig(
        background_strategy="controls_only",
        max_background_samples=50,
    )
    rng = np.random.default_rng(42)
    bg = _sample_background(X, y, config, rng)
    assert bg.shape == (50, 5)


def test_sample_background_controls_only_insufficient():
    """_sample_background controls_only fills with other samples when controls insufficient."""
    X = np.random.randn(100, 5)
    y = np.concatenate([np.zeros(20), np.ones(80)])
    config = SHAPConfig(
        background_strategy="controls_only",
        max_background_samples=50,
    )
    rng = np.random.default_rng(42)
    bg = _sample_background(X, y, config, rng)
    assert bg.shape == (50, 5)


def test_sample_background_stratified():
    """_sample_background stratified respects class proportions."""
    X = np.random.randn(100, 5)
    y = np.concatenate([np.zeros(80), np.ones(20)])
    config = SHAPConfig(
        background_strategy="stratified",
        max_background_samples=40,
    )
    rng = np.random.default_rng(42)
    bg = _sample_background(X, y, config, rng)
    assert bg.shape[0] <= 40
    assert bg.shape[1] == 5


def test_sample_background_cap_at_max():
    """_sample_background caps at max_background_samples."""
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, size=20)
    config = SHAPConfig(
        background_strategy="random_train",
        max_background_samples=100,
    )
    rng = np.random.default_rng(42)
    bg = _sample_background(X, y, config, rng)
    assert bg.shape == (20, 5)


# =============================================================================
# aggregate_fold_shap (C9)
# =============================================================================


def test_aggregate_fold_shap_empty():
    """aggregate_fold_shap returns empty DataFrame with correct columns when no folds."""
    config = SHAPConfig()
    result = aggregate_fold_shap([], config)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "feature",
        "mean_abs_shap",
        "std_abs_shap",
        "median_abs_shap",
        "n_folds_nonzero",
    ]
    assert len(result) == 0


def test_aggregate_fold_shap_single_fold():
    """aggregate_fold_shap computes correct stats for single fold."""
    fold_result = SHAPFoldResult(
        values=np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b", "feat_c"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
    )
    config = SHAPConfig()
    result = aggregate_fold_shap([fold_result], config)
    assert len(result) == 3
    assert result.loc[0, "feature"] == "feat_c"
    np.testing.assert_almost_equal(result.loc[0, "mean_abs_shap"], 2.25)
    assert result.loc[0, "n_folds_nonzero"] == 1


def test_aggregate_fold_shap_multiple_folds():
    """aggregate_fold_shap aggregates across multiple folds."""
    fold1 = SHAPFoldResult(
        values=np.array([[1.0, 2.0], [0.5, 1.0]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
    )
    fold2 = SHAPFoldResult(
        values=np.array([[0.8, 1.5], [0.3, 0.7]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
    )
    config = SHAPConfig()
    result = aggregate_fold_shap([fold1, fold2], config)
    assert len(result) == 2
    assert set(result["feature"]) == {"feat_a", "feat_b"}
    assert result["n_folds_nonzero"].min() >= 2


def test_aggregate_fold_shap_mixed_scale_guard():
    """aggregate_fold_shap raises ValueError on mixed scales by default."""
    fold1 = SHAPFoldResult(
        values=np.array([[1.0, 2.0]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
    )
    fold2 = SHAPFoldResult(
        values=np.array([[1.0, 2.0]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="raw",
        model_name="RF",
        explainer_type="TreeExplainer",
    )
    config = SHAPConfig(allow_mixed_scales=False)
    with pytest.raises(ValueError, match="Mixed SHAP output scales"):
        aggregate_fold_shap([fold1, fold2], config)


def test_aggregate_fold_shap_mixed_scale_allowed():
    """aggregate_fold_shap succeeds when allow_mixed_scales=True."""
    fold1 = SHAPFoldResult(
        values=np.array([[1.0, 2.0]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
    )
    fold2 = SHAPFoldResult(
        values=np.array([[1.0, 2.0]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="raw",
        model_name="RF",
        explainer_type="TreeExplainer",
    )
    config = SHAPConfig(allow_mixed_scales=True)
    result = aggregate_fold_shap([fold1, fold2], config)
    assert len(result) == 2


def test_aggregate_fold_shap_descriptive_stats_only():
    """aggregate_fold_shap returns only descriptive stats (no p-values or CIs) (C9)."""
    fold_result = SHAPFoldResult(
        values=np.random.randn(10, 3),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b", "feat_c"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
    )
    config = SHAPConfig()
    result = aggregate_fold_shap([fold_result], config)
    assert "p_value" not in result.columns
    assert "ci_lower" not in result.columns
    assert "ci_upper" not in result.columns


# =============================================================================
# select_waterfall_samples (C8)
# =============================================================================


def test_select_waterfall_samples_all_categories():
    """select_waterfall_samples returns TP, FP, FN, near-threshold TN."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred_proba = np.array([0.9, 0.55, 0.3, 0.8, 0.6, 0.45, 0.2, 0.1])
    threshold = 0.5
    samples = select_waterfall_samples(y_pred_proba, y_true, threshold, n=4)
    assert len(samples) <= 4
    categories = {s["category"] for s in samples}
    assert "TP (highest risk)" in categories
    assert "FP (highest risk)" in categories
    assert "FN (highest risk missed)" in categories
    assert "TN (near threshold)" in categories


def test_select_waterfall_samples_fn_included():
    """select_waterfall_samples includes FN (clinically critical)."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_proba = np.array([0.9, 0.3, 0.8, 0.2])
    threshold = 0.5
    samples = select_waterfall_samples(y_pred_proba, y_true, threshold, n=4)
    categories = {s["category"] for s in samples}
    assert "FN (highest risk missed)" in categories


def test_select_waterfall_samples_uses_calibrated_threshold():
    """select_waterfall_samples uses threshold to classify predictions (C8)."""
    y_true = np.array([1, 0])
    y_pred_proba = np.array([0.51, 0.49])
    threshold = 0.5
    samples = select_waterfall_samples(y_pred_proba, y_true, threshold, n=2)
    assert samples[0]["category"] == "TP (highest risk)"
    assert samples[1]["category"] == "TN (near threshold)"


def test_select_waterfall_samples_limit_n():
    """select_waterfall_samples returns up to n samples."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_proba = np.array([0.9, 0.8, 0.7, 0.2])
    threshold = 0.5
    samples = select_waterfall_samples(y_pred_proba, y_true, threshold, n=2)
    assert len(samples) <= 2


# =============================================================================
# Pipeline unwrapping (_unwrap_calibrated_for_shap)
# =============================================================================


def test_unwrap_calibrated_for_shap_linear_svc_cal(fitted_linear_svc_cal):
    """_unwrap_calibrated_for_shap averages LinearSVC coefficients."""
    unwrapped, scale = _unwrap_calibrated_for_shap(fitted_linear_svc_cal)
    assert isinstance(unwrapped, tuple)
    assert scale == "margin"
    avg_coef, avg_intercept = unwrapped
    assert isinstance(avg_coef, np.ndarray)
    assert isinstance(avg_intercept, float)
    assert avg_coef.ndim == 1


def test_unwrap_calibrated_for_shap_raw_lr(fitted_lr_model):
    """_unwrap_calibrated_for_shap passes through raw LogisticRegression."""
    unwrapped, scale = _unwrap_calibrated_for_shap(fitted_lr_model)
    assert unwrapped is fitted_lr_model
    assert scale == "log_odds"


def test_unwrap_calibrated_for_shap_oof_calibrated(fitted_lr_model, simple_classification_data):
    """_unwrap_calibrated_for_shap unwraps OOFCalibratedModel to base_model."""
    from ced_ml.models.calibration import OOFCalibrator

    X, y = simple_classification_data
    oof_preds = fitted_lr_model.predict_proba(X)[:, 1]

    calibrator = OOFCalibrator(method="isotonic")
    calibrator.fit(oof_preds, y)

    oof_wrapped = OOFCalibratedModel(
        base_model=fitted_lr_model,
        calibrator=calibrator,
    )
    unwrapped, scale = _unwrap_calibrated_for_shap(oof_wrapped)
    assert unwrapped is fitted_lr_model
    assert scale == "log_odds"


def test_unwrap_calibrated_for_shap_pipeline(fitted_pipeline_lr):
    """_unwrap_calibrated_for_shap unwraps Pipeline to clf step."""
    unwrapped, scale = _unwrap_calibrated_for_shap(fitted_pipeline_lr)
    assert unwrapped is fitted_pipeline_lr.named_steps["clf"]
    assert scale == "log_odds"


# =============================================================================
# Additivity tests (REQUIRE shap)
# =============================================================================


@pytest.mark.skipif(shap is None, reason="shap not installed")
@pytest.mark.skipif(xgb is None, reason="xgboost not installed")
def test_additivity_xgboost_tree_path_dependent_raw():
    """XGBoost (tree_path_dependent, raw) additivity: sum(SHAP) + E[f(x)] = f(x)."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(20),
            "feat_b": np.random.randn(20),
            "feat_c": np.random.randn(20),
            "feat_d": np.random.randn(20),
            "feat_e": np.random.randn(20),
        }
    )
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values
    X_train = X.iloc[:15]
    X_test = X.iloc[15:]
    y_train = y[:15]

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    explanation = explainer(X_test.values)

    shap_values_norm = _normalize_shap_values(explanation.values, model.classes_, positive_label=1)
    expected_value_norm = _normalize_expected_value(
        explanation.base_values, model.classes_, positive_label=1
    )

    model_output = model.predict(X_test, output_margin=True).ravel()

    np.testing.assert_allclose(
        shap_values_norm.sum(axis=1) + expected_value_norm,
        model_output,
        rtol=1e-4,
    )


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_additivity_rf_interventional_raw():
    """RF (interventional, raw) additivity with relaxed tolerance."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(20),
            "feat_b": np.random.randn(20),
            "feat_c": np.random.randn(20),
            "feat_d": np.random.randn(20),
            "feat_e": np.random.randn(20),
        }
    )
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values
    X_train = X.iloc[:15]
    X_test = X.iloc[15:]
    y_train = y[:15]

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        max_depth=3,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(
        model,
        data=X_train.values,
        feature_perturbation="interventional",
        model_output="raw",
    )
    explanation = explainer(X_test.values)

    shap_values_norm = _normalize_shap_values(explanation.values, model.classes_, positive_label=1)
    expected_value_norm = _normalize_expected_value(
        explanation.base_values, model.classes_, positive_label=1
    )

    model_output = model.predict_proba(X_test)[:, 1]

    np.testing.assert_allclose(
        shap_values_norm.sum(axis=1) + expected_value_norm,
        model_output,
        rtol=1e-2,
    )


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_additivity_lr_log_odds():
    """LR (log-odds) additivity: sum(SHAP) + E[f(x)] = decision_function(X)."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(20),
            "feat_b": np.random.randn(20),
            "feat_c": np.random.randn(20),
            "feat_d": np.random.randn(20),
            "feat_e": np.random.randn(20),
        }
    )
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values
    X_train = X.iloc[:15]
    X_test = X.iloc[15:]
    y_train = y[:15]

    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    explainer = shap.LinearExplainer(model, X_train.values)
    explanation = explainer(X_test.values)

    shap_values_norm = _normalize_shap_values(explanation.values, model.classes_, positive_label=1)
    expected_value_norm = _normalize_expected_value(
        explanation.base_values, model.classes_, positive_label=1
    )

    model_output = model.decision_function(X_test).ravel()

    np.testing.assert_allclose(
        shap_values_norm.sum(axis=1) + expected_value_norm,
        model_output,
        rtol=1e-6,
    )


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_additivity_svm_margin():
    """SVM (margin) additivity: sum(SHAP) + E[f(x)] = averaged margin."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(20),
            "feat_b": np.random.randn(20),
            "feat_c": np.random.randn(20),
            "feat_d": np.random.randn(20),
            "feat_e": np.random.randn(20),
        }
    )
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values
    X_train = X.iloc[:15]
    X_test = X.iloc[15:]
    y_train = y[:15]

    base = LinearSVC(random_state=42, max_iter=200, dual=False)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    avg_coef = np.mean(
        [cc.estimator.coef_.ravel() for cc in clf.calibrated_classifiers_],
        axis=0,
    )
    avg_intercept = np.mean([cc.estimator.intercept_[0] for cc in clf.calibrated_classifiers_])

    explainer = shap.LinearExplainer(
        (avg_coef.reshape(1, -1), np.array([avg_intercept])),
        X_train.values,
    )
    explanation = explainer(X_test.values)

    shap_values_norm = _normalize_shap_values(explanation.values, clf.classes_, positive_label=1)
    expected_value_norm = _normalize_expected_value(
        explanation.base_values, clf.classes_, positive_label=1
    )

    model_output = (avg_coef @ X_test.T + avg_intercept).ravel()

    np.testing.assert_allclose(
        shap_values_norm.sum(axis=1) + expected_value_norm,
        model_output,
        rtol=1e-6,
    )


# =============================================================================
# Config: background sensitivity fields
# =============================================================================


def test_config_background_sensitivity_defaults():
    """SHAPConfig has background_sensitivity_mode=False and n_background_replicates=3."""
    config = SHAPConfig()
    assert config.background_sensitivity_mode is False
    assert config.n_background_replicates == 3


def test_config_background_sensitivity_accepts_valid():
    """SHAPConfig accepts valid sensitivity settings."""
    config = SHAPConfig(background_sensitivity_mode=True, n_background_replicates=5)
    assert config.background_sensitivity_mode is True
    assert config.n_background_replicates == 5


def test_config_background_sensitivity_rejects_below_min():
    """SHAPConfig rejects n_background_replicates < 2."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SHAPConfig(n_background_replicates=1)


def test_config_background_sensitivity_rejects_above_max():
    """SHAPConfig rejects n_background_replicates > 10."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SHAPConfig(n_background_replicates=11)


def test_config_backward_compatible_without_sensitivity_fields():
    """SHAPConfig constructed without new fields parses correctly (defaults used)."""
    config = SHAPConfig(
        enabled=True,
        max_background_samples=50,
        background_strategy="random_train",
    )
    assert config.background_sensitivity_mode is False
    assert config.n_background_replicates == 3


# =============================================================================
# SHAPTestPayload: background_sensitivity_result field
# =============================================================================


def test_shap_test_payload_sensitivity_result_default_none():
    """SHAPTestPayload.background_sensitivity_result defaults to None."""
    payload = SHAPTestPayload(
        values=np.array([[1.0]]),
        expected_value=0.0,
        feature_names=["feat_a"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
        split="test",
    )
    assert payload.background_sensitivity_result is None


def test_shap_test_payload_sensitivity_result_accepts_dict():
    """SHAPTestPayload accepts a dict for background_sensitivity_result."""
    result_dict = {
        "mean_rank_correlation": 0.95,
        "rank_std": np.array([0.1, 0.2]),
        "attributions": [np.array([[1.0, 2.0]])],
        "n_replicates": 3,
    }
    payload = SHAPTestPayload(
        values=np.array([[1.0, 2.0]]),
        expected_value=0.0,
        feature_names=["feat_a", "feat_b"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
        split="test",
        background_sensitivity_result=result_dict,
    )
    assert payload.background_sensitivity_result is not None
    assert payload.background_sensitivity_result["mean_rank_correlation"] == 0.95
    assert payload.background_sensitivity_result["n_replicates"] == 3


# =============================================================================
# compute_background_sensitivity (REQUIRE shap)
# =============================================================================


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_compute_background_sensitivity_lr():
    """compute_background_sensitivity produces valid correlation for LogisticRegression."""
    from ced_ml.features.shap_values import compute_background_sensitivity

    np.random.seed(42)
    n_features = 5
    X = pd.DataFrame({f"feat_{i}": np.random.randn(100) for i in range(n_features)})
    y = (X["feat_0"] + 0.5 * X["feat_1"] > 0).astype(int).values

    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("num", StandardScaler(), [f"feat_{i}" for i in range(n_features)])]
                ),
            ),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    X_train_t, _ = get_model_matrix_and_feature_names(pipeline, X.iloc[:80])
    X_eval_t, _ = get_model_matrix_and_feature_names(pipeline, X.iloc[80:])

    config = SHAPConfig(
        enabled=True,
        background_sensitivity_mode=True,
        n_background_replicates=3,
        max_background_samples=30,
    )
    rng = np.random.default_rng(42)

    result = compute_background_sensitivity(
        fitted_pipeline=pipeline,
        X_train_transformed=X_train_t,
        y_train=y[:80],
        X_eval_transformed=X_eval_t,
        config=config,
        model_name="LR_EN",
        rng=rng,
    )

    assert 0.0 <= result["mean_rank_correlation"] <= 1.0
    assert len(result["attributions"]) == 3
    assert result["rank_std"].shape == (n_features,)
    assert result["n_replicates"] == 3


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_compute_background_sensitivity_two_replicates():
    """compute_background_sensitivity works with minimum n_replicates=2."""
    from ced_ml.features.shap_values import compute_background_sensitivity

    np.random.seed(123)
    n_features = 3
    X = pd.DataFrame({f"feat_{i}": np.random.randn(60) for i in range(n_features)})
    y = (X["feat_0"] > 0).astype(int).values

    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("num", StandardScaler(), [f"feat_{i}" for i in range(n_features)])]
                ),
            ),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    X_train_t, _ = get_model_matrix_and_feature_names(pipeline, X.iloc[:40])
    X_eval_t, _ = get_model_matrix_and_feature_names(pipeline, X.iloc[40:])

    config = SHAPConfig(
        enabled=True,
        background_sensitivity_mode=True,
        n_background_replicates=2,
        max_background_samples=20,
    )
    rng = np.random.default_rng(99)

    result = compute_background_sensitivity(
        fitted_pipeline=pipeline,
        X_train_transformed=X_train_t,
        y_train=y[:40],
        X_eval_transformed=X_eval_t,
        config=config,
        model_name="LR_EN",
        rng=rng,
    )

    assert result["n_replicates"] == 2
    assert len(result["attributions"]) == 2
    # With 2 replicates there is exactly 1 pairwise correlation
    assert -1.0 <= result["mean_rank_correlation"] <= 1.0


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_compute_final_shap_sensitivity_disabled_by_default():
    """compute_final_shap returns None for sensitivity when disabled."""
    from ced_ml.features.shap_values import compute_final_shap

    np.random.seed(42)
    n_features = 3
    X = pd.DataFrame({f"feat_{i}": np.random.randn(60) for i in range(n_features)})
    y = (X["feat_0"] > 0).astype(int).values

    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("num", StandardScaler(), [f"feat_{i}" for i in range(n_features)])]
                ),
            ),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    config = SHAPConfig(
        enabled=True,
        background_sensitivity_mode=False,
        max_background_samples=20,
    )

    payload = compute_final_shap(
        fitted_pipeline=pipeline,
        model_name="LR_EN",
        X_eval=X.iloc[50:],
        y_eval=y[50:],
        X_train=X.iloc[:50],
        config=config,
        split="test",
        y_train=y[:50],
    )

    assert payload.background_sensitivity_result is None


@pytest.mark.skipif(shap is None, reason="shap not installed")
def test_compute_final_shap_sensitivity_enabled():
    """compute_final_shap populates sensitivity result when enabled."""
    from ced_ml.features.shap_values import compute_final_shap

    np.random.seed(42)
    n_features = 3
    X = pd.DataFrame({f"feat_{i}": np.random.randn(60) for i in range(n_features)})
    y = (X["feat_0"] > 0).astype(int).values

    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("num", StandardScaler(), [f"feat_{i}" for i in range(n_features)])]
                ),
            ),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)

    config = SHAPConfig(
        enabled=True,
        background_sensitivity_mode=True,
        n_background_replicates=2,
        max_background_samples=20,
    )

    payload = compute_final_shap(
        fitted_pipeline=pipeline,
        model_name="LR_EN",
        X_eval=X.iloc[50:],
        y_eval=y[50:],
        X_train=X.iloc[:50],
        config=config,
        split="test",
        y_train=y[:50],
    )

    assert payload.background_sensitivity_result is not None
    sens = payload.background_sensitivity_result
    assert sens["n_replicates"] == 2
    assert 0.0 <= sens["mean_rank_correlation"] <= 1.0
    assert sens["rank_std"].shape == (n_features,)
    assert len(sens["attributions"]) == 2
