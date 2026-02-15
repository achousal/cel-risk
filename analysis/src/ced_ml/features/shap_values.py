"""SHAP explainability computation for CeD-ML pipeline.

Provides per-fold OOF SHAP, final-model SHAP, fold aggregation,
and waterfall sample selection. All SHAP values are in the base
model's native output scale (log-odds, margin, or raw), NOT
calibrated probability.

Key design decisions:
- Pipeline unwrapping: transform X through preprocessing, explain classifier only
- LinSVM_cal: always unwrap CalibratedClassifierCV, explain averaged linear surrogate
- Output scale metadata on every result to prevent cross-model comparison bugs
- Class-axis normalization for binary tree classifiers (C13)
- Descriptive fold aggregation only (C9): no inferential stats across CV folds
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ..data.schema import ModelName

if TYPE_CHECKING:
    from ..config.shap_schema import SHAPConfig

logger = logging.getLogger(__name__)

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

_SHAP_INSTALL_MSG = (
    "SHAP is required for explainability features. " "Install with: pip install -e 'analysis[shap]'"
)


def _require_shap() -> None:
    """Raise ImportError with install instructions if shap is not available."""
    if shap is None:
        raise ImportError(_SHAP_INSTALL_MSG)


# ---------------------------------------------------------------------------
# Explained model state enum
# ---------------------------------------------------------------------------


class ExplainedModelState(str, Enum):
    """State of the model being explained by SHAP.

    Tracks whether SHAP values explain the pre-calibration classifier
    or a calibration surrogate (e.g., averaged linear coefficients).
    """

    PRE_CALIBRATION = "pre_calibration"
    LINEAR_SURROGATE_MEAN_COEF = (
        "linear_surrogate_mean_coef"  # LinSVM_cal unwrapped mean coefficients
    )


# ---------------------------------------------------------------------------
# Normalization helpers (C13)
# ---------------------------------------------------------------------------


def _normalize_expected_value(
    ev: Any,
    classes: np.ndarray | None = None,
    positive_label: int = 1,
) -> float:
    """Normalize SHAP expected_value to positive-class scalar.

    SHAP explainers return expected_value in inconsistent shapes:
    scalar, length-2 array (binary classifiers), or list wrapping.
    This helper always returns a float for the positive class.

    When ev is array-like with length 2 (binary classifier), uses classes
    (from clf.classes_) to find the index matching positive_label rather
    than hardcoding index [1]. This prevents silent sign flips if a model's
    class ordering is [1, 0] instead of the conventional [0, 1].
    """
    # Unwrap single-element list/array
    if isinstance(ev, list):
        ev = ev[0] if len(ev) == 1 else np.array(ev)

    if isinstance(ev, np.ndarray):
        ev = ev.ravel()
        if ev.shape[0] == 1:
            return float(ev[0])
        if ev.shape[0] == 2:
            idx = _positive_class_index(classes, positive_label)
            return float(ev[idx])

    return float(ev)


def _normalize_shap_values(
    values: np.ndarray,
    classes: np.ndarray | None = None,
    positive_label: int = 1,
) -> np.ndarray:
    """Normalize SHAP values to a 2D positive-class matrix.

    Handles common SHAP return shapes:
    - (n_samples, n_features): already normalized
    - (n_samples, n_features, 2): binary classifier with class axis
    - list-wrapped binary outputs from older SHAP APIs

    Uses classes (from clf.classes_) to select the positive_label axis.
    Returns shape (n_samples, n_features).
    """
    # Handle list wrapping (older SHAP versions)
    if isinstance(values, list):
        if len(values) == 2:
            # Binary class list: [neg_class_array, pos_class_array]
            idx = _positive_class_index(classes, positive_label)
            return np.asarray(values[idx])
        if len(values) == 1:
            return np.asarray(values[0])
        values = np.asarray(values)

    values = np.asarray(values)

    if values.ndim == 2:
        return values

    if values.ndim == 3 and values.shape[2] == 2:
        idx = _positive_class_index(classes, positive_label)
        return values[:, :, idx]

    if values.ndim == 3 and values.shape[2] == 1:
        # Single-output model (e.g. CalibratedClassifierCV wrapping LinearSVC):
        # squeeze the trailing dimension to get (n_samples, n_features)
        return values[:, :, 0]

    # Unexpected shape -- return as-is with warning
    logger.warning("Unexpected SHAP values shape: %s, returning as-is", values.shape)
    return values


def _positive_class_index(
    classes: np.ndarray | None,
    positive_label: int = 1,
) -> int:
    """Find index of positive_label in classes array.

    Falls back to index 1 if classes is None (conventional binary ordering).
    """
    if classes is not None:
        matches = np.where(classes == positive_label)[0]
        if len(matches) == 1:
            return int(matches[0])
        logger.warning(
            "Could not find positive_label=%d in classes=%s; defaulting to index 1",
            positive_label,
            classes,
        )
    return 1


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SHAPFoldResult:
    """Result of SHAP computation for a single CV fold."""

    values: np.ndarray  # (n_samples, n_features) positive-class
    expected_value: float  # E[f(x)] baseline
    feature_names: list[str]
    shap_output_scale: str  # "margin" | "log_odds" | "probability" | "raw"
    model_name: str
    explainer_type: str  # "TreeExplainer" | "LinearExplainer"
    repeat: int = -1
    outer_split: int = -1
    explained_model_state: str = (
        ExplainedModelState.PRE_CALIBRATION
    )  # State of model being explained


@dataclass
class SHAPTestPayload:
    """Full SHAP result for final model on test (or val).

    NOTE: Does NOT carry y_pred. Waterfall sample selection uses
    ctx.test_preds_df (calibrated predictions at operating threshold).
    SHAP values are in the base model's native output scale.
    """

    values: np.ndarray  # (n_samples, n_features) positive-class
    expected_value: float
    feature_names: list[str]
    shap_output_scale: str
    model_name: str
    explainer_type: str
    split: str  # "test" | "val"
    y_true: np.ndarray | None = None
    sample_indices: np.ndarray | None = None  # Row indices if subsampled (max_eval_samples)
    X_transformed: np.ndarray | None = None  # Transformed feature matrix for plot color axis
    explained_model_state: str = (
        ExplainedModelState.PRE_CALIBRATION
    )  # State of model being explained
    background_sensitivity_result: dict | None = None  # Multi-background sensitivity analysis


# ---------------------------------------------------------------------------
# Pipeline unwrapping
# ---------------------------------------------------------------------------


def get_model_matrix_and_feature_names(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Transform X through pipeline preprocessing, return (X_transformed, feature_names).

    Steps:
    1. Apply 'pre' (ColumnTransformer) to get transformed matrix
    2. Apply 'sel' (KBest selector) mask if present
    3. Apply 'model_sel' mask if present
    4. Return transformed X and resolved feature names

    Mirrors _get_final_feature_names() logic from importance.py:56-107.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected Pipeline, got {type(pipeline).__name__}")

    # Step 1: get preprocessor output
    if "pre" not in pipeline.named_steps:
        raise ValueError("Pipeline missing 'pre' step")

    pre = pipeline.named_steps["pre"]
    X_transformed = pre.transform(X)
    if hasattr(pre, "get_feature_names_out"):
        feature_names = list(pre.get_feature_names_out())
    else:
        raise ValueError("Preprocessor does not support get_feature_names_out()")

    # Convert sparse to dense if needed
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed, dtype=float)

    # Step 2: apply sel mask
    if "sel" in pipeline.named_steps:
        sel = pipeline.named_steps["sel"]
        if hasattr(sel, "selected_proteins_"):
            # ProteinOnlySelector
            out_names = list(sel.get_feature_names_out())
            mask = np.isin(feature_names, out_names)
            X_transformed = X_transformed[:, mask]
            feature_names = out_names
        elif hasattr(sel, "get_support"):
            support = sel.get_support()
            X_transformed = X_transformed[:, support]
            feature_names = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    # Step 3: apply model_sel mask
    if "model_sel" in pipeline.named_steps:
        model_sel = pipeline.named_steps["model_sel"]
        if hasattr(model_sel, "selected_proteins_"):
            out_names = list(model_sel.get_feature_names_out())
            mask = np.isin(feature_names, out_names)
            X_transformed = X_transformed[:, mask]
            feature_names = out_names
        elif hasattr(model_sel, "get_support"):
            support = model_sel.get_support()
            X_transformed = X_transformed[:, support]
            feature_names = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    return X_transformed, feature_names


def _unwrap_calibrated_for_shap(clf: Any) -> tuple[Any, str]:
    """Unwrap calibration wrapper to get base estimator + output scale.

    Handles:
    - CalibratedClassifierCV with LinearSVC: average coef_/intercept_
      (mirrors importance.py:148-168 pattern).
    - CalibratedClassifierCV with tree models: access base estimator.
    - OOFCalibratedModel: access base_model (defensive).
    - Raw estimator: passthrough.
    """
    # Import lazily to avoid circular imports
    from ..models.calibration import OOFCalibratedModel

    # OOFCalibratedModel wrapping
    if isinstance(clf, OOFCalibratedModel):
        logger.debug("Unwrapping OOFCalibratedModel to base_model")
        return _unwrap_calibrated_for_shap(clf.base_model)

    # CalibratedClassifierCV wrapping
    if isinstance(clf, CalibratedClassifierCV):
        if not hasattr(clf, "calibrated_classifiers_"):
            raise ValueError("CalibratedClassifierCV is not fitted")

        # Check if base estimator is linear (LinearSVC)
        sample_est = None
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if est is not None:
                sample_est = est
                break

        if sample_est is None:
            raise ValueError("No base estimators found in CalibratedClassifierCV")

        if hasattr(sample_est, "coef_"):
            # Linear model (LinearSVC) -- average coefficients
            coefs_list = []
            intercepts_list = []
            for cc in clf.calibrated_classifiers_:
                est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if est and hasattr(est, "coef_"):
                    coefs_list.append(est.coef_.ravel())
                    intercepts_list.append(
                        float(est.intercept_[0]) if hasattr(est, "intercept_") else 0.0
                    )

            avg_coef = np.mean(np.vstack(coefs_list), axis=0)
            avg_intercept = np.mean(intercepts_list)
            return (avg_coef, avg_intercept), "margin"

        # Tree model inside CalibratedClassifierCV -- return base estimator
        return sample_est, "raw"

    # Pipeline wrapping
    if isinstance(clf, Pipeline):
        if "clf" in clf.named_steps:
            return _unwrap_calibrated_for_shap(clf.named_steps["clf"])

    # Raw estimator -- infer scale
    return clf, _infer_output_scale(clf)


def _infer_output_scale(clf: Any) -> str:
    """Infer the SHAP output scale for an unwrapped estimator."""
    clf_type = type(clf).__name__

    if "XGB" in clf_type:
        return "log_odds"
    if "LogisticRegression" in clf_type:
        return "log_odds"
    if "RandomForest" in clf_type:
        return "raw"
    if "LinearSVC" in clf_type or "SVC" in clf_type:
        return "margin"

    return "raw"


# ---------------------------------------------------------------------------
# Background data sampling
# ---------------------------------------------------------------------------


def _sample_background(
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    config: SHAPConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample background data for SHAP explainers."""
    n = min(config.max_background_samples, len(X_train))

    if config.background_strategy == "controls_only" and y_train is not None:
        control_idx = np.where(y_train == 0)[0]
        if len(control_idx) >= n:
            chosen = rng.choice(control_idx, size=n, replace=False)
            return X_train[chosen]
        logger.warning(
            "Only %d controls available (requested %d); using all controls + random fill",
            len(control_idx),
            n,
        )
        remaining = n - len(control_idx)
        other_idx = np.where(y_train != 0)[0]
        fill = rng.choice(other_idx, size=min(remaining, len(other_idx)), replace=False)
        chosen = np.concatenate([control_idx, fill])
        return X_train[chosen]

    if config.background_strategy == "stratified" and y_train is not None:
        classes = np.unique(y_train)
        indices = []
        for c in classes:
            c_idx = np.where(y_train == c)[0]
            n_c = max(1, int(n * len(c_idx) / len(y_train)))
            chosen_c = rng.choice(c_idx, size=min(n_c, len(c_idx)), replace=False)
            indices.append(chosen_c)
        all_idx = np.concatenate(indices)
        if len(all_idx) > n:
            all_idx = rng.choice(all_idx, size=n, replace=False)
        return X_train[all_idx]

    # Default: random_train
    chosen = rng.choice(len(X_train), size=n, replace=False)
    return X_train[chosen]


# ---------------------------------------------------------------------------
# Background sensitivity analysis
# ---------------------------------------------------------------------------


def compute_background_sensitivity(
    fitted_pipeline: Any,
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    X_eval_transformed: np.ndarray,
    config: SHAPConfig,
    model_name: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compute SHAP with multiple background samples and assess rank stability.

    Tests whether feature attributions are robust to baseline sample choice.
    Reports pairwise Spearman rank correlations across backgrounds (higher = more stable).

    WARNING: This is N times slower than standard SHAP computation, where N is
    config.n_background_replicates.

    Args:
        fitted_pipeline: Fitted ML pipeline (or raw estimator).
        X_train_transformed: Training features (for background sampling).
        y_train: Training labels (for stratified background sampling).
        X_eval_transformed: Test/eval features to explain.
        config: SHAPConfig with n_background_replicates and background settings.
        model_name: Model identifier (e.g., "LR_EN", "XGBoost").
        rng: numpy random generator for reproducible sampling.

    Returns:
        dict with keys:
        - "mean_rank_correlation": float, average pairwise Spearman rank correlation
        - "rank_std": np.ndarray, per-feature rank standard deviation across backgrounds
        - "attributions": list[np.ndarray], SHAP values per background (n_reps entries)
        - "n_replicates": int, number of backgrounds tested
    """
    from scipy.stats import spearmanr

    _require_shap()

    n_reps = config.n_background_replicates
    attributions: list[np.ndarray] = []

    for _rep in range(n_reps):
        X_bg = _sample_background(X_train_transformed, y_train, config, rng)
        explainer, _, classes = get_shap_explainer(fitted_pipeline, model_name, X_bg, config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_shap = explainer.shap_values(X_eval_transformed)

        attr_values = _normalize_shap_values(raw_shap, classes, config.positive_label)
        attributions.append(np.asarray(attr_values))

    # Compute mean |SHAP| rankings for each background (descending importance)
    rankings: list[np.ndarray] = []
    for attr in attributions:
        mean_abs_shap = np.mean(np.abs(attr), axis=0)
        # rankdata gives ascending ranks; negate for descending importance order
        from scipy.stats import rankdata

        ranks = rankdata(-mean_abs_shap, method="average")
        rankings.append(ranks)

    # Pairwise Spearman correlations between ranking vectors
    correlations: list[float] = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            corr, _ = spearmanr(rankings[i], rankings[j])
            correlations.append(float(corr))

    mean_corr = float(np.mean(correlations)) if correlations else 1.0

    # Per-feature rank variability
    rank_array = np.array(rankings)
    rank_std = np.std(rank_array, axis=0)

    logger.info(
        "Background sensitivity: %d replicates, mean rank correlation=%.4f",
        n_reps,
        mean_corr,
    )

    return {
        "mean_rank_correlation": mean_corr,
        "rank_std": rank_std,
        "attributions": attributions,
        "n_replicates": n_reps,
    }


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------


def _resolve_tree_model_output(model_name: str, config: SHAPConfig) -> str:
    """Resolve effective TreeExplainer model_output from config.

    "auto" uses model-specific defaults for readability while preserving
    safer defaults for boosted trees:
    - RF -> "probability" (interventional is forced anyway)
    - XGBoost -> "raw" (keeps path-dependent compatibility by default)
    """
    if config.tree_model_output != "auto":
        return config.tree_model_output
    if model_name == ModelName.RF:
        return "probability"
    return "raw"


def get_shap_explainer(
    clf: Any,
    model_name: str,
    X_background: np.ndarray,
    config: SHAPConfig,
) -> tuple[Any, str, np.ndarray | None]:
    """Select and create SHAP explainer by model type.

    Returns (explainer, shap_output_scale, classes_).
    classes_ is used for normalization of class-axis outputs.
    """
    _require_shap()

    # Extract classifier from pipeline if needed
    raw_clf = clf
    if isinstance(clf, Pipeline):
        if "clf" in clf.named_steps:
            raw_clf = clf.named_steps["clf"]

    # Unwrap calibration wrappers
    unwrapped, output_scale = _unwrap_calibrated_for_shap(raw_clf)

    # Get classes_ for normalization
    classes = getattr(raw_clf, "classes_", None)
    if classes is None and not isinstance(unwrapped, tuple):
        classes = getattr(unwrapped, "classes_", None)

    # Validate positive_label is in classes
    if classes is not None and config.positive_label not in classes:
        raise ValueError(
            f"positive_label={config.positive_label} not found in classifier classes {classes}. "
            f"Check SHAPConfig.positive_label."
        )

    # LinSVM_cal: unwrapped is (avg_coef, avg_intercept) tuple
    if isinstance(unwrapped, tuple):
        avg_coef, avg_intercept = unwrapped
        # Build masker for LinearExplainer
        explainer = shap.LinearExplainer(
            (avg_coef.reshape(1, -1), np.array([avg_intercept])),
            X_background,
        )
        return explainer, "margin", classes

    # Tree models
    if model_name in (ModelName.XGBoost, ModelName.RF):
        effective_model_output = _resolve_tree_model_output(model_name, config)
        if model_name == ModelName.RF:
            # RF always uses interventional perturbation
            explainer = shap.TreeExplainer(
                unwrapped,
                data=X_background,
                feature_perturbation="interventional",
                model_output=effective_model_output,
            )
        else:
            # XGBoost
            tree_kwargs = {
                "feature_perturbation": config.tree_feature_perturbation,
                "model_output": effective_model_output,
            }
            # Only pass data for interventional
            if config.tree_feature_perturbation == "interventional":
                tree_kwargs["data"] = X_background
            explainer = shap.TreeExplainer(unwrapped, **tree_kwargs)

        # Determine output scale
        if effective_model_output == "probability":
            output_scale = "probability"
        elif model_name == ModelName.XGBoost:
            output_scale = "log_odds"
        else:
            output_scale = "raw"

        return explainer, output_scale, classes

    # Linear models (LR_EN, LR_L1)
    if hasattr(unwrapped, "coef_"):
        explainer = shap.LinearExplainer(unwrapped, X_background)
        return explainer, "log_odds", classes

    raise ValueError(
        f"No SHAP explainer available for model_name={model_name}, "
        f"estimator type={type(unwrapped).__name__}"
    )


# ---------------------------------------------------------------------------
# Per-fold SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_for_fold(
    fitted_model: Pipeline,
    model_name: str,
    X_val: pd.DataFrame,
    X_train: pd.DataFrame,
    config: SHAPConfig,
    random_state: int = 42,
    y_train: np.ndarray | None = None,
) -> SHAPFoldResult:
    """SHAP on single CV fold.

    Steps:
    1. Transform X_val/X_train through pipeline preprocessing
    2. Sample background from X_train per config strategy
    3. Create explainer via get_shap_explainer()
    4. Compute SHAP values
    5. Normalize to positive class
    6. Return SHAPFoldResult with scale metadata
    """
    _require_shap()
    rng = np.random.default_rng(random_state)

    # Transform through pipeline preprocessing
    X_val_transformed, feature_names = get_model_matrix_and_feature_names(fitted_model, X_val)
    X_train_transformed, _ = get_model_matrix_and_feature_names(fitted_model, X_train)

    if len(feature_names) > config.max_features_warn:
        logger.warning(
            "SHAP: %d features exceeds warning threshold (%d). "
            "Consider feature selection before SHAP computation.",
            len(feature_names),
            config.max_features_warn,
        )

    # Cap samples if configured
    if config.max_eval_samples > 0 and len(X_val_transformed) > config.max_eval_samples:
        idx = rng.choice(len(X_val_transformed), size=config.max_eval_samples, replace=False)
        X_val_transformed = X_val_transformed[idx]

    # Sample background
    X_bg = _sample_background(X_train_transformed, y_train, config, rng)

    # Create explainer
    explainer, output_scale, classes = get_shap_explainer(
        fitted_model,
        model_name,
        X_bg,
        config,
    )

    # Compute SHAP values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_shap = explainer.shap_values(X_val_transformed)

    raw_ev = explainer.expected_value

    # Normalize to positive class (C13)
    values = _normalize_shap_values(raw_shap, classes, config.positive_label)
    expected_value = _normalize_expected_value(raw_ev, classes, config.positive_label)

    # Cast dtype
    values = values.astype(config.raw_dtype)
    explained_model_state = (
        ExplainedModelState.LINEAR_SURROGATE_MEAN_COEF
        if model_name == ModelName.LinSVM_cal
        else ExplainedModelState.PRE_CALIBRATION
    )

    return SHAPFoldResult(
        values=values,
        expected_value=expected_value,
        feature_names=list(feature_names),
        shap_output_scale=output_scale,
        model_name=model_name,
        explainer_type=type(explainer).__name__,
        explained_model_state=explained_model_state,
    )


# ---------------------------------------------------------------------------
# Final model SHAP
# ---------------------------------------------------------------------------


def compute_final_shap(
    fitted_pipeline: Pipeline,
    model_name: str,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    X_train: pd.DataFrame,
    config: SHAPConfig,
    split: str = "test",
    y_train: np.ndarray | None = None,
) -> SHAPTestPayload:
    """SHAP on final fitted model for test (or val) set.

    Same logic as compute_shap_for_fold but returns SHAPTestPayload
    with y_true for reference. Does NOT include y_pred.
    """
    _require_shap()
    rng = np.random.default_rng(42)

    X_eval_transformed, feature_names = get_model_matrix_and_feature_names(
        fitted_pipeline,
        X_eval,
    )
    X_train_transformed, _ = get_model_matrix_and_feature_names(fitted_pipeline, X_train)

    if len(feature_names) > config.max_features_warn:
        logger.warning(
            "SHAP: %d features exceeds warning threshold (%d).",
            len(feature_names),
            config.max_features_warn,
        )

    # Cap samples if configured
    eval_idx = None
    if config.max_eval_samples > 0 and len(X_eval_transformed) > config.max_eval_samples:
        eval_idx = rng.choice(
            len(X_eval_transformed),
            size=config.max_eval_samples,
            replace=False,
        )
        X_eval_transformed = X_eval_transformed[eval_idx]
        y_eval = y_eval[eval_idx] if y_eval is not None else None

    # Sample background
    X_bg = _sample_background(X_train_transformed, y_train, config, rng)

    # Create explainer
    explainer, output_scale, classes = get_shap_explainer(
        fitted_pipeline,
        model_name,
        X_bg,
        config,
    )

    # Compute SHAP values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_shap = explainer.shap_values(X_eval_transformed)

    raw_ev = explainer.expected_value

    # Normalize
    values = _normalize_shap_values(raw_shap, classes, config.positive_label)
    expected_value = _normalize_expected_value(raw_ev, classes, config.positive_label)
    values = values.astype(config.raw_dtype)

    # Background sensitivity analysis (optional, disabled by default)
    bg_sensitivity_result = None
    if config.background_sensitivity_mode:
        bg_sensitivity_result = compute_background_sensitivity(
            fitted_pipeline=fitted_pipeline,
            X_train_transformed=X_train_transformed,
            y_train=y_train,
            X_eval_transformed=X_eval_transformed,
            config=config,
            model_name=model_name,
            rng=rng,
        )
    explained_model_state = (
        ExplainedModelState.LINEAR_SURROGATE_MEAN_COEF
        if model_name == ModelName.LinSVM_cal
        else ExplainedModelState.PRE_CALIBRATION
    )

    return SHAPTestPayload(
        values=values,
        expected_value=expected_value,
        feature_names=list(feature_names),
        shap_output_scale=output_scale,
        model_name=model_name,
        explainer_type=type(explainer).__name__,
        split=split,
        explained_model_state=explained_model_state,
        y_true=y_eval,
        sample_indices=eval_idx,
        X_transformed=X_eval_transformed,
        background_sensitivity_result=bg_sensitivity_result,
    )


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------


def aggregate_fold_shap(
    fold_results: list[SHAPFoldResult],
    config: SHAPConfig,
) -> pd.DataFrame:
    """Aggregate SHAP across CV folds.

    Mirrors aggregate_fold_importances() (importance.py:673-836):
    - Per fold: compute mean(|shap_values|) per feature
    - Cross-fold: mean, std, median, n_folds_nonzero

    NOTE (C9): Descriptive statistics only. No inferential statistics
    (p-values, CIs) across repeated-CV folds -- folds are not independent.

    Output schema: [feature, mean_abs_shap, std_abs_shap, median_abs_shap, n_folds_nonzero]
    """
    if not fold_results:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_abs_shap",
                "std_abs_shap",
                "median_abs_shap",
                "n_folds_nonzero",
            ]
        )

    # Scale guard
    scales = {r.shap_output_scale for r in fold_results}
    if len(scales) > 1 and not config.allow_mixed_scales:
        raise ValueError(
            f"Mixed SHAP output scales across folds: {scales}. "
            f"Cannot aggregate meaningfully. Set allow_mixed_scales=True to override."
        )

    # Collect all feature names
    all_features: set[str] = set()
    for r in fold_results:
        all_features.update(r.feature_names)

    # Build per-feature importance matrix
    importance_matrix: dict[str, list[float]] = {f: [] for f in all_features}

    for r in fold_results:
        # mean(|SHAP|) per feature for this fold
        mean_abs = np.mean(np.abs(r.values), axis=0)
        name_to_val = dict(zip(r.feature_names, mean_abs, strict=False))

        for f in all_features:
            importance_matrix[f].append(float(name_to_val.get(f, 0.0)))

    # Compute statistics
    rows = []
    for feature, values in importance_matrix.items():
        arr = np.array(values)
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(arr)),
                "std_abs_shap": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median_abs_shap": float(np.median(arr)),
                "n_folds_nonzero": int(np.sum(arr > 0)),
            }
        )

    df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Waterfall sample selection
# ---------------------------------------------------------------------------


def select_waterfall_samples(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    n: int = 4,
) -> list[dict]:
    """Select clinically informative samples for waterfall plots.

    NOTE: y_pred_proba should be CALIBRATED predictions (from ctx.test_preds_df),
    not raw SHAP-scale predictions.

    Selection (priority order):
    1. Highest-risk true positive (TP)
    2. Highest-risk false positive (FP)
    3. Highest-risk false negative (FN) -- clinically critical
    4. Near-threshold negative

    Returns list of dicts: [{index, label, pred_proba, category}, ...]
    """
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    results = []

    # TP: predicted positive AND actually positive
    tp_mask = (y_pred_binary == 1) & (y_true == 1)
    if tp_mask.any():
        tp_idx = np.where(tp_mask)[0]
        best_tp = tp_idx[np.argmax(y_pred_proba[tp_idx])]
        results.append(
            {
                "index": int(best_tp),
                "label": int(y_true[best_tp]),
                "pred_proba": float(y_pred_proba[best_tp]),
                "category": "TP (highest risk)",
            }
        )

    # FP: predicted positive AND actually negative
    fp_mask = (y_pred_binary == 1) & (y_true == 0)
    if fp_mask.any():
        fp_idx = np.where(fp_mask)[0]
        best_fp = fp_idx[np.argmax(y_pred_proba[fp_idx])]
        results.append(
            {
                "index": int(best_fp),
                "label": int(y_true[best_fp]),
                "pred_proba": float(y_pred_proba[best_fp]),
                "category": "FP (highest risk)",
            }
        )

    # FN: predicted negative AND actually positive (clinically critical)
    fn_mask = (y_pred_binary == 0) & (y_true == 1)
    if fn_mask.any():
        fn_idx = np.where(fn_mask)[0]
        best_fn = fn_idx[np.argmax(y_pred_proba[fn_idx])]
        results.append(
            {
                "index": int(best_fn),
                "label": int(y_true[best_fn]),
                "pred_proba": float(y_pred_proba[best_fn]),
                "category": "FN (highest risk missed)",
            }
        )

    # Near-threshold negative: closest to threshold among true negatives
    tn_mask = (y_pred_binary == 0) & (y_true == 0)
    if tn_mask.any():
        tn_idx = np.where(tn_mask)[0]
        distances = np.abs(y_pred_proba[tn_idx] - threshold)
        nearest_tn = tn_idx[np.argmin(distances)]
        results.append(
            {
                "index": int(nearest_tn),
                "label": int(y_true[nearest_tn]),
                "pred_proba": float(y_pred_proba[nearest_tn]),
                "category": "TN (near threshold)",
            }
        )

    return results[:n]
