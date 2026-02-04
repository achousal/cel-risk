"""Feature importance computation for OOF predictions.

This module provides correlation-robust feature importance computation on held-out data:
- Linear models (LR_EN, LR_L1, LinSVM_cal): Standardized absolute coefficients
- Tree models (RF, XGBoost): Built-in feature importances (Gini/gain)

All importance values are extracted from fitted models and aggregated across outer CV folds
to produce robust, unbiased importance estimates. This approach avoids data leakage by
using only OOF data for importance computation.

Key functions:
    extract_linear_importance: Extract standardized |coef| from linear models
    extract_tree_importance: Extract feature_importances_ from tree models
    extract_importance_from_model: Unified dispatcher based on model_name
    aggregate_fold_importances: Aggregate importance across CV folds

Design notes:
    - Handles sklearn Pipeline wrappers (preprocessing, feature selection)
    - Handles CalibratedClassifierCV wrappers (averages across calibration folds)
    - Returns empty DataFrame on errors rather than raising exceptions
    - Uses logging for diagnostics
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ..data.schema import ModelName

logger = logging.getLogger(__name__)

__all__ = [
    "extract_linear_importance",
    "extract_tree_importance",
    "extract_importance_from_model",
    "aggregate_fold_importances",
]


def _get_final_feature_names(pipeline: Pipeline) -> np.ndarray | None:
    """Extract feature names from pipeline after all transformations.

    Handles sklearn pipelines with preprocessing and feature selection steps.

    Args:
        pipeline: Fitted sklearn Pipeline

    Returns:
        Array of feature names after all transformations, or None if unavailable
    """
    if not isinstance(pipeline, Pipeline):
        logger.warning("Expected Pipeline object, got %s", type(pipeline).__name__)
        return None

    # Start with preprocessor feature names
    if "pre" in pipeline.named_steps:
        pre = pipeline.named_steps["pre"]
        if hasattr(pre, "get_feature_names_out"):
            feature_names = pre.get_feature_names_out()
        else:
            logger.warning("Preprocessor does not have get_feature_names_out method")
            return None
    else:
        logger.warning("Pipeline does not have 'pre' step")
        return None

    # Apply K-best selector mask if present
    if "sel" in pipeline.named_steps:
        sel = pipeline.named_steps["sel"]
        if hasattr(sel, "get_support"):
            support = sel.get_support()
            feature_names = feature_names[support]
        else:
            logger.warning("Selector 'sel' does not have get_support method")

    # Apply model-specific selector mask if present
    if "model_sel" in pipeline.named_steps:
        model_sel = pipeline.named_steps["model_sel"]
        if hasattr(model_sel, "get_support"):
            support = model_sel.get_support()
            feature_names = feature_names[support]
        else:
            logger.warning("Selector 'model_sel' does not have get_support method")

    return feature_names


def extract_linear_importance(
    estimator,
    feature_names: np.ndarray | list[str],
) -> pd.DataFrame:
    """Extract standardized absolute coefficients from linear models.

    This function handles:
    - Standard linear models (LogisticRegression, LinearSVC)
    - CalibratedClassifierCV wrappers (averages across calibration folds)
    - Pipeline wrappers (extracts final classifier step)

    Args:
        estimator: Fitted sklearn estimator (may be wrapped in Pipeline or CalibratedClassifierCV)
        feature_names: Feature names corresponding to coefficients

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - importance: float, standardized |coef|
            - importance_type: str, always "abs_coef"

        Empty DataFrame if coefficients cannot be extracted.

    Notes:
        Standardization is performed by dividing |coef| by the sum of all |coef|,
        producing importance values that sum to 1.0. This makes importance
        values comparable across different models and folds.
    """
    # Handle Pipeline wrapper
    if isinstance(estimator, Pipeline):
        if "clf" not in estimator.named_steps:
            logger.warning("Pipeline does not have 'clf' step; cannot extract coefficients")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator

    # Handle CalibratedClassifierCV wrapper (e.g., LinSVM_cal)
    if isinstance(clf, CalibratedClassifierCV):
        if not hasattr(clf, "calibrated_classifiers_"):
            logger.warning("CalibratedClassifierCV is not fitted; cannot extract coefficients")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])

        # Average coefficients across calibration folds
        coefs_list = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                est = getattr(cc, "base_estimator", None)  # Older sklearn versions
            if est and hasattr(est, "coef_"):
                coefs_list.append(est.coef_.ravel())

        if not coefs_list:
            logger.warning(
                "No coefficients found in CalibratedClassifierCV calibrated_classifiers_"
            )
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])

        coefs = np.mean(np.vstack(coefs_list), axis=0)

    elif hasattr(clf, "coef_"):
        # Standard linear model
        coefs = clf.coef_.ravel()
    else:
        logger.warning(
            "Estimator %s does not have coef_ attribute; cannot extract coefficients",
            type(clf).__name__,
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Validate dimensions
    feature_names = np.asarray(feature_names)
    if len(feature_names) != len(coefs):
        logger.warning(
            "Feature names length (%d) != coef length (%d); cannot extract importance",
            len(feature_names),
            len(coefs),
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Compute standardized |coef|
    abs_coefs = np.abs(coefs)
    coef_sum = np.sum(abs_coefs)
    if coef_sum == 0:
        logger.warning("All coefficients are zero; cannot standardize")
        standardized = abs_coefs
    else:
        standardized = abs_coefs / coef_sum

    # Build DataFrame
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": standardized,
            "importance_type": "abs_coef",
        }
    )


def extract_tree_importance(
    estimator,
    feature_names: np.ndarray | list[str],
) -> pd.DataFrame:
    """Extract feature importances from tree-based models.

    This function handles:
    - RandomForestClassifier (Gini importance)
    - XGBClassifier (gain importance)
    - Pipeline wrappers (extracts final classifier step)

    Args:
        estimator: Fitted sklearn estimator (may be wrapped in Pipeline)
        feature_names: Feature names corresponding to importances

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - importance: float, normalized importance (sums to 1.0)
            - importance_type: str, "gini" for RF, "gain" for XGBoost

        Empty DataFrame if importances cannot be extracted.

    Notes:
        Tree importances are already normalized by sklearn/XGBoost to sum to 1.0.
        For RF, this is mean decrease in impurity (Gini importance).
        For XGBoost, this is total gain from splits on each feature.
    """
    # Handle Pipeline wrapper
    if isinstance(estimator, Pipeline):
        if "clf" not in estimator.named_steps:
            logger.warning("Pipeline does not have 'clf' step; cannot extract importances")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator

    # Extract feature_importances_
    if not hasattr(clf, "feature_importances_"):
        logger.warning(
            "Estimator %s does not have feature_importances_ attribute; cannot extract importance",
            type(clf).__name__,
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    importances = clf.feature_importances_

    # Validate dimensions
    feature_names = np.asarray(feature_names)
    if len(feature_names) != len(importances):
        logger.warning(
            "Feature names length (%d) != importance length (%d); cannot extract importance",
            len(feature_names),
            len(importances),
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Determine importance type based on estimator class
    clf_type = type(clf).__name__
    if "RandomForest" in clf_type:
        importance_type = "gini"
    elif "XGB" in clf_type:
        importance_type = "gain"
    else:
        importance_type = "tree"  # Generic fallback

    # Build DataFrame
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "importance_type": importance_type,
        }
    )


def extract_importance_from_model(
    estimator,
    model_name: Literal["LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"],
    feature_names: np.ndarray | list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature importance from fitted model (dispatcher function).

    This function dispatches to the appropriate extraction method based on model_name:
    - Linear models (LR_EN, LR_L1, LinSVM_cal) -> extract_linear_importance
    - Tree models (RF, XGBoost) -> extract_tree_importance

    If estimator is a Pipeline, feature names are extracted automatically from the
    pipeline's preprocessing and feature selection steps.

    Args:
        estimator: Fitted sklearn estimator (may be Pipeline)
        model_name: Model identifier (one of: "LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost")
        feature_names: Optional feature names. If None and estimator is Pipeline,
                      feature names are extracted from pipeline steps.

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - importance: float, importance value
            - importance_type: str, "abs_coef", "gini", or "gain"

        Empty DataFrame if importance cannot be extracted.

    Raises:
        ValueError: If model_name is unknown

    Examples:
        >>> # Extract from Pipeline
        >>> df = extract_importance_from_model(fitted_pipeline, "LR_EN")
        >>>
        >>> # Extract with explicit feature names
        >>> df = extract_importance_from_model(fitted_model, "RF", feature_names=["age", "bmi"])
    """
    # Extract feature names from Pipeline if needed
    if feature_names is None:
        if isinstance(estimator, Pipeline):
            feature_names = _get_final_feature_names(estimator)
            if feature_names is None:
                logger.warning("Could not extract feature names from Pipeline")
                return pd.DataFrame(columns=["feature", "importance", "importance_type"])
        else:
            logger.warning("feature_names is required for non-Pipeline estimators")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Dispatch based on model type
    model_name_clean = str(model_name).strip()

    if model_name_clean in (ModelName.LR_EN, ModelName.LR_L1, ModelName.LinSVM_cal):
        return extract_linear_importance(estimator, feature_names)

    elif model_name_clean in (ModelName.RF, ModelName.XGBoost):
        return extract_tree_importance(estimator, feature_names)

    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Expected one of: LR_EN, LR_L1, LinSVM_cal, RF, XGBoost"
        )


def aggregate_fold_importances(fold_importances: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate feature importances across CV folds.

    Computes mean, std, and non-zero count for each feature across folds.
    This produces robust importance estimates that account for variability
    across different train/test splits.

    Args:
        fold_importances: List of importance DataFrames (one per fold)
                         Each DataFrame must have columns: feature, importance, importance_type

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - mean_importance: float, mean importance across folds
            - std_importance: float, standard deviation across folds
            - n_folds_nonzero: int, number of folds where importance > 0
            - importance_type: str, importance type (taken from first fold)

        Sorted by mean_importance descending.
        Empty DataFrame if input list is empty.

    Notes:
        - Features present in some folds but not others are assigned importance=0 for missing folds
        - Features with higher n_folds_nonzero are more consistently selected
        - Standard deviation indicates importance stability across folds

    Examples:
        >>> fold_dfs = [
        ...     extract_importance_from_model(pipeline1, "LR_EN"),
        ...     extract_importance_from_model(pipeline2, "LR_EN"),
        ... ]
        >>> agg_df = aggregate_fold_importances(fold_dfs)
        >>> print(agg_df.head())
    """
    if not fold_importances:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "std_importance",
                "n_folds_nonzero",
                "importance_type",
            ]
        )

    # Filter out empty DataFrames
    valid_dfs = [df for df in fold_importances if not df.empty]
    if not valid_dfs:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "std_importance",
                "n_folds_nonzero",
                "importance_type",
            ]
        )

    # Collect all unique features
    all_features = set()
    for df in valid_dfs:
        all_features.update(df["feature"].values)

    # Determine importance type (should be consistent across folds)
    importance_type = (
        valid_dfs[0]["importance_type"].iloc[0]
        if "importance_type" in valid_dfs[0].columns
        else "unknown"
    )

    # Build importance matrix: rows=features, cols=folds
    importance_matrix = {}
    for feature in all_features:
        importance_matrix[feature] = []
        for df in valid_dfs:
            if feature in df["feature"].values:
                val = float(df[df["feature"] == feature]["importance"].iloc[0])
            else:
                val = 0.0
            importance_matrix[feature].append(val)

    # Compute statistics
    results = []
    for feature, values in importance_matrix.items():
        values_array = np.array(values)
        results.append(
            {
                "feature": feature,
                "mean_importance": float(np.mean(values_array)),
                "std_importance": (
                    float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
                ),
                "n_folds_nonzero": int(np.sum(values_array > 0)),
                "importance_type": importance_type,
            }
        )

    # Build DataFrame and sort
    result_df = pd.DataFrame(results).sort_values(
        "mean_importance", ascending=False, ignore_index=True
    )

    return result_df
