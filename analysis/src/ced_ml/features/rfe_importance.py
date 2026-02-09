"""Feature importance extraction for RFE.

Provides model-specific importance calculation strategies:
- Linear models: coefficient-based importance
- Tree models: permutation importance
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ced_ml.data.schema import ModelName
from ced_ml.utils.feature_names import extract_protein_name

logger = logging.getLogger(__name__)


def compute_feature_importance(
    pipeline: Pipeline,
    model_name: str,
    protein_cols: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42,
    n_perm_repeats: int = 5,
) -> dict[str, float]:
    """Extract feature importances for ranking during RFE.

    Uses model-specific strategy:
    - Linear models (LR_EN, LR_L1, LinSVM_cal): Absolute coefficient values.
    - Tree models (RF, XGBoost): Permutation importance.

    Args:
        pipeline: Fitted sklearn Pipeline with steps [pre, clf] or [screen, pre, sel, clf].
        model_name: Model identifier.
        protein_cols: List of protein column names in current panel.
        X: Feature DataFrame.
        y: Target labels.
        random_state: Random seed for permutation importance.
        n_perm_repeats: Number of permutation repeats (trees only).

    Returns:
        Dict mapping protein -> importance score (higher = more important).
        Proteins not found return 0.0.
    """
    importance: dict[str, float] = dict.fromkeys(protein_cols, 0.0)

    clf = pipeline.named_steps.get("clf")
    if clf is None:
        logger.warning("No 'clf' step found in pipeline")
        return importance

    pre = pipeline.named_steps.get("pre")
    if pre is None or not hasattr(pre, "get_feature_names_out"):
        logger.warning("No preprocessor with feature names found")
        return importance

    feature_names = list(pre.get_feature_names_out())

    if "sel" in pipeline.named_steps:
        support = pipeline.named_steps["sel"].get_support()
        feature_names = [f for f, s in zip(feature_names, support, strict=False) if s]

    # Strategy 1: Coefficient-based (linear models)
    if model_name in (ModelName.LR_EN, ModelName.LR_L1, ModelName.LinSVM_cal):
        importance = _importance_from_coefficients(clf, feature_names, protein_cols, model_name)
        if importance:
            return importance

    # Strategy 2: Permutation importance (tree models or fallback)
    importance = _importance_from_permutation(
        pipeline, X, y, feature_names, protein_cols, random_state, n_perm_repeats
    )
    return importance


def _importance_from_coefficients(
    clf: Any,
    feature_names: list[str],
    protein_cols: list[str],
    model_name: str,
) -> dict[str, float]:
    """Extract importance from linear model coefficients."""
    importance: dict[str, float] = dict.fromkeys(protein_cols, 0.0)

    coefs = None

    if model_name == ModelName.LinSVM_cal and hasattr(clf, "calibrated_classifiers_"):
        coefs_list = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est and hasattr(est, "coef_"):
                coefs_list.append(est.coef_.ravel())
        if coefs_list:
            coefs = np.mean(np.vstack(coefs_list), axis=0)

    elif hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()

    if coefs is None or len(coefs) != len(feature_names):
        return {}

    for name, c in zip(feature_names, coefs, strict=False):
        orig = extract_protein_name(name)
        if orig and orig in protein_cols:
            importance[orig] = abs(float(c))

    return importance


def _importance_from_permutation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    protein_cols: list[str],
    random_state: int,
    n_repeats: int,
) -> dict[str, float]:
    """Extract importance via permutation importance."""
    importance: dict[str, float] = dict.fromkeys(protein_cols, 0.0)

    try:
        perm_result = permutation_importance(
            pipeline,
            X,
            y,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        importances = perm_result.importances_mean
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        return importance

    if len(importances) != len(feature_names):
        logger.warning(
            f"Importance length ({len(importances)}) != feature names ({len(feature_names)})"
        )
        return importance

    for name, imp in zip(feature_names, importances, strict=False):
        if not np.isfinite(imp):
            continue
        orig = extract_protein_name(name)
        if orig and orig in protein_cols:
            importance[orig] = importance.get(orig, 0.0) + float(max(0, imp))

    return importance
