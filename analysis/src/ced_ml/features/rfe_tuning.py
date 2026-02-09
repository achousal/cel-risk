"""Hyperparameter tuning utilities for RFE.

Provides functions for building pipelines, creating fresh estimators,
and running Optuna-based hyperparameter searches at specific panel sizes.
"""

import logging
import time

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ced_ml.models.hyperparams import get_rfe_tune_space
from ced_ml.models.optuna_search import OptunaSearchCV

logger = logging.getLogger(__name__)


def build_lightweight_pipeline(
    base_pipeline: Pipeline,
    protein_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Build a simplified pipeline for RFE evaluation.

    Clones the classifier from base_pipeline and builds a new pipeline
    with only the specified proteins (no screening, no k-best tuning).

    Args:
        base_pipeline: Original trained pipeline to clone classifier from.
        protein_cols: Proteins for this RFE iteration (unused but kept for API consistency).
        cat_cols: Categorical metadata columns.

    Returns:
        New unfitted Pipeline with [pre, clf] steps.
        Numeric columns pass through automatically via preprocessor.
    """
    from ced_ml.cli.train import build_preprocessor
    from ced_ml.models.calibration import OOFCalibratedModel

    pipeline = base_pipeline
    if isinstance(base_pipeline, OOFCalibratedModel):
        pipeline = base_pipeline.base_model

    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("Base pipeline has no 'clf' step")

    cloned_clf = clone(clf)
    preprocessor = build_preprocessor(cat_cols)

    return Pipeline([("pre", preprocessor), ("clf", cloned_clf)])


def make_fresh_estimator(model_name: str, random_state: int = 42) -> object:
    """Create a fresh classifier from registry defaults.

    Builds a new estimator instance (not cloned from a fitted model) suitable
    for hyperparameter tuning at a given panel size.

    Args:
        model_name: Model identifier (e.g., "LR_EN", "RF", "XGBoost").
        random_state: Random seed for the estimator.

    Returns:
        Unfitted sklearn-compatible estimator.
    """
    from ced_ml.models.registry import (
        build_linear_svm_calibrated,
        build_logistic_regression,
        build_random_forest,
        build_xgboost,
    )

    if model_name == "LR_EN":
        return build_logistic_regression(random_state=random_state)
    elif model_name == "LR_L1":
        return build_logistic_regression(l1_ratio=1.0, penalty="l1", random_state=random_state)
    elif model_name == "LinSVM_cal":
        return build_linear_svm_calibrated(random_state=random_state)
    elif model_name == "RF":
        return build_random_forest(n_estimators=300, random_state=random_state)
    elif model_name == "XGBoost":
        return build_xgboost(n_estimators=300, random_state=random_state)
    else:
        raise ValueError(f"Unknown model for RFE tuning: {model_name}")


def quick_tune_at_k(
    model_name: str,
    X_train: pd.DataFrame,
    y_train,
    feature_cols: list[str],
    cat_cols: list[str],
    cv_folds: int = 3,
    n_trials: int = 20,
    n_jobs: int = 1,
    random_state: int = 42,
    rfe_tune_spaces: dict[str, dict[str, dict]] | None = None,
) -> tuple[Pipeline, dict]:
    """Re-tune hyperparameters at a specific panel size k.

    Builds a fresh estimator, constructs a reduced search space, and runs
    OptunaSearchCV on the TRAIN set only. Returns the fitted pipeline and
    best hyperparameters found.

    Args:
        model_name: Model identifier (e.g., "LR_EN", "RF", "XGBoost").
        X_train: Training features (already subset to feature_cols).
        y_train: Training labels.
        feature_cols: Column names in X_train (for logging only).
        cat_cols: Categorical metadata columns.
        cv_folds: CV folds for the Optuna search.
        n_trials: Number of Optuna trials.
        n_jobs: Parallel jobs for Optuna CV evaluation.
        random_state: Random seed.
        rfe_tune_spaces: Optional config-driven search spaces from
            optimize_panel.yaml. Passed to get_rfe_tune_space.

    Returns:
        Tuple of (fitted Pipeline with best params, best_params dict).
    """
    from ced_ml.cli.train import build_preprocessor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tune_space = get_rfe_tune_space(model_name, config_overrides=rfe_tune_spaces)
    param_names = [k.replace("clf__", "") for k in tune_space]
    logger.info(
        f"[RFE k={len(feature_cols)}] Tuning {len(tune_space)} hyperparams "
        f"({', '.join(param_names)}) | {n_trials} trials, cv={cv_folds}, n_jobs={n_jobs}"
    )

    tune_start = time.time()

    clf = make_fresh_estimator(model_name, random_state=random_state)
    preprocessor = build_preprocessor(cat_cols)
    pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])

    def _run_optuna_search(search_n_jobs: int) -> OptunaSearchCV:
        search = OptunaSearchCV(
            estimator=pipeline,
            param_distributions=tune_space,
            n_trials=n_trials,
            scoring="roc_auc",
            cv=cv_folds,
            n_jobs=search_n_jobs,
            random_state=random_state,
            refit=True,
            direction="maximize",
            sampler="tpe",
            sampler_seed=random_state,
            pruner="hyperband",
            verbose=0,
        )
        search.fit(X_train, y_train)
        return search

    try:
        search = _run_optuna_search(n_jobs)
    except (PermissionError, NotImplementedError, OSError) as exc:
        if n_jobs == 1:
            raise
        logger.warning(
            "[RFE k=%d] Parallel Optuna CV unavailable (%s). Retrying with n_jobs=1.",
            len(feature_cols),
            exc,
        )
        search = _run_optuna_search(1)
    except RuntimeError as exc:
        if n_jobs == 1 or "Optuna trials failed" not in str(exc):
            raise
        logger.warning(
            "[RFE k=%d] Optuna search failed with n_jobs=%d (%s). Retrying with n_jobs=1.",
            len(feature_cols),
            n_jobs,
            exc,
        )
        search = _run_optuna_search(1)

    best_params = search.best_params_
    best_score = search.best_score_
    tune_elapsed = time.time() - tune_start

    param_summary = ", ".join(
        (
            f"{k.replace('clf__', '')}={v:.4g}"
            if isinstance(v, float)
            else f"{k.replace('clf__', '')}={v}"
        )
        for k, v in best_params.items()
    )
    logger.info(
        f"[RFE k={len(feature_cols)}] Best: {param_summary} " f"(CV AUROC={best_score:.3f})"
    )
    logger.info(f"[RFE k={len(feature_cols)}] Tuning completed in {tune_elapsed:.1f}s")

    return search.best_estimator_, best_params
