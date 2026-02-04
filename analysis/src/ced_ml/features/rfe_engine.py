"""Core RFE elimination loop, evaluation engine, and utility functions.

Extracted from rfe.py to reduce module complexity. Contains the core
recursive feature elimination algorithm, per-size evaluation logic, feature
importance computation, and panel analysis utilities.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ced_ml.data.schema import ModelName
from ced_ml.metrics.discrimination import (
    alpha_sensitivity_at_specificity,
    auroc,
    compute_brier_score,
    prauc,
)
from ced_ml.utils.feature_names import extract_protein_name

logger = logging.getLogger(__name__)


def compute_eval_sizes(
    max_size: int,
    min_size: int,
    strategy: str = "geometric",
) -> list[int]:
    """Compute panel sizes to evaluate based on elimination strategy.

    Args:
        max_size: Starting panel size.
        min_size: Smallest panel size to evaluate.
        strategy: One of:
            - "geometric": Evaluate at powers of 2 (100, 50, 25, 12, 6, ...). [DEFAULT]
            - "fine": More granular than geometric, evaluates intermediate points.
              Adds quarter-steps between geometric levels (e.g., 100, 75, 50, 37, 25, 18, 12, ...).
            - "linear": Evaluate every size from max to min (slowest, most data).

    Returns:
        List of panel sizes to evaluate, sorted descending.

    Raises:
        ValueError: If strategy is not one of the valid options.

    Example:
        >>> compute_eval_sizes(100, 5, "geometric")
        [100, 50, 25, 12, 6, 5]
        >>> compute_eval_sizes(100, 5, "fine")
        [100, 75, 50, 37, 25, 18, 12, 9, 6, 5]
    """
    # Validate strategy
    valid_strategies = {"geometric", "fine", "linear"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid step_strategy: '{strategy}'. " f"Must be one of {sorted(valid_strategies)}."
        )

    if max_size <= min_size:
        return [max_size]

    if strategy == "linear":
        return list(range(max_size, min_size - 1, -1))

    if strategy == "fine":
        # Fine-grained strategy: geometric + quarter-step interpolation
        sizes = []
        current = max_size
        while current > min_size:
            sizes.append(current)
            # Add intermediate point at 3/4 of current (between current and current//2)
            quarter_step = max(min_size, int(current * 0.75))
            if quarter_step not in sizes and quarter_step > min_size:
                sizes.append(quarter_step)
            # Add point at half
            half_step = max(min_size, current // 2)
            if half_step not in sizes and half_step > min_size:
                sizes.append(half_step)
            current = half_step
            if current in sizes and current <= min_size:
                break

        # Ensure min_size is included
        if min_size not in sizes:
            sizes.append(min_size)

        return sorted(set(sizes), reverse=True)

    # Geometric: powers of 2 plus min_size
    sizes = []
    current = max_size
    while current > min_size:
        sizes.append(current)
        current = max(min_size, current // 2)
        if current in sizes:  # Prevent infinite loop if we hit min_size
            break

    # Ensure min_size is included
    if min_size not in sizes:
        sizes.append(min_size)

    return sorted(set(sizes), reverse=True)


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

    # Get classifier from pipeline
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        logger.warning("No 'clf' step found in pipeline")
        return importance

    # Get feature names after preprocessing
    pre = pipeline.named_steps.get("pre")
    if pre is None or not hasattr(pre, "get_feature_names_out"):
        logger.warning("No preprocessor with feature names found")
        return importance

    feature_names = list(pre.get_feature_names_out())

    # Apply K-best mask if present
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

    # Handle CalibratedClassifierCV wrapper for LinSVM
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

    # Map to protein names with absolute values
    for name, c in zip(feature_names, coefs, strict=False):
        orig = extract_protein_name(name)
        if orig and orig in protein_cols:
            # Use absolute value for importance (magnitude matters, not direction)
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

    # Aggregate importance per protein (handles multi-feature transforms)
    for name, imp in zip(feature_names, importances, strict=False):
        if not np.isfinite(imp):
            continue
        orig = extract_protein_name(name)
        if orig and orig in protein_cols:
            importance[orig] = importance.get(orig, 0.0) + float(max(0, imp))

    return importance


def find_recommended_panels(
    curve: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> dict[str, int]:
    """Find smallest panel sizes maintaining various AUROC thresholds.

    Args:
        curve: List of dicts with keys "size" and "auroc_val".
        thresholds: Fractions of max AUROC to target (default: [0.95, 0.90, 0.85]).

    Returns:
        Dict with keys like "min_size_95pct" -> panel size, plus "knee_point".

    Example:
        >>> curve = [
        ...     {"size": 100, "auroc_val": 0.90},
        ...     {"size": 50, "auroc_val": 0.88},
        ...     {"size": 25, "auroc_val": 0.85},
        ... ]
        >>> find_recommended_panels(curve)
        {'min_size_95pct': 50, 'min_size_90pct': 25, 'knee_point': 50, ...}
    """
    if thresholds is None:
        thresholds = [0.95, 0.90, 0.85]

    if not curve:
        return {}

    max_auroc = max(p["auroc_val"] for p in curve)
    recommended: dict[str, int] = {}

    for thresh in thresholds:
        target = max_auroc * thresh
        # Find smallest panel meeting threshold
        valid = [p for p in curve if p["auroc_val"] >= target]
        if valid:
            smallest = min(valid, key=lambda x: x["size"])
            key = f"min_size_{int(thresh * 100)}pct"
            recommended[key] = smallest["size"]

    # Knee point detection (elbow method)
    recommended["knee_point"] = detect_knee_point(curve)

    return recommended


def detect_knee_point(curve: list[dict[str, Any]]) -> int:
    """Detect knee point (elbow) in the AUROC vs size curve.

    Uses the perpendicular distance method: finds the point with maximum
    distance from the line connecting the first and last points.

    Args:
        curve: List of dicts with "size" and "auroc_val".

    Returns:
        Panel size at the knee point.
    """
    if len(curve) < 3:
        return curve[0]["size"] if curve else 0

    # Sort by size descending
    sorted_curve = sorted(curve, key=lambda x: -x["size"])

    # Normalize to [0, 1] for distance calculation
    sizes = np.array([p["size"] for p in sorted_curve], dtype=float)
    aurocs = np.array([p["auroc_val"] for p in sorted_curve], dtype=float)

    # Normalize
    size_range = sizes.max() - sizes.min()
    auroc_range = aurocs.max() - aurocs.min()

    if size_range == 0 or auroc_range == 0:
        return sorted_curve[0]["size"]

    x = (sizes - sizes.min()) / size_range
    y = (aurocs - aurocs.min()) / auroc_range

    # Line from first to last point
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # Perpendicular distance from each point to the line
    # d = |ax + by + c| / sqrt(a^2 + b^2) where line is ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    denom = np.sqrt(a**2 + b**2)

    if denom == 0:
        return sorted_curve[0]["size"]

    distances = np.abs(a * x + b * y + c) / denom

    # Find maximum distance point
    knee_idx = int(np.argmax(distances))
    return int(sorted_curve[knee_idx]["size"])


def cluster_correlated_proteins_for_rfe(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
    selection_freq: dict[str, float] | None = None,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
) -> tuple[list[str], dict[str, dict]]:
    """Cluster correlated proteins and select representatives for RFE.

    Uses graph-based connected components (from corr_prune) to group
    proteins with |correlation| >= threshold, then selects one
    representative per cluster using composite criterion: stability
    frequency (primary) + Mann-Whitney p-value (tiebreak).

    All correlation analysis uses TRAIN data only.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    protein_cols : list[str]
        Input proteins to cluster.
    selection_freq : dict[str, float] or None
        Selection frequency from CV folds {protein: frequency}.
        If None, all proteins get equal weight.
    corr_threshold : float
        Absolute correlation threshold for clustering (0.0-1.0).
    corr_method : str
        Correlation method ("spearman" or "pearson").

    Returns
    -------
    representative_proteins : list[str]
        Cluster representatives to pass to RFE.
    cluster_map : dict[str, dict]
        Mapping representative -> {cluster_id, cluster_size, members}.
        Empty if clustering was skipped.
    """
    from ced_ml.features.corr_prune import prune_correlated_proteins

    if corr_threshold >= 1.0 or len(protein_cols) < 2:
        logger.info("Skipping correlation clustering (threshold >= 1.0 or < 2 proteins)")
        return list(protein_cols), {}

    df_map, kept_proteins = prune_correlated_proteins(
        df=X_train,
        y=y_train,
        proteins=protein_cols,
        selection_freq=selection_freq,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        tiebreak_method="freq_then_univariate",
    )

    if df_map.empty:
        return list(protein_cols), {}

    # Build cluster_map from df_map
    cluster_map: dict[str, dict] = {}
    for rep in kept_proteins:
        comp_rows = df_map[df_map["rep_protein"] == rep]
        if comp_rows.empty:
            logger.warning(f"No component rows found for representative '{rep}'; skipping")
            continue
        comp_id = int(comp_rows["component_id"].iloc[0])
        members = sorted(comp_rows["protein"].tolist())
        cluster_map[rep] = {
            "cluster_id": comp_id,
            "cluster_size": len(members),
            "members": members,
        }

    n_multi = sum(1 for v in cluster_map.values() if v["cluster_size"] > 1)
    logger.info(
        f"Clustered {len(protein_cols)} proteins into {len(cluster_map)} clusters "
        f"({len(cluster_map) - n_multi} singletons, {n_multi} multi-protein)"
    )
    if n_multi > 0:
        top_clusters = sorted(
            [(rep, info) for rep, info in cluster_map.items() if info["cluster_size"] > 1],
            key=lambda x: -x[1]["cluster_size"],
        )[:5]
        for rep, info in top_clusters:
            preview = ", ".join(info["members"][:3])
            ellipsis = "..." if len(info["members"]) > 3 else ""
            logger.info(f"  {rep}: {info['cluster_size']} members ({preview}{ellipsis})")

    return kept_proteins, cluster_map


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
    y_train: np.ndarray,
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
    import optuna

    from ced_ml.cli.train import build_preprocessor
    from ced_ml.models.hyperparams import get_rfe_tune_space
    from ced_ml.models.optuna_search import OptunaSearchCV

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

    search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=tune_space,
        n_trials=n_trials,
        scoring="roc_auc",
        cv=cv_folds,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        direction="maximize",
        sampler="tpe",
        sampler_seed=random_state,
        pruner="hyperband",
        verbose=0,
    )
    search.fit(X_train, y_train)

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


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap percentile confidence interval for AUROC.

    Returns:
        Tuple of (std, ci_low, ci_high) where ci_low/ci_high are the
        percentile-based bounds at the requested confidence level.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]

        if len(np.unique(y_t)) < 2:
            continue

        try:
            aurocs.append(auroc(y_t, y_p))
        except Exception as e:
            logger.debug("Bootstrap AUROC calculation skipped: %s", e)
            continue

    if not aurocs:
        return 0.0, 0.0, 0.0

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(aurocs, 100 * alpha / 2))
    ci_high = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return float(np.std(aurocs)), ci_low, ci_high


def evaluate_panel_size(
    pipeline: Pipeline,
    X_train_subset: pd.DataFrame,
    y_train: np.ndarray,
    X_val_subset: pd.DataFrame,
    y_val: np.ndarray,
    cv_folds: int,
    random_state: int,
    panel_size: int,
) -> dict[str, float]:
    """Evaluate metrics for a given panel size.

    Args:
        pipeline: Fitted pipeline to evaluate.
        X_train_subset: Training features subset.
        y_train: Training labels.
        X_val_subset: Validation features subset.
        y_val: Validation labels.
        cv_folds: CV folds for OOF estimation (0 to skip).
        random_state: Random seed.
        panel_size: Current panel size (for logging).

    Returns:
        Dict with all evaluation metrics.
    """
    metrics: dict[str, float] = {}

    if cv_folds > 1:
        logger.info(f"  Running {cv_folds}-fold CV for size {panel_size}...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        oof_probs = cross_val_predict(
            clone(pipeline),
            X_train_subset,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        metrics["auroc_cv"] = auroc(y_train, oof_probs)
        metrics["prauc_cv"] = prauc(y_train, oof_probs)
        metrics["brier_cv"] = compute_brier_score(y_train, oof_probs)
        metrics["sens_at_95spec_cv"] = alpha_sensitivity_at_specificity(
            y_train, oof_probs, target_specificity=0.95
        )
        logger.info(
            f"  CV AUROC: {metrics['auroc_cv']:.4f}, "
            f"PR-AUC: {metrics['prauc_cv']:.4f}, "
            f"Brier: {metrics['brier_cv']:.4f}"
        )
        auroc_cv_std, _, _ = bootstrap_auroc_ci(y_train, oof_probs, n_bootstrap=50)
        metrics["auroc_cv_std"] = auroc_cv_std
    else:
        train_probs = pipeline.predict_proba(X_train_subset)[:, 1]
        metrics["auroc_cv"] = auroc(y_train, train_probs)
        metrics["prauc_cv"] = prauc(y_train, train_probs)
        metrics["brier_cv"] = compute_brier_score(y_train, train_probs)
        metrics["sens_at_95spec_cv"] = alpha_sensitivity_at_specificity(
            y_train, train_probs, target_specificity=0.95
        )
        metrics["auroc_cv_std"] = 0.0
        logger.info(
            f"  Train AUROC: {metrics['auroc_cv']:.4f}, "
            f"PR-AUC: {metrics['prauc_cv']:.4f}, "
            f"Brier: {metrics['brier_cv']:.4f}"
        )

    logger.info("  Computing validation metrics...")
    val_probs = pipeline.predict_proba(X_val_subset)[:, 1]
    metrics["auroc_val"] = auroc(y_val, val_probs)
    metrics["prauc_val"] = prauc(y_val, val_probs)
    metrics["brier_val"] = compute_brier_score(y_val, val_probs)
    metrics["sens_at_95spec_val"] = alpha_sensitivity_at_specificity(
        y_val, val_probs, target_specificity=0.95
    )
    auroc_val_std, auroc_val_ci_low, auroc_val_ci_high = bootstrap_auroc_ci(
        y_val, val_probs, n_bootstrap=200
    )
    metrics["auroc_val_std"] = auroc_val_std
    metrics["auroc_val_ci_low"] = auroc_val_ci_low
    metrics["auroc_val_ci_high"] = auroc_val_ci_high

    logger.info(
        f"[RFE k={panel_size}] Val AUROC={metrics['auroc_val']:.3f} "
        f"(95% CI: {auroc_val_ci_low:.3f}-{auroc_val_ci_high:.3f}), "
        f"PR-AUC={metrics['prauc_val']:.3f}, Brier={metrics['brier_val']:.4f}"
    )

    return metrics


def elimination_loop(
    current_proteins: list[str],
    target_size: int,
    min_size: int,
    base_pipeline: Pipeline,
    cat_cols: list[str],
    meta_num_cols: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    random_state: int,
    n_perm_repeats: int,
    feature_ranking: dict[str, int],
    elimination_order: int,
) -> tuple[list[str], dict[str, int], int]:
    """Execute elimination loop to reduce panel to target size.

    Args:
        current_proteins: List of proteins currently in panel.
        target_size: Target panel size.
        min_size: Minimum allowed panel size.
        base_pipeline: Base pipeline to clone classifier from.
        cat_cols: Categorical metadata columns.
        meta_num_cols: Numeric metadata columns.
        X_train: Training features.
        y_train: Training labels.
        model_name: Model identifier.
        random_state: Random seed.
        n_perm_repeats: Permutation repeats for importance.
        feature_ranking: Dict to update with eliminated proteins.
        elimination_order: Current elimination order counter.

    Returns:
        Tuple of (updated_proteins, updated_ranking, updated_order).
    """
    from ced_ml.features.rfe import compute_feature_importance

    proteins_to_eliminate = len(current_proteins) - target_size
    logger.info(f"Eliminating {proteins_to_eliminate} proteins to reach target size {target_size}")

    elimination_count = 0
    while len(current_proteins) > target_size and len(current_proteins) > min_size:
        elimination_count += 1

        if elimination_count % 10 == 0:
            logger.info(
                f"  Elimination progress: {elimination_count}/{proteins_to_eliminate} "
                f"({elimination_count/proteins_to_eliminate*100:.1f}%), "
                f"current panel size: {len(current_proteins)}"
            )

        try:
            pipeline = build_lightweight_pipeline(base_pipeline, current_proteins, cat_cols)
        except Exception as e:
            logger.error(f"Failed to build pipeline: {e}")
            break

        feature_cols = current_proteins + cat_cols + meta_num_cols
        X_train_subset = X_train[feature_cols]

        try:
            pipeline.fit(X_train_subset, y_train)
        except Exception as e:
            logger.error(f"Pipeline fit failed at size {len(current_proteins)}: {e}")
            break

        importances = compute_feature_importance(
            pipeline,
            model_name,
            current_proteins,
            X_train_subset,
            y_train,
            random_state,
            n_perm_repeats,
        )

        protein_importances = {p: importances.get(p, 0.0) for p in current_proteins}
        worst_protein = min(protein_importances, key=protein_importances.get)

        feature_ranking[worst_protein] = elimination_order
        current_proteins.remove(worst_protein)
        elimination_order += 1

        logger.debug(
            f"Eliminated {worst_protein} (importance={protein_importances[worst_protein]:.4f}), "
            f"panel size now {len(current_proteins)}"
        )

    return current_proteins, feature_ranking, elimination_order


def run_elimination_with_evaluation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_pipeline: Pipeline,
    model_name: str,
    current_proteins: list[str],
    cat_cols: list[str],
    meta_num_cols: list[str],
    eval_sizes: list[int],
    min_size: int,
    cv_folds: int,
    random_state: int,
    n_perm_repeats: int,
    can_retune: bool,
    retune_n_trials: int,
    retune_cv_folds: int,
    retune_n_jobs: int,
    rfe_tune_spaces: dict[str, dict[str, dict]] | None,
    min_auroc_frac: float,
) -> tuple[list[dict[str, Any]], dict[str, int], float, list[dict]]:
    """Run RFE elimination loop with per-size evaluation.

    Core engine that executes elimination, evaluation, and optional tuning
    at each target panel size.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        base_pipeline: Base pipeline to clone from.
        model_name: Model identifier.
        current_proteins: Starting protein list.
        cat_cols: Categorical metadata columns.
        meta_num_cols: Numeric metadata columns.
        eval_sizes: List of panel sizes to evaluate.
        min_size: Minimum panel size.
        cv_folds: CV folds for OOF estimation.
        random_state: Random seed.
        n_perm_repeats: Permutation repeats.
        can_retune: Whether per-k tuning is enabled.
        retune_n_trials: Optuna trials for tuning.
        retune_cv_folds: CV folds for tuning.
        retune_n_jobs: Parallel jobs for tuning.
        rfe_tune_spaces: Optional config-driven tune spaces.
        min_auroc_frac: Early stopping threshold.

    Returns:
        Tuple of (curve, feature_ranking, max_auroc, all_best_params).
    """
    curve: list[dict[str, Any]] = []
    feature_ranking: dict[str, int] = {}
    elimination_order = 0
    max_auroc_seen = 0.0
    all_best_params: list[dict] = []

    start_time = time.time()
    total_eval_points = len(eval_sizes)
    eval_point_idx = 0
    total_tune_time = 0.0

    for target_size in eval_sizes:
        eval_point_idx += 1
        elapsed = time.time() - start_time
        logger.info(
            f"\n{'='*60}\n"
            f"RFE Progress: {eval_point_idx}/{total_eval_points} evaluation points "
            f"({eval_point_idx/total_eval_points*100:.1f}%)\n"
            f"Elapsed time: {elapsed/60:.1f} min\n"
            f"Target panel size: {target_size}\n"
            f"{'='*60}"
        )

        current_proteins, feature_ranking, elimination_order = elimination_loop(
            current_proteins=current_proteins,
            target_size=target_size,
            min_size=min_size,
            base_pipeline=base_pipeline,
            cat_cols=cat_cols,
            meta_num_cols=meta_num_cols,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            random_state=random_state,
            n_perm_repeats=n_perm_repeats,
            feature_ranking=feature_ranking,
            elimination_order=elimination_order,
        )

        if len(current_proteins) < min_size:
            logger.warning(
                f"Panel size {len(current_proteins)} below min_size {min_size}, stopping"
            )
            break

        logger.info(f"\nEvaluating panel size {len(current_proteins)}...")

        feature_cols = current_proteins + cat_cols + meta_num_cols
        X_train_subset = X_train[feature_cols]
        X_val_subset = X_val[feature_cols]

        try:
            eval_start = time.time()

            best_params = {}
            if can_retune:
                tune_start = time.time()
                pipeline, best_params = quick_tune_at_k(
                    model_name=model_name,
                    X_train=X_train_subset,
                    y_train=y_train,
                    feature_cols=feature_cols,
                    cat_cols=cat_cols,
                    cv_folds=retune_cv_folds,
                    n_trials=retune_n_trials,
                    n_jobs=retune_n_jobs,
                    random_state=random_state,
                    rfe_tune_spaces=rfe_tune_spaces,
                )
                total_tune_time += time.time() - tune_start
                all_best_params.append(best_params)
            else:
                pipeline = build_lightweight_pipeline(base_pipeline, current_proteins, cat_cols)
                pipeline.fit(X_train_subset, y_train)

            metrics = evaluate_panel_size(
                pipeline=pipeline,
                X_train_subset=X_train_subset,
                y_train=y_train,
                X_val_subset=X_val_subset,
                y_val=y_val,
                cv_folds=cv_folds,
                random_state=random_state,
                panel_size=len(current_proteins),
            )

            eval_elapsed = time.time() - eval_start
            logger.info(f"[RFE k={len(current_proteins)}] Completed in {eval_elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Evaluation failed at size {len(current_proteins)}: {e}")
            continue

        point = {
            "size": len(current_proteins),
            "auroc_cv": metrics["auroc_cv"],
            "auroc_cv_std": metrics["auroc_cv_std"],
            "auroc_val": metrics["auroc_val"],
            "auroc_val_std": metrics["auroc_val_std"],
            "auroc_val_ci_low": metrics["auroc_val_ci_low"],
            "auroc_val_ci_high": metrics["auroc_val_ci_high"],
            "prauc_cv": metrics["prauc_cv"],
            "prauc_val": metrics["prauc_val"],
            "brier_cv": metrics["brier_cv"],
            "brier_val": metrics["brier_val"],
            "sens_at_95spec_cv": metrics["sens_at_95spec_cv"],
            "sens_at_95spec_val": metrics["sens_at_95spec_val"],
            "proteins": list(current_proteins),
            "best_params": best_params,
        }
        curve.append(point)
        logger.info(
            f"\n  *** Panel size {len(current_proteins)} results: ***\n"
            f"    AUROC_cv:        {metrics['auroc_cv']:.4f} +/- {metrics['auroc_cv_std']:.4f}\n"
            f"    AUROC_val:       {metrics['auroc_val']:.4f} "
            f"(95% CI: {metrics['auroc_val_ci_low']:.4f}-{metrics['auroc_val_ci_high']:.4f})\n"
            f"    PR-AUC_val:      {metrics['prauc_val']:.4f}\n"
            f"    Brier_val:       {metrics['brier_val']:.4f}\n"
            f"    Sens@95%Spec:    {metrics['sens_at_95spec_val']:.4f}\n"
        )

        if metrics["auroc_val"] > max_auroc_seen:
            max_auroc_seen = metrics["auroc_val"]
            logger.info(f"  New maximum AUROC: {max_auroc_seen:.4f}")

        if metrics["auroc_val"] < max_auroc_seen * min_auroc_frac:
            logger.info(
                f"\n*** Early stopping triggered ***\n"
                f"AUROC {metrics['auroc_val']:.4f} < "
                f"{min_auroc_frac:.0%} of max {max_auroc_seen:.4f}\n"
            )
            break

    return curve, feature_ranking, max_auroc_seen, all_best_params
