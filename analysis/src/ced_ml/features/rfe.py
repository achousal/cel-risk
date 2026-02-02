"""Post-hoc RFE for clinical deployment panel optimization.

Identifies the smallest protein panel that maintains acceptable AUROC through
iterative feature removal. Runs AFTER training on a single trained model to
generate Pareto curves showing size-performance trade-offs for clinical
decision-making.

Design:
- Uses validation set AUROC for elimination decisions (test reserved for final eval)
- Supports geometric step strategy for efficiency (~45× faster than nested RFECV)
- Model-specific importance: coefficients for linear, permutation for trees
- Outputs: curve CSV, recommendations JSON, feature ranking, Pareto plots

Complementary to features/nested_rfe.py (robust feature discovery):
- rfe.py (this module): Clinical deployment after training
  → "What's the minimum panel size maintaining AUROC ≥ 0.90?"
  → Use for: Stakeholder decisions, cost-benefit analysis, rapid iteration
  → Output: Pareto curve (panel size vs. AUROC)
  → Speed: Fast (single model evaluation per size)

- nested_rfe.py (during training): Scientific discovery within CV
  → "What features are robustly selected across CV folds?"
  → Use for: Publishing, understanding stability, early discovery
  → Output: Consensus panel (features in ≥80% of folds)
  → Speed: Slower (~45× more model fits due to nested CV)

Typical workflow: Use both sequentially
  1. Enable rfe_enabled: true during training (robust discovery)
  2. Run ced optimize-panel after training (deployment trade-offs)
"""

import json
import logging
from dataclasses import dataclass, field
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


@dataclass
class RFEResult:
    """Results from recursive feature elimination.

    Attributes:
        curve: List of evaluation points with size, AUROC, and selected proteins.
        feature_ranking: Dict mapping protein -> elimination order (0 = removed first).
        recommended_panels: Dict with minimum sizes at various AUROC thresholds.
        max_auroc: Maximum AUROC achieved.
        model_name: Name of model used.
        cluster_map: Dict mapping representative protein -> cluster metadata
            (cluster_id, cluster_size, members). Empty if correlation-aware
            pre-filtering was not used.
    """

    curve: list[dict[str, Any]] = field(default_factory=list)
    feature_ranking: dict[str, int] = field(default_factory=dict)
    recommended_panels: dict[str, int] = field(default_factory=dict)
    max_auroc: float = 0.0
    model_name: str = ""
    cluster_map: dict[str, dict] = field(default_factory=dict)


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

    # Unwrap OOFCalibratedModel if necessary
    pipeline = base_pipeline
    if isinstance(base_pipeline, OOFCalibratedModel):
        pipeline = base_pipeline.base_model

    # Clone the classifier
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("Base pipeline has no 'clf' step")

    cloned_clf = clone(clf)

    # Build preprocessor for reduced feature set
    # Note: build_preprocessor only takes cat_cols; numeric cols pass through automatically
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


def _quick_tune_at_k(
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
            optimize_panel.yaml. Passed to ``get_rfe_tune_space``.

    Returns:
        Tuple of (fitted Pipeline with best params, best_params dict).
    """
    import time

    import optuna

    from ced_ml.cli.train import build_preprocessor
    from ced_ml.models.hyperparams import get_rfe_tune_space
    from ced_ml.models.optuna_search import OptunaSearchCV

    # Suppress Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tune_space = get_rfe_tune_space(model_name, config_overrides=rfe_tune_spaces)
    param_names = [k.replace("clf__", "") for k in tune_space]
    logger.info(
        f"[RFE k={len(feature_cols)}] Tuning {len(tune_space)} hyperparams "
        f"({', '.join(param_names)}) | {n_trials} trials, cv={cv_folds}, n_jobs={n_jobs}"
    )

    tune_start = time.time()

    # Build fresh pipeline
    clf = make_fresh_estimator(model_name, random_state=random_state)
    preprocessor = build_preprocessor(cat_cols)
    pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])

    # Run Optuna search
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

    # Log results
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


def recursive_feature_elimination(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_pipeline: Pipeline,
    model_name: str,
    initial_proteins: list[str],
    cat_cols: list[str],
    meta_num_cols: list[str],
    min_size: int = 5,
    cv_folds: int = 0,
    step_strategy: str = "geometric",
    min_auroc_frac: float = 0.90,
    random_state: int = 42,
    n_perm_repeats: int = 5,
    retune_n_trials: int = 40,
    retune_cv_folds: int = 3,
    retune_n_jobs: int = 1,
    corr_aware: bool = True,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
    selection_freq: dict[str, float] | None = None,
    rfe_tune_spaces: dict[str, dict[str, dict]] | None = None,
) -> RFEResult:
    """Perform recursive feature elimination to find minimum viable panel.

    Iteratively removes least important proteins, evaluating AUROC at each
    step. At each evaluation point, hyperparameters are re-tuned on TRAIN
    via a quick Optuna search so each Pareto curve point reflects the best
    achievable AUROC at that panel size.

    WARNING: When cv_folds > 0, the OOF AUROC estimates are optimistically
    biased because feature ranking is computed on the full training set before
    CV. The validation AUROC (auroc_val) is the honest metric. Default
    cv_folds=0 skips CV entirely to avoid confusion.

    Args:
        X_train: Training features (DataFrame with protein + metadata columns).
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        base_pipeline: Trained pipeline to clone classifier from.
        model_name: Model identifier (e.g., "LR_EN", "RF").
        initial_proteins: Starting protein panel.
        cat_cols: Categorical metadata columns.
        meta_num_cols: Numeric metadata columns.
        min_size: Smallest panel to evaluate.
        cv_folds: CV folds for OOF AUROC estimation (default 0 = skip CV).
        step_strategy: Elimination strategy ("geometric", "fine", "linear").
        min_auroc_frac: Early stop if AUROC drops below this fraction of max.
        random_state: Random seed.
        n_perm_repeats: Permutation repeats for tree importance.
        retune_n_trials: Optuna trials for per-k hyperparameter re-tuning.
        retune_cv_folds: CV folds for per-k Optuna search.
        retune_n_jobs: Parallel jobs for per-k Optuna CV evaluation.
        corr_aware: If True, cluster correlated proteins before RFE and
            run elimination on representatives only.
        corr_threshold: Correlation threshold for clustering (0.0-1.0).
        corr_method: Correlation method ("spearman" or "pearson").
        selection_freq: Stability selection frequencies for representative
            selection. If None, uses uniform weights.
        rfe_tune_spaces: Optional config-driven per-model search spaces from
            optimize_panel.yaml. Overrides hardcoded RFE_TUNE_SPACES defaults.

    Returns:
        RFEResult with curve, feature_ranking, and recommended_panels.
    """
    import time

    from ced_ml.models.hyperparams import RFE_TUNE_SPACES

    start_time = time.time()
    logger.info("Starting recursive feature elimination")
    logger.info(f"  initial_proteins: {len(initial_proteins)} proteins")
    logger.info(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"  model_name: {model_name}")

    # Check if per-k tuning is available for this model.
    # When config overrides are provided (rfe_tune_spaces from optimize_panel.yaml),
    # only models listed there get retuned. Models absent from config use fixed HPs.
    if rfe_tune_spaces:
        can_retune = model_name in rfe_tune_spaces
    else:
        can_retune = model_name in RFE_TUNE_SPACES
    if can_retune:
        logger.info(
            f"  Per-k hyperparameter re-tuning: enabled "
            f"({retune_n_trials} trials, cv={retune_cv_folds}, n_jobs={retune_n_jobs})"
        )
    else:
        logger.info(
            f"  Per-k hyperparameter re-tuning: disabled "
            f"(no tune space defined for {model_name})"
        )

    # Correlation-aware pre-filtering
    cluster_map: dict[str, dict] = {}
    if corr_aware:
        logger.info(
            f"Correlation-aware pre-filtering: threshold={corr_threshold}, " f"method={corr_method}"
        )
        current_proteins, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train=X_train,
            y_train=y_train,
            protein_cols=initial_proteins,
            selection_freq=selection_freq,
            corr_threshold=corr_threshold,
            corr_method=corr_method,
        )
    else:
        current_proteins = list(initial_proteins)

    curve: list[dict[str, Any]] = []
    feature_ranking: dict[str, int] = {}
    elimination_order = 0
    max_auroc_seen = 0.0
    all_best_params: list[dict] = []

    # Determine evaluation points
    logger.debug(f"Computing eval sizes: current={len(current_proteins)}, min={min_size}")
    eval_sizes = compute_eval_sizes(len(current_proteins), min_size, step_strategy)
    logger.debug(f"Eval sizes computed: {eval_sizes}")
    logger.info(f"RFE: Starting with {len(current_proteins)} proteins, target sizes: {eval_sizes}")
    logger.info(f"RFE: Will evaluate {len(eval_sizes)} panel sizes")

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

        # Eliminate proteins until we reach target_size
        proteins_to_eliminate = len(current_proteins) - target_size
        logger.info(
            f"Eliminating {proteins_to_eliminate} proteins to reach target size {target_size}"
        )

        elimination_count = 0
        while len(current_proteins) > target_size and len(current_proteins) > min_size:
            elimination_count += 1

            # Progress checkpoint every 10 eliminations
            if elimination_count % 10 == 0:
                logger.info(
                    f"  Elimination progress: {elimination_count}/{proteins_to_eliminate} "
                    f"({elimination_count/proteins_to_eliminate*100:.1f}%), "
                    f"current panel size: {len(current_proteins)}"
                )

            # Build pipeline with current panel (frozen hyperparams for elimination steps)
            try:
                pipeline = build_lightweight_pipeline(base_pipeline, current_proteins, cat_cols)
            except Exception as e:
                logger.error(f"Failed to build pipeline: {e}")
                break

            # Subset data to current features
            feature_cols = current_proteins + cat_cols + meta_num_cols
            X_train_subset = X_train[feature_cols]

            # Fit pipeline
            try:
                pipeline.fit(X_train_subset, y_train)
            except Exception as e:
                logger.error(f"Pipeline fit failed at size {len(current_proteins)}: {e}")
                break

            # Compute feature importances
            importances = compute_feature_importance(
                pipeline,
                model_name,
                current_proteins,
                X_train_subset,
                y_train,
                random_state,
                n_perm_repeats,
            )

            # Find least important protein (minimum non-zero, or any if all zero)
            protein_importances = {p: importances.get(p, 0.0) for p in current_proteins}
            worst_protein = min(protein_importances, key=protein_importances.get)

            # Record and remove
            feature_ranking[worst_protein] = elimination_order
            current_proteins.remove(worst_protein)
            elimination_order += 1

            logger.debug(
                f"Eliminated {worst_protein} (importance={protein_importances[worst_protein]:.4f}), "
                f"panel size now {len(current_proteins)}"
            )

        # Evaluate at this panel size
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

            # Per-k hyperparameter re-tuning at evaluation points
            best_params = {}
            if can_retune:
                tune_start = time.time()
                pipeline, best_params = _quick_tune_at_k(
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
                # Fallback: use frozen hyperparams from base pipeline
                pipeline = build_lightweight_pipeline(base_pipeline, current_proteins, cat_cols)
                pipeline.fit(X_train_subset, y_train)

            # CV metrics (OOF estimate) - skip if cv_folds <= 1 for speed
            if cv_folds > 1:
                logger.info(f"  Running {cv_folds}-fold CV for size {len(current_proteins)}...")
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                oof_probs = cross_val_predict(
                    clone(pipeline),
                    X_train_subset,
                    y_train,
                    cv=cv,
                    method="predict_proba",
                )[:, 1]
                auroc_cv = auroc(y_train, oof_probs)
                prauc_cv = prauc(y_train, oof_probs)
                brier_cv = compute_brier_score(y_train, oof_probs)
                sens_cv = alpha_sensitivity_at_specificity(
                    y_train, oof_probs, target_specificity=0.95
                )
                logger.info(
                    f"  CV AUROC: {auroc_cv:.4f}, PR-AUC: {prauc_cv:.4f}, Brier: {brier_cv:.4f}"
                )
                auroc_cv_std, _, _ = _bootstrap_auroc_ci(y_train, oof_probs, n_bootstrap=50)
            else:
                # No CV: use training set metrics (biased but fast)
                train_probs = pipeline.predict_proba(X_train_subset)[:, 1]
                auroc_cv = auroc(y_train, train_probs)
                prauc_cv = prauc(y_train, train_probs)
                brier_cv = compute_brier_score(y_train, train_probs)
                sens_cv = alpha_sensitivity_at_specificity(
                    y_train, train_probs, target_specificity=0.95
                )
                auroc_cv_std = 0.0
                logger.info(
                    f"  Train AUROC: {auroc_cv:.4f}, PR-AUC: {prauc_cv:.4f}, Brier: {brier_cv:.4f}"
                )

            # Validation metrics
            logger.info("  Computing validation metrics...")
            val_probs = pipeline.predict_proba(X_val_subset)[:, 1]
            auroc_val = auroc(y_val, val_probs)
            prauc_val = prauc(y_val, val_probs)
            brier_val = compute_brier_score(y_val, val_probs)
            sens_val = alpha_sensitivity_at_specificity(y_val, val_probs, target_specificity=0.95)
            auroc_val_std, auroc_val_ci_low, auroc_val_ci_high = _bootstrap_auroc_ci(
                y_val, val_probs, n_bootstrap=200
            )

            eval_elapsed = time.time() - eval_start
            logger.info(
                f"[RFE k={len(current_proteins)}] Val AUROC={auroc_val:.3f} "
                f"(95% CI: {auroc_val_ci_low:.3f}-{auroc_val_ci_high:.3f}), "
                f"PR-AUC={prauc_val:.3f}, Brier={brier_val:.4f}"
            )
            logger.info(f"[RFE k={len(current_proteins)}] Completed in {eval_elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Evaluation failed at size {len(current_proteins)}: {e}")
            continue

        # Record evaluation point
        point = {
            "size": len(current_proteins),
            "auroc_cv": auroc_cv,
            "auroc_cv_std": auroc_cv_std,
            "auroc_val": auroc_val,
            "auroc_val_std": auroc_val_std,
            "auroc_val_ci_low": auroc_val_ci_low,
            "auroc_val_ci_high": auroc_val_ci_high,
            "prauc_cv": prauc_cv,
            "prauc_val": prauc_val,
            "brier_cv": brier_cv,
            "brier_val": brier_val,
            "sens_at_95spec_cv": sens_cv,
            "sens_at_95spec_val": sens_val,
            "proteins": list(current_proteins),
            "best_params": best_params,
        }
        curve.append(point)
        logger.info(
            f"\n  *** Panel size {len(current_proteins)} results: ***\n"
            f"    AUROC_cv:        {auroc_cv:.4f} +/- {auroc_cv_std:.4f}\n"
            f"    AUROC_val:       {auroc_val:.4f} (95% CI: {auroc_val_ci_low:.4f}-{auroc_val_ci_high:.4f})\n"
            f"    PR-AUC_val:      {prauc_val:.4f}\n"
            f"    Brier_val:       {brier_val:.4f}\n"
            f"    Sens@95%Spec:    {sens_val:.4f}\n"
        )

        # Track max AUROC
        if auroc_val > max_auroc_seen:
            max_auroc_seen = auroc_val
            logger.info(f"  New maximum AUROC: {max_auroc_seen:.4f}")

        # Early stopping check
        if auroc_val < max_auroc_seen * min_auroc_frac:
            logger.info(
                f"\n*** Early stopping triggered ***\n"
                f"AUROC {auroc_val:.4f} < {min_auroc_frac:.0%} of max {max_auroc_seen:.4f}\n"
            )
            break

    # Per-seed summary
    if can_retune and all_best_params:
        # Summarize hyperparameter ranges across evaluation points
        param_ranges: dict[str, list] = {}
        for bp in all_best_params:
            for k, v in bp.items():
                param_ranges.setdefault(k, []).append(v)

        range_summary = ", ".join(
            (
                f"{k.replace('clf__', '')} [{min(vs):.4g}, {max(vs):.4g}]"
                if isinstance(vs[0], float)
                else f"{k.replace('clf__', '')} [{min(vs)}, {max(vs)}]"
            )
            for k, vs in param_ranges.items()
        )
        logger.info(
            f"\nRFE Summary (seed {random_state}):\n"
            f"  Eval points: {len(curve)} | Total tune time: {total_tune_time/60:.1f} min | "
            f"Total elapsed: {(time.time() - start_time)/60:.1f} min\n"
            f"  Hyperparams varied: {range_summary}"
        )

    # Compute recommendations
    logger.info("\nComputing recommended panel sizes...")
    recommended = find_recommended_panels(curve)

    total_elapsed = time.time() - start_time
    logger.info(
        f"\n{'='*60}\n"
        f"RFE Completed\n"
        f"Total time: {total_elapsed/60:.1f} min\n"
        f"Evaluation points: {len(curve)}\n"
        f"Max AUROC: {max_auroc_seen:.4f}\n"
        f"{'='*60}\n"
    )

    return RFEResult(
        curve=curve,
        feature_ranking=feature_ranking,
        recommended_panels=recommended,
        max_auroc=max_auroc_seen,
        model_name=model_name,
        cluster_map=cluster_map,
    )


def _bootstrap_auroc_ci(
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

        # Need both classes
        if len(np.unique(y_t)) < 2:
            continue

        try:
            aurocs.append(auroc(y_t, y_p))
        except Exception:
            continue

    if not aurocs:
        return 0.0, 0.0, 0.0

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(aurocs, 100 * alpha / 2))
    ci_high = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return float(np.std(aurocs)), ci_low, ci_high


def save_rfe_results(
    result: RFEResult,
    output_dir: str,
    model_name: str,
    split_seed: int,
) -> dict[str, str]:
    """Save RFE results to output directory.

    Args:
        result: RFEResult from recursive_feature_elimination.
        output_dir: Directory to save outputs.
        model_name: Model name for metadata.
        split_seed: Split seed for metadata.

    Returns:
        Dict mapping artifact name -> file path.
    """
    import os
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # Determine suffix for aggregated results (split_seed=-1 indicates aggregated)
    suffix = "_aggregated" if split_seed == -1 else ""

    # 1. Panel curve CSV
    curve_path = os.path.join(output_dir, f"panel_curve{suffix}.csv")
    curve_records = []
    for p in result.curve:
        record = {
            "size": p["size"],
            "auroc_cv": p["auroc_cv"],
            "auroc_cv_std": p["auroc_cv_std"],
            "auroc_val": p["auroc_val"],
            "auroc_val_std": p.get("auroc_val_std", 0.0),
            "auroc_val_ci_low": p.get("auroc_val_ci_low", 0.0),
            "auroc_val_ci_high": p.get("auroc_val_ci_high", 0.0),
            "prauc_cv": p.get("prauc_cv", float("nan")),
            "prauc_val": p.get("prauc_val", float("nan")),
            "brier_cv": p.get("brier_cv", float("nan")),
            "brier_val": p.get("brier_val", float("nan")),
            "sens_at_95spec_cv": p.get("sens_at_95spec_cv", float("nan")),
            "sens_at_95spec_val": p.get("sens_at_95spec_val", float("nan")),
            "proteins": json.dumps(p["proteins"]),
        }
        if result.cluster_map:
            total_members = sum(
                result.cluster_map.get(prot, {}).get("cluster_size", 1) for prot in p["proteins"]
            )
            record["total_cluster_members"] = total_members
            members_map = {
                prot: result.cluster_map[prot]["members"]
                for prot in p["proteins"]
                if prot in result.cluster_map
            }
            record["cluster_members_json"] = json.dumps(members_map)
        curve_records.append(record)
    curve_df = pd.DataFrame(curve_records)
    curve_df.to_csv(curve_path, index=False)
    paths["panel_curve"] = curve_path

    # 2. Feature ranking CSV
    ranking_path = os.path.join(output_dir, f"feature_ranking{suffix}.csv")
    ranking_records = []
    for p, order in sorted(result.feature_ranking.items(), key=lambda x: x[1]):
        record = {"protein": p, "elimination_order": order}
        if result.cluster_map and p in result.cluster_map:
            info = result.cluster_map[p]
            record["cluster_id"] = info["cluster_id"]
            record["cluster_size"] = info["cluster_size"]
            record["cluster_members"] = json.dumps(info["members"])
        elif result.cluster_map:
            record["cluster_id"] = None
            record["cluster_size"] = 1
            record["cluster_members"] = json.dumps([p])
        ranking_records.append(record)
    ranking_df = pd.DataFrame(ranking_records)
    ranking_df.to_csv(ranking_path, index=False)
    paths["feature_ranking"] = ranking_path

    # 2b. Cluster mapping CSV (if clusters were used)
    if result.cluster_map:
        cluster_map_path = os.path.join(output_dir, f"cluster_mapping{suffix}.csv")
        cluster_records = []
        for rep, info in result.cluster_map.items():
            for member in info["members"]:
                cluster_records.append(
                    {
                        "representative": rep,
                        "member_protein": member,
                        "cluster_id": info["cluster_id"],
                        "cluster_size": info["cluster_size"],
                        "is_representative": member == rep,
                    }
                )
        cluster_df = pd.DataFrame(cluster_records).sort_values(
            ["cluster_id", "is_representative"],
            ascending=[True, False],
        )
        cluster_df.to_csv(cluster_map_path, index=False)
        paths["cluster_mapping"] = cluster_map_path
        logger.info(f"Saved cluster mapping to {cluster_map_path}")

    # 3. Recommended panels JSON
    rec_path = os.path.join(output_dir, f"recommended_panels{suffix}.json")
    rec_data = {
        "model": model_name,
        "split_seed": split_seed,
        "max_auroc": result.max_auroc,
        "recommended_panels": result.recommended_panels,
        "timestamp": datetime.now().isoformat(),
    }
    with open(rec_path, "w") as f:
        json.dump(rec_data, f, indent=2)
    paths["recommended_panels"] = rec_path

    # 4. Metrics summary CSV (panel size vs all metrics)
    metrics_summary_path = os.path.join(output_dir, f"metrics_summary{suffix}.csv")
    metrics_df = pd.DataFrame(
        [
            {
                "size": p["size"],
                "auroc_cv": p["auroc_cv"],
                "auroc_cv_std": p["auroc_cv_std"],
                "auroc_val": p["auroc_val"],
                "auroc_val_std": p.get("auroc_val_std", 0.0),
                "auroc_val_ci_low": p.get("auroc_val_ci_low", 0.0),
                "auroc_val_ci_high": p.get("auroc_val_ci_high", 0.0),
                "prauc_cv": p.get("prauc_cv", float("nan")),
                "prauc_val": p.get("prauc_val", float("nan")),
                "brier_cv": p.get("brier_cv", float("nan")),
                "brier_val": p.get("brier_val", float("nan")),
                "sens_at_95spec_cv": p.get("sens_at_95spec_cv", float("nan")),
                "sens_at_95spec_val": p.get("sens_at_95spec_val", float("nan")),
            }
            for p in result.curve
        ]
    )
    metrics_df.to_csv(metrics_summary_path, index=False)
    paths["metrics_summary"] = metrics_summary_path

    logger.info(f"Saved RFE results to {output_dir}")
    return paths


def aggregate_rfe_results(results: list[RFEResult]) -> RFEResult:
    """Aggregate RFE results across multiple split seeds.

    Combines per-seed Pareto curves into a single curve with cross-seed
    mean and percentile-based 95% confidence intervals. Feature rankings
    are aggregated via mean elimination order.

    Args:
        results: List of RFEResult objects, one per split seed.

    Returns:
        Aggregated RFEResult with mean validation metrics, cross-seed CIs,
        mean feature rankings, and recommendations from the aggregated curve.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    if len(results) == 1:
        logger.info("Single seed: skipping aggregation, returning as-is")
        return results[0]

    n_seeds = len(results)
    logger.info(f"Aggregating RFE curves across {n_seeds} seeds")

    # -- Aggregate curves by panel size --
    # Collect all curve points keyed by size
    size_to_metrics: dict[int, list[dict[str, Any]]] = {}
    size_to_proteins: dict[int, list[list[str]]] = {}
    for r in results:
        for point in r.curve:
            size = point["size"]
            if size not in size_to_metrics:
                size_to_metrics[size] = []
                size_to_proteins[size] = []
            size_to_metrics[size].append(point)
            size_to_proteins[size].append(point["proteins"])

    # Only keep sizes present in ALL seeds for a clean curve
    all_seed_sizes = [size for size, points in size_to_metrics.items() if len(points) == n_seeds]
    all_seed_sizes.sort(reverse=True)

    if not all_seed_sizes:
        # Fallback: use sizes present in at least half the seeds
        all_seed_sizes = [
            size for size, points in size_to_metrics.items() if len(points) >= max(1, n_seeds // 2)
        ]
        all_seed_sizes.sort(reverse=True)
        logger.warning(
            f"No panel sizes common to all {n_seeds} seeds; "
            f"using {len(all_seed_sizes)} sizes present in >= {max(1, n_seeds // 2)} seeds"
        )

    val_metrics = ["auroc_val", "prauc_val", "brier_val", "sens_at_95spec_val"]
    cv_metrics = ["auroc_cv", "prauc_cv", "brier_cv", "sens_at_95spec_cv"]

    aggregated_curve: list[dict[str, Any]] = []
    for size in all_seed_sizes:
        points = size_to_metrics[size]
        agg: dict[str, Any] = {"size": size}

        for metric in val_metrics + cv_metrics:
            values = np.array([p.get(metric, np.nan) for p in points])
            values = values[~np.isnan(values)]
            if len(values) > 0:
                agg[metric] = float(np.mean(values))
                agg[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                agg[metric] = np.nan
                agg[f"{metric}_std"] = 0.0

        # Cross-seed percentile CI for auroc_val
        auroc_vals = np.array([p["auroc_val"] for p in points])
        if len(auroc_vals) > 1:
            agg["auroc_val_ci_low"] = float(np.percentile(auroc_vals, 2.5))
            agg["auroc_val_ci_high"] = float(np.percentile(auroc_vals, 97.5))
        else:
            agg["auroc_val_ci_low"] = agg["auroc_val"]
            agg["auroc_val_ci_high"] = agg["auroc_val"]

        agg["auroc_cv_std"] = agg.get("auroc_cv_std", 0.0)
        agg["n_seeds"] = len(points)

        # Use proteins from the first seed at this size (ordering is seed-dependent)
        agg["proteins"] = size_to_proteins[size][0]

        aggregated_curve.append(agg)

    # -- Aggregate feature rankings via mean elimination order --
    all_proteins: set[str] = set()
    for r in results:
        all_proteins.update(r.feature_ranking.keys())

    aggregated_ranking: dict[str, float] = {}
    for protein in all_proteins:
        orders = [r.feature_ranking[protein] for r in results if protein in r.feature_ranking]
        aggregated_ranking[protein] = float(np.mean(orders))

    # Convert to int-keyed dict sorted by mean order (for compatibility)
    sorted_ranking = {
        p: rank
        for rank, (p, _) in enumerate(sorted(aggregated_ranking.items(), key=lambda x: x[1]))
    }

    # -- Recommendations from aggregated curve --
    recommended = find_recommended_panels(aggregated_curve)

    # -- Max AUROC from aggregated curve --
    max_auroc = max(
        (p["auroc_val"] for p in aggregated_curve),
        default=0.0,
    )

    model_name = results[0].model_name

    logger.info(
        f"Aggregated {len(aggregated_curve)} panel sizes across {n_seeds} seeds, "
        f"max mean AUROC={max_auroc:.4f}"
    )

    # Use first seed's cluster map as canonical (same proteins -> same clusters)
    cluster_map = results[0].cluster_map if results[0].cluster_map else {}

    return RFEResult(
        curve=aggregated_curve,
        feature_ranking=sorted_ranking,
        recommended_panels=recommended,
        max_auroc=max_auroc,
        model_name=model_name,
        cluster_map=cluster_map,
    )
