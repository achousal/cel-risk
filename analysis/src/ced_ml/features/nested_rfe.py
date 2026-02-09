"""Nested RFECV for robust feature discovery within CV folds.

RFECV runs within each outer CV fold during training to ensure:
- No data leakage (feature selection uses only train fold data)
- Automatic optimal size (internal CV finds best panel size per fold)
- Unbiased evaluation (validation on held-out fold)
- Consensus panel (aggregates selections across folds)

Workflow:
1. Outer CV fold splits data into train/val
2. RFECV runs on train fold with internal CV to find optimal panel size
3. Selected features evaluated on held-out val fold
4. Aggregate: consensus panel = features in >= threshold folds

Complementary to rfe.py:
- nested_rfe.py: Discovery during training, slower, consensus panel
- rfe.py: Post-hoc optimization, faster, Pareto curve for deployment
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ced_ml.metrics.discrimination import auroc
from ced_ml.utils.serialization import save_json

logger = logging.getLogger(__name__)


@dataclass
class RFECVFoldResult:
    """Results from RFECV on a single CV fold."""

    fold_idx: int = 0
    optimal_n_features: int = 0
    selected_features: list[str] = field(default_factory=list)
    cv_scores: list[float] = field(default_factory=list)
    feature_ranking: dict[str, int] = field(default_factory=dict)
    val_auroc: float = 0.0


@dataclass
class NestedRFECVResult:
    """Aggregated results from nested RFECV across CV folds."""

    fold_results: list[RFECVFoldResult] = field(default_factory=list)
    consensus_panel: list[str] = field(default_factory=list)
    feature_stability: dict[str, float] = field(default_factory=dict)
    optimal_sizes: list[int] = field(default_factory=list)
    mean_optimal_size: float = 0.0
    fold_val_aurocs: list[float] = field(default_factory=list)


def _get_predictions(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Get predictions from estimator, handling both predict_proba and decision_function.

    Args:
        estimator: Fitted estimator.
        X: Feature matrix.

    Returns:
        1D array of predictions (probabilities or decision scores).
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def _fit_rfecv_with_fallback(rfecv: RFECV, X: np.ndarray, y: np.ndarray, n_jobs: int) -> RFECV:
    """Fit RFECV with automatic fallback to serial execution if parallel fails.

    Args:
        rfecv: RFECV instance to fit.
        X: Feature matrix.
        y: Labels.
        n_jobs: Number of parallel jobs requested.

    Returns:
        Fitted RFECV instance.
    """
    parallel_requested = n_jobs is not None and int(n_jobs) != 1
    try:
        rfecv.fit(X, y)
        return rfecv
    except (PermissionError, NotImplementedError, OSError) as exc:
        if not parallel_requested:
            raise
        logger.warning(
            "RFECV parallel execution unavailable in current runtime (%s). "
            "Retrying with n_jobs=1.",
            exc,
        )
        rfecv_serial = clone(rfecv)
        rfecv_serial.n_jobs = 1
        rfecv_serial.fit(X, y)
        return rfecv_serial


def run_rfecv_within_fold(
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    X_val_fold: pd.DataFrame,
    y_val_fold: np.ndarray,
    estimator: Any,
    feature_names: list[str],
    fold_idx: int = 0,
    min_features: int = 5,
    step: int | float = 1,
    cv_folds: int = 3,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: int = 42,
) -> RFECVFoldResult:
    """Run RFECV within a single outer CV fold.

    Args:
        X_train_fold: Training features for this fold.
        y_train_fold: Training labels for this fold.
        X_val_fold: Validation features for this fold.
        y_val_fold: Validation labels for this fold.
        estimator: Base estimator (must have coef_ or feature_importances_).
        feature_names: List of feature names matching X columns.
        fold_idx: Outer fold index for tracking.
        min_features: Minimum number of features to select.
        step: Features to remove at each iteration (int or fraction).
        cv_folds: Internal CV folds for RFECV.
        scoring: Scoring metric for CV.
        n_jobs: Parallel jobs for CV.
        random_state: Random seed.

    Returns:
        RFECVFoldResult with selected features and performance.
    """
    if len(feature_names) != X_train_fold.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != X columns ({X_train_fold.shape[1]})"
        )

    if min_features < 1:
        logger.warning(
            f"min_features_to_select={min_features} is invalid (must be >= 1). Clamping to 1."
        )
        min_features = 1

    inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_estimator = clone(estimator)

    rfecv = RFECV(
        estimator=base_estimator,
        step=step,
        cv=inner_cv,
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=n_jobs,
    )

    logger.debug(f"Fold {fold_idx}: Running RFECV on {len(feature_names)} features...")

    rfecv = _fit_rfecv_with_fallback(rfecv, X_train_fold.values, y_train_fold, n_jobs)

    optimal_n = rfecv.n_features_
    support_mask = rfecv.support_
    selected_features = [
        f for f, selected in zip(feature_names, support_mask, strict=False) if selected
    ]
    feature_ranking = {f: int(r) for f, r in zip(feature_names, rfecv.ranking_, strict=False)}
    cv_scores = list(rfecv.cv_results_["mean_test_score"])

    X_train_selected = X_train_fold.iloc[:, support_mask]
    X_val_selected = X_val_fold.iloc[:, support_mask]
    rfecv.estimator_.fit(X_train_selected.values, y_train_fold)

    val_probs = _get_predictions(rfecv.estimator_, X_val_selected.values)
    val_auroc = auroc(y_val_fold, val_probs)

    logger.info(
        f"Fold {fold_idx}: RFECV selected {optimal_n} features, val AUROC = {val_auroc:.4f}"
    )

    return RFECVFoldResult(
        fold_idx=fold_idx,
        optimal_n_features=optimal_n,
        selected_features=selected_features,
        cv_scores=cv_scores,
        feature_ranking=feature_ranking,
        val_auroc=val_auroc,
    )


def _unwrap_calibrated_classifier(obj: Any) -> Any:
    """Unwrap CalibratedClassifierCV to get base estimator.

    Args:
        obj: Potentially wrapped estimator.

    Returns:
        Unwrapped estimator.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if not isinstance(obj, CalibratedClassifierCV):
        return obj

    if hasattr(obj, "calibrated_classifiers_") and obj.calibrated_classifiers_:
        return obj.calibrated_classifiers_[0].estimator
    if hasattr(obj, "estimator"):
        return obj.estimator
    return getattr(obj, "base_estimator", obj)


def _validate_estimator_for_rfecv(clf: Any) -> None:
    """Validate estimator has required attributes for RFECV.

    Args:
        clf: Estimator to validate.
    """
    estimator_class = type(clf)
    has_coef = hasattr(clf, "coef_") or "coef_" in dir(estimator_class)
    has_importance = hasattr(clf, "feature_importances_") or "feature_importances_" in dir(
        estimator_class
    )

    if not (has_coef or has_importance):
        logger.warning(
            f"Estimator {type(clf).__name__} may not have coef_ or feature_importances_. "
            "RFECV may fail."
        )


def extract_estimator_for_rfecv(fitted_pipeline: Pipeline) -> Any:
    """Extract the classifier from a fitted pipeline for use with RFECV.

    Args:
        fitted_pipeline: Fitted sklearn Pipeline.

    Returns:
        Cloned estimator suitable for RFECV.

    Raises:
        ValueError: If no suitable estimator found.
    """
    if hasattr(fitted_pipeline, "base_model"):
        fitted_pipeline = fitted_pipeline.base_model

    pipeline = _unwrap_calibrated_classifier(fitted_pipeline)

    if isinstance(pipeline, Pipeline):
        clf = pipeline.named_steps.get("clf")
        if clf is None:
            raise ValueError("Pipeline has no 'clf' step")
    else:
        clf = pipeline

    clf = _unwrap_calibrated_classifier(clf)
    _validate_estimator_for_rfecv(clf)

    return clone(clf)


def compute_consensus_panel(
    fold_selections: list[list[str]],
    threshold: float = 0.80,
) -> tuple[list[str], dict[str, float]]:
    """Compute consensus panel from fold-wise selections.

    Args:
        fold_selections: List of selected feature lists per fold.
        threshold: Minimum selection fraction for consensus.

    Returns:
        Tuple of (consensus_panel, stability_dict) where stability_dict
        maps feature to selection fraction.
    """
    if not fold_selections:
        return [], {}

    n_folds = len(fold_selections)
    feature_counts: dict[str, int] = {}

    for selection in fold_selections:
        for feature in selection:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    stability = {f: count / n_folds for f, count in feature_counts.items()}
    consensus = [f for f, frac in stability.items() if frac >= threshold]
    consensus.sort(key=lambda f: (-stability[f], f))

    return consensus, stability


def aggregate_rfecv_results(
    fold_results: list[RFECVFoldResult],
    consensus_threshold: float = 0.80,
) -> NestedRFECVResult:
    """Aggregate RFECV results across CV folds.

    Args:
        fold_results: RFECVFoldResult per fold.
        consensus_threshold: Threshold for consensus panel.

    Returns:
        NestedRFECVResult with aggregated statistics.
    """
    fold_selections = [r.selected_features for r in fold_results]
    consensus_panel, stability = compute_consensus_panel(fold_selections, consensus_threshold)

    optimal_sizes = [r.optimal_n_features for r in fold_results]
    fold_aurocs = [r.val_auroc for r in fold_results]

    return NestedRFECVResult(
        fold_results=fold_results,
        consensus_panel=consensus_panel,
        feature_stability=stability,
        optimal_sizes=optimal_sizes,
        mean_optimal_size=float(np.mean(optimal_sizes)) if optimal_sizes else 0.0,
        fold_val_aurocs=fold_aurocs,
    )


def _save_consensus_panel(result: NestedRFECVResult, output_dir: Path) -> tuple[str, str]:
    """Save consensus panel and feature stability CSVs.

    Returns:
        Tuple of (consensus_panel_path, stability_path).
    """
    consensus_path = output_dir / "consensus_panel.csv"
    consensus_df = pd.DataFrame(
        [
            {"protein": p, "stability": result.feature_stability.get(p, 0.0)}
            for p in result.consensus_panel
        ]
    )
    consensus_df.to_csv(consensus_path, index=False)

    stability_path = output_dir / "feature_stability.csv"
    stability_df = pd.DataFrame(
        [
            {"protein": p, "selection_fraction": frac}
            for p, frac in sorted(result.feature_stability.items(), key=lambda x: -x[1])
        ]
    )
    stability_df.to_csv(stability_path, index=False)

    return str(consensus_path), str(stability_path)


def _save_fold_results(result: NestedRFECVResult, output_dir: Path) -> str:
    """Save fold-wise results CSV.

    Returns:
        Path to fold_results.csv.
    """
    fold_results_path = output_dir / "fold_results.csv"
    fold_rows = [
        {
            "fold": r.fold_idx,
            "optimal_n_features": r.optimal_n_features,
            "val_auroc": r.val_auroc,
            "n_cv_scores": len(r.cv_scores),
            "selected_features": json.dumps(r.selected_features),
        }
        for r in result.fold_results
    ]
    pd.DataFrame(fold_rows).to_csv(fold_results_path, index=False)
    return str(fold_results_path)


def _save_cv_scores_curve(result: NestedRFECVResult, output_dir: Path) -> str | None:
    """Save CV scores curve CSV.

    Returns:
        Path to cv_scores_curve.csv if data exists, else None.
    """
    cv_rows = [
        {"fold": r.fold_idx, "n_features": i + 1, "cv_score": score}
        for r in result.fold_results
        for i, score in enumerate(r.cv_scores)
    ]
    if not cv_rows:
        return None

    cv_scores_path = output_dir / "cv_scores_curve.csv"
    pd.DataFrame(cv_rows).to_csv(cv_scores_path, index=False)
    return str(cv_scores_path)


def _save_summary_json(
    result: NestedRFECVResult, output_dir: Path, model_name: str, split_seed: int
) -> str:
    """Save summary JSON.

    Returns:
        Path to nested_rfecv_summary.json.
    """
    from datetime import datetime

    summary_path = output_dir / "nested_rfecv_summary.json"
    summary = {
        "model": model_name,
        "split_seed": int(split_seed),
        "n_folds": len(result.fold_results),
        "consensus_panel_size": len(result.consensus_panel),
        "mean_optimal_size": float(result.mean_optimal_size),
        "std_optimal_size": (float(np.std(result.optimal_sizes)) if result.optimal_sizes else 0.0),
        "optimal_sizes_per_fold": [int(x) for x in result.optimal_sizes],
        "mean_val_auroc": (
            float(np.mean(result.fold_val_aurocs)) if result.fold_val_aurocs else 0.0
        ),
        "std_val_auroc": (float(np.std(result.fold_val_aurocs)) if result.fold_val_aurocs else 0.0),
        "val_aurocs_per_fold": [float(x) for x in result.fold_val_aurocs],
        "consensus_threshold": 0.80,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(summary, summary_path)
    return str(summary_path)


def _save_selection_curve_plot(
    cv_scores_path: str, output_dir: Path, model_name: str, split_seed: int
) -> str | None:
    """Save RFECV selection curve plot.

    Returns:
        Path to plot if successful, else None.
    """
    try:
        from ced_ml.plotting.panel_curve import plot_rfecv_selection_curve

        selection_curve_plot = output_dir / "rfecv_selection_curve.png"
        plot_rfecv_selection_curve(
            cv_scores_curve_path=cv_scores_path,
            out_path=selection_curve_plot,
            title="RFECV Feature Selection Curve",
            model_name=f"{model_name} (split_seed={split_seed})",
        )
        logger.info(f"Saved RFECV selection curve plot to {selection_curve_plot}")
        return str(selection_curve_plot)
    except ImportError:
        logger.warning("Matplotlib not available, skipping RFECV selection curve plot")
        return None
    except Exception as e:
        logger.warning(f"Failed to generate RFECV selection curve plot: {e}")
        return None


def save_nested_rfecv_results(
    result: NestedRFECVResult,
    output_dir: str | Path,
    model_name: str,
    split_seed: int,
) -> dict[str, str]:
    """Save nested RFECV results to output directory.

    Args:
        result: NestedRFECVResult from nested CV.
        output_dir: Directory to save outputs.
        model_name: Model name for metadata.
        split_seed: Split seed for metadata.

    Returns:
        Dict mapping artifact name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    consensus_path, stability_path = _save_consensus_panel(result, output_dir)
    paths["consensus_panel"] = consensus_path
    paths["feature_stability"] = stability_path

    paths["fold_results"] = _save_fold_results(result, output_dir)

    cv_scores_path = _save_cv_scores_curve(result, output_dir)
    if cv_scores_path:
        paths["cv_scores_curve"] = cv_scores_path
        plot_path = _save_selection_curve_plot(cv_scores_path, output_dir, model_name, split_seed)
        if plot_path:
            paths["selection_curve_plot"] = plot_path

    paths["summary"] = _save_summary_json(result, output_dir, model_name, split_seed)

    logger.info(f"Saved nested RFECV results to {output_dir}")
    return paths
