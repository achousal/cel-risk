"""Nested RFECV for robust feature discovery within CV folds.

This module provides RFECV (Recursive Feature Elimination with Cross-Validation)
that operates WITHIN each outer CV fold during training, ensuring:
1. No data leakage - feature selection uses only train fold data
2. Automatic optimal size - internal CV finds best panel size per fold
3. Unbiased evaluation - validation happens on held-out fold
4. Consensus panel - aggregates selections across folds

Workflow:
1. Outer CV fold splits data into train/val
2. RFECV runs on train fold with internal CV to find optimal panel size
3. Selected features evaluated on held-out val fold
4. Aggregate: consensus panel = features in >= threshold folds

Complementary to features/rfe.py (post-hoc panel optimization):
- nested_rfe.py (this module): Scientific discovery during training
  → "What features are robustly selected across CV folds?"
  → Use for: Publishing, understanding stability, early discovery
  → Output: Consensus panel (features in ≥80% of folds)
  → Speed: Slower (~45× more model fits)

- rfe.py (ced optimize-panel): Clinical deployment after training
  → "What's the minimum panel size maintaining AUROC ≥ 0.90?"
  → Use for: Stakeholder decisions, cost-benefit trade-offs
  → Output: Pareto curve (panel size vs. AUROC)
  → Speed: Faster (single model evaluation per size)

Typical workflow: Use both sequentially
  1. Enable rfe_enabled: true during training (robust discovery)
  2. Run ced optimize-panel after training (deployment trade-offs)
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
    """Results from RFECV on a single CV fold.

    Attributes:
        fold_idx: Outer fold index.
        optimal_n_features: Number of features selected by RFECV.
        selected_features: List of selected feature names.
        cv_scores: Cross-validation scores at each feature count.
        feature_ranking: Dict mapping feature -> rank (1 = selected).
        val_auroc: AUROC on held-out validation fold.
    """

    fold_idx: int = 0
    optimal_n_features: int = 0
    selected_features: list[str] = field(default_factory=list)
    cv_scores: list[float] = field(default_factory=list)
    feature_ranking: dict[str, int] = field(default_factory=dict)
    val_auroc: float = 0.0


@dataclass
class NestedRFECVResult:
    """Aggregated results from nested RFECV across CV folds.

    Attributes:
        fold_results: RFECVFoldResult per outer fold.
        consensus_panel: Features selected in >= threshold folds.
        feature_stability: Dict mapping feature -> selection fraction.
        optimal_sizes: List of optimal sizes per fold.
        mean_optimal_size: Mean optimal panel size across folds.
        fold_val_aurocs: Validation AUROC per fold.
    """

    fold_results: list[RFECVFoldResult] = field(default_factory=list)
    consensus_panel: list[str] = field(default_factory=list)
    feature_stability: dict[str, float] = field(default_factory=dict)
    optimal_sizes: list[int] = field(default_factory=list)
    mean_optimal_size: float = 0.0
    fold_val_aurocs: list[float] = field(default_factory=list)


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

    RFECV uses internal CV on train_fold to find optimal feature count,
    then evaluates on val_fold for unbiased performance estimate.

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
    # Validate inputs
    if len(feature_names) != X_train_fold.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != X columns ({X_train_fold.shape[1]})"
        )

    # Setup internal CV
    inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Validate min_features
    if min_features < 1:
        logger.warning(
            f"min_features_to_select={min_features} is invalid (must be >= 1). " "Clamping to 1."
        )
        min_features = 1

    # Clone estimator to avoid mutation
    base_estimator = clone(estimator)

    # Run RFECV
    rfecv = RFECV(
        estimator=base_estimator,
        step=step,
        cv=inner_cv,
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=n_jobs,
    )

    logger.debug(f"Fold {fold_idx}: Running RFECV on {len(feature_names)} features...")

    # Fit RFECV on train fold
    rfecv.fit(X_train_fold.values, y_train_fold)

    # Extract results
    optimal_n = rfecv.n_features_
    support_mask = rfecv.support_
    ranking = rfecv.ranking_

    # Get selected feature names
    selected_features = [
        f for f, selected in zip(feature_names, support_mask, strict=False) if selected
    ]

    # Build feature ranking dict (1 = selected, higher = eliminated earlier)
    feature_ranking = {f: int(r) for f, r in zip(feature_names, ranking, strict=False)}

    # Get CV scores curve
    cv_scores = list(rfecv.cv_results_["mean_test_score"])

    # Evaluate on held-out validation fold
    X_val_selected = X_val_fold.iloc[:, support_mask]
    rfecv.estimator_.fit(X_train_fold.iloc[:, support_mask].values, y_train_fold)

    # Get predictions (use decision_function for models without predict_proba)
    if hasattr(rfecv.estimator_, "predict_proba"):
        val_probs = rfecv.estimator_.predict_proba(X_val_selected.values)[:, 1]
    else:
        # Use decision_function for models like LinearSVC
        val_probs = rfecv.estimator_.decision_function(X_val_selected.values)

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


def extract_estimator_for_rfecv(
    fitted_pipeline: Pipeline,
) -> Any:
    """Extract the classifier from a fitted pipeline for use with RFECV.

    RFECV requires an estimator with coef_ or feature_importances_.
    This extracts and clones the appropriate estimator.

    Args:
        fitted_pipeline: Fitted sklearn Pipeline.

    Returns:
        Cloned estimator suitable for RFECV.

    Raises:
        ValueError: If no suitable estimator found.
    """
    from sklearn.calibration import CalibratedClassifierCV

    # Unwrap OOFCalibratedModel if present
    if hasattr(fitted_pipeline, "base_model"):
        fitted_pipeline = fitted_pipeline.base_model

    # Unwrap CalibratedClassifierCV
    if isinstance(fitted_pipeline, CalibratedClassifierCV):
        if hasattr(fitted_pipeline, "estimator"):
            pipeline = fitted_pipeline.estimator
        else:
            pipeline = getattr(fitted_pipeline, "base_estimator", fitted_pipeline)
    else:
        pipeline = fitted_pipeline

    # Extract classifier from pipeline
    if isinstance(pipeline, Pipeline):
        clf = pipeline.named_steps.get("clf")
        if clf is None:
            raise ValueError("Pipeline has no 'clf' step")
    else:
        clf = pipeline

    # Unwrap CalibratedClassifierCV from classifier
    if isinstance(clf, CalibratedClassifierCV):
        if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
            # Get the base estimator from first calibrated classifier
            clf = clf.calibrated_classifiers_[0].estimator
        elif hasattr(clf, "estimator"):
            clf = clf.estimator

    # Validate estimator has required attributes
    # Note: We check the class, not instance, since we'll be fitting a fresh clone
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

    return clone(clf)


def compute_consensus_panel(
    fold_selections: list[list[str]],
    threshold: float = 0.80,
) -> tuple[list[str], dict[str, float]]:
    """Compute consensus panel from fold-wise selections.

    A feature is included in the consensus panel if it was selected in
    >= threshold fraction of folds. This ensures robust feature selection.

    Args:
        fold_selections: List of selected feature lists per fold.
        threshold: Minimum selection fraction for consensus (default 0.80).

    Returns:
        Tuple of (consensus_panel, stability_dict).
        stability_dict maps feature -> selection fraction.
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

    # Sort by stability (most stable first), then alphabetically
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
        Dict mapping artifact name -> file path.
    """
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 1. Consensus panel CSV
    consensus_path = output_dir / "consensus_panel.csv"
    consensus_df = pd.DataFrame(
        [
            {"protein": p, "stability": result.feature_stability.get(p, 0.0)}
            for p in result.consensus_panel
        ]
    )
    consensus_df.to_csv(consensus_path, index=False)
    paths["consensus_panel"] = str(consensus_path)

    # 2. Feature stability CSV (all features)
    stability_path = output_dir / "feature_stability.csv"
    stability_df = pd.DataFrame(
        [
            {"protein": p, "selection_fraction": frac}
            for p, frac in sorted(result.feature_stability.items(), key=lambda x: -x[1])
        ]
    )
    stability_df.to_csv(stability_path, index=False)
    paths["feature_stability"] = str(stability_path)

    # 3. Fold-wise results CSV
    fold_results_path = output_dir / "fold_results.csv"
    fold_rows = []
    for r in result.fold_results:
        fold_rows.append(
            {
                "fold": r.fold_idx,
                "optimal_n_features": r.optimal_n_features,
                "val_auroc": r.val_auroc,
                "n_cv_scores": len(r.cv_scores),
                "selected_features": json.dumps(r.selected_features),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(fold_results_path, index=False)
    paths["fold_results"] = str(fold_results_path)

    # 4. CV scores curve per fold (for plotting)
    cv_scores_path = output_dir / "cv_scores_curve.csv"
    cv_rows = []
    for r in result.fold_results:
        for i, score in enumerate(r.cv_scores):
            cv_rows.append(
                {
                    "fold": r.fold_idx,
                    "n_features": i + 1,  # RFECV scores are 1-indexed by n_features
                    "cv_score": score,
                }
            )
    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv(cv_scores_path, index=False)
        paths["cv_scores_curve"] = str(cv_scores_path)

    # 5. Summary JSON
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
    paths["summary"] = str(summary_path)

    # 6. RFECV selection curve plot (if cv_scores_curve exists)
    if "cv_scores_curve" in paths:
        try:
            from ced_ml.plotting.panel_curve import plot_rfecv_selection_curve

            selection_curve_plot = output_dir / "rfecv_selection_curve.png"
            plot_rfecv_selection_curve(
                cv_scores_curve_path=paths["cv_scores_curve"],
                out_path=selection_curve_plot,
                title="RFECV Feature Selection Curve",
                model_name=f"{model_name} (split_seed={split_seed})",
            )
            paths["selection_curve_plot"] = str(selection_curve_plot)
            logger.info(f"Saved RFECV selection curve plot to {selection_curve_plot}")
        except ImportError:
            logger.warning("Matplotlib not available, skipping RFECV selection curve plot")
        except Exception as e:
            logger.warning(f"Failed to generate RFECV selection curve plot: {e}")

    logger.info(f"Saved nested RFECV results to {output_dir}")
    return paths
