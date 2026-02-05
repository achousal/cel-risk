"""
Nested cross-validation orchestration for out-of-fold prediction generation.

This module handles the outer CV loop for generating OOF predictions,
including hyperparameter tuning (inner CV), feature selection tracking,
and optional post-hoc calibration.

Key components:
- OOF prediction generation with repeated stratified K-fold CV
- Nested hyperparameter tuning (inner CV with Optuna or RandomizedSearchCV)
- RFECV integration for fold-wise feature selection
- Per-fold and OOF-posthoc calibration strategies
- Protein selection extraction from fitted models
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ..config import TrainingConfig
from ..data.schema import ModelName
from ..metrics.scorers import get_scorer
from ..models.registry import compute_scale_pos_weight_from_y
from ..utils.constants import MAX_SAFE_PREVALENCE, MIN_SAFE_PREVALENCE
from ..utils.feature_names import extract_protein_name
from ..utils.logging import log_fold_header

if TYPE_CHECKING:
    from ced_ml.features.nested_rfe import NestedRFECVResult

    from .calibration import OOFCalibrator

logger = logging.getLogger(__name__)


_DEFAULT_N_ITER = 30  # Fallback when neither global nor per-model n_iter is set


def get_model_n_iter(model_name: str, config: TrainingConfig) -> int:
    """
    Get n_iter for a model.

    Priority order:
    1. Global cv.n_iter (if set, overrides all per-model values)
    2. Per-model n_iter (lr.n_iter, rf.n_iter, etc.)
    3. Default fallback (30)

    Args:
        model_name: Model identifier (LR_EN, LR_L1, LinSVM_cal, RF, XGBoost)
        config: TrainingConfig object

    Returns:
        n_iter value (>= 1)
    """
    # Global override takes precedence
    if config.cv.n_iter is not None:
        return config.cv.n_iter

    # Per-model setting
    model_configs = {
        ModelName.LR_EN: config.lr,
        ModelName.LR_L1: config.lr,
        ModelName.LinSVM_cal: config.svm,
        ModelName.RF: config.rf,
        ModelName.XGBoost: config.xgboost,
    }
    model_cfg = model_configs.get(model_name)
    if model_cfg is not None and getattr(model_cfg, "n_iter", None) is not None:
        return model_cfg.n_iter

    return _DEFAULT_N_ITER


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def oof_predictions_with_nested_cv(
    pipeline: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    protein_cols: list[str],
    config: TrainingConfig,
    random_state: int,
    grid_rng: np.random.Generator | None = None,
) -> tuple[
    np.ndarray,
    float,
    pd.DataFrame,
    pd.DataFrame,
    "OOFCalibrator | None",
    "NestedRFECVResult | None",
    pd.DataFrame | None,
]:
    """
    Generate out-of-fold predictions using nested cross-validation.

    Outer CV: Repeated stratified K-fold for robust OOF predictions
    Inner CV: Hyperparameter tuning via RandomizedSearchCV
    Optional RFECV: Feature selection within each fold (no leakage)

    Args:
        pipeline: Unfitted sklearn pipeline (pre + feature selection + clf)
        model_name: Model identifier (RF, XGBoost, LR_EN, LinSVM_cal)
        X: Training features (N x D)
        y: Training labels (N,)
        protein_cols: List of protein column names
        config: TrainingConfiguration object
        random_state: Random seed
        grid_rng: Optional RNG for grid randomization

    Returns:
        preds: OOF predictions (n_repeats x N) - each row is one repeat's predictions
               If calibration strategy is "oof_posthoc", predictions are calibrated.
        elapsed_sec: Training time in seconds
        best_params_df: DataFrame with best hyperparameters per fold
        selected_proteins_df: DataFrame with selected proteins per fold
        oof_calibrator: OOFCalibrator instance if strategy is "oof_posthoc", else None.
                        Use this calibrator for val/test predictions.
        rfecv_result: NestedRFECVResult if rfe_enabled, else None.
                      Contains consensus panel and fold-wise feature selection.
        oof_importance_df: DataFrame with OOF feature importance if compute_oof_importance=True,
                           else None. Contains aggregated importance across folds.

    Raises:
        RuntimeError: If any repeat has missing OOF predictions (CV split bug)
    """
    from .calibration import OOFCalibrator

    # Import RFECV utilities (lazy to avoid circular imports)
    rfecv_enabled = config.features.feature_selection_strategy == "rfecv"
    if rfecv_enabled:
        from ced_ml.features.nested_rfe import (
            RFECVFoldResult,
            aggregate_rfecv_results,
            extract_estimator_for_rfecv,
            run_rfecv_within_fold,
        )

    n_splits = config.cv.folds
    n_repeats = config.cv.repeats
    calibration_strategy = config.calibration.get_strategy_for_model(model_name)

    if n_repeats < 1:
        raise ValueError(f"cv.repeats must be >= 1, got {n_repeats}")

    n_samples = len(y)
    preds = np.full((n_repeats, n_samples), np.nan, dtype=float)
    best_params_rows: list[dict[str, Any]] = []
    selected_proteins_rows: list[dict[str, Any]] = []

    # RFECV tracking (if enabled)
    rfecv_fold_results: list[RFECVFoldResult] = [] if rfecv_enabled else []

    # OOF importance tracking (if enabled)
    fold_importances: list[pd.DataFrame] = []

    # Validate outer CV folds
    if n_splits < 2:
        raise ValueError(
            f"cv.folds must be >= 2 for cross-validation, got {n_splits}. "
            "In-sample predictions defeat the purpose of CV and lead to overfitting."
        )

    total_outer_folds = n_splits * n_repeats
    split_idx = 0
    t0 = time.perf_counter()

    # Setup outer CV splitter
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    split_iterator = rskf.split(X, y)
    split_divisor = n_splits

    # Outer CV loop
    for train_idx, test_idx in split_iterator:
        repeat_num = split_idx // split_divisor
        base_pipeline = clone(pipeline)

        # Log fold start and statistics
        fold_start = time.perf_counter()
        log_fold_header(logger, split_idx + 1, total_outer_folds, repeat_num)

        # Compute class distribution
        y_train = y[train_idx]
        y_val = y[test_idx]
        n_train_cases = int(np.sum(y_train))
        n_val_cases = int(np.sum(y_val))
        train_prevalence = n_train_cases / len(y_train) if len(y_train) > 0 else 0.0

        # Compact one-liner for train/val sizes
        logger.info(
            f"  Train: N={len(y_train):,} ({n_train_cases} cases) | "
            f"Val: N={len(y_val):,} ({n_val_cases} cases)"
        )

        # Warn about extreme class imbalance
        if train_prevalence < MIN_SAFE_PREVALENCE or train_prevalence > MAX_SAFE_PREVALENCE:
            logger.warning(
                f"Fold {split_idx+1}: Extreme class imbalance (prevalence={train_prevalence:.2%})"
            )

        # Handle XGBoost scale_pos_weight
        xgb_spw = None
        if model_name == ModelName.XGBoost:
            # Check if user specified explicit value via scale_pos_weight_grid
            spw_grid = getattr(config.xgboost, "scale_pos_weight_grid", None)
            if spw_grid and len(spw_grid) == 1 and spw_grid[0] > 0:
                xgb_spw = float(spw_grid[0])
            else:
                # Auto: compute from class distribution
                xgb_spw = compute_scale_pos_weight_from_y(y[train_idx])
            # Set scale_pos_weight - fail fast if parameter doesn't exist (structural bug)
            base_pipeline.set_params(clf__scale_pos_weight=float(xgb_spw))

        # Build inner CV hyperparameter search
        search = _build_hyperparameter_search(
            base_pipeline, model_name, config, random_state, xgb_spw, grid_rng
        )

        # Fit model (with or without search)
        if search is None:
            base_pipeline.fit(X.iloc[train_idx], y[train_idx])
            fitted_model = base_pipeline
            best_params, best_score = {}, np.nan
        else:
            # Use loky backend for multi-threaded search to avoid thread oversubscription
            if getattr(search, "n_jobs", 1) and int(search.n_jobs) > 1:
                with parallel_backend("loky", inner_max_num_threads=1):
                    search.fit(X.iloc[train_idx], y[train_idx])
            else:
                search.fit(X.iloc[train_idx], y[train_idx])

            fitted_model = search.best_estimator_
            best_params = search.best_params_
            best_score = float(search.best_score_)

        # Optional post-hoc calibration (only for per_fold strategy)
        # For oof_posthoc, calibration happens after CV loop completes
        fitted_model = _apply_per_fold_calibration(
            fitted_model,
            model_name,
            config,
            X.iloc[train_idx],
            y[train_idx],
        )

        # Extract selected proteins from this fold (initial feature selection)
        selected_proteins = _extract_selected_proteins_from_fold(
            fitted_model,
            model_name,
            protein_cols,
            config,
            X.iloc[train_idx],
            y[train_idx],
            random_state,
        )

        # --- OOF Importance computation (if enabled) ---
        if getattr(config.features, "compute_oof_importance", False):
            from ced_ml.features.importance import extract_importance_from_model

            try:
                # Use grouped (cluster-aware) importance to handle correlated features
                # Trees: OOF grouped permutation importance on held-out fold
                # Linear: Standardized |coef| aggregated by correlation clusters
                oof_grouped = getattr(config.features, "oof_importance_grouped", True)
                oof_corr_threshold = getattr(config.features, "oof_corr_threshold", 0.85)

                # Prepare validation data for grouped permutation importance (trees)
                X_val_fold = X.iloc[test_idx][protein_cols] if oof_grouped else None
                y_val_fold = y[test_idx] if oof_grouped else None

                fold_importance = extract_importance_from_model(
                    fitted_model,
                    model_name,
                    protein_cols,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    grouped=oof_grouped,
                    corr_threshold=oof_corr_threshold,
                )
                if not fold_importance.empty:
                    fold_importance["repeat"] = repeat_num
                    fold_importance["outer_split"] = split_idx
                    fold_importances.append(fold_importance)
            except Exception as e:
                logger.warning(f"Fold {split_idx}: Importance extraction failed: {e}")

        # --- RFECV within fold (if enabled) ---
        if rfecv_enabled and selected_proteins:
            try:
                logger.info(
                    f"Fold {split_idx}: Running RFECV on {len(selected_proteins)} proteins..."
                )

                # Extract estimator for RFECV
                estimator = extract_estimator_for_rfecv(fitted_model)

                # Prepare data with only selected proteins
                X_train_proteins = X.iloc[train_idx][selected_proteins]
                X_test_proteins = X.iloc[test_idx][selected_proteins]

                # Optional k-best pre-filter to reduce computational cost
                if getattr(config.features, "rfe_kbest_prefilter", True):
                    k_max = getattr(config.features, "rfe_kbest_k", 100)
                    if len(selected_proteins) > k_max:
                        from sklearn.feature_selection import SelectKBest, f_classif

                        logger.info(
                            f"Fold {split_idx}: Applying k-best pre-filter "
                            f"({len(selected_proteins)} → {k_max} proteins) before RFECV..."
                        )
                        selector = SelectKBest(f_classif, k=k_max)
                        selector.fit(X_train_proteins, y[train_idx])
                        kbest_mask = selector.get_support()
                        selected_proteins = [
                            p
                            for p, keep in zip(selected_proteins, kbest_mask, strict=False)
                            if keep
                        ]
                        X_train_proteins = X_train_proteins.iloc[:, kbest_mask]
                        X_test_proteins = X_test_proteins.iloc[:, kbest_mask]
                        logger.info(
                            f"Fold {split_idx}: K-best pre-filter retained {len(selected_proteins)} proteins"
                        )

                # Resolve RFECV parameters from config
                rfecv_min_features = max(5, getattr(config.features, "rfe_target_size", 50) // 2)
                rfecv_step = _resolve_rfecv_step(
                    getattr(config.features, "rfe_step_strategy", "adaptive"),
                    len(selected_proteins),
                )

                # Run RFECV within this fold
                rfecv_result = run_rfecv_within_fold(
                    X_train_fold=X_train_proteins,
                    y_train_fold=y[train_idx],
                    X_val_fold=X_test_proteins,
                    y_val_fold=y[test_idx],
                    estimator=estimator,
                    feature_names=selected_proteins,
                    fold_idx=split_idx,
                    min_features=rfecv_min_features,
                    step=rfecv_step,
                    cv_folds=getattr(config.cv, "inner_folds", 3),
                    scoring="roc_auc",
                    n_jobs=getattr(config, "n_jobs", -1),
                    random_state=random_state,
                )
                rfecv_fold_results.append(rfecv_result)

                # Update selected_proteins to RFECV-selected features
                selected_proteins = rfecv_result.selected_features
                logger.info(
                    f"Fold {split_idx}: RFECV reduced to {len(selected_proteins)} proteins "
                    f"(val AUROC = {rfecv_result.val_auroc:.4f})"
                )

                # CRITICAL: Retrain model on RFECV-selected features only
                # Prepare RFECV-filtered data (proteins + metadata)
                # Extract metadata columns (non-protein columns)
                metadata_cols = [c for c in X.columns if c not in protein_cols]
                rfecv_feature_cols = selected_proteins + metadata_cols

                X_train_rfecv = X.iloc[train_idx][rfecv_feature_cols]
                y_train_rfecv = y[train_idx]

                # Clone the pipeline and retrain on RFECV features
                pipeline_rfecv = clone(search.best_estimator_)

                # Update screener's protein_cols to match RFECV-selected proteins
                # (the original screener expects all protein_cols, but X_train_rfecv
                # only contains the RFECV-selected subset)
                if (
                    hasattr(pipeline_rfecv, "named_steps")
                    and "screen" in pipeline_rfecv.named_steps
                ):
                    pipeline_rfecv.named_steps["screen"].protein_cols = selected_proteins

                pipeline_rfecv.fit(X_train_rfecv, y_train_rfecv)

                # Apply calibration if needed
                fitted_model = _apply_per_fold_calibration(
                    pipeline_rfecv, model_name, config, X_train_rfecv, y_train_rfecv
                )

                logger.info(
                    f"Fold {split_idx}: Retrained model on {len(selected_proteins)} RFECV-selected proteins"
                )

            except (ValueError, RuntimeError, MemoryError) as e:
                logger.warning(
                    f"Fold {split_idx}: RFECV failed ({type(e).__name__}: {e}). "
                    "Using initial selection."
                )

        # Generate OOF predictions for this fold
        # Note: If RFECV was applied, fitted_model was retrained on selected features
        proba = fitted_model.predict_proba(X.iloc[test_idx])[:, 1]
        proba = np.clip(proba, 0.0, 1.0)
        preds[repeat_num, test_idx] = proba

        # Record best hyperparameters
        best_params_row = {
            "model": model_name,
            "repeat": repeat_num,
            "outer_split": split_idx,
            "best_score_inner": best_score,
            "best_params": json.dumps(_convert_numpy_types(best_params), sort_keys=True),
        }

        # Add Optuna-specific metadata if available
        if (
            search is not None
            and hasattr(search, "study_")
            and search.study_ is not None
            and hasattr(search, "n_trials_")
        ):
            best_params_row["optuna_n_trials"] = search.n_trials_
            best_params_row["optuna_sampler"] = config.optuna.sampler
            best_params_row["optuna_pruner"] = config.optuna.pruner

        best_params_rows.append(best_params_row)

        # Record selected proteins (post-RFECV if enabled)
        if selected_proteins:
            selected_proteins_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat_num,
                    "outer_split": split_idx,
                    "n_selected_proteins": len(selected_proteins),
                    # Store as JSON string (CSV cannot store native lists)
                    "selected_proteins": json.dumps(sorted(selected_proteins)),
                    "rfecv_applied": rfecv_enabled,
                }
            )

        # Log fold completion summary
        fold_duration = time.perf_counter() - fold_start

        # Get actual feature count used by the model
        n_features_used = _get_model_feature_count(fitted_model)

        pct = 100 * (split_idx + 1) / total_outer_folds
        logger.info(
            f"  Fold completed in {fold_duration:.1f}s | score={best_score:.3f} | "
            f"features={n_features_used} | [{pct:.0f}%]"
        )

        split_idx += 1

    elapsed_sec = time.perf_counter() - t0

    # Explicit memory cleanup after CV loop
    import gc

    gc.collect()

    # Validate: no missing OOF predictions
    for r in range(n_repeats):
        if np.isnan(preds[r]).any():
            raise RuntimeError(f"Repeat {r} has missing OOF predictions. Check CV splitting logic.")

    # Handle oof_posthoc calibration: fit calibrator on pooled OOF predictions
    oof_calibrator = None
    if calibration_strategy == "oof_posthoc":
        logger.info(
            f"Fitting OOF calibrator (method={config.calibration.method}) on pooled OOF predictions..."
        )
        # Use mean across repeats for calibration fitting
        mean_oof_preds = np.nanmean(preds, axis=0)
        oof_calibrator = OOFCalibrator(method=config.calibration.method)
        oof_calibrator.fit(mean_oof_preds, y)
        logger.info("OOF calibrator fitted successfully")

        # Apply calibration to OOF predictions for consistent reporting
        for r in range(n_repeats):
            preds[r] = oof_calibrator.transform(preds[r])

    # Aggregate RFECV results (if enabled)
    nested_rfecv_result = None
    if rfecv_enabled and rfecv_fold_results:
        consensus_threshold = getattr(config.features, "rfe_consensus_thresh", 0.80)
        nested_rfecv_result = aggregate_rfecv_results(
            rfecv_fold_results, consensus_threshold=consensus_threshold
        )
        logger.info(
            f"RFECV complete: consensus panel has {len(nested_rfecv_result.consensus_panel)} proteins "
            f"(mean optimal size: {nested_rfecv_result.mean_optimal_size:.1f})"
        )

    # Aggregate OOF importance across folds
    oof_importance_df = None
    if fold_importances and getattr(config.features, "compute_oof_importance", False):
        from ced_ml.features.importance import aggregate_fold_importances

        oof_importance_df = aggregate_fold_importances(fold_importances)
        logger.info(f"OOF importance computed for {len(oof_importance_df)} features")

    return (
        preds,
        elapsed_sec,
        pd.DataFrame(best_params_rows),
        pd.DataFrame(selected_proteins_rows),
        oof_calibrator,
        nested_rfecv_result,
        oof_importance_df,
    )


def _resolve_rfecv_step(strategy: str, n_features: int) -> int | float:
    """
    Resolve RFECV step parameter from strategy string.

    Args:
        strategy: Step strategy ("adaptive", "linear", "geometric", or integer string)
        n_features: Number of features being evaluated

    Returns:
        Step size (int) or fraction (float) for RFECV

    Notes:
        - "adaptive"/"geometric": Remove ~10% of features per iteration (fast)
        - "linear": Remove 1 feature per iteration (thorough but slow)
        - Integer string: Use as fixed step size
    """
    if strategy in ("adaptive", "geometric"):
        # Remove ~10% per iteration (sklearn RFECV convention for geometric)
        return max(1, int(0.1 * n_features))
    elif strategy == "linear":
        # Remove 1 feature per iteration (thorough)
        return 1
    else:
        # Try parsing as integer
        try:
            step = int(strategy)
            return max(1, step)
        except (ValueError, TypeError):
            logger.warning(f"Invalid rfe_step_strategy '{strategy}', defaulting to adaptive (10%)")
            return max(1, int(0.1 * n_features))


def _scoring_to_direction(scoring: str) -> str:
    """
    Infer Optuna optimization direction from sklearn scoring metric.

    Args:
        scoring: sklearn scoring string (e.g., "roc_auc", "average_precision")

    Returns:
        "maximize" or "minimize"
    """
    # Metrics that should be maximized
    maximize_metrics = {
        "roc_auc",
        "average_precision",
        "f1",
        "f1_weighted",
        "f1_micro",
        "f1_macro",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "jaccard",
        "tpr_at_fpr",
    }

    # neg_* metrics are also maximized (sklearn convention)
    if scoring in maximize_metrics or scoring.startswith("neg_"):
        return "maximize"

    return "maximize"  # Default to maximize for unknown metrics


def _build_hyperparameter_search(
    pipeline: Pipeline,
    model_name: str,
    config: TrainingConfig,
    random_state: int,
    xgb_spw: float | None,
    grid_rng: np.random.Generator | None,
):
    """
    Build hyperparameter search object (Optuna or RandomizedSearchCV).

    Returns None if:
    - Model has no hyperparameters to tune
    - config.cv.inner_folds < 2 (tuning disabled)
    - n_iter < 1 (tuning disabled, for RandomizedSearchCV; uses get_model_n_iter())

    Args:
        pipeline: Base pipeline to tune
        model_name: Model identifier
        config: TrainingConfiguration object
        random_state: Random seed
        xgb_spw: XGBoost scale_pos_weight (if applicable)
        grid_rng: Optional RNG for grid randomization

    Returns:
        OptunaSearchCV, RandomizedSearchCV, or None
    """
    # Validate inner CV settings
    inner_folds = config.cv.inner_folds
    if inner_folds < 2:
        logger.info(
            f"[tune] WARNING: inner_folds={inner_folds} < 2; skipping hyperparameter search."
        )
        return None

    # Build inner CV splitter
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    # === Optuna path ===
    if config.optuna.enabled:
        from .hyperparams import get_param_distributions_optuna
        from .optuna_search import OptunaSearchCV

        # Get Optuna-format parameter distributions
        param_dists = get_param_distributions_optuna(model_name, config, xgb_spw=xgb_spw)

        if not param_dists:
            logger.info(f"[optuna] No tunable params for {model_name}; skipping search.")
            return None

        # Determine direction from config or scoring metric
        direction = config.optuna.direction
        if direction is None:
            direction = _scoring_to_direction(config.cv.scoring)

        logger.info(
            f"[optuna] Using Optuna: {config.optuna.n_trials} trials, "
            f"sampler={config.optuna.sampler}, direction={direction}"
        )

        # Get scorer (handles custom scorers like tpr_at_fpr)
        scorer = get_scorer(config.cv.scoring, target_fpr=config.cv.scoring_target_fpr)

        return OptunaSearchCV(
            estimator=pipeline,
            param_distributions=param_dists,
            n_trials=config.optuna.n_trials,
            timeout=config.optuna.timeout,
            scoring=scorer,
            cv=inner_cv,
            n_jobs=config.optuna.n_jobs,
            random_state=random_state,
            refit=True,
            direction=direction,
            sampler=config.optuna.sampler,
            sampler_seed=config.optuna.sampler_seed,
            pruner=config.optuna.pruner,
            pruner_n_startup_trials=config.optuna.pruner_n_startup_trials,
            pruner_percentile=config.optuna.pruner_percentile,
            storage=config.optuna.storage,
            study_name=config.optuna.study_name,
            load_if_exists=config.optuna.load_if_exists,
            verbose=0,
            # Multi-objective parameters
            multi_objective=config.optuna.multi_objective,
            objectives=config.optuna.objectives,
            pareto_selection=config.optuna.pareto_selection,
        )

    # === RandomizedSearchCV path (default) ===
    from sklearn.model_selection import RandomizedSearchCV

    from .hyperparams import get_param_distributions

    n_iter = get_model_n_iter(model_name, config)
    if n_iter < 1:
        logger.warning(f"[tune] WARNING: n_iter={n_iter} < 1; skipping hyperparameter search.")
        return None

    logger.info(f"[tune] {model_name}: using n_iter={n_iter}")

    # Get sklearn-format parameter distributions
    param_dists = get_param_distributions(model_name, config, xgb_spw=xgb_spw, grid_rng=grid_rng)

    if not param_dists:
        return None

    # Determine parallelization strategy
    n_jobs = _get_search_n_jobs(model_name, config)

    # Get scorer (handles custom scorers like tpr_at_fpr)
    scorer = get_scorer(config.cv.scoring, target_fpr=config.cv.scoring_target_fpr)

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dists,
        n_iter=n_iter,
        scoring=scorer,
        cv=inner_cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        refit=True,
        random_state=random_state,
        error_score="raise",  # Fail fast on errors
        verbose=0,
    )


def _get_search_n_jobs(model_name: str, config: TrainingConfig) -> int:
    """
    Determine n_jobs for RandomizedSearchCV.

    Strategy:
    - LR/SVM: Parallelize search (models are single-threaded)
    - RF/XGBoost: Keep search single-threaded (estimators use internal parallelism)

    Args:
        model_name: Model identifier
        config: TrainingConfiguration object

    Returns:
        n_jobs value (>= 1)
    """
    tune_n_jobs = config.compute.tune_n_jobs
    cpus = config.compute.cpus

    if tune_n_jobs is not None:
        # Explicit override
        return max(1, min(cpus, tune_n_jobs))

    # Auto strategy
    if model_name in (ModelName.LR_EN, ModelName.LR_L1, ModelName.LinSVM_cal):
        # Parallelize search for single-threaded models
        return max(1, cpus)
    else:
        # Keep search single-threaded for models with internal parallelism
        return 1


def _apply_per_fold_calibration(
    estimator,
    model_name: str,
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
):
    """
    Apply optional post-hoc probability calibration (per-fold strategy only).

    This function is called during the CV loop. It only applies calibration
    when the strategy is "per_fold". For "oof_posthoc" and "none" strategies,
    it returns the estimator unchanged.

    Rules:
    - strategy="none": Return estimator unchanged
    - strategy="oof_posthoc": Return estimator unchanged (calibration happens post-CV)
    - strategy="per_fold": Apply CalibratedClassifierCV (current behavior)
    - LinSVM_cal: Already calibrated (skip regardless of strategy)
    - Already wrapped: Skip double-calibration

    Args:
        estimator: Fitted estimator or pipeline
        model_name: Model identifier
        config: TrainingConfiguration object
        X_train: Training features (for calibration CV)
        y_train: Training labels (for calibration CV)

    Returns:
        Calibrated or original estimator
    """
    # Get effective strategy for this model
    strategy = config.calibration.get_strategy_for_model(model_name)

    # For none or oof_posthoc, return unchanged
    if strategy in ("none", "oof_posthoc"):
        return estimator

    # per_fold strategy: apply CalibratedClassifierCV
    # SVM is already calibrated
    if model_name == ModelName.LinSVM_cal:
        return estimator

    # Don't double-calibrate
    if isinstance(estimator, CalibratedClassifierCV):
        return estimator

    # Determine appropriate number of CV folds based on class sizes
    # Need at least 2 samples per class per fold
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = int(np.min(class_counts))

    # Use min(configured_cv, min_class_count) to ensure we have enough samples
    # Need at least 2 samples per fold for the minority class
    max_safe_cv = max(2, min_class_count)
    effective_cv = min(config.calibration.cv, max_safe_cv)

    # Wrap with calibration
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method=config.calibration.method, cv=effective_cv
    )

    # Fit on training fold
    calibrated.fit(X_train, y_train)
    return calibrated


def _get_model_feature_count(fitted_model) -> int:
    """
    Get the number of features used by a fitted model.

    Args:
        fitted_model: Fitted pipeline or calibrated estimator

    Returns:
        Number of features the model was trained on
    """
    # Unwrap CalibratedClassifierCV if needed
    if isinstance(fitted_model, CalibratedClassifierCV):
        if hasattr(fitted_model, "estimator"):
            model = fitted_model.estimator
        else:
            model = getattr(fitted_model, "base_estimator", fitted_model)
    else:
        model = fitted_model

    # Try to get from the final estimator
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][1]
    else:
        final_step = model

    # Try different attributes
    if hasattr(final_step, "n_features_in_"):
        return int(final_step.n_features_in_)
    elif hasattr(final_step, "coef_"):
        # Linear models
        coef = final_step.coef_
        if coef.ndim == 1:
            return len(coef)
        else:
            return coef.shape[1]
    elif hasattr(final_step, "feature_importances_"):
        # Tree models
        return len(final_step.feature_importances_)

    return 0


def _extract_selected_proteins_from_fold(
    fitted_model,
    model_name: str,
    protein_cols: list[str],
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
    nested_rfecv_result=None,
) -> list[str]:
    """
    Extract selected proteins from a fitted model/pipeline.

    Strategy depends on model type and feature selection method:
    - Linear models (LR/SVM): Extract proteins with |coef| > threshold
    - Tree models (RF/XGBoost): Use permutation importance (if enabled)
    - K-best selection: Extract from SelectKBest step
    - Screening: Extract from TrainOnlyScreenSelector step
    - RFECV: Use consensus panel from nested CV

    Args:
        fitted_model: Fitted pipeline or calibrated estimator
        model_name: Model identifier
        protein_cols: List of protein column names
        config: TrainingConfiguration object
        X_train: Training fold features (for permutation importance)
        y_train: Training fold labels (for permutation importance)
        random_state: Random seed
        nested_rfecv_result: NestedRFECVResult with consensus panel (if RFECV enabled)

    Returns:
        List of selected protein names (sorted)
    """
    # Import here to avoid circular dependency
    from ..models.calibration import OOFCalibratedModel

    # Unwrap OOFCalibratedModel if needed
    if isinstance(fitted_model, OOFCalibratedModel):
        pipeline = fitted_model.base_model
    # Unwrap CalibratedClassifierCV if needed
    elif isinstance(fitted_model, CalibratedClassifierCV):
        if hasattr(fitted_model, "estimator"):
            pipeline = fitted_model.estimator
        else:
            # Older sklearn uses base_estimator
            pipeline = getattr(fitted_model, "base_estimator", fitted_model)
    else:
        pipeline = fitted_model

    if not isinstance(pipeline, Pipeline):
        return []

    selected_proteins = set()

    # Strategy 1: Extract from K-best selection (if present)
    # NOTE: Screening step is NOT included - it's a pre-filter, not the final selection
    strategy = config.features.feature_selection_strategy

    # For RFECV strategy: use consensus panel from nested CV if available
    if strategy == "rfecv" and nested_rfecv_result is not None:
        consensus_panel = nested_rfecv_result.consensus_panel
        if consensus_panel:
            return sorted(consensus_panel)
        # Fallback: if no consensus panel, return empty (screening is not the final selection)
        return []

    if strategy in ("hybrid_stability", "rfecv"):
        if "sel" in pipeline.named_steps:
            # K-best selection
            kbest_proteins = _extract_from_kbest_transformed(pipeline, protein_cols)
            if kbest_proteins:
                selected_proteins.update(kbest_proteins)

    # For RFECV strategy: if no k-best proteins extracted, use screening output
    # This ensures RFECV has proteins to work with when there's no "sel" step
    if strategy == "rfecv" and not selected_proteins and "screen" in pipeline.named_steps:
        screen_proteins = getattr(pipeline.named_steps["screen"], "selected_features_", [])
        if screen_proteins:
            selected_proteins.update(screen_proteins)

    # Strategy 1b: Extract from model-specific selector (if present)
    if "model_sel" in pipeline.named_steps:
        model_sel_step = pipeline.named_steps["model_sel"]
        if hasattr(model_sel_step, "get_feature_names_out"):
            sel_names = model_sel_step.get_feature_names_out()
            # Map back to protein names
            model_sel_proteins = set()
            for name in sel_names:
                if name in protein_cols:
                    model_sel_proteins.add(name)
                elif name.startswith("num__"):
                    orig = name[len("num__") :]
                    if orig in protein_cols:
                        model_sel_proteins.add(orig)
            if model_sel_proteins:
                # model_sel is the final selection -- override kbest output
                return sorted(model_sel_proteins)

    # Strategy 2: Extract from model coefficients (linear models)
    # Only relevant for hybrid_stability strategy
    if strategy == "hybrid_stability":
        model_proteins = _extract_from_model_coefficients(
            pipeline, model_name, protein_cols, config
        )
        if model_proteins:
            selected_proteins.update(model_proteins)

    # Strategy 3: Permutation importance for RF (if enabled and hybrid mode)
    if strategy == "hybrid_stability" and model_name == "RF" and config.features.rf_use_permutation:
        perm_proteins = _extract_from_rf_permutation(
            pipeline, X_train, y_train, protein_cols, config, random_state
        )
        if perm_proteins:
            selected_proteins.update(perm_proteins)

    return sorted(selected_proteins)


def _extract_from_kbest_transformed(pipeline: Pipeline, protein_cols: list[str]) -> set:
    """Extract protein names from SelectKBest in transformed space."""
    if "sel" not in pipeline.named_steps:
        return set()

    # Get feature names from preprocessing step
    pre = pipeline.named_steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return set()

    feature_names = pre.get_feature_names_out()
    support = pipeline.named_steps["sel"].get_support()
    selected_names = feature_names[support]

    # Extract protein columns (handle different feature naming patterns)
    proteins = set()
    for name in selected_names:
        # Handle different feature naming patterns via extract_protein_name
        if name in protein_cols:
            proteins.add(name)
        else:
            orig = extract_protein_name(name)
            if orig in protein_cols:
                proteins.add(orig)

    return proteins


def _extract_from_model_coefficients(
    pipeline: Pipeline, model_name: str, protein_cols: list[str], config: TrainingConfig
) -> set:
    """Extract protein names from linear model coefficients."""
    coef_thresh = config.features.coef_threshold

    # Get feature names
    pre = pipeline.named_steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return set()

    feature_names = pre.get_feature_names_out()

    # Apply K-best mask if present
    if "sel" in pipeline.named_steps:
        support = pipeline.named_steps["sel"].get_support()
        feature_names = feature_names[support]

    # Apply model-specific selector mask if present
    if "model_sel" in pipeline.named_steps:
        support = pipeline.named_steps["model_sel"].get_support()
        feature_names = feature_names[support]

    # Extract coefficients
    clf = pipeline.named_steps["clf"]

    # Handle CalibratedClassifierCV wrapper for LinSVM
    if model_name == ModelName.LinSVM_cal and hasattr(clf, "calibrated_classifiers_"):
        # Average coefficients across calibration folds
        coefs_list = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est and hasattr(est, "coef_"):
                coefs_list.append(est.coef_.ravel())

        if not coefs_list:
            return set()

        coefs = np.mean(np.vstack(coefs_list), axis=0)

    elif hasattr(clf, "coef_"):
        # Standard linear model
        coefs = clf.coef_.ravel()

    else:
        return set()

    # Sanity check
    if len(feature_names) != len(coefs):
        logger.warning(
            f"[extract] WARNING: Feature names length ({len(feature_names)}) != "
            f"coef length ({len(coefs)}); skipping extraction"
        )
        return set()

    # Extract proteins with |coef| > threshold
    proteins = set()
    for name, c in zip(feature_names, coefs, strict=False):
        orig = extract_protein_name(name)
        if orig in protein_cols and abs(c) > coef_thresh:
            proteins.add(orig)

    return proteins


def _extract_from_rf_permutation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
    config: TrainingConfig,
    random_state: int,
) -> set:
    """Extract important proteins from RF using permutation importance."""
    from sklearn.inspection import permutation_importance

    # Get feature names
    pre = pipeline.named_steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return set()

    feature_names = pre.get_feature_names_out()

    # Apply K-best mask if present
    if "sel" in pipeline.named_steps:
        support = pipeline.named_steps["sel"].get_support()
        feature_names = feature_names[support]

    # Apply model-specific selector mask if present
    if "model_sel" in pipeline.named_steps:
        support = pipeline.named_steps["model_sel"].get_support()
        feature_names = feature_names[support]

    # Compute permutation importance
    try:
        # Get scorer (handles custom scorers like tpr_at_fpr)
        scorer = get_scorer(config.cv.scoring, target_fpr=config.cv.scoring_target_fpr)

        perm_result = permutation_importance(
            pipeline,
            X_train,
            y_train,
            scoring=scorer,
            n_repeats=config.features.rf_perm_repeats,
            random_state=random_state,
            n_jobs=1,  # Already inside parallel context
        )
        importances = perm_result.importances_mean
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.warning(
            f"[perm] Permutation importance failed ({type(e).__name__}: {e}). "
            "Skipping permutation-based selection."
        )
        return set()

    # Sanity check
    if len(feature_names) != len(importances):
        return set()

    # Aggregate protein importances (sum across transformed features)
    protein_importance: dict[str, float] = {}
    for name, imp in zip(feature_names, importances, strict=False):
        if not np.isfinite(imp):
            continue

        orig = extract_protein_name(name)
        if orig in protein_cols:
            protein_importance[orig] = protein_importance.get(orig, 0.0) + float(imp)

    if not protein_importance:
        return set()

    # Filter by minimum importance
    min_imp = config.features.rf_perm_min_importance
    filtered = [(p, v) for p, v in protein_importance.items() if v >= min_imp]

    if not filtered:
        # If all below threshold, keep all
        filtered = list(protein_importance.items())

    # Sort by importance and take top N
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_n = config.features.rf_perm_top_n
    top_proteins = [p for p, _ in filtered[:top_n]]

    return set(top_proteins)
