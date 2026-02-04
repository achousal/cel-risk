"""
CLI implementation for train command.

This module provides the train command for the CeD-ML pipeline.
The training workflow is orchestrated through discrete stages in
the cli/orchestration package.

Helper functions in this module (validate_protein_columns, build_preprocessor,
build_training_pipeline, load_split_indices, evaluate_on_split) are used by
the orchestration stages.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ced_ml.config.loader import load_training_config
from ced_ml.config.validation import check_split_overlap, validate_training_config
from ced_ml.data.schema import METRIC_BRIER
from ced_ml.features.kbest import build_kbest_pipeline_step
from ced_ml.metrics.discrimination import (
    compute_brier_score,
    compute_discrimination_metrics,
)
from ced_ml.metrics.threshold_strategy import get_threshold_strategy
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    compute_multi_target_specificity_metrics,
)
from ced_ml.models.calibration import (
    calibration_intercept_slope,
    expected_calibration_error,
)
from ced_ml.models.prevalence import adjust_probabilities_for_prevalence
from ced_ml.utils.logging import log_section, setup_logger

logger = logging.getLogger(__name__)


def validate_protein_columns(
    df: pd.DataFrame,
    protein_cols: list[str],
    logger: Any,
) -> None:
    """
    Validate that protein columns are numeric and log NaN statistics.

    Args:
        df: DataFrame containing protein columns
        protein_cols: List of protein column names to validate
        logger: Logger instance for warnings

    Raises:
        ValueError: If protein columns contain non-numeric dtypes
    """
    # Check dtypes
    non_numeric = []
    for col in protein_cols:
        if col in df.columns:
            dtype = df[col].dtype
            if not pd.api.types.is_numeric_dtype(dtype):
                non_numeric.append(f"{col} ({dtype})")

    if non_numeric:
        raise ValueError(
            f"Protein columns must be numeric but found non-numeric dtypes: {non_numeric[:5]}"
        )

    # Log NaN statistics (warning if >0)
    protein_df = df[protein_cols]
    nan_counts = protein_df.isna().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        cols_with_nans = nan_counts[nan_counts > 0]
        logger.warning(
            f"Found {total_nans:,} NaN values across {len(cols_with_nans)} protein columns"
        )
        logger.warning(f"Top 5 columns with NaNs: {cols_with_nans.nlargest(5).to_dict()}")
        logger.warning("StandardScaler will propagate NaNs - consider imputation if needed")
    else:
        logger.info("Protein columns validated: all numeric, no NaN values")


def build_preprocessor(cat_cols: list[str]) -> ColumnTransformer:
    """
    Build preprocessing pipeline for model training.

    Uses sklearn.compose.make_column_selector to dynamically select numeric and
    categorical columns at fit time, making the preprocessor robust to upstream
    feature screening that reduces the protein column set.

    Args:
        cat_cols: List of categorical column names

    Returns:
        ColumnTransformer with StandardScaler for numeric and OneHotEncoder for categorical
    """
    from sklearn.compose import make_column_selector

    # Dynamic selector: scale all numeric columns present at fit time
    # This handles cases where screening reduces protein columns
    numeric_selector = make_column_selector(dtype_include="number")

    transformers = [
        ("num", StandardScaler(), numeric_selector),
    ]

    if cat_cols:
        # Categorical columns are explicit (not reduced by screening)
        transformers.append(
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)


def build_training_pipeline(
    config: Any,
    classifier: Any,
    protein_cols: list[str],
    cat_cols: list[str],
    model_name: str | None = None,
) -> Pipeline:
    """Build complete training pipeline with preprocessing, feature selection, and classifier.

    Two mutually exclusive feature selection strategies:
    1. hybrid_stability: screen → kbest (tuned k_grid) → stability → model
    2. rfecv: screen → (light kbest cap) → RFECV (within CV) → model

    Pipeline order (hybrid_stability):
    1. Screening: Univariate Mann-Whitney/F-test → keep top-N proteins (operates on raw DataFrame)
    2. Preprocessing: StandardScaler + OneHotEncoder (adds metadata columns)
    3. SelectKBest: Tuned k value (from k_grid) during CV
    4. Classifier

    Pipeline order (rfecv):
    1. Screening: Univariate Mann-Whitney/F-test → keep top-N proteins
    2. Preprocessing: StandardScaler + OneHotEncoder
    3. (Optional light kbest cap if enabled)
    4. RFECV (handled within CV loop, not as pipeline step)
    5. Classifier

    Args:
        config: TrainingConfig object
        classifier: Unfitted sklearn classifier
        protein_cols: List of protein column names
        cat_cols: List of categorical column names

    Returns:
        Pipeline with named steps: [screen], pre, [sel], clf

    Example (hybrid_stability):
        >>> config.features.feature_selection_strategy = "hybrid_stability"
        >>> config.features.screen_top_n = 1000
        >>> config.features.k_grid = [25, 50, 100, 200, 400, 800]
        Pipeline stages: screen → pre → sel → clf

    Example (rfecv):
        >>> config.features.feature_selection_strategy = "rfecv"
        >>> config.features.screen_top_n = 1000
        Pipeline stages: screen → pre → clf (RFECV runs within CV loop)
    """
    logger = logging.getLogger(__name__)

    strategy = config.features.feature_selection_strategy
    steps = []

    # Stage 0: Screening (BEFORE preprocessing, operates on raw DataFrame)
    # Common to both strategies
    screen_top_n = getattr(config.features, "screen_top_n", 0)
    screen_method = getattr(config.features, "screen_method", "mannwhitney")

    if strategy != "none" and screen_top_n > 0 and screen_method:
        from ced_ml.features.kbest import ScreeningTransformer

        screener = ScreeningTransformer(
            method=screen_method,
            top_n=screen_top_n,
            protein_cols=protein_cols,
        )
        steps.append(("screen", screener))
        logger.debug(f"Feature screening: {screen_method} → top {screen_top_n} features")

    # Stage 1: Preprocessing (StandardScaler + OneHotEncoder)
    # Note: If screening is enabled, protein_cols will be reduced by screener
    preprocessor = build_preprocessor(cat_cols)
    steps.append(("pre", preprocessor))

    # Stage 2: Strategy-specific feature selection
    if strategy == "hybrid_stability":
        # Hybrid+Stability: SelectKBest with k_grid tuning
        k_grid = getattr(config.features, "k_grid", None)
        if k_grid:
            # Use first k value as default (will be tuned during CV)
            k_default = k_grid[0] if isinstance(k_grid, list | tuple) else k_grid
            kbest = build_kbest_pipeline_step(k=k_default)
            steps.append(("sel", kbest))
            logger.debug(f"Feature selection: hybrid_stability with k_grid={k_grid}")
        else:
            # Fallback: use kbest_max as fixed k (no tuning)
            k_val = getattr(config.features, "kbest_max", 500)
            kbest = build_kbest_pipeline_step(k=k_val)
            steps.append(("sel", kbest))
            logger.debug(f"Feature selection: hybrid_stability with fixed k={k_val}")

    elif strategy == "rfecv":
        # RFECV: No SelectKBest here (RFECV runs within CV loop in training.py)
        # Optional: add a light kbest cap if you want to limit RFECV search space
        # For now, RFECV operates on all screened features
        logger.debug("Feature selection: rfecv (RFECV runs within CV loop)")

    elif strategy == "none":
        logger.debug("Feature selection: none")

    # Stage 2b: Model-specific selector (hybrid_stability only, opt-in)
    if (
        strategy == "hybrid_stability"
        and getattr(config.features, "model_selector", False)
        and model_name is not None
    ):
        from ced_ml.features.model_selector import ModelSpecificSelector

        model_sel = ModelSpecificSelector(
            model_name=model_name,
            threshold=getattr(config.features, "model_selector_threshold", "median"),
            max_features=getattr(config.features, "model_selector_max_features", None),
            min_features=getattr(config.features, "model_selector_min_features", 10),
        )
        steps.append(("model_sel", model_sel))
        logger.debug(
            "Model-specific selector: %s (threshold=%s, min=%d)",
            model_name,
            config.features.model_selector_threshold,
            config.features.model_selector_min_features,
        )

    # Stage 3: Classifier
    steps.append(("clf", classifier))

    return Pipeline(steps=steps)


def load_split_indices(
    split_dir: str, scenario: str | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load train/val/test split indices from CSVs.

    Args:
        split_dir: Directory containing split CSV files
        scenario: Scenario name (e.g., IncidentOnly). If None, auto-detect from files.
        seed: Random seed used for splits

    Returns:
        (train_idx, val_idx, test_idx, detected_scenario) as tuple

    Raises:
        FileNotFoundError: If any split file is missing
    """
    split_path = Path(split_dir)

    # Auto-detect scenario if not provided
    if scenario is None:
        import glob

        pattern = str(split_path / f"train_idx_*_seed{seed}.csv")
        matches = glob.glob(pattern)
        if matches:
            # Extract scenario from filename: train_idx_{scenario}_seed{seed}.csv
            filename = Path(matches[0]).name
            scenario = filename.replace("train_idx_", "").replace(f"_seed{seed}.csv", "")
            logger.info(f"Auto-detected scenario from split files: {scenario}")
        else:
            raise FileNotFoundError(
                f"No split files found for seed={seed} in {split_dir}.\n"
                f"Run 'ced save-splits' first."
            )

    # Load split files (new format with scenario)
    train_file = split_path / f"train_idx_{scenario}_seed{seed}.csv"
    val_file = split_path / f"val_idx_{scenario}_seed{seed}.csv"
    test_file = split_path / f"test_idx_{scenario}_seed{seed}.csv"

    # Validate required files exist
    missing_files = []
    if not train_file.exists():
        missing_files.append(str(train_file))
    if not test_file.exists():
        missing_files.append(str(test_file))

    if missing_files:
        raise FileNotFoundError(
            f"Split files not found: {', '.join(missing_files)}\n"
            f"Run 'ced save-splits' to generate splits with scenario={scenario}"
        )

    train_idx = pd.read_csv(train_file)["idx"].values
    # Val file is optional (may not exist if val_size=0)
    if val_file.exists():
        val_idx = pd.read_csv(val_file)["idx"].values
    else:
        logger.warning(f"Validation split file not found: {val_file}. Using empty validation set.")
        val_idx = np.array([], dtype=int)
    test_idx = pd.read_csv(test_file)["idx"].values

    # Validate split indices before returning
    from ced_ml.data.persistence import validate_split_indices

    is_valid, error_msg = validate_split_indices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        total_samples=None,  # Will check bounds later when data is loaded
    )

    if not is_valid:
        raise ValueError(f"Split index validation failed: {error_msg}")

    # Check for split overlap (data leakage detection)
    check_split_overlap(
        train_idx=train_idx.tolist(),
        val_idx=val_idx.tolist(),
        test_idx=test_idx.tolist(),
        strictness="error",  # Overlap is always an error
    )

    return train_idx, val_idx, test_idx, scenario


def evaluate_on_split(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    train_prev: float,
    target_prev: float,
    config: Any,
    precomputed_threshold: float | None = None,
) -> dict[str, float]:
    """
    Evaluate model on a data split.

    Args:
        model: Fitted sklearn model with predict_proba method
        X: Features
        y: True labels
        train_prev: Training prevalence
        target_prev: Target prevalence for adjustment
        config: TrainingConfig object
        precomputed_threshold: Optional precomputed threshold (e.g., from validation set).
                              If provided, this threshold is used instead of computing a new one.

    Returns:
        Dictionary of metrics
    """
    y_probs = model.predict_proba(X)[:, 1]

    y_probs_adj = adjust_probabilities_for_prevalence(
        y_probs, sample_prev=train_prev, target_prev=target_prev
    )

    metrics = compute_discrimination_metrics(y, y_probs_adj)

    # Add calibration metrics
    brier = compute_brier_score(y, y_probs_adj)
    cal_metrics = calibration_intercept_slope(y, y_probs_adj)
    ece = expected_calibration_error(y, y_probs_adj)

    metrics.update(
        {
            METRIC_BRIER: brier,
            "calibration_intercept": cal_metrics.intercept,
            "calibration_slope": cal_metrics.slope,
            "ECE": ece,
        }
    )

    # Use precomputed threshold if provided (e.g., from validation set)
    # Otherwise compute threshold on this split using strategy pattern
    if precomputed_threshold is not None:
        threshold = precomputed_threshold
    else:
        # Use strategy pattern for threshold selection
        strategy = get_threshold_strategy(config.thresholds)
        threshold = strategy.find_threshold(y, y_probs_adj)
        logger.info(f"Threshold selection: {strategy.name}, threshold={threshold:.4f}")

    binary_metrics = binary_metrics_at_threshold(y, y_probs_adj, threshold)

    metrics.update(
        {
            "threshold": binary_metrics.threshold,
            "precision": binary_metrics.precision,
            "sensitivity": binary_metrics.sensitivity,
            "f1": binary_metrics.f1,
            "specificity": binary_metrics.specificity,
            "fpr": binary_metrics.fpr,
            "tpr": binary_metrics.tpr,
            "tp": binary_metrics.tp,
            "fp": binary_metrics.fp,
            "tn": binary_metrics.tn,
            "fn": binary_metrics.fn,
        }
    )

    # Multi-target specificity metrics
    if hasattr(config, "evaluation") and hasattr(config.evaluation, "control_spec_targets"):
        spec_targets = config.evaluation.control_spec_targets
        if spec_targets:
            multi_target_metrics = compute_multi_target_specificity_metrics(
                y_true=y, y_pred=y_probs_adj, spec_targets=spec_targets
            )
            metrics.update(multi_target_metrics)

    return metrics


def run_train(
    config_file: str | None = None,
    cli_args: dict[str, Any] | None = None,
    overrides: list[str] | None = None,
    log_level: int | None = None,
):
    """
    Run model training with staged orchestration.

    This function orchestrates the training pipeline through discrete stages:
    1. Configuration loading and validation
    2. Data loading and column resolution
    3. Split loading and validation
    4. Feature preparation
    5. Model training with nested CV
    6. Evaluation on val/test splits
    7. Artifact persistence
    8. Plot generation

    Args:
        config_file: Path to YAML config file (optional)
        cli_args: Dictionary of CLI arguments (optional)
        overrides: List of config overrides (optional)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
    """
    from ced_ml.cli.orchestration import (
        TrainingContext,
        evaluate_splits,
        generate_plots,
        load_data,
        load_splits,
        prepare_features,
        save_artifacts,
        train_models,
    )

    # Setup logger
    if log_level is None:
        log_level = logging.INFO
    logger = setup_logger("ced_ml", level=log_level)

    log_section(logger, "CeD-ML Model Training")

    # Build overrides list from CLI args
    all_overrides = list(overrides) if overrides else []
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:
                all_overrides.append(f"{key}={value}")

    # Load and validate config
    logger.info("Loading configuration...")
    config = load_training_config(config_file=config_file, overrides=all_overrides)

    # Require infile for training
    if config.infile is None:
        raise ValueError(
            "Training requires an input file. Provide 'infile' in config or via --infile CLI arg."
        )

    logger.info("Validating configuration...")
    validate_training_config(config)

    # Create base output directory
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Initialize context from config
    ctx = TrainingContext.from_config(
        config=config,
        cli_args=cli_args or {},
        log_level=log_level,
    )

    # Stage 1: Load data
    ctx = load_data(ctx)

    # Stage 2: Load and validate splits
    ctx = load_splits(ctx)

    # Stage 3: Prepare features
    ctx = prepare_features(ctx)

    # Stage 4: Train models
    ctx = train_models(ctx)

    # Stage 5: Evaluate on splits
    ctx = evaluate_splits(ctx)

    # Stage 6: Save all artifacts
    ctx = save_artifacts(ctx)

    # Stage 7: Generate plots
    ctx = generate_plots(ctx)

    # File logging disabled (using shell tee for live logs)
    # finalize_live_log(logger)
