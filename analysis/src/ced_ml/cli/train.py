"""
CLI implementation for train command.

Thin wrapper around existing celiacML_faith.py logic with new config system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ced_ml.config.loader import load_training_config, save_config
from ced_ml.config.validation import validate_training_config
from ced_ml.data.columns import get_available_columns_from_file, resolve_columns
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file, usecols_for_proteomics
from ced_ml.data.schema import (
    CONTROL_LABEL,
    METRIC_AUROC,
    METRIC_BRIER,
    METRIC_PRAUC,
    SCENARIO_DEFINITIONS,
    TARGET_COL,
    ModelName,
)
from ced_ml.evaluation.reports import OutputDirectories, ResultsWriter
from ced_ml.features.kbest import (
    build_kbest_pipeline_step,
)
from ced_ml.features.panels import build_multi_size_panels
from ced_ml.features.stability import (
    compute_selection_frequencies,
    extract_stable_panel,
)
from ced_ml.metrics.bootstrap import stratified_bootstrap_ci
from ced_ml.metrics.dca import save_dca_results, threshold_dca_zero_crossing

# Feature selection modules
# Metrics modules
from ced_ml.metrics.discrimination import (
    compute_brier_score,
    compute_discrimination_metrics,
)
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    choose_threshold_objective,
    compute_multi_target_specificity_metrics,
    compute_threshold_bundle,
)
from ced_ml.models.calibration import (
    OOFCalibratedModel,
    calibration_intercept_slope,
    expected_calibration_error,
)
from ced_ml.models.prevalence import adjust_probabilities_for_prevalence

# Model modules
from ced_ml.models.registry import (
    build_models,
)
from ced_ml.models.training import (
    _apply_per_fold_calibration,
    _extract_selected_proteins_from_fold,
    get_model_n_iter,
    oof_predictions_with_nested_cv,
)

# Plotting modules
from ced_ml.plotting import (
    plot_calibration_curve,
    plot_oof_combined,
    plot_pr_curve,
    plot_risk_distribution,
    plot_roc_curve,
)
from ced_ml.plotting.dca import plot_dca_curve
from ced_ml.plotting.learning_curve import save_learning_curve_csv
from ced_ml.utils.logging import auto_log_path, log_hpc_context, log_section, setup_logger
from ced_ml.utils.metadata import (
    build_oof_metadata,
    build_plot_metadata,
    count_category_breakdown,
)

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


def _log_unwired_feature_selection_settings(config: Any, logger: logging.Logger) -> None:
    """Log warning if advanced feature selection configs are set but not implemented.

    Note: Screening (screen_top_n + screen_method) is now wired into the pipeline.
    """
    unwired = []

    # Stability threshold is post-hoc only (not used during training)
    if getattr(config.features, "stability_thresh", 0.75) != 0.75:
        unwired.append("stability_thresh (post-hoc panel building only)")

    # Correlation threshold is post-hoc only
    if getattr(config.features, "stable_corr_thresh", 0.85) != 0.85:
        unwired.append("stable_corr_thresh (post-hoc panel building only)")

    if unwired:
        logger.warning(
            f"Feature selection options set but not used in training pipeline: {', '.join(unwired)}."
        )


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
    # Otherwise compute threshold on this split
    if precomputed_threshold is not None:
        threshold = precomputed_threshold
    else:
        threshold_obj = config.thresholds.objective if hasattr(config, "thresholds") else "youden"
        # Pass config threshold parameters (fixed_spec, fbeta, fixed_ppv)
        fixed_spec = (
            config.thresholds.fixed_spec if hasattr(config.thresholds, "fixed_spec") else 0.90
        )
        fbeta = config.thresholds.fbeta if hasattr(config.thresholds, "fbeta") else 1.0
        fixed_ppv = config.thresholds.fixed_ppv if hasattr(config.thresholds, "fixed_ppv") else 0.5

        _, threshold = choose_threshold_objective(
            y,
            y_probs_adj,
            objective=threshold_obj,
            fixed_spec=fixed_spec,
            fbeta=fbeta,
            fixed_ppv=fixed_ppv,
            log_details=True,
        )

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
    Run model training with new config system.

    Args:
        config_file: Path to YAML config file (optional)
        cli_args: Dictionary of CLI arguments (optional)
        overrides: List of config overrides (optional)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
    """
    # Setup logger
    if log_level is None:
        log_level = logging.INFO
    # Use parent logger "ced_ml" so all child modules inherit the level
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

    # Require infile for training (not needed for ensemble)
    if config.infile is None:
        raise ValueError(
            "Training requires an input file. Provide 'infile' in config or via --infile CLI arg."
        )

    logger.info("Validating configuration...")
    validate_training_config(config)

    # Create base output directory
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Resolve columns (auto-detect or explicit)
    log_section(logger, "Resolving Columns")
    logger.info(f"Column mode: {config.columns.mode}")
    available_columns = get_available_columns_from_file(str(config.infile))
    resolved = resolve_columns(available_columns, config.columns)

    logger.info("Resolved columns:")
    logger.info(f"  Proteins: {len(resolved.protein_cols)}")
    logger.info(f"  Numeric metadata: {resolved.numeric_metadata}")
    logger.info(f"  Categorical metadata: {resolved.categorical_metadata}")

    # Step 2: Load data with resolved columns
    log_section(logger, "Loading Data")
    logger.info(f"Reading: {config.infile}")
    usecols_fn = usecols_for_proteomics(
        numeric_metadata=resolved.numeric_metadata,
        categorical_metadata=resolved.categorical_metadata,
    )
    df_raw = read_proteomics_file(config.infile, usecols=usecols_fn)

    # Step 3: Apply row filters (defaults: drop_uncertain_controls=True, dropna_meta_num=True)
    logger.info("Applying row filters...")
    df_filtered, filter_stats = apply_row_filters(df_raw, meta_num_cols=resolved.numeric_metadata)
    logger.info(f"Filtered: {filter_stats['n_in']:,} → {filter_stats['n_out']:,} rows")
    logger.info(f"  Removed {filter_stats['n_removed_uncertain_controls']} uncertain controls")
    logger.info(f"  Removed {filter_stats['n_removed_dropna_meta_num']} rows with missing metadata")

    # Step 4: Use resolved columns
    protein_cols = resolved.protein_cols
    logger.info(f"Using {len(protein_cols)} protein columns")

    # Validate protein columns (dtype and NaN check)
    validate_protein_columns(df_filtered, protein_cols, logger)

    # Determine split seed (used for output subdirs and loading splits)
    seed = getattr(config, "split_seed", getattr(config, "seed", 0))

    # Determine run_id (if provided, otherwise generate timestamp-based)
    run_id = getattr(config, "run_id", None)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 4: Create output directories (with run-specific and split-specific subdirs)
    log_section(logger, "Setting Up Output Structure")
    outdirs = OutputDirectories.create(
        config.outdir, exist_ok=True, split_seed=seed, run_id=run_id, model=config.model
    )
    logger.info(f"Output root: {outdirs.root}")
    logger.info(f"Split seed: {seed}")
    logger.info(f"Run ID: {run_id}")

    # Auto-file-logging: write to logs/training/run_{ID}/{model}_seed{N}.log
    log_file = auto_log_path(
        command="train",
        outdir=config.outdir,
        run_id=run_id,
        model=config.model,
        split_seed=seed,
    )
    logger = setup_logger("ced_ml", level=log_level, log_file=log_file)
    logger.info(f"Logging to file: {log_file}")

    log_hpc_context(logger)

    # Save resolved config to run-specific directory
    config_path = Path(outdirs.root) / "training_config.yaml"
    save_config(config, config_path)
    logger.info(f"Saved resolved config to: {config_path}")

    # Log config summary
    logger.info(f"Model: {config.model}")
    logger.info(f"CV: {config.cv.folds} folds × {config.cv.repeats} repeats")
    logger.info(f"Scoring: {config.cv.scoring}")

    # Step 5: Load split indices (auto-detect scenario from split files)
    log_section(logger, "Loading Splits")
    try:
        # Auto-detect scenario from split files (or use config if provided)
        scenario = getattr(config, "scenario", None)
        train_idx, val_idx, test_idx, detected_scenario = load_split_indices(
            str(config.split_dir), scenario, seed
        )
        scenario = detected_scenario  # Use detected scenario for all subsequent operations
        logger.info(f"Scenario: {scenario}")
        logger.info(f"Loaded splits for seed {seed}:")
        logger.info(f"  Train: {len(train_idx):,} samples")
        logger.info(f"  Val:   {len(val_idx):,} samples")
        logger.info(f"  Test:  {len(test_idx):,} samples")

        # Save split trace
        split_trace_df = pd.DataFrame(
            {
                "idx": np.concatenate([train_idx, val_idx, test_idx]),
                "split": (
                    ["train"] * len(train_idx) + ["val"] * len(val_idx) + ["test"] * len(test_idx)
                ),
                "scenario": scenario,
                "seed": seed,
            }
        )
        split_trace_path = Path(outdirs.diag_splits) / "train_test_split_trace.csv"
        split_trace_df.to_csv(split_trace_path, index=False)
        logger.info(f"Split trace saved: {split_trace_path}")

        # Validate row filter alignment (C3 fix)
        # Load split metadata and check that meta_num_cols_used matches current config
        from ced_ml.data.persistence import load_split_metadata

        split_meta = load_split_metadata(str(config.split_dir), scenario, seed)
        if split_meta is not None:
            row_filters = split_meta.get("row_filters", {})
            split_meta_num_cols = set(row_filters.get("meta_num_cols_used", []))
            current_meta_num_cols = set(resolved.numeric_metadata)

            if split_meta_num_cols and split_meta_num_cols != current_meta_num_cols:
                logger.error("Row filter column mismatch detected!")
                logger.error(f"  Splits used:   {sorted(split_meta_num_cols)}")
                logger.error(f"  Training uses: {sorted(current_meta_num_cols)}")
                logger.error("This can cause train/val/test contamination.")
                logger.error(
                    "Please regenerate splits with matching config or update training config."
                )
                raise ValueError(
                    f"Row filter alignment error: splits used {sorted(split_meta_num_cols)}, "
                    f"but training uses {sorted(current_meta_num_cols)}. "
                    "Regenerate splits or update config to match."
                )
            elif split_meta_num_cols:
                logger.info(f"Row filter alignment verified: {sorted(split_meta_num_cols)}")
        else:
            logger.warning("Split metadata not found - cannot verify row filter alignment")
    except FileNotFoundError as e:
        logger.error(f"Split files not found: {e}")
        logger.error("Please run 'ced save-splits' first to generate splits")
        raise

    # Step 6: Prepare X, y for each split
    scenario_def = SCENARIO_DEFINITIONS[scenario]
    target_labels = scenario_def["labels"]
    scenario_def["positive_label"]

    # Validate that all expected labels are present in data
    unique_labels = set(df_filtered[TARGET_COL].unique())
    missing_labels = set(target_labels) - unique_labels
    if missing_labels:
        logger.warning(
            f"Expected labels {missing_labels} not found in filtered data. Available: {unique_labels}"
        )

    # Check for unknown labels (not in scenario definition)
    unknown_labels = unique_labels - set(target_labels)
    if unknown_labels:
        logger.warning(
            f"Found labels not in scenario definition: {unknown_labels}. Will be filtered out."
        )

    mask_scenario = df_filtered[TARGET_COL].isin(target_labels)
    n_filtered = (~mask_scenario).sum()
    if n_filtered > 0:
        logger.info(f"Filtered out {n_filtered:,} samples with labels not in scenario {scenario}")

    df_scenario = df_filtered[mask_scenario].copy()
    df_scenario["y"] = (df_scenario[TARGET_COL] != CONTROL_LABEL).astype(int)

    # Validate split indices are within bounds now that data is loaded
    from ced_ml.data.persistence import validate_split_indices

    is_valid, error_msg = validate_split_indices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        total_samples=len(df_scenario),
    )

    if not is_valid:
        logger.error(f"Split index bounds validation failed: {error_msg}")
        logger.error(f"Data shape: {len(df_scenario)} samples after filtering")
        logger.error(f"Max train_idx: {train_idx.max() if len(train_idx) > 0 else 'N/A'}")
        logger.error(f"Max val_idx: {val_idx.max() if len(val_idx) > 0 else 'N/A'}")
        logger.error(f"Max test_idx: {test_idx.max() if len(test_idx) > 0 else 'N/A'}")
        raise ValueError(f"Split index bounds validation failed: {error_msg}")

    feature_cols = resolved.all_feature_cols

    # Handle fixed panel (from CLI or config)
    fixed_panel_path = cli_args.get("fixed_panel") if cli_args else None

    # Check if fixed panel is specified in config
    if not fixed_panel_path and config.features.feature_selection_strategy == "fixed_panel":
        if config.features.fixed_panel_csv:
            # Resolve path relative to data/ directory if not absolute
            fixed_panel_path = Path(config.features.fixed_panel_csv)
            if not fixed_panel_path.is_absolute():
                # Assume path is relative to data/ directory
                data_dir = Path(__file__).parent.parent.parent.parent.parent / "data"
                fixed_panel_path = data_dir / fixed_panel_path
        else:
            raise ValueError(
                "feature_selection_strategy='fixed_panel' but fixed_panel_csv not specified in config. "
                "Set features.fixed_panel_csv in training_config.yaml"
            )

    fixed_panel_proteins = None

    if fixed_panel_path:
        log_section(logger, "Fixed Panel Mode")
        logger.info(f"Loading fixed panel from: {fixed_panel_path}")

        # Load fixed panel CSV
        fixed_panel_df = pd.read_csv(fixed_panel_path)

        # Expect a column named 'protein' or use first column
        if "protein" in fixed_panel_df.columns:
            fixed_panel_proteins = fixed_panel_df["protein"].tolist()
        else:
            fixed_panel_proteins = fixed_panel_df.iloc[:, 0].tolist()

        # Validate that all fixed panel proteins exist in data
        missing_proteins = set(fixed_panel_proteins) - set(protein_cols)
        if missing_proteins:
            logger.error(f"Fixed panel contains {len(missing_proteins)} proteins not in dataset")
            logger.error(f"Missing proteins (first 10): {list(missing_proteins)[:10]}")
            raise ValueError(
                f"Fixed panel contains {len(missing_proteins)} proteins not found in dataset. "
                f"Check that protein names match exactly (e.g., 'PROT_123_resid')."
            )

        # Override protein_cols to use only fixed panel
        protein_cols = fixed_panel_proteins
        logger.info(f"Fixed panel loaded: {len(fixed_panel_proteins)} proteins")
        logger.info("Feature selection will be BYPASSED (using pre-specified panel)")

        # Override config to disable feature selection
        config.features.feature_selection_strategy = "none"
        config.features.screen_top_n = 0
        logger.info("Feature selection strategy set to 'none'")

        # Update feature_cols to use fixed panel proteins + metadata
        feature_cols = (
            list(protein_cols) + resolved.numeric_metadata + resolved.categorical_metadata
        )
        logger.info(f"Feature columns updated: {len(feature_cols)} total features")

    X_train = df_scenario.iloc[train_idx][feature_cols]
    y_train = df_scenario.iloc[train_idx]["y"].values

    X_val = df_scenario.iloc[val_idx][feature_cols]
    y_val = df_scenario.iloc[val_idx]["y"].values

    X_test = df_scenario.iloc[test_idx][feature_cols]
    y_test = df_scenario.iloc[test_idx]["y"].values

    # Extract original category labels for 3-panel KDE plots
    cat_train = df_scenario.iloc[train_idx][TARGET_COL].values
    cat_val = df_scenario.iloc[val_idx][TARGET_COL].values
    cat_test = df_scenario.iloc[test_idx][TARGET_COL].values

    train_prev = float(y_train.mean())
    logger.info(f"Training prevalence: {train_prev:.3f}")

    # Step 7: Build classifier
    log_section(logger, "Building Model")
    logger.info(f"Model type: {config.model}")

    classifier = build_models(
        model_name=config.model,
        config=config,
        random_state=seed,
        n_jobs=config.n_jobs,
    )

    # Step 8: Build full pipeline
    pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols,
        resolved.categorical_metadata,
        model_name=config.model,
    )
    logger.info(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")

    # Step 9: Run nested CV for OOF predictions
    log_section(logger, "Nested Cross-Validation")
    total_folds = config.cv.folds * config.cv.repeats
    logger.info(
        f"Config: {config.model} | {config.cv.folds}-fold x {config.cv.repeats} repeats "
        f"= {total_folds} outer folds | scoring={config.cv.scoring}"
    )
    strategy = config.features.feature_selection_strategy
    screen_top = getattr(config.features, "screen_top_n", "?")
    logger.info(
        f"Features: {strategy} | screen={getattr(config.features, 'screen_method', 'none')} top-{screen_top}"
    )
    optuna_cfg = config.optuna
    cal_strategy = getattr(config.calibration, "strategy", "none")
    cal_method = getattr(config.calibration, "method", "none")
    if optuna_cfg.enabled:
        logger.info(
            f"Optuna: {optuna_cfg.n_trials} trials ({optuna_cfg.sampler}/{optuna_cfg.pruner}) | "
            f"calibration: {cal_strategy} ({cal_method})"
        )
    else:
        logger.info(f"Optuna: disabled | calibration: {cal_strategy} ({cal_method})")
    logger.info(f"Running {total_folds} folds...")

    # Create grid RNG if grid randomization is enabled
    grid_rng = np.random.default_rng(seed) if config.cv.grid_randomize else None

    (
        oof_preds,
        elapsed_sec,
        best_params_df,
        selected_proteins_df,
        oof_calibrator,
        nested_rfecv_result,
    ) = oof_predictions_with_nested_cv(
        pipeline=pipeline,
        model_name=config.model,
        X=X_train,
        y=y_train,
        protein_cols=protein_cols,
        config=config,
        random_state=seed,
        grid_rng=grid_rng,
    )
    # oof_calibrator is saved to model bundle for stacking ensemble consumption
    # nested_rfecv_result contains consensus panel if rfe_enabled=true

    logger.info(f"CV completed in {elapsed_sec:.1f}s")

    # Step 10: Fit final model on full train set
    log_section(logger, "Training Final Model")
    logger.info("Fitting on full training set...")

    final_pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols,
        resolved.categorical_metadata,
        model_name=config.model,
    )

    # Determine best k value for final model (if using hybrid_stability with k_grid)
    strategy = config.features.feature_selection_strategy
    if strategy == "hybrid_stability" and "sel__k" in best_params_df.columns:
        k_grid = getattr(config.features, "k_grid", None)

        if k_grid and len(k_grid) > 1:
            # Multiple k values were tuned: use most frequently selected k (mode)
            best_k = int(best_params_df["sel__k"].mode()[0])
            logger.info(f"Multiple k values tuned: using mode k={best_k} for final model")
        elif k_grid and len(k_grid) == 1:
            # Single k value: use that value
            best_k = k_grid[0]
            logger.info(f"Single k value in config: using k={best_k} for final model")
        else:
            # Fallback: use first k from CV results
            best_k = int(best_params_df["sel__k"].iloc[0])
            logger.info(f"Using k={best_k} from CV results for final model")

        # Set the k parameter in the final pipeline
        final_pipeline.set_params(sel__k=best_k)
        logger.info(f"Final model k-best set to k={best_k}")

    final_pipeline.fit(X_train, y_train)
    logger.info("Final model fitted")

    # Apply calibration to final model if enabled (consistent with CV behavior)
    final_pipeline = _apply_per_fold_calibration(
        estimator=final_pipeline,
        model_name=config.model,
        config=config,
        X_train=X_train,
        y_train=y_train,
    )

    # For oof_posthoc strategy, wrap final model with the OOF calibrator
    # This ensures val/test predictions are calibrated consistently with OOF predictions
    if oof_calibrator is not None:
        final_pipeline = OOFCalibratedModel(
            base_model=final_pipeline,
            calibrator=oof_calibrator,
        )
        logger.info(f"Final model wrapped with OOF calibrator (method={oof_calibrator.method})")
    elif config.calibration.enabled and config.model != ModelName.LinSVM_cal:
        logger.info(f"Final model calibrated using {config.calibration.method}")

    # Extract selected proteins from final model for test panel
    try:
        final_selected_proteins = _extract_selected_proteins_from_fold(
            fitted_model=final_pipeline,
            model_name=config.model,
            protein_cols=protein_cols,
            config=config,
            X_train=X_train,
            y_train=y_train,
            random_state=seed,
            nested_rfecv_result=nested_rfecv_result,
        )
        logger.info(f"Final test panel: {len(final_selected_proteins)} proteins selected")
    except Exception as e:
        logger.warning(f"Could not extract final test panel: {e}")
        final_selected_proteins = []

    # Step 11: Evaluate on validation set (threshold selection)
    log_section(logger, "Validation Set Evaluation")

    # Check if validation set exists
    if len(val_idx) == 0:
        logger.warning("No validation set available (val_size=0). Skipping validation evaluation.")
        logger.warning("Threshold will be computed on test set (not recommended for production).")
        val_metrics = None
        val_threshold = None
        val_target_prev = train_prev  # Default for metadata logging
    else:
        # Determine target prevalence for validation based on config
        if config.thresholds.target_prevalence_source == "fixed":
            val_target_prev = config.thresholds.target_prevalence_fixed
        elif config.thresholds.target_prevalence_source == "train":
            val_target_prev = train_prev
        elif config.thresholds.target_prevalence_source == "val":
            val_target_prev = float(np.asarray(y_val).mean())
        elif config.thresholds.target_prevalence_source == "test":
            # Using test prevalence for val is unusual but supported
            val_target_prev = float(np.asarray(y_test).mean())
        else:
            val_target_prev = train_prev

        val_metrics = evaluate_on_split(
            final_pipeline, X_val, y_val, train_prev, val_target_prev, config
        )

        logger.info(f"Val AUROC: {val_metrics[METRIC_AUROC]:.3f}")
        logger.info(f"Val PRAUC: {val_metrics[METRIC_PRAUC]:.3f}")
        logger.info(f"Selected threshold: {val_metrics['threshold']:.3f}")
        if final_selected_proteins:
            logger.info(f"Val evaluation using {len(final_selected_proteins)} selected proteins")

        # Store validation threshold for reuse on test set (prevents leakage)
        val_threshold = val_metrics["threshold"]

    # Step 12: Evaluate on test set
    log_section(logger, "Test Set Evaluation")

    # Determine target prevalence for test based on config
    if config.thresholds.target_prevalence_source == "fixed":
        test_target_prev = config.thresholds.target_prevalence_fixed
    elif config.thresholds.target_prevalence_source == "train":
        test_target_prev = train_prev
    elif config.thresholds.target_prevalence_source == "val":
        test_target_prev = float(np.asarray(y_val).mean())
    elif config.thresholds.target_prevalence_source == "test":
        test_target_prev = float(np.asarray(y_test).mean())
    else:
        test_target_prev = train_prev

    # Reuse validation threshold on test set (standard practice, prevents leakage)
    # If no validation threshold (val_size=0), compute threshold on test set
    test_metrics = evaluate_on_split(
        final_pipeline,
        X_test,
        y_test,
        train_prev,
        test_target_prev,
        config,
        precomputed_threshold=val_threshold,
    )

    logger.info(f"Test AUROC: {test_metrics[METRIC_AUROC]:.3f}")
    logger.info(f"Test PRAUC: {test_metrics[METRIC_PRAUC]:.3f}")
    if val_threshold is not None:
        logger.info(f"Test threshold (from val): {test_metrics['threshold']:.3f}")
    else:
        logger.info(f"Test threshold (computed on test): {test_metrics['threshold']:.3f}")
    if final_selected_proteins:
        logger.info(f"Test evaluation using {len(final_selected_proteins)} selected proteins")

    # Use test prevalence for downstream logging
    target_prev = test_target_prev

    # Step 13: Save outputs
    # Model saved as-is without wrappers. All training/validation/test sets are at
    # the same prevalence (16.7%), so no adjustment is needed during pipeline execution.
    # Note: Real-world deployment prevalence is ~0.34%. Prevalence adjustment for
    # deployment is a future concern (see ADR-010 and DEPLOYMENT.md).
    log_section(logger, "Saving Results")

    writer = ResultsWriter(outdirs)

    # Save model bundle (instead of bare model for holdout compatibility)
    import joblib
    import sklearn

    # Determine effective calibration strategy for this model
    calibration_strategy = config.calibration.get_strategy_for_model(config.model)

    model_bundle = {
        "model": final_pipeline,
        "scenario": scenario,
        "model_name": config.model,
        "thresholds": {
            "val_threshold": val_threshold,
            "test_threshold": test_metrics["threshold"],
            "objective": config.thresholds.objective,
            "fixed_spec": (
                config.thresholds.fixed_spec if hasattr(config.thresholds, "fixed_spec") else None
            ),
        },
        "metadata": {
            # Training/test prevalence (all at same 16.7% level after downsampling)
            "train_prevalence": train_prev,
            "val_prevalence": val_target_prev,
            "test_prevalence": test_target_prev,
        },
        "calibration": {
            "enabled": config.calibration.enabled,
            "method": config.calibration.method if config.calibration.enabled else None,
            "strategy": calibration_strategy,
            "oof_calibrator": oof_calibrator,  # For stacking ensemble consumption
        },
        # Store resolved columns for holdout compatibility (C6 fix)
        # These are the actual columns used during training, not just config values
        "resolved_columns": {
            "protein_cols": protein_cols,
            "numeric_metadata": resolved.numeric_metadata,
            "categorical_metadata": resolved.categorical_metadata,
        },
        # Fixed panel metadata (if used)
        "fixed_panel": {
            "enabled": fixed_panel_path is not None,
            "path": str(fixed_panel_path) if fixed_panel_path else None,
            "source": (
                "cli"
                if (cli_args and cli_args.get("fixed_panel"))
                else "config" if fixed_panel_path else None
            ),
            "n_proteins": len(fixed_panel_proteins) if fixed_panel_proteins else None,
        },
        "config": (config.model_dump() if hasattr(config, "model_dump") else config.dict()),
        "seed": seed,
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    model_filename = f"{config.model}__final_model.joblib"
    model_path = Path(outdirs.core) / model_filename
    joblib.dump(model_bundle, model_path)
    logger.info(f"Model bundle saved: {model_path}")

    # Save metrics (using ResultsWriter with append mode)
    if val_metrics is not None:
        writer.save_val_metrics(val_metrics, scenario, config.model)
    writer.save_test_metrics(test_metrics, scenario, config.model)

    # Save CV artifacts
    best_params_path = Path(outdirs.cv) / "best_params_per_split.csv"
    best_params_df.to_csv(best_params_path, index=False)
    logger.info(f"Best params saved: {best_params_path}")

    selected_proteins_path = Path(outdirs.cv) / "selected_proteins_per_split.csv"
    selected_proteins_df.to_csv(selected_proteins_path, index=False)
    logger.info(f"Selected proteins saved: {selected_proteins_path}")

    # Save nested RFECV results (if enabled)
    if nested_rfecv_result is not None:
        from ced_ml.features.nested_rfe import save_nested_rfecv_results

        rfecv_dir = Path(outdirs.cv) / "rfecv"
        save_nested_rfecv_results(
            result=nested_rfecv_result,
            output_dir=rfecv_dir,
            model_name=config.model,
            split_seed=seed,
        )
        logger.info(f"Nested RFECV results saved: {rfecv_dir}")
        logger.info(
            f"  Consensus panel: {len(nested_rfecv_result.consensus_panel)} proteins "
            f"(selected in >= {config.features.rfe_consensus_thresh:.0%} of folds)"
        )
        logger.info(f"  Mean optimal size: {nested_rfecv_result.mean_optimal_size:.1f} proteins")
        logger.info(
            f"  Mean val AUROC: {np.mean(nested_rfecv_result.fold_val_aurocs):.4f} "
            f"(std: {np.std(nested_rfecv_result.fold_val_aurocs):.4f})"
        )

    # Save final test panel (proteins used in final model for test predictions)
    if final_selected_proteins:
        panel_metadata = {
            "selection_method": config.features.feature_selection_strategy,
            "n_train": len(y_train),
            "n_train_pos": int(y_train.sum()),
            "train_prevalence": float(train_prev),
            "random_state": seed,
            "timestamp": datetime.now().isoformat(),
        }
        writer.save_final_test_panel(
            panel_proteins=final_selected_proteins,
            scenario=scenario,
            model=config.model,
            metadata=panel_metadata,
        )

    # Save Optuna artifacts if enabled
    if config.optuna.enabled:
        # Save artifacts flat at cv level (no optuna subdirectory)
        cv_dir = Path(outdirs.cv)

        # Check if Optuna metadata was collected
        if "optuna_n_trials" in best_params_df.columns:
            # Save Optuna summary
            optuna_summary = {
                "enabled": True,
                "sampler": config.optuna.sampler,
                "pruner": config.optuna.pruner,
                "n_trials_per_fold": int(config.optuna.n_trials),
                "timeout": config.optuna.timeout,
                "direction": config.optuna.direction or "maximize",
                "total_folds": len(best_params_df),
            }
            optuna_summary_path = cv_dir / "optuna_config.json"
            with open(optuna_summary_path, "w") as f:
                json.dump(optuna_summary, f, indent=2)
            logger.info(f"Optuna config saved: {optuna_summary_path}")

            # Save best params with Optuna metadata
            optuna_params_path = cv_dir / "best_params_optuna.csv"
            best_params_df.to_csv(optuna_params_path, index=False)
            logger.info(f"Optuna best params saved: {optuna_params_path}")

            # Generate Optuna plots using existing study (no refitting)
            if config.output.plot_optuna:
                try:
                    from ced_ml.plotting.optuna_plots import save_optuna_plots

                    logger.info("Generating Optuna hyperparameter tuning plots...")

                    # Try to load study from persistent storage if configured
                    study_loaded = False
                    if config.optuna.storage and config.optuna.study_name:
                        try:
                            import optuna

                            logger.info(
                                f"Loading existing Optuna study from storage: {config.optuna.study_name}"
                            )
                            study = optuna.load_study(
                                study_name=config.optuna.study_name,
                                storage=config.optuna.storage,
                            )
                            study_loaded = True
                            logger.info(
                                f"Successfully loaded study with {len(study.trials)} trials "
                                "(reusing from CV, no refitting needed)"
                            )
                        except Exception as e:
                            logger.warning(f"Could not load study from storage: {e}")

                    # Fallback: refit if study not available in storage
                    if not study_loaded:
                        logger.info(
                            "No persistent study storage configured, refitting for plots "
                            "(consider setting optuna.storage and optuna.study_name to avoid this)"
                        )
                        from ced_ml.models.training import _build_hyperparameter_search

                        # Build and fit hyperparameter search on full training set
                        optuna_pipeline = build_training_pipeline(
                            config,
                            classifier,
                            protein_cols,
                            resolved.categorical_metadata,
                            model_name=config.model,
                        )

                        xgb_spw = None
                        if config.model == ModelName.XGBoost:
                            from ced_ml.models.registry import (
                                compute_scale_pos_weight_from_y,
                            )

                            # Check if user specified explicit value via scale_pos_weight_grid
                            spw_grid = getattr(config.xgboost, "scale_pos_weight_grid", None)
                            if spw_grid and len(spw_grid) == 1 and spw_grid[0] > 0:
                                xgb_spw = float(spw_grid[0])
                            else:
                                # Auto: compute from class distribution
                                xgb_spw = compute_scale_pos_weight_from_y(y_train)

                        optuna_search = _build_hyperparameter_search(
                            optuna_pipeline,
                            config.model,
                            config,
                            seed,
                            xgb_spw,
                            grid_rng=None,
                        )

                        if optuna_search is not None:
                            optuna_search.fit(X_train, y_train)
                            if (
                                hasattr(optuna_search, "study_")
                                and optuna_search.study_ is not None
                            ):
                                study = optuna_search.study_
                            else:
                                study = None
                        else:
                            study = None

                    # Generate plots if we have a study
                    if study is not None:
                        # Use optuna_plot_format if available, otherwise fall back to plot_format
                        optuna_fmt = getattr(
                            config.output,
                            "optuna_plot_format",
                            config.output.plot_format,
                        )
                        save_optuna_plots(
                            study=study,
                            out_dir=cv_dir,
                            prefix=f"{config.model}__",
                            plot_format=optuna_fmt,
                            fallback_to_html=True,
                        )
                        logger.info(f"Optuna plots saved to: {cv_dir}")

                        # Generate Pareto frontier plot if multi-objective
                        # Note: Pareto plot needs the search_cv object, not just the study
                        # Only available if we had to refit (fallback path)
                        if (
                            hasattr(config.optuna, "multi_objective")
                            and config.optuna.multi_objective
                            and not study_loaded
                            and "optuna_search" in locals()
                        ):
                            from ced_ml.plotting.optuna_plots import (
                                plot_pareto_frontier,
                            )

                            try:
                                plot_pareto_frontier(
                                    search_cv=optuna_search,
                                    outdir=cv_dir,
                                    plot_format=config.output.plot_format,
                                    dpi=config.output.plot_dpi,
                                )
                                logger.info(f"Pareto frontier plot saved to: {cv_dir}")
                            except Exception as e:
                                logger.warning(f"Failed to generate Pareto frontier plot: {e}")
                        elif (
                            hasattr(config.optuna, "multi_objective")
                            and config.optuna.multi_objective
                            and study_loaded
                        ):
                            logger.info(
                                "Pareto frontier plot skipped (requires refitting, use persistent storage to avoid)"
                            )
                    else:
                        logger.warning("No Optuna study available for plotting")

                except Exception as e:
                    logger.warning(f"Failed to generate Optuna plots: {e}")
            else:
                logger.info("Optuna plots disabled (output.plot_optuna=False)")

        else:
            logger.warning(
                "[optuna] Optuna was enabled but no trial metadata found. "
                "Check if optuna is installed: pip install optuna"
            )

    # Save cv_repeat_metrics.csv (per-repeat OOF metrics)
    cv_repeat_rows = []
    for repeat in range(oof_preds.shape[0]):
        repeat_preds = oof_preds[repeat, :]
        valid_mask = ~np.isnan(repeat_preds)
        if valid_mask.sum() > 0:
            y_repeat = y_train[valid_mask]
            p_repeat = repeat_preds[valid_mask]
            auroc = roc_auc_score(y_repeat, p_repeat) if len(np.unique(y_repeat)) > 1 else np.nan
            prauc = (
                average_precision_score(y_repeat, p_repeat)
                if len(np.unique(y_repeat)) > 1
                else np.nan
            )
            brier = float(np.mean((y_repeat - p_repeat) ** 2))
            cv_repeat_rows.append(
                {
                    "scenario": scenario,
                    "model": config.model,
                    "repeat": repeat,
                    "folds": config.cv.folds,
                    "repeats": config.cv.repeats,
                    "n_train": len(y_train),
                    "n_train_pos": int(y_train.sum()),
                    "AUROC_oof": auroc,
                    "PR_AUC_oof": prauc,
                    "Brier_oof": brier,
                    "cv_seconds": elapsed_sec,
                    "feature_selection_strategy": config.features.feature_selection_strategy,
                    "random_state": seed,
                }
            )
    if cv_repeat_rows:
        # Use ResultsWriter with append mode
        writer.save_cv_repeat_metrics(cv_repeat_rows, scenario, config.model)

    # Save run settings
    run_settings = {
        "model": config.model,
        "scenario": scenario,
        "seed": seed,
        "train_prevalence": float(train_prev),
        "target_prevalence": float(target_prev),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "cv_elapsed_sec": elapsed_sec,
        "columns": {
            "mode": config.columns.mode,
            "n_proteins": len(resolved.protein_cols),
            "numeric_metadata": resolved.numeric_metadata,
            "categorical_metadata": resolved.categorical_metadata,
        },
    }
    run_settings_path = Path(outdirs.core) / "run_settings.json"
    with open(run_settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)
    logger.info(f"Run settings saved: {run_settings_path}")

    # Save config_metadata.json at root (comprehensive run configuration)
    config_metadata = {
        "pipeline_version": "ced_ml_v2",
        "scenario": scenario,
        "model": config.model,
        "folds": config.cv.folds,
        "repeats": config.cv.repeats,
        "val_size": getattr(config, "val_size", 0.25),
        "test_size": getattr(config, "test_size", 0.25),
        "random_state": seed,
        "scoring": config.cv.scoring,
        "inner_folds": getattr(config.cv, "inner_folds", 5),
        "n_iter": getattr(config.cv, "n_iter", 50),
        "feature_selection_strategy": config.features.feature_selection_strategy,
        "kbest_max": getattr(config.features, "kbest_max", 500),
        "screen_method": getattr(config.features, "screen_method", "none"),
        "screen_top_n": getattr(config.features, "screen_top_n", 1000),
        "calibrate_final_models": int(getattr(config.calibration, "enabled", False)),
        "threshold_source": getattr(config.thresholds, "threshold_source", "val"),
        "target_prevalence_source": getattr(config.thresholds, "target_prevalence_source", "train"),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_prevalence": float(train_prev),
        "target_prevalence": float(target_prev),
        "cv_elapsed_sec": elapsed_sec,
        "n_proteins": len(resolved.protein_cols),
        "bootstrap_seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    config_metadata_path = Path(outdirs.root) / "config_metadata.json"
    with open(config_metadata_path, "w") as f:
        json.dump(config_metadata, f, indent=2, sort_keys=True)
    logger.info(f"Config metadata saved: {config_metadata_path}")

    # Save run_metadata.json (shared across models, at run level)
    # New layout: root = .../run_{id}/{model}/splits/split_seed{N}/
    # Go up 3 levels (split_seed -> splits -> model) to reach run-level dir
    if seed is not None:
        run_level_dir = Path(outdirs.root).parent.parent.parent
    else:
        # No split seed: root = .../run_{id}/{model}/
        run_level_dir = Path(outdirs.root).parent

    # Infer seed_start and n_splits from available split files
    seed_start = None
    n_splits = None
    if config.split_dir:
        split_dir_path = Path(config.split_dir)
        if split_dir_path.exists():
            meta_pattern = f"split_meta_{scenario}_seed*.json"
            meta_files = list(split_dir_path.glob(meta_pattern))

            if not meta_files:
                meta_files = list(split_dir_path.glob("split_meta_seed*.json"))

            if meta_files:
                seeds = []
                for meta_file in meta_files:
                    seed_match = meta_file.stem.split("seed")[-1]
                    try:
                        seeds.append(int(seed_match))
                    except ValueError:
                        continue

                if seeds:
                    seed_start = min(seeds)
                    n_splits = len(seeds)
                    logger.info(
                        f"Inferred split config: seed_start={seed_start}, n_splits={n_splits}"
                    )

    # Shared run_metadata.json: read-merge-write for HPC safety
    run_metadata_path = run_level_dir / "run_metadata.json"
    run_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    existing_metadata: dict = {}
    if run_metadata_path.exists():
        try:
            with open(run_metadata_path) as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_metadata = {}

    model_entry = {
        "scenario": scenario,
        "infile": str(config.infile),
        "split_dir": str(config.split_dir),
        "split_seed": seed,
        "seed_start": seed_start,
        "n_splits": n_splits,
        "timestamp": datetime.now().isoformat(),
    }

    existing_metadata.setdefault("run_id", run_id)
    existing_metadata.setdefault("infile", str(config.infile))
    existing_metadata.setdefault("split_dir", str(config.split_dir))
    existing_metadata.setdefault("models", {})
    existing_metadata["models"][config.model] = model_entry

    # Atomic write: temp file + rename
    tmp_path = run_metadata_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)
    tmp_path.rename(run_metadata_path)
    logger.info(f"Run metadata saved: {run_metadata_path}")

    # Save predictions
    test_probs_raw = final_pipeline.predict_proba(X_test)[:, 1]
    test_probs_adj = adjust_probabilities_for_prevalence(
        test_probs_raw, sample_prev=train_prev, target_prev=test_target_prev
    )
    test_preds_df = pd.DataFrame(
        {
            "idx": test_idx,
            "y_true": y_test,
            "y_prob": test_probs_raw,
            "y_prob_adjusted": test_probs_adj,
            "category": cat_test,
        }
    )
    # H4: Validate test predictions (NaN, Inf, bounds)
    test_probs = test_preds_df["y_prob"].values
    if not np.isfinite(test_probs).all():
        raise ValueError("Test predictions contain NaN or Inf values")
    if (test_probs < 0).any() or (test_probs > 1).any():
        raise ValueError(
            f"Test predictions out of [0,1] bounds: min={test_probs.min():.4f}, max={test_probs.max():.4f}"
        )
    test_preds_path = Path(outdirs.preds_test) / f"test_preds__{config.model}.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    logger.info(f"Test predictions saved: {test_preds_path}")

    # Only generate validation predictions if validation set exists
    if len(val_idx) > 0:
        val_probs_raw = final_pipeline.predict_proba(X_val)[:, 1]
        val_probs_adj = adjust_probabilities_for_prevalence(
            val_probs_raw, sample_prev=train_prev, target_prev=val_target_prev
        )
        val_preds_df = pd.DataFrame(
            {
                "idx": val_idx,
                "y_true": y_val,
                "y_prob": val_probs_raw,
                "y_prob_adjusted": val_probs_adj,
                "category": cat_val,
            }
        )
        # H4: Validate val predictions (NaN, Inf, bounds)
        val_probs = val_preds_df["y_prob"].values
        if not np.isfinite(val_probs).all():
            raise ValueError("Val predictions contain NaN or Inf values")
        if (val_probs < 0).any() or (val_probs > 1).any():
            raise ValueError(
                f"Val predictions out of [0,1] bounds: min={val_probs.min():.4f}, max={val_probs.max():.4f}"
            )
        val_preds_path = Path(outdirs.preds_val) / f"val_preds__{config.model}.csv"
        val_preds_df.to_csv(val_preds_path, index=False)
        logger.info(f"Val predictions saved: {val_preds_path}")
    else:
        logger.warning("Skipping validation predictions (no validation set)")
        val_preds_df = pd.DataFrame(
            columns=["idx", "y_true", "y_prob", "y_prob_adjusted", "category"]
        )

    # Save OOF predictions
    oof_preds_df = pd.DataFrame(
        {
            "idx": train_idx,
            "y_true": y_train,
            "category": cat_train,
        }
    )
    for repeat in range(oof_preds.shape[0]):
        oof_preds_df[f"y_prob_repeat{repeat}"] = oof_preds[repeat, :]
    # H4: Validate OOF predictions (NaN, Inf, bounds)
    if not np.isfinite(oof_preds).all():
        raise ValueError("OOF predictions contain NaN or Inf values")
    if (oof_preds < 0).any() or (oof_preds > 1).any():
        raise ValueError(
            f"OOF predictions out of [0,1] bounds: min={oof_preds.min():.4f}, max={oof_preds.max():.4f}"
        )
    oof_preds_path = Path(outdirs.preds_train_oof) / f"train_oof__{config.model}.csv"
    oof_preds_df.to_csv(oof_preds_path, index=False)
    logger.info(f"OOF predictions saved: {oof_preds_path}")

    # Save controls OOF predictions (mean across repeats)
    controls_mask = y_train == 0
    if controls_mask.sum() > 0:
        controls_idx = train_idx[controls_mask]
        controls_oof_mean = oof_preds[:, controls_mask].mean(axis=0)
        controls_oof_df = pd.DataFrame(
            {
                "idx": controls_idx,
                "y_true": y_train[controls_mask],
                "y_prob_oof_mean": controls_oof_mean,
            }
        )
        controls_oof_path = (
            Path(outdirs.preds_controls) / f"controls_risk__{config.model}__oof_mean.csv"
        )
        controls_oof_df.to_csv(controls_oof_path, index=False)
        logger.info(f"Controls OOF predictions saved: {controls_oof_path}")

    # Extract category breakdowns from splits (needed for plots AND learning curve metadata)
    train_cat_df = pd.DataFrame({"category": cat_train})
    val_cat_df = pd.DataFrame({"category": cat_val})
    test_cat_df = pd.DataFrame({"category": cat_test})

    train_breakdown = count_category_breakdown(train_cat_df)
    val_breakdown = count_category_breakdown(val_cat_df)
    test_breakdown = count_category_breakdown(test_cat_df)

    # Step 15: Generate plots (if enabled and within max_plot_splits limit)
    should_plot = config.output.save_plots and (
        config.output.max_plot_splits == 0 or seed < config.output.max_plot_splits
    )
    if should_plot:
        log_section(logger, "Generating Plots")
        plots_dir = Path(outdirs.plots)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Common metadata for plots (enriched with run context)
        meta_lines = build_plot_metadata(
            model=config.model,
            scenario=scenario,
            seed=seed,
            train_prev=train_prev,
            target_prev=target_prev,
            cv_folds=config.cv.folds,
            cv_repeats=config.cv.repeats,
            cv_scoring=config.cv.scoring,
            n_features=(len(final_selected_proteins) if final_selected_proteins else None),
            feature_method=config.features.feature_selection_strategy,
            n_train=len(y_train),
            n_val=len(y_val),
            n_test=len(y_test),
            n_train_pos=int(y_train.sum()),
            n_val_pos=int(y_val.sum()),
            n_test_pos=int(y_test.sum()),
            n_train_controls=train_breakdown.get("controls"),
            n_train_incident=train_breakdown.get("incident"),
            n_train_prevalent=train_breakdown.get("prevalent"),
            n_val_controls=val_breakdown.get("controls"),
            n_val_incident=val_breakdown.get("incident"),
            n_val_prevalent=val_breakdown.get("prevalent"),
            n_test_controls=test_breakdown.get("controls"),
            n_test_incident=test_breakdown.get("incident"),
            n_test_prevalent=test_breakdown.get("prevalent"),
            split_mode="development",
            optuna_enabled=config.optuna.enabled,
            n_trials=config.optuna.n_trials if config.optuna.enabled else None,
            n_iter=(get_model_n_iter(config.model, config) if not config.optuna.enabled else None),
            threshold_objective=config.thresholds.objective,
            prevalence_adjusted=True,
        )

        # Validation set plots (only if validation set exists)
        if len(val_idx) > 0:
            val_y_prob = val_preds_df["y_prob"].values
            val_title = f"{config.model} - Validation Set"

            # Compute validation threshold bundle (standardized interface)
            val_dca_thr = threshold_dca_zero_crossing(y_val, val_y_prob)
            val_bundle = compute_threshold_bundle(
                y_val,
                val_y_prob,
                target_spec=config.thresholds.fixed_spec,
                dca_threshold=val_dca_thr,
            )

            if config.output.plot_roc:
                plot_roc_curve(
                    y_true=y_val,
                    y_pred=val_y_prob,
                    out_path=plots_dir / f"{config.model}__val_roc.{config.output.plot_format}",
                    title=val_title,
                    subtitle="ROC Curve",
                    meta_lines=meta_lines,
                    threshold_bundle=val_bundle,
                )
                logger.info("Val ROC curve saved")

            if config.output.plot_pr:
                plot_pr_curve(
                    y_true=y_val,
                    y_pred=val_y_prob,
                    out_path=plots_dir / f"{config.model}__val_pr.{config.output.plot_format}",
                    title=val_title,
                    subtitle="Precision-Recall Curve",
                    meta_lines=meta_lines,
                )
                logger.info("Val PR curve saved")

            if config.output.plot_calibration:
                plot_calibration_curve(
                    y_true=y_val,
                    y_pred=val_y_prob,
                    out_path=plots_dir
                    / f"{config.model}__val_calibration.{config.output.plot_format}",
                    title=val_title,
                    subtitle="Calibration",
                    n_bins=config.output.calib_bins,
                    meta_lines=meta_lines,
                )
                logger.info("Val calibration plot saved")
        else:
            logger.info("Skipping validation plots (no validation set)")
            val_bundle = None

        # Test set plots
        test_y_prob = test_preds_df["y_prob"].values
        test_title = f"{config.model} - Test Set"

        # Compute test threshold bundle (standardized interface)
        dca_thr = threshold_dca_zero_crossing(y_test, test_y_prob)
        test_bundle = compute_threshold_bundle(
            y_test,
            test_y_prob,
            target_spec=config.thresholds.fixed_spec,
            dca_threshold=dca_thr,
        )
        youden_thr = test_bundle["youden_threshold"]
        spec_target_thr = test_bundle["spec_target_threshold"]
        dca_str = f"{dca_thr:.4f}" if dca_thr is not None else "N/A"
        logger.info(
            f"Thresholds: Youden={youden_thr:.4f}, SpecTarget={spec_target_thr:.4f}, DCA={dca_str}"
        )

        if config.output.plot_roc:
            plot_roc_curve(
                y_true=y_test,
                y_pred=test_y_prob,
                out_path=plots_dir / f"{config.model}__test_roc.{config.output.plot_format}",
                title=test_title,
                subtitle="ROC Curve",
                meta_lines=meta_lines,
                threshold_bundle=test_bundle,
            )
            logger.info("Test ROC curve saved")

        if config.output.plot_pr:
            plot_pr_curve(
                y_true=y_test,
                y_pred=test_y_prob,
                out_path=plots_dir / f"{config.model}__test_pr.{config.output.plot_format}",
                title=test_title,
                subtitle="Precision-Recall Curve",
                meta_lines=meta_lines,
            )
            logger.info("Test PR curve saved")

        if config.output.plot_calibration:
            plot_calibration_curve(
                y_true=y_test,
                y_pred=test_y_prob,
                out_path=plots_dir
                / f"{config.model}__test_calibration.{config.output.plot_format}",
                title=test_title,
                subtitle="Calibration",
                n_bins=config.output.calib_bins,
                meta_lines=meta_lines,
            )
            logger.info("Test calibration plot saved")

        # DCA plots for test set
        if config.output.plot_dca:
            plot_dca_curve(
                y_true=y_test,
                y_pred=test_y_prob,
                out_path=str(plots_dir / f"{config.model}__test_dca.{config.output.plot_format}"),
                title=test_title,
                subtitle="Decision Curve Analysis",
                meta_lines=meta_lines,
            )
            logger.info("Test DCA plot saved")

        # DCA plots for validation set (only if validation set exists)
        if config.output.plot_dca and len(val_idx) > 0:
            plot_dca_curve(
                y_true=y_val,
                y_pred=val_preds_df["y_prob"].values,
                out_path=str(plots_dir / f"{config.model}__val_dca.{config.output.plot_format}"),
                title=f"{config.model} - Validation Set",
                subtitle="Decision Curve Analysis",
                meta_lines=meta_lines,
            )
            logger.info("Validation DCA plot saved")

        # Combined OOF plots across CV repeats
        oof_meta = build_oof_metadata(
            model=config.model,
            scenario=scenario,
            seed=seed,
            cv_folds=config.cv.folds,
            cv_repeats=config.cv.repeats,
            train_prev=train_prev,
            n_train=len(y_train),
            n_train_pos=int(y_train.sum()),
            n_train_controls=train_breakdown.get("controls"),
            n_train_incident=train_breakdown.get("incident"),
            n_train_prevalent=train_breakdown.get("prevalent"),
            n_features=(len(final_selected_proteins) if final_selected_proteins else None),
            feature_method=config.features.feature_selection_strategy,
            cv_scoring=config.cv.scoring,
        )
        if config.output.plot_oof_combined:
            plot_oof_combined(
                y_true=y_train,
                oof_preds=oof_preds,
                out_dir=plots_dir,
                model_name=config.model,
                plot_format=config.output.plot_format,
                calib_bins=config.output.calib_bins,
                meta_lines=oof_meta,
            )
            logger.info("OOF combined plots saved")

        # Generate risk distribution plots (consolidated with other diagnostic plots)
        # Test set risk distribution
        if config.output.plot_risk_distribution:
            plot_risk_distribution(
                y_true=y_test,
                scores=test_preds_df["y_prob"].values,
                out_path=plots_dir
                / f"{config.model}__TEST_risk_distribution.{config.output.plot_format}",
                title=f"{config.model} - Test Set",
                subtitle="Risk Score Distribution",
                meta_lines=meta_lines,
                category_col=cat_test,
                threshold_bundle=test_bundle,
            )
            logger.info("Test risk distribution plot saved")

        # Val set risk distribution (use VAL bundle) - only if validation set exists
        if config.output.plot_risk_distribution and len(val_idx) > 0:
            plot_risk_distribution(
                y_true=y_val,
                scores=val_preds_df["y_prob"].values,
                out_path=plots_dir
                / f"{config.model}__VAL_risk_distribution.{config.output.plot_format}",
                title=f"{config.model} - Validation Set",
                subtitle="Risk Score Distribution",
                meta_lines=meta_lines,
                category_col=cat_val,
                threshold_bundle=val_bundle,
            )
            logger.info("Val risk distribution plot saved")

        # Train OOF risk distribution (mean across repeats)
        if config.output.plot_risk_distribution:
            oof_mean = oof_preds.mean(axis=0)
            # Compute OOF-specific threshold bundle
            oof_dca_thr = threshold_dca_zero_crossing(y_train, oof_mean)
            oof_bundle = compute_threshold_bundle(
                y_train,
                oof_mean,
                target_spec=config.thresholds.fixed_spec,
                dca_threshold=oof_dca_thr,
            )
            plot_risk_distribution(
                y_true=y_train,
                scores=oof_mean,
                out_path=plots_dir
                / f"{config.model}__TRAIN_OOF_risk_distribution.{config.output.plot_format}",
                title=f"{config.model} - Train OOF",
                subtitle="Risk Score Distribution (mean across repeats)",
                meta_lines=meta_lines,
                category_col=cat_train,
                threshold_bundle=oof_bundle,
            )
            logger.info("Train OOF risk distribution plot saved")

        logger.info(f"All diagnostic plots saved to: {plots_dir}")

    # Step 16: Generate additional artifacts
    log_section(logger, "Generating Additional Artifacts")

    # --- Calibration CSV export (raw + adjusted) ---
    try:
        from sklearn.calibration import calibration_curve

        # Test set calibration data
        test_y_prob = test_preds_df["y_prob"].values
        prob_true_test, prob_pred_test = calibration_curve(
            y_test, test_y_prob, n_bins=config.output.calib_bins, strategy="uniform"
        )
        calib_df_test = pd.DataFrame(
            {
                "bin_center": prob_pred_test,
                "observed_freq": prob_true_test,
                "split": "test",
                "scenario": scenario,
                "model": config.model,
            }
        )

        # Val set calibration data
        val_y_prob = val_preds_df["y_prob"].values
        prob_true_val, prob_pred_val = calibration_curve(
            y_val, val_y_prob, n_bins=config.output.calib_bins, strategy="uniform"
        )
        calib_df_val = pd.DataFrame(
            {
                "bin_center": prob_pred_val,
                "observed_freq": prob_true_val,
                "split": "val",
                "scenario": scenario,
                "model": config.model,
            }
        )

        # Combine and save
        calib_df = pd.concat([calib_df_test, calib_df_val], ignore_index=True)
        calib_csv_path = Path(outdirs.diag_calibration) / f"{config.model}__calibration.csv"
        calib_df.to_csv(calib_csv_path, index=False)
        logger.info(f"Calibration data saved: {calib_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save calibration CSV: {e}")

    # --- DCA data export ---
    try:
        dca_summary = save_dca_results(
            y_true=y_test,
            y_pred_prob=test_preds_df["y_prob"].values,
            out_dir=str(outdirs.diag_dca),
            prefix=f"{config.model}__test__",
            thresholds=None,  # Use default thresholds
            report_points=None,
            prevalence_adjustment=target_prev,
        )
        logger.info(f"DCA results saved: {dca_summary.get('dca_csv_path', 'N/A')}")

        # Also compute DCA for validation set (only if validation set exists)
        if len(val_idx) > 0:
            dca_summary_val = save_dca_results(
                y_true=y_val,
                y_pred_prob=val_preds_df["y_prob"].values,
                out_dir=str(outdirs.diag_dca),
                prefix=f"{config.model}__val__",
                thresholds=None,
                report_points=None,
                prevalence_adjustment=target_prev,
            )
            logger.info(f"DCA (val) results saved: {dca_summary_val.get('dca_csv_path', 'N/A')}")
    except Exception as e:
        logger.warning(f"Failed to save DCA results: {e}")

    # --- Learning curve CSV export ---
    try:
        lc_enabled = getattr(config.evaluation, "learning_curve", False)
        plot_lc = getattr(config.output, "plot_learning_curve", True)
        if lc_enabled and plot_lc:
            # CSV goes to diagnostics/, plots go to plots/
            lc_csv_path = Path(outdirs.diag_learning) / f"{config.model}__learning_curve.csv"
            # Only generate plot if within max_plot_splits limit
            lc_plot_path = (
                Path(outdirs.plots) / f"{config.model}__learning_curve.{config.output.plot_format}"
                if should_plot
                else None
            )
            lc_meta = build_plot_metadata(
                model=config.model,
                scenario=scenario,
                seed=seed,
                train_prev=train_prev,
                cv_folds=min(config.cv.folds, 5),  # Matches actual CV used in learning curve
                cv_repeats=1,  # Learning curve doesn't use repeats
                cv_scoring=config.cv.scoring,
                n_features=(len(final_selected_proteins) if final_selected_proteins else None),
                feature_method=config.features.feature_selection_strategy,
                n_train=len(y_train),
                n_train_pos=int(y_train.sum()),
                n_train_controls=train_breakdown.get("controls"),
                n_train_incident=train_breakdown.get("incident"),
                n_train_prevalent=train_breakdown.get("prevalent"),
                split_mode="development",
            )
            # Precompute screening on full training set so learning curve
            # iterations reuse the result instead of re-running Mann-Whitney
            # 25 times (5 sizes x 5 folds).
            lc_precomputed_screen = None
            screen_method = getattr(config.features, "screen_method", "none")
            screen_top_n = getattr(config.features, "screen_top_n", 0)
            if screen_method and screen_method != "none" and screen_top_n > 0:
                from ced_ml.features.screening import screen_proteins

                lc_precomputed_screen, _, _ = screen_proteins(
                    X_train=X_train,
                    y_train=y_train,
                    protein_cols=protein_cols,
                    method=screen_method,
                    top_n=screen_top_n,
                )
                logger.info(
                    f"Precomputed screening for learning curve: "
                    f"{len(lc_precomputed_screen)} proteins"
                )

            # Build a fresh pipeline for learning curve (don't use fitted one)
            lc_pipeline = build_training_pipeline(
                config,
                build_models(config.model, config, seed, config.n_jobs),
                protein_cols,
                resolved.categorical_metadata,
                model_name=config.model,
            )

            # Inject precomputed screening to avoid redundant computation
            if lc_precomputed_screen is not None and "screen" in dict(lc_pipeline.steps):
                lc_pipeline.named_steps["screen"].precomputed_features = lc_precomputed_screen
            save_learning_curve_csv(
                estimator=lc_pipeline,
                X=X_train,
                y=y_train,
                out_csv=lc_csv_path,
                scoring=config.cv.scoring,
                cv=min(config.cv.folds, 5),  # Use at most 5 folds for speed
                min_frac=0.3,
                n_points=5,
                seed=seed,
                out_plot=lc_plot_path,
                meta_lines=lc_meta,
            )
            logger.info(f"Learning curve saved: {lc_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save learning curve: {e}")

    # --- Screening results export (compute once and reuse) ---
    screening_stats = pd.DataFrame()
    try:
        screen_method = getattr(config.features, "screen_method", "none")
        if screen_method and screen_method != "none":
            # Compute screening stats once (will be reused for feature report below)
            _, screening_stats, _ = screen_proteins(
                X_train=X_train,
                y_train=y_train,
                protein_cols=protein_cols,
                method=screen_method,
                top_n=0,  # Get all proteins (no filtering)
            )
            if not screening_stats.empty:
                # Save standalone screening results CSV
                screening_stats_export = screening_stats.copy()
                screening_stats_export["scenario"] = scenario
                screening_stats_export["model"] = config.model
                screening_path = (
                    Path(outdirs.diag_screening) / f"{config.model}__screening_results.csv"
                )
                screening_stats_export.to_csv(screening_path, index=False)
                logger.info(f"Screening results saved: {screening_path}")
    except Exception as e:
        logger.warning(f"Failed to save screening results: {e}")

    # --- Feature reports and stable panel export ---
    try:
        # Compute selection frequencies from CV results
        selection_freq = compute_selection_frequencies(
            selected_proteins_df,
            selection_col="selected_proteins",
        )

        if selection_freq:
            # Build base feature report with selection frequencies
            feature_report = pd.DataFrame(
                [{"protein": p, "selection_freq": f} for p, f in selection_freq.items()]
            )

            # Merge with screening statistics if available (reuse from above)
            if not screening_stats.empty:
                # screening_stats has columns: protein, effect_size, p_value, rank
                feature_report = feature_report.merge(
                    screening_stats[["protein", "effect_size", "p_value"]],
                    on="protein",
                    how="left",
                )

            # Add NaN columns if not present (when screening is disabled)
            if "effect_size" not in feature_report.columns:
                feature_report["effect_size"] = np.nan
            if "p_value" not in feature_report.columns:
                feature_report["p_value"] = np.nan

            # Sort by selection frequency and add rank
            feature_report = feature_report.sort_values(
                "selection_freq", ascending=False
            ).reset_index(drop=True)
            feature_report["rank"] = range(1, len(feature_report) + 1)
            feature_report["scenario"] = scenario
            feature_report["model"] = config.model

            # Reorder columns for readability
            col_order = [
                "rank",
                "protein",
                "selection_freq",
                "effect_size",
                "p_value",
                "scenario",
                "model",
            ]
            feature_report = feature_report[col_order]

            # Use ResultsWriter to save
            writer.save_feature_report(feature_report, config.model)

            # Stable panel extraction
            stable_panel_df, stable_proteins, _ = extract_stable_panel(
                selection_log=selected_proteins_df,
                n_repeats=config.cv.repeats,
                stability_threshold=0.75,
                selection_col="selected_proteins",
                fallback_top_n=20,
            )
            if not stable_panel_df.empty:
                stable_panel_df["scenario"] = scenario
                # Use ResultsWriter to save
                writer.save_stable_panel_report(stable_panel_df, panel_type="KBest")

            # Panel manifests (multiple sizes)
            panels_config = getattr(config, "panels", None)
            panel_sizes = (
                getattr(panels_config, "panel_sizes", [10, 25, 50])
                if panels_config
                else [10, 25, 50]
            )
            if panel_sizes and len(selection_freq) >= min(panel_sizes):
                corr_threshold = (
                    getattr(panels_config, "panel_corr_thresh", 0.80) if panels_config else 0.80
                )
                corr_method = (
                    getattr(panels_config, "panel_corr_method", "spearman")
                    if panels_config
                    else "spearman"
                )
                panels = build_multi_size_panels(
                    df=X_train,
                    y=y_train,
                    selection_freq=selection_freq,
                    panel_sizes=panel_sizes,
                    corr_threshold=corr_threshold,
                    corr_method=corr_method,
                    pool_limit=1000,
                )
                for size, (_comp_map, panel_proteins) in panels.items():
                    manifest = {
                        "scenario": scenario,
                        "model": config.model,
                        "panel_size": size,
                        "actual_size": len(panel_proteins),
                        "corr_threshold": corr_threshold,
                        "proteins": panel_proteins,
                    }
                    # Use ResultsWriter to save
                    writer.save_panel_manifest(manifest, config.model, size)
    except Exception as e:
        logger.warning(f"Failed to save feature reports/panels: {e}")

    # --- Bootstrap CI for small test sets ---
    try:
        # Only run bootstrap if test set is small (< threshold samples)
        min_bootstrap_threshold = getattr(config.evaluation, "bootstrap_min_samples", 100)
        if len(y_test) < min_bootstrap_threshold:
            logger.info(f"Test set small ({len(y_test)} samples) - computing bootstrap CI")

            # Bootstrap CI for AUROC
            auroc_lo, auroc_hi = stratified_bootstrap_ci(
                y_true=y_test,
                y_pred=test_preds_df["y_prob"].values,
                metric_fn=roc_auc_score,
                n_boot=1000,
                seed=seed,
            )

            # Bootstrap CI for PR-AUC
            prauc_lo, prauc_hi = stratified_bootstrap_ci(
                y_true=y_test,
                y_pred=test_preds_df["y_prob"].values,
                metric_fn=average_precision_score,
                n_boot=1000,
                seed=seed,
            )

            bootstrap_ci_df = pd.DataFrame(
                [
                    {
                        "scenario": scenario,
                        "model": config.model,
                        "n_test": len(y_test),
                        "n_boot": 1000,
                        "bootstrap_seed": seed,
                        METRIC_AUROC: test_metrics[METRIC_AUROC],
                        "AUROC_ci_lo": auroc_lo,
                        "AUROC_ci_hi": auroc_hi,
                        METRIC_PRAUC: test_metrics[METRIC_PRAUC],
                        "PR_AUC_ci_lo": prauc_lo,
                        "PR_AUC_ci_hi": prauc_hi,
                    }
                ]
            )
            bootstrap_ci_path = (
                Path(outdirs.diag_test_ci) / f"{config.model}__test_bootstrap_ci.csv"
            )
            bootstrap_ci_df.to_csv(bootstrap_ci_path, index=False)
            logger.info(
                f"Bootstrap CI saved: {bootstrap_ci_path} "
                f"(AUROC: {auroc_lo:.3f}-{auroc_hi:.3f}, PR-AUC: {prauc_lo:.3f}-{prauc_hi:.3f})"
            )
    except Exception as e:
        logger.warning(f"Failed to compute bootstrap CI: {e}")

    log_section(logger, "Training Complete")
    logger.info(f"All results saved to: {config.outdir}")

    # File logging disabled (using shell tee for live logs)
    # finalize_live_log(logger)
