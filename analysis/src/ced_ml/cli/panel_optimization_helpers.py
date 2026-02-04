"""Helper functions for panel optimization workflows.

This module contains shared functionality used by both single-seed and
aggregated panel optimization workflows in optimize_panel.py.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label
from ced_ml.features.rfe import RFEResult, recursive_feature_elimination

logger = logging.getLogger(__name__)


def load_stability_panel(
    stability_file: Path,
    stability_threshold: float,
    start_size: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Load and filter stability panel from aggregated results.

    Args:
        stability_file: Path to feature_stability_summary.csv
        stability_threshold: Minimum selection frequency threshold
        start_size: Cap to top N proteins by selection frequency (None = no cap)

    Returns:
        Tuple of (stable_proteins, selection_freq_dict)

    Raises:
        FileNotFoundError: If stability file not found
        ValueError: If no proteins meet threshold
    """
    if not stability_file.exists():
        raise FileNotFoundError(
            f"Feature stability file not found: {stability_file}\n"
            "Run 'ced aggregate-splits' first to generate aggregated results."
        )

    logger.info(f"Loading stability panel from {stability_file}")
    stability_df = pd.read_csv(stability_file)

    stable_proteins = stability_df[stability_df["selection_fraction"] >= stability_threshold][
        "protein"
    ].tolist()

    if not stable_proteins:
        raise ValueError(
            f"No proteins meet stability threshold {stability_threshold:.2f}. "
            f"Try lowering --stability-threshold."
        )

    if start_size and start_size > 0 and len(stable_proteins) > start_size:
        stable_sorted = stability_df[stability_df["protein"].isin(stable_proteins)].sort_values(
            "selection_fraction", ascending=False
        )
        stable_proteins = stable_sorted["protein"].head(start_size).tolist()
        logger.info(f"Capped {len(stable_sorted)} stable proteins to start_size={start_size}")

    selection_freq = dict(
        zip(stability_df["protein"], stability_df["selection_fraction"], strict=False)
    )

    logger.info(f"Loaded {len(stable_proteins)} stable proteins (>={stability_threshold:.2f})")

    return stable_proteins, selection_freq


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate model bundle.

    Args:
        model_path: Path to model joblib file

    Returns:
        Model bundle dictionary

    Raises:
        FileNotFoundError: If model not found
        ValueError: If bundle is invalid
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")

    if bundle.get("model") is None:
        raise ValueError("Model bundle missing 'model' key")

    return bundle


def extract_retune_cv_folds(bundle_config: Any) -> int:
    """Extract inner CV folds from bundle config.

    Args:
        bundle_config: Config object from model bundle

    Returns:
        Number of inner CV folds (default: 5)
    """
    if isinstance(bundle_config, dict):
        return bundle_config.get("cv", {}).get("inner_folds", 5)
    elif hasattr(bundle_config, "cv"):
        return getattr(bundle_config.cv, "inner_folds", 5)
    return 5


def load_and_filter_data(
    infile: str,
    feature_cols: list[str],
    protein_cols: list[str],
    meta_num_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load data and filter columns.

    Args:
        infile: Path to input data file
        feature_cols: List of all feature columns
        protein_cols: List of protein columns
        meta_num_cols: List of numeric metadata columns

    Returns:
        Tuple of (filtered_df, filtered_feature_cols, filtered_protein_cols)
    """
    logger.info(f"Loading data from {infile}")
    df_raw = read_proteomics_file(infile, validate=True)

    logger.info("Applying row filters...")
    df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
    logger.info(f"Filtered: {filter_stats['n_in']:,} -> {filter_stats['n_out']:,} rows")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing[:10]}...")
        feature_cols = [c for c in feature_cols if c in df.columns]
        protein_cols = [c for c in protein_cols if c in df.columns]

    return df, feature_cols, protein_cols


def load_split_indices(
    split_dir: Path,
    scenario: str,
    seed: int,
) -> tuple[Any, Any]:
    """Load train and validation split indices.

    Args:
        split_dir: Directory containing split files
        scenario: Data scenario (e.g., "IncidentOnly")
        seed: Split seed

    Returns:
        Tuple of (train_idx, val_idx)

    Raises:
        FileNotFoundError: If split files not found
    """
    train_file = split_dir / f"train_idx_{scenario}_seed{seed}.csv"
    val_file = split_dir / f"val_idx_{scenario}_seed{seed}.csv"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Split files not found for seed {seed}: {train_file}, {val_file}")

    train_idx = pd.read_csv(train_file).squeeze().values
    val_idx = pd.read_csv(val_file).squeeze().values

    return train_idx, val_idx


def run_rfe_for_seed(
    seed: int,
    model_path: Path,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_dir: Path,
    scenario: str,
    model_name: str,
    initial_proteins: list[str],
    cat_cols: list[str],
    meta_num_cols: list[str],
    min_size: int,
    cv_folds: int,
    step_strategy: str,
    min_auroc_frac: float,
    retune_n_trials: int,
    retune_cv_folds: int,
    retune_n_jobs: int,
    corr_aware: bool,
    corr_threshold: float,
    corr_method: str,
    selection_freq: dict[str, float] | None,
    rfe_tune_spaces: dict[str, dict[str, dict]] | None,
) -> RFEResult:
    """Run RFE for a single split seed.

    This is the core RFE execution logic shared between single-seed and
    aggregated workflows.

    Args:
        seed: Split seed number
        model_path: Path to model bundle for this seed
        df: Full dataframe (pre-loaded and filtered)
        feature_cols: Feature column names
        split_dir: Directory containing split indices
        scenario: Data scenario
        model_name: Model name
        initial_proteins: Starting protein panel
        cat_cols: Categorical column names
        meta_num_cols: Numeric metadata column names
        min_size: Minimum panel size
        cv_folds: CV folds for RFE
        step_strategy: Elimination strategy
        min_auroc_frac: Early stop threshold
        retune_n_trials: Optuna trials
        retune_cv_folds: CV folds for retuning
        retune_n_jobs: Parallel jobs for Optuna
        corr_aware: Use correlation-aware elimination
        corr_threshold: Correlation threshold
        corr_method: Correlation method
        selection_freq: Selection frequency dict
        rfe_tune_spaces: Custom hyperparameter spaces

    Returns:
        RFEResult for this seed
    """
    import time

    logger.info(f"\n{'='*60}\nStarting RFE for seed {seed}\n{'='*60}")
    seed_start = time.time()

    bundle = load_model_bundle(model_path)
    pipeline = bundle.get("model")

    if pipeline is None:
        raise ValueError(f"Model bundle missing 'model' key for seed {seed}")

    train_idx, val_idx = load_split_indices(Path(split_dir), scenario, seed)

    positive_label = get_positive_label(scenario)
    y_all = (df[TARGET_COL] == positive_label).astype(int).values

    X_train = df.iloc[train_idx][feature_cols].copy()
    y_train = y_all[train_idx]
    X_val = df.iloc[val_idx][feature_cols].copy()
    y_val = y_all[val_idx]

    logger.info(
        f"Seed {seed}: train={len(X_train)} ({y_train.sum()} cases), "
        f"val={len(X_val)} ({y_val.sum()} cases)"
    )

    result = recursive_feature_elimination(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        base_pipeline=pipeline,
        model_name=model_name,
        initial_proteins=initial_proteins,
        cat_cols=cat_cols,
        meta_num_cols=meta_num_cols,
        min_size=min_size,
        cv_folds=cv_folds,
        step_strategy=step_strategy,
        min_auroc_frac=min_auroc_frac,
        random_state=seed,
        retune_n_trials=retune_n_trials,
        retune_cv_folds=retune_cv_folds,
        retune_n_jobs=retune_n_jobs,
        corr_aware=corr_aware,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        selection_freq=selection_freq,
        rfe_tune_spaces=rfe_tune_spaces,
    )

    seed_elapsed = time.time() - seed_start
    logger.info(
        f"\n{'='*60}\n"
        f"Completed RFE for seed {seed}\n"
        f"Time: {seed_elapsed/60:.1f} min\n"
        f"Max AUROC: {result.max_auroc:.4f}\n"
        f"Evaluation points: {len(result.curve)}\n"
        f"{'='*60}\n"
    )

    return result


def print_optimization_summary(
    model_name: str,
    result: RFEResult,
    n_seeds: int | None = None,
    initial_panel_size: int | None = None,
    outdir: Path | None = None,
    aggregated: bool = False,
) -> None:
    """Print standardized optimization summary.

    Args:
        model_name: Model name
        result: RFE result
        n_seeds: Number of seeds (for aggregated results)
        initial_panel_size: Starting panel size
        outdir: Output directory
        aggregated: Whether this is aggregated results
    """
    title = (
        "Aggregated Panel Optimization Complete" if aggregated else "Panel Optimization Complete"
    )
    print(f"\n{'='*60}")
    print(f"{title}: {model_name}")
    print(f"{'='*60}")

    if n_seeds:
        print(f"Seeds aggregated: {n_seeds}")

    if initial_panel_size:
        desc = "aggregated stable proteins" if aggregated else "proteins"
        print(f"Starting panel size: {initial_panel_size} ({desc})")

    print(f"Max {'mean ' if aggregated else ''}AUROC: {result.max_auroc:.4f}")

    if result.curve:
        best_point = max(result.curve, key=lambda x: x["auroc_val"])
        print(f"\nBest panel (size={best_point['size']}):")
        print(f"  AUROC:           {best_point['auroc_val']:.4f}")
        print(f"  PR-AUC:          {best_point.get('prauc_val', float('nan')):.4f}")
        print(f"  Brier Score:     {best_point.get('brier_val', float('nan')):.4f}")
        print(f"  Sens@95%Spec:    {best_point.get('sens_at_95spec_val', float('nan')):.4f}")

    print("\nRecommended panel sizes:")
    for key, size in result.recommended_panels.items():
        print(f"  {key}: {size} proteins")

    if outdir:
        suffix = "_aggregated" if aggregated else ""
        print(f"\nResults saved to: {outdir}")
        print(f"  - panel_curve{suffix}.csv (full curve with all metrics)")
        print(f"  - metrics_summary{suffix}.csv (metrics at each panel size)")
        print(f"  - recommended_panels{suffix}.json (threshold-based recommendations)")
        print(f"  - feature_ranking{suffix}.csv (protein elimination order)")

    print(f"{'='*60}\n")
