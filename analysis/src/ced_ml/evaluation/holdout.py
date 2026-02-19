"""
Holdout evaluation module.

This module provides functionality for evaluating trained models on holdout sets.
It handles:
- Loading holdout indices and model artifacts
- Computing discrimination, calibration, and clinical utility metrics
- Generating predictions with prevalence adjustment
- Saving comprehensive evaluation results
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import identify_protein_columns, read_proteomics_file
from ced_ml.data.schema import (
    METRIC_AUROC,
    METRIC_PRAUC,
    TARGET_COL,
    get_positive_label,
    get_scenario_labels,
)
from ced_ml.data.splits import temporal_order_indices
from ced_ml.metrics import (
    binary_metrics_at_threshold,
    compute_brier_score,
    compute_discrimination_metrics,
    generate_dca_thresholds,
    parse_dca_report_points,
    save_dca_results,
    top_risk_capture,
)
from ced_ml.models.calibration import (
    adaptive_expected_calibration_error,
    brier_score_decomposition,
    calibration_intercept_slope,
    expected_calibration_error,
    integrated_calibration_index,
    spiegelhalter_z_test,
)
from ced_ml.models.prevalence import adjust_probabilities_for_prevalence
from ced_ml.utils.math_utils import EPSILON_BOUNDS

logger = logging.getLogger(__name__)


@dataclass
class HoldoutResult:
    """Result from loading holdout indices.

    Attributes:
        indices: Array of holdout sample indices
        metadata: Dictionary of metadata from companion JSON file
    """

    indices: np.ndarray
    metadata: dict[str, Any]


def load_holdout_indices(path: str) -> HoldoutResult:
    """
    Load holdout indices and associated metadata from CSV file.

    Args:
        path: Path to holdout index CSV with 'idx' column

    Returns:
        HoldoutResult dataclass with indices and metadata

    Raises:
        ValueError: If file missing 'idx' column
    """
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Holdout index file {path} must contain an 'idx' column. "
            f"Available columns: [{available_cols}]"
        )

    indices = df["idx"].to_numpy(dtype=int)

    # Try to load companion metadata JSON
    metadata = {}
    path_obj = Path(path)
    meta_path = path_obj.parent / (path_obj.stem.replace("_idx", "_meta") + ".json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return HoldoutResult(indices=indices, metadata=metadata)


def load_model_artifact(path: str) -> dict[str, Any]:
    """
    Load saved model artifact (joblib bundle).

    Args:
        path: Path to model artifact (.joblib file)

    Returns:
        Dictionary containing model and metadata

    Raises:
        ValueError: If artifact is not in modern bundle format (dict)
    """
    artifact = joblib.load(path)

    # Require modern bundle format
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Model artifact at {path} is in legacy format (bare model). "
            "Modern bundle format (dict) is required. "
            "Please retrain the model with current training pipeline."
        )

    return artifact


def extract_holdout_data(
    df_filtered: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    holdout_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Extract holdout subset from full dataset.

    Args:
        df_filtered: Full filtered dataframe
        X_all: Full feature matrix
        y_all: Full target array
        holdout_idx: Indices for holdout set

    Returns:
        (df_holdout, X_holdout, y_holdout) tuple

    Raises:
        ValueError: If holdout indices exceed dataset size
    """
    if len(holdout_idx) > 0 and holdout_idx.max() >= len(df_filtered):
        raise ValueError(
            f"Holdout index exceeds dataset rows " f"({holdout_idx.max()} >= {len(df_filtered)})."
        )

    # Convert to int indices for proper indexing
    idx_int = holdout_idx.astype(int) if len(holdout_idx) > 0 else np.array([], dtype=int)

    df_holdout = df_filtered.iloc[idx_int].reset_index(drop=True)
    X_holdout = X_all.iloc[idx_int].reset_index(drop=True)
    y_holdout = y_all[idx_int]

    return df_holdout, X_holdout, y_holdout


def compute_holdout_metrics(
    y_true: np.ndarray,
    proba_eval: np.ndarray,
    bundle: dict[str, Any],
    scenario: str,
    clinical_points: list[float],
) -> dict[str, Any]:
    """
    Compute comprehensive holdout metrics.

    Args:
        y_true: True labels for holdout set
        proba_eval: Predicted probabilities
        bundle: Model artifact bundle with metadata
        scenario: Scenario name (e.g., 'IncidentOnly')
        clinical_points: Clinical probability thresholds to evaluate

    Returns:
        Dictionary of holdout metrics
    """
    # Discrimination metrics
    disc_metrics = compute_discrimination_metrics(y_true, proba_eval)
    brier = compute_brier_score(y_true, proba_eval)

    # Calibration metrics
    cal_metrics = calibration_intercept_slope(y_true, proba_eval)
    ece = expected_calibration_error(y_true, proba_eval)
    ici = integrated_calibration_index(y_true, proba_eval)
    spieg = spiegelhalter_z_test(y_true, proba_eval)
    ece_adaptive = adaptive_expected_calibration_error(y_true, proba_eval)
    brier_decomp = brier_score_decomposition(y_true, proba_eval)

    # Extract threshold metadata
    thresholds_meta = bundle.get("thresholds", {})
    # Use val_threshold as primary threshold (chosen on validation set)
    val_threshold = thresholds_meta.get("val_threshold")
    if val_threshold is None:
        # Never silently use test_threshold as replacement (ADR-009 violation).
        # Using test_threshold would leak test-set information into holdout decisions.
        # Log on the root logger so test harnesses using caplog/root handlers
        # reliably capture this warning even when ced_ml logger propagation is disabled.
        logging.getLogger().warning(
            "val_threshold missing from model bundle. Using default threshold=0.5. "
            "This indicates an older model bundle format. "
            "Models should be retrained with validation-based threshold selection (ADR-009)."
        )
        thr_primary = 0.5
    else:
        thr_primary = val_threshold
    objective_name = thresholds_meta.get("objective", "unknown")
    fixed_spec_value = thresholds_meta.get("fixed_spec")

    # Check if threshold was selected on adjusted or raw probabilities (F2 fix)
    threshold_prob_scale = thresholds_meta.get("threshold_prob_scale", "raw")

    # For backward compatibility with old bundles
    thr_f1 = thresholds_meta.get("max_f1", thr_primary)
    thr_spec90 = thresholds_meta.get("spec90", thr_primary)
    ctrl_specs = thresholds_meta.get("control_specs", {})

    # Prevalence metadata (needed for F2: probability scale matching)
    prevalence_meta = bundle.get("prevalence", {})
    train_prev = prevalence_meta.get("train_prevalence")
    if train_prev is None or not np.isfinite(train_prev):
        raise ValueError(
            "Model bundle missing valid 'prevalence.train_prevalence'. "
            "Bundle must be from modern training pipeline (post-F1 fix). "
            "Please retrain the model."
        )
    target_prev = prevalence_meta.get("test_prevalence")
    if target_prev is None or not np.isfinite(target_prev):
        raise ValueError(
            "Model bundle missing valid 'prevalence.test_prevalence'. "
            "Bundle must be from modern training pipeline (post-F1 fix). "
            "Please retrain the model."
        )
    train_prev = float(train_prev)
    target_prev = float(np.clip(target_prev, EPSILON_BOUNDS, 1.0 - EPSILON_BOUNDS))

    # F2 fix: Use adjusted probabilities for threshold-based metrics if threshold was
    # selected on adjusted scale. Discrimination/calibration metrics stay on raw scale.
    if threshold_prob_scale == "adjusted":
        from ced_ml.models.prevalence import adjust_probabilities_for_prevalence

        proba_for_thresholds = adjust_probabilities_for_prevalence(
            proba_eval, sample_prev=train_prev, target_prev=target_prev
        )
        logger.info(
            f"Using adjusted probabilities for threshold-based metrics (scale={threshold_prob_scale})"
        )
    else:
        proba_for_thresholds = proba_eval

    # Metrics at key thresholds
    m_primary = binary_metrics_at_threshold(y_true, proba_for_thresholds, thr_primary)
    m_f1 = binary_metrics_at_threshold(y_true, proba_for_thresholds, thr_f1)
    m_spec90 = binary_metrics_at_threshold(y_true, proba_for_thresholds, thr_spec90)

    # Build metrics dictionary
    metrics = {
        "scenario": scenario,
        "model_name": bundle.get("model_name"),
        "model_label": bundle.get("model_label"),
        "split_id": bundle.get("split_id"),
        "n_holdout": int(len(y_true)),
        "n_holdout_pos": int(y_true.sum()),
        "train_prevalence_sample": (float(train_prev) if np.isfinite(train_prev) else np.nan),
        "target_prevalence": float(target_prev),
        "AUROC_holdout": disc_metrics[METRIC_AUROC],
        "PR_AUC_holdout": disc_metrics[METRIC_PRAUC],
        "Brier_holdout": float(brier),
        "calibration_intercept_holdout": (
            float(cal_metrics.intercept) if np.isfinite(cal_metrics.intercept) else np.nan
        ),
        "calibration_slope_holdout": (
            float(cal_metrics.slope) if np.isfinite(cal_metrics.slope) else np.nan
        ),
        "ECE_holdout": float(ece),
        "ICI_holdout": float(ici),
        "ECE_adaptive_holdout": float(ece_adaptive),
        "spiegelhalter_z_holdout": float(spieg.z_statistic),
        "spiegelhalter_p_holdout": float(spieg.p_value),
        "brier_reliability_holdout": float(brier_decomp.reliability),
        "brier_resolution_holdout": float(brier_decomp.resolution),
        "brier_uncertainty_holdout": float(brier_decomp.uncertainty),
        "thr_objective_name": objective_name,
        "thr_primary": float(thr_primary),
        "precision_holdout_at_thr_primary": float(m_primary.precision),
        "recall_holdout_at_thr_primary": float(m_primary.sensitivity),
        "specificity_holdout_at_thr_primary": float(m_primary.specificity),
        "fixed_spec_value": (float(fixed_spec_value) if fixed_spec_value is not None else np.nan),
        "thr_maxF1": float(thr_f1),
        "f1_holdout_at_thr_maxF1": float(m_f1.f1),
        "precision_holdout_at_thr_maxF1": float(m_f1.precision),
        "recall_holdout_at_thr_maxF1": float(m_f1.sensitivity),
        "thr_spec90": float(thr_spec90),
        "sensitivity_holdout_at_spec90": float(m_spec90.sensitivity),
        "specificity_holdout_at_spec90": float(m_spec90.specificity),
    }

    # Control specificity thresholds
    for key, val in ctrl_specs.items():
        try:
            thr_val = float(val)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping invalid threshold '%s': %s", val, e)
            continue
        m_ctrl = binary_metrics_at_threshold(y_true, proba_for_thresholds, thr_val)
        tag = str(key).replace("0.", "")
        metrics[f"thr_ctrl_{tag}"] = float(thr_val)
        metrics[f"precision_holdout_ctrl_{tag}"] = float(m_ctrl.precision)
        metrics[f"recall_holdout_ctrl_{tag}"] = float(m_ctrl.sensitivity)
        metrics[f"specificity_holdout_ctrl_{tag}"] = float(m_ctrl.specificity)

    # Clinical thresholds
    for thr in clinical_points:
        if not (0.0 < thr < 1.0):
            continue
        m_thr = binary_metrics_at_threshold(y_true, proba_for_thresholds, thr)
        tag = f"clin_{str(thr).replace('.', 'p')}"
        metrics[f"{tag}_threshold"] = float(thr)
        metrics[f"{tag}_precision"] = float(m_thr.precision)
        metrics[f"{tag}_recall"] = float(m_thr.sensitivity)
        metrics[f"{tag}_specificity"] = float(m_thr.specificity)
        metrics[f"{tag}_f1"] = float(m_thr.f1)

    return metrics


def compute_top_risk_capture(
    y_true: np.ndarray,
    proba_eval: np.ndarray,
    top_fracs: list[float],
) -> pd.DataFrame:
    """
    Compute top-risk capture statistics.

    Args:
        y_true: True labels
        proba_eval: Predicted probabilities
        top_fracs: List of top fractions to evaluate (e.g., [0.01, 0.05])

    Returns:
        DataFrame with capture statistics per fraction
    """
    rows = []
    for frac in sorted(top_fracs):
        capture = top_risk_capture(y_true, proba_eval, frac=frac)
        rows.append({"frac": frac, **capture})
    return pd.DataFrame(rows)


def save_holdout_predictions(
    outdir: str,
    holdout_idx: np.ndarray,
    df_holdout: pd.DataFrame,
    y_true: np.ndarray,
    proba_eval: np.ndarray,
    proba_adjusted: np.ndarray,
) -> None:
    """
    Save holdout predictions to CSV.

    Args:
        outdir: Output directory
        holdout_idx: Original holdout indices
        df_holdout: Holdout dataframe
        y_true: True labels
        proba_eval: Predicted probabilities
        proba_adjusted: Prevalence-adjusted probabilities
    """
    out = pd.DataFrame(
        {
            "idx": holdout_idx,
            TARGET_COL: df_holdout[TARGET_COL].astype(str),
            "y_true": y_true.astype(int),
            "risk_holdout": proba_eval,
            "risk_holdout_adjusted": proba_adjusted,
            "risk_holdout_raw": proba_eval,
        }
    )
    out.to_csv(Path(outdir) / "holdout_predictions.csv", index=False)


def evaluate_holdout(
    infile: str,
    holdout_idx_file: str,
    model_artifact_path: str,
    outdir: str,
    scenario: str | None = None,
    compute_dca: bool = False,
    dca_threshold_min: float | None = None,
    dca_threshold_max: float | None = None,
    dca_threshold_step: float | None = None,
    dca_report_points: str = "",
    dca_use_target_prevalence: bool = False,
    save_preds: bool = False,
    toprisk_fracs: str = "0.01",
    target_prevalence: float | None = None,
    clinical_threshold_points: str = "",
) -> dict[str, Any]:
    """
    Evaluate trained model on holdout set.

    This is the main entry point for holdout evaluation. It:
    1. Loads the model artifact and holdout data
    2. Generates predictions
    3. Computes comprehensive metrics
    4. Optionally computes DCA and subgroup analyses
    5. Saves all results to disk

    Args:
        infile: Path to full dataset CSV
        holdout_idx_file: Path to holdout indices CSV
        model_artifact_path: Path to trained model artifact (.joblib)
        outdir: Output directory for results
        scenario: Override scenario (if not in artifact)
        compute_dca: Whether to compute decision curve analysis
        dca_threshold_min: Min threshold for DCA
        dca_threshold_max: Max threshold for DCA
        dca_threshold_step: Step size for DCA thresholds
        dca_report_points: Comma-separated thresholds to report
        dca_use_target_prevalence: Use prevalence-adjusted probs for DCA
        save_preds: Save individual predictions to CSV
        toprisk_fracs: Comma-separated top-risk fractions (e.g., "0.01,0.05")
        target_prevalence: Override target prevalence
        clinical_threshold_points: Comma-separated clinical thresholds

    Returns:
        Dictionary of holdout metrics
    """
    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Load model artifact
    bundle = load_model_artifact(model_artifact_path)
    model = bundle["model"]
    scenario_final = scenario or bundle.get("scenario", "IncidentPlusPrevalent")

    # Load data (supports both CSV and Parquet)
    positive_label = get_positive_label(scenario_final)
    df_raw = read_proteomics_file(infile, validate=True)

    # Filter to relevant classes
    keep_labels = get_scenario_labels(scenario_final)
    df_scenario_raw = df_raw[df_raw[TARGET_COL].isin(keep_labels)].copy()

    # Load holdout indices and metadata
    holdout_result = load_holdout_indices(holdout_idx_file)
    holdout_idx = holdout_result.indices
    split_meta = holdout_result.metadata

    # Apply row filters (matching save_splits.py and train.py)
    # Use metadata if available, otherwise use defaults
    meta_num_cols = None
    if split_meta.get("row_filters"):
        meta_num_cols = split_meta["row_filters"].get("meta_num_cols_used")

    df_filtered, filter_stats = apply_row_filters(df_scenario_raw, meta_num_cols=meta_num_cols)

    # Apply temporal ordering if specified in split metadata
    # This ensures indices align exactly with split generation
    temporal_split = split_meta.get("temporal_split", False)
    if temporal_split:
        temporal_col = split_meta.get("temporal_col", "CeD_date")
        if temporal_col in df_filtered.columns:
            order_idx = temporal_order_indices(df_filtered, temporal_col)
            df_filtered = df_filtered.iloc[order_idx].reset_index(drop=True)

    # Create binary outcome
    df_filtered["y"] = (df_filtered[TARGET_COL] == positive_label).astype(int)
    y_all = df_filtered["y"].to_numpy()

    # Extract resolved columns from bundle
    # The model was trained with: protein_cols + numeric_metadata + categorical_metadata
    resolved_cols = bundle.get("resolved_columns")
    if not resolved_cols:
        raise ValueError(
            f"Model artifact at {model_artifact_path} is missing 'resolved_columns' metadata. "
            "This indicates an old bundle format. "
            "Please retrain the model with current training pipeline."
        )

    # Use resolved columns from training (C6 fix + protein validation)
    prot_cols = resolved_cols.get("protein_cols", [])
    numeric_metadata = resolved_cols.get("numeric_metadata", [])
    categorical_metadata = resolved_cols.get("categorical_metadata", [])

    # Validate that holdout data contains all required protein columns
    holdout_prot_cols = set(identify_protein_columns(df_filtered))
    missing_proteins = set(prot_cols) - holdout_prot_cols
    if missing_proteins:
        raise ValueError(
            f"Holdout data missing {len(missing_proteins)} protein columns required by model. "
            f"First 5 missing: {sorted(missing_proteins)[:5]}. "
            "Ensure holdout data was processed with same protein panel as training data."
        )

    # Build feature matrix matching training columns
    feature_cols = list(prot_cols)
    for col in numeric_metadata:
        if col in df_filtered.columns and col not in feature_cols:
            feature_cols.append(col)
    for col in categorical_metadata:
        if col in df_filtered.columns and col not in feature_cols:
            feature_cols.append(col)

    X_all = df_filtered[feature_cols]

    # Extract holdout subset
    df_holdout, X_holdout, y_holdout = extract_holdout_data(df_filtered, X_all, y_all, holdout_idx)

    # Generate predictions
    proba_eval = np.clip(model.predict_proba(X_holdout)[:, 1], 0.0, 1.0)

    # Prevalence adjustment
    prevalence_meta = bundle.get("prevalence", {})
    train_prev = prevalence_meta.get("train_prevalence")
    if train_prev is None or not np.isfinite(train_prev):
        raise ValueError(
            "Model bundle missing valid 'prevalence.train_prevalence'. "
            "Bundle must be from modern training pipeline (post-F1 fix). "
            "Please retrain the model."
        )

    target_prev = (
        target_prevalence
        if target_prevalence is not None
        else prevalence_meta.get("test_prevalence")
    )
    if target_prev is None or not np.isfinite(target_prev):
        raise ValueError(
            "Model bundle missing valid 'prevalence.test_prevalence'. "
            "Bundle must be from modern training pipeline (post-F1 fix). "
            "Please retrain the model."
        )
    train_prev = float(train_prev)
    target_prev = float(np.clip(target_prev, EPSILON_BOUNDS, 1.0 - EPSILON_BOUNDS))

    proba_adjusted = adjust_probabilities_for_prevalence(proba_eval, train_prev, target_prev)

    # Parse clinical thresholds
    clinical_points_src = clinical_threshold_points or bundle.get("args", {}).get(
        "clinical_threshold_points", ""
    )
    clinical_points = sorted(
        {float(t.strip()) for t in (clinical_points_src or "").split(",") if t.strip()}
    )

    # Compute metrics
    metrics = compute_holdout_metrics(
        y_holdout,
        proba_eval,
        bundle,
        scenario_final,
        clinical_points,
    )

    # Save metrics
    pd.DataFrame([metrics]).to_csv(Path(outdir) / "holdout_metrics.csv", index=False)

    # Top-risk capture
    top_fracs = sorted({float(t.strip()) for t in (toprisk_fracs or "").split(",") if t.strip()})
    if top_fracs:
        top_risk_df = compute_top_risk_capture(y_holdout, proba_eval, top_fracs)
        top_risk_df.to_csv(Path(outdir) / "holdout_toprisk_capture.csv", index=False)

    # Save predictions if requested
    if save_preds:
        save_holdout_predictions(
            outdir,
            holdout_idx,
            df_holdout,
            y_holdout,
            proba_eval,
            proba_adjusted,
        )

    # Decision curve analysis
    if compute_dca:
        min_thr = (
            dca_threshold_min
            if dca_threshold_min is not None
            else bundle.get("args", {}).get("dca_threshold_min", 0.001)
        )
        max_thr = (
            dca_threshold_max
            if dca_threshold_max is not None
            else bundle.get("args", {}).get("dca_threshold_max", 0.10)
        )
        step_thr = (
            dca_threshold_step
            if dca_threshold_step is not None
            else bundle.get("args", {}).get("dca_threshold_step", 0.001)
        )

        dca_thresholds = generate_dca_thresholds(min_thr, max_thr, step_thr)
        report_points = parse_dca_report_points(dca_report_points) or parse_dca_report_points(
            bundle.get("args", {}).get("dca_report_points", "")
        )

        dca_dir = str(Path(outdir) / "diagnostics")
        dca_probs = proba_adjusted if dca_use_target_prevalence else proba_eval
        dca_prev = target_prev if dca_use_target_prevalence else None

        summary = save_dca_results(
            y_holdout,
            dca_probs,
            out_dir=dca_dir,
            prefix="holdout__",
            thresholds=dca_thresholds,
            report_points=report_points,
            prevalence_adjustment=dca_prev,
        )

        with open(Path(dca_dir) / "holdout_dca_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return metrics
