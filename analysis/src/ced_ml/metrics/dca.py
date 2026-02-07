"""
Decision Curve Analysis (DCA) for clinical utility assessment.

Implements net benefit calculations and DCA computations for evaluating
prediction models against treat-all and treat-none strategies across
a range of threshold probabilities.

Reference:
    Vickers AJ, Elkin EB (2006). Decision curve analysis: a novel method
    for evaluating prediction models. Med Decis Making, 26(6):565-574.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Core DCA Computations
# =============================================================================


def net_benefit(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute net benefit at a single threshold.

    Net Benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))

    Args:
        y_true: Binary labels (0/1)
        y_pred_prob: Predicted probabilities
        threshold: Classification threshold (0 < t < 1)

    Returns:
        Net benefit value (can be negative)
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred_prob).astype(float)
    t = float(threshold)

    if t <= 0.0 or t >= 1.0 or len(y) == 0:
        return np.nan

    # Classify and count
    tp = int(((p >= t) & (y == 1)).sum())
    fp = int(((p >= t) & (y == 0)).sum())
    n = len(y)

    # Net benefit calculation
    odds = t / (1.0 - t)
    return (tp / n) - (fp / n) * odds


def _validate_prevalence(prevalence: float, param_name: str = "prevalence") -> None:
    """Validate that prevalence is in [0, 1] range."""
    if prevalence < 0.0 or prevalence > 1.0:
        raise ValueError(f"{param_name} must be in [0, 1] range, got {prevalence}")


def net_benefit_treat_all(
    prevalence: float,
    threshold: float,
) -> float:
    """
    Compute net benefit of "treat all" strategy.

    NB_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))

    Args:
        prevalence: Observed prevalence (proportion of positive cases)
        threshold: Classification threshold (0 < t < 1)

    Returns:
        Net benefit of treating all patients
    """
    _validate_prevalence(prevalence)

    if threshold <= 0.0 or threshold >= 1.0:
        return np.nan

    odds = threshold / (1.0 - threshold)
    return prevalence - (1.0 - prevalence) * odds


def decision_curve_analysis(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    prevalence_adjustment: float | None = None,
    prevalence: float | None = None,
    min_threshold: float = 0.001,
    max_threshold: float = 0.10,
    threshold_step: float = 0.001,
) -> pd.DataFrame:
    """
    Compute Decision Curve Analysis across threshold range.

    Evaluates net benefit of using a prediction model vs. treating all
    or treating none, across a range of threshold probabilities.

    Args:
        y_true: True binary labels (0/1)
        y_pred_prob: Predicted probabilities
        thresholds: Array of threshold probabilities. If None and prevalence
            is provided, auto-configures range based on prevalence. Otherwise
            uses min_threshold to max_threshold.
        prevalence_adjustment: Optional prevalence for calibration adjustment (affects treat-all NB)
        prevalence: Optional prevalence for auto-threshold range (computes min/max from prevalence)
        min_threshold: Minimum threshold for auto-generated range (default: 0.001)
        max_threshold: Maximum threshold for auto-generated range (default: 0.10)
        threshold_step: Step size for auto-generated range (default: 0.001)

    Returns:
        DataFrame with DCA metrics
    """
    # Validate prevalence parameters at function entry (fail fast)
    if prevalence_adjustment is not None:
        _validate_prevalence(prevalence_adjustment, "prevalence_adjustment")
    if prevalence is not None:
        _validate_prevalence(prevalence, "prevalence")

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred_prob).astype(float)
    n = len(y)

    if n == 0:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = generate_dca_thresholds(
            min_thr=min_threshold,
            max_thr=max_threshold,
            step=threshold_step,
            prevalence=prevalence,
        )

    observed_prevalence = np.mean(y)
    if prevalence_adjustment is not None:
        actual_prevalence = float(np.mean(y))
        ratio = prevalence_adjustment / actual_prevalence if actual_prevalence > 0 else np.inf
        if ratio > 2.0 or ratio < 0.5:
            logger.warning(
                "prevalence_adjustment (%.4f) deviates significantly from observed prevalence (%.4f). "
                "Ratio: %.2fx. This may indicate miscalibration or inappropriate adjustment.",
                prevalence_adjustment,
                actual_prevalence,
                ratio,
            )
        observed_prevalence = float(prevalence_adjustment)

    results = []
    for t in thresholds:
        if t <= 0 or t >= 1:
            continue

        # Classify based on threshold
        y_pred_binary = (p >= t).astype(int)

        # Calculate confusion matrix
        tp = np.sum((y_pred_binary == 1) & (y == 1))
        fp = np.sum((y_pred_binary == 1) & (y == 0))
        tn = np.sum((y_pred_binary == 0) & (y == 0))
        fn = np.sum((y_pred_binary == 0) & (y == 1))

        # Net benefit calculations
        odds = t / (1 - t)
        nb_model = (tp / n) - (fp / n) * odds
        nb_all = observed_prevalence - (1 - observed_prevalence) * odds
        nb_none = 0.0

        # Relative utility
        if nb_all > 0:
            relative_utility = nb_model / nb_all
        else:
            relative_utility = np.nan

        results.append(
            {
                "threshold": t,
                "threshold_pct": t * 100,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": nb_none,
                "relative_utility": relative_utility,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "n_treat": int(tp + fp),
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            }
        )

    return pd.DataFrame(results)


def threshold_dca_zero_crossing(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    prevalence_adjustment: float | None = None,
    prevalence: float | None = None,
    min_threshold: float = 0.001,
    max_threshold: float = 0.10,
    threshold_step: float = 0.001,
) -> float | None:
    """
    Find the threshold where model net benefit crosses zero.

    This threshold represents the point where the model transitions from
    providing clinical utility (positive net benefit) to no utility or harm.
    Useful for visualizing the practical decision threshold range.

    Args:
        y_true: True binary labels (0/1)
        y_pred_prob: Predicted probabilities
        thresholds: Array of threshold probabilities. If None and prevalence
            is provided, auto-configures range based on prevalence. Otherwise
            uses min_threshold to max_threshold.
        prevalence_adjustment: Optional prevalence for calibration adjustment
        prevalence: Optional prevalence for auto-threshold range
        min_threshold: Minimum threshold for auto-generated range (default: 0.001)
        max_threshold: Maximum threshold for auto-generated range (default: 0.10)
        threshold_step: Step size for auto-generated range (default: 0.001)

    Returns:
        Threshold where net benefit crosses zero, or None if no crossing found
    """
    # Validate prevalence parameters at function entry (fail fast)
    if prevalence_adjustment is not None:
        _validate_prevalence(prevalence_adjustment, "prevalence_adjustment")
    if prevalence is not None:
        _validate_prevalence(prevalence, "prevalence")

    if thresholds is None:
        thresholds = generate_dca_thresholds(
            min_thr=min_threshold,
            max_thr=max_threshold,
            step=threshold_step,
            prevalence=prevalence,
        )

    dca_df = decision_curve_analysis(
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        thresholds=thresholds,
        prevalence_adjustment=prevalence_adjustment,
    )

    if dca_df.empty:
        return None

    nb_model = dca_df["net_benefit_model"].values
    thr = dca_df["threshold"].values

    # Find last positive net benefit point
    positive_mask = nb_model > 0
    if not np.any(positive_mask):
        return None

    # Get the last threshold where net benefit is positive
    last_positive_idx = np.where(positive_mask)[0][-1]

    # If at the end of array, return that threshold
    if last_positive_idx >= len(thr) - 1:
        return thr[last_positive_idx]

    # Otherwise, interpolate between last positive and first non-positive
    next_idx = last_positive_idx + 1
    t1, t2 = thr[last_positive_idx], thr[next_idx]
    nb1, nb2 = nb_model[last_positive_idx], nb_model[next_idx]

    # Linear interpolation to find zero crossing
    if nb1 == nb2:
        return t1
    zero_crossing = t1 + (t2 - t1) * (-nb1 / (nb2 - nb1))

    logger.warning(
        "DCA zero-crossing computed via linear interpolation on nonlinear curve. "
        "Crossing at %.6f (bracket: [%.6f, %.6f], NB: [%.6f, %.6f]). "
        "Approximation may have ~10-50%% relative error at low thresholds.",
        zero_crossing,
        t1,
        t2,
        nb1,
        nb2,
    )

    return float(zero_crossing)


def decision_curve_table(
    scenario: str,
    y_true: np.ndarray,
    pred_dict: dict[str, np.ndarray],
    max_pt: float = 0.20,
    step: float = 0.005,
) -> pd.DataFrame:
    """
    Generate cross-model DCA comparison table.

    Computes net benefit for multiple models at each threshold, formatted
    for visualization and comparison. Used in postprocessing to compare
    models side-by-side.

    Args:
        scenario: Scenario name (e.g., "IncidentPlusPrevalent")
        y_true: True binary labels
        pred_dict: Dict mapping model names to predicted probabilities
        max_pt: Maximum threshold for comparison (default: 0.20)
        step: Threshold step size (default: 0.005)

    Returns:
        DataFrame with columns: scenario, threshold, model, net_benefit
        Models include: "treat_none", "treat_all", and all models in pred_dict
    """
    y = np.asarray(y_true).astype(int)
    prev = float(np.mean(y)) if len(y) else np.nan
    thresholds = np.arange(step, max_pt + 1e-12, step)

    rows = []
    for pt in thresholds:
        # Treat none strategy (net benefit = 0)
        rows.append(
            {
                "scenario": scenario,
                "threshold": float(pt),
                "model": "treat_none",
                "net_benefit": 0.0,
            }
        )

        # Treat all strategy
        if np.isfinite(prev):
            nb_all = net_benefit_treat_all(prev, pt)
        else:
            nb_all = np.nan
        rows.append(
            {
                "scenario": scenario,
                "threshold": float(pt),
                "model": "treat_all",
                "net_benefit": float(nb_all),
            }
        )

        # Each model
        for model_name, p in pred_dict.items():
            nb = net_benefit(y, p, pt)
            rows.append(
                {
                    "scenario": scenario,
                    "threshold": float(pt),
                    "model": model_name,
                    "net_benefit": float(nb),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# DCA Summary and Reporting
# =============================================================================


def compute_dca_summary(
    dca_df: pd.DataFrame,
    report_points: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute summary statistics from DCA results.

    Identifies key clinical utility metrics:
    - Range where model beats treat-all/treat-none
    - Integrated net benefit (area under curve)
    - Net benefit at key clinical thresholds

    Args:
        dca_df: DataFrame from decision_curve_analysis()
        report_points: Key thresholds for reporting (default: [0.005, 0.01, 0.02, 0.05])

    Returns:
        Dictionary with DCA summary metrics:
            - dca_computed: Whether DCA was successfully computed
            - n_thresholds: Number of thresholds evaluated
            - threshold_range: Min-max threshold range
            - model_beats_all_from/to: Range where model > treat-all
            - model_beats_none_from/to: Range where model > treat-none
            - integrated_nb_model/all/improvement: Area under curves
            - nb_model_at_X, nb_all_at_X: Net benefit at key thresholds
    """
    if dca_df.empty:
        return {"dca_computed": False}

    summary = {
        "dca_computed": True,
        "n_thresholds": len(dca_df),
        "threshold_range": f"{dca_df['threshold'].min():.3f}-{dca_df['threshold'].max():.3f}",
    }

    # Find where model beats "treat all"
    model_beats_all = dca_df[dca_df["net_benefit_model"] > dca_df["net_benefit_all"]]
    if len(model_beats_all) > 0:
        summary["model_beats_all_from"] = float(model_beats_all["threshold"].min())
        summary["model_beats_all_to"] = float(model_beats_all["threshold"].max())
        summary["model_beats_all_range"] = (
            f"{summary['model_beats_all_from']:.3f}-{summary['model_beats_all_to']:.3f}"
        )
    else:
        summary["model_beats_all_from"] = np.nan
        summary["model_beats_all_to"] = np.nan
        summary["model_beats_all_range"] = "Never"

    # Find where model beats "treat none" (positive net benefit)
    model_beats_none = dca_df[dca_df["net_benefit_model"] > 0]
    if len(model_beats_none) > 0:
        summary["model_beats_none_from"] = float(model_beats_none["threshold"].min())
        summary["model_beats_none_to"] = float(model_beats_none["threshold"].max())
    else:
        summary["model_beats_none_from"] = np.nan
        summary["model_beats_none_to"] = np.nan

    # Integrated net benefit (area under curve)
    thresholds = dca_df["threshold"].values
    nb_model = dca_df["net_benefit_model"].values
    nb_all = dca_df["net_benefit_all"].values

    if len(thresholds) > 1:
        summary["integrated_nb_model"] = float(np.trapezoid(nb_model, thresholds))
        summary["integrated_nb_all"] = float(np.trapezoid(nb_all, thresholds))

        # Net benefit improvement over treat-all
        nb_improvement = nb_model - nb_all
        summary["integrated_nb_improvement"] = float(np.trapezoid(nb_improvement, thresholds))

    # Net benefit at key clinical thresholds
    report_points = report_points or [0.005, 0.01, 0.02, 0.05]
    for key_t in report_points:
        row = dca_df[np.isclose(dca_df["threshold"], key_t, atol=0.001)]
        if len(row) > 0:
            r = row.iloc[0]
            key_str = f"{key_t:.1%}".replace(".", "p")
            summary[f"nb_model_at_{key_str}"] = float(r["net_benefit_model"])
            summary[f"nb_all_at_{key_str}"] = float(r["net_benefit_all"])

    return summary


def find_dca_zero_crossing(dca_curve_path: str) -> float | None:
    """
    Find DCA zero-crossing threshold from saved DCA curve.

    Identifies where model net benefit crosses zero (net_benefit_model = 0).
    Uses linear interpolation for accurate crossing detection.

    Args:
        dca_curve_path: Path to DCA curve CSV with columns:
            threshold, net_benefit_model, net_benefit_all, net_benefit_none

    Returns:
        Threshold where zero-crossing occurs, or None if:
        - File doesn't exist
        - File can't be parsed
        - No crossing found and fallback fails

    Example:
        >>> threshold = find_dca_zero_crossing("dca_curve.csv")
        >>> if threshold:
        ...     print(f"Optimal DCA threshold: {threshold:.4f}")
    """
    curve_path = Path(dca_curve_path)
    if not curve_path.exists():
        logger.debug(f"DCA curve not found: {dca_curve_path}")
        return None

    try:
        dca_df = pd.read_csv(curve_path)
        if "net_benefit_model" not in dca_df.columns:
            logger.warning(f"DCA curve missing net_benefit_model column: {dca_curve_path}")
            return None

        nb = dca_df["net_benefit_model"].values
        thr = dca_df["threshold"].values

        if len(nb) == 0 or len(thr) == 0:
            return None

        # Find sign changes (zero crossings)
        sign_changes = np.where(np.diff(np.sign(nb)))[0]

        if len(sign_changes) > 0:
            # Use first crossing (where model benefit becomes positive)
            idx = sign_changes[0]
            if idx + 1 < len(thr):
                # Linear interpolation
                thr_lo, thr_hi = thr[idx], thr[idx + 1]
                nb_lo, nb_hi = nb[idx], nb[idx + 1]

                if abs(nb_hi - nb_lo) > 1e-10:
                    crossing = thr_lo + (0 - nb_lo) * (thr_hi - thr_lo) / (nb_hi - nb_lo)
                    return float(crossing)
        else:
            # No crossing - find closest to zero as fallback
            abs_nb = np.abs(nb)
            closest_idx = np.nanargmin(abs_nb)
            closest_nb = nb[closest_idx]

            # Only use if reasonably close to zero
            if abs(closest_nb) < 0.05:
                return float(thr[closest_idx])

    except Exception as e:
        logger.warning(f"Failed to parse DCA curve {dca_curve_path}: {e}")

    return None


# =============================================================================
# DCA Persistence
# =============================================================================


def save_dca_results(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    out_dir: str,
    prefix: str = "",
    thresholds: np.ndarray | None = None,
    report_points: list[float] | None = None,
    prevalence_adjustment: float | None = None,
    prevalence: float | None = None,
) -> dict[str, Any]:
    """
    Compute and save DCA results to files.

    Writes DCA curve CSV, summary JSON, and returns summary dictionary
    for immediate use.

    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        out_dir: Output directory
        prefix: Filename prefix (e.g., "RF__")
        thresholds: Threshold array. If None and prevalence is provided,
            auto-configures range based on prevalence. Otherwise defaults
            to 0.0005 to 0.20, step 0.001.
        report_points: Key thresholds for summary (default: [0.005, 0.01, 0.02, 0.05])
        prevalence_adjustment: Optional prevalence for calibration, must be in [0, 1]
        prevalence: Optional prevalence for auto-configuring threshold range,
            must be in [0, 1].
            When provided and thresholds is None, computes:
            - min_thr = max(0.0001, prevalence / 10)
            - max_thr = min(0.5, prevalence * 10)

    Returns:
        DCA summary dictionary with additional keys:
            - dca_csv_path: Path to saved curve CSV
            - dca_json_path: Path to saved summary JSON
            - error: Error message if computation failed

    Raises:
        ValueError: If prevalence_adjustment or prevalence is outside [0, 1] range

    Example:
        >>> summary = save_dca_results(
        ...     y_test, y_pred_prob,
        ...     out_dir="results/diagnostics/dca",
        ...     prefix="RF__",
        ...     thresholds=np.linspace(0.0005, 1.0, 1000),
        ... )
        >>> print(summary["model_beats_all_range"])
        '0.003-0.156'
    """
    # Validate prevalence parameters at function entry (fail fast)
    if prevalence_adjustment is not None:
        _validate_prevalence(prevalence_adjustment, "prevalence_adjustment")
    if prevalence is not None:
        _validate_prevalence(prevalence, "prevalence")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Compute DCA
    dca_df = decision_curve_analysis(
        y_true,
        y_pred_prob,
        thresholds=thresholds,
        prevalence_adjustment=prevalence_adjustment,
        prevalence=prevalence,
    )

    if dca_df.empty:
        return {"dca_computed": False, "error": "Empty DCA results"}

    # Save curve CSV
    csv_path = out_path / f"{prefix}dca_curve.csv"
    dca_df.to_csv(csv_path, index=False)

    # Compute summary
    summary = compute_dca_summary(dca_df, report_points=report_points)
    if prevalence_adjustment is not None:
        summary["prevalence_adjustment"] = float(prevalence_adjustment)
    if prevalence is not None:
        summary["auto_range_prevalence"] = float(prevalence)
    summary["dca_csv_path"] = str(csv_path)

    # Save summary JSON
    json_path = out_path / f"{prefix}dca_summary.json"
    with open(json_path, "w") as f:
        # Convert numpy types for JSON serialization
        json_summary = {}
        for k, v in summary.items():
            if isinstance(v, np.integer | np.floating):
                json_summary[k] = float(v) if np.isfinite(v) else None
            else:
                json_summary[k] = v
        json.dump(json_summary, f, indent=2)

    summary["dca_json_path"] = str(json_path)

    logger.info(f"Saved DCA results to {csv_path}")
    return summary


# =============================================================================
# Utility Functions
# =============================================================================


def generate_dca_thresholds(
    min_thr: float = 0.001,
    max_thr: float = 0.10,
    step: float = 0.001,
    prevalence: float | None = None,
) -> np.ndarray:
    """
    Generate threshold array for DCA analysis.

    Args:
        min_thr: Minimum threshold (clamped to 0.0001, default: 0.001)
        max_thr: Maximum threshold (clamped to 0.999, default: 0.10)
        step: Step size between thresholds (minimum 0.0001, default: 0.001)
        prevalence: Optional disease prevalence for auto-configuring threshold range,
            must be in [0, 1].
            When provided, min_thr and max_thr are computed as:
            - min_thr = max(0.0001, prevalence / 10)
            - max_thr = min(0.5, prevalence * 10)
            This ensures the threshold range captures clinically relevant
            decision points for low-prevalence scenarios.

    Returns:
        Array of threshold values for DCA computation

    Raises:
        ValueError: If prevalence is outside [0, 1] range

    Note:
        Default range 0.1% to 10% covers clinically relevant thresholds
        for most prediction tasks while maintaining computational efficiency.
        For low-prevalence conditions (e.g., 0.34%), the auto-range feature
        ensures thresholds are not missed.
    """
    # Validate and auto-configure range based on prevalence if provided
    if prevalence is not None:
        _validate_prevalence(prevalence, "prevalence")
        prevalence = float(prevalence)
        if prevalence > 0 and prevalence < 1:
            min_thr = max(0.0001, prevalence / 10)
            max_thr = min(0.5, prevalence * 10)
            logger.debug(
                "Auto-configured DCA thresholds from prevalence %.4f: "
                "range [%.4f, %.4f]. For low-prevalence scenarios, verify this range captures "
                "clinically relevant decision points.",
                prevalence,
                min_thr,
                max_thr,
            )

    min_thr = max(1e-4, float(min_thr))
    max_thr = min(0.999, float(max_thr))
    step = max(1e-4, float(step))

    if min_thr >= max_thr:
        return np.array([min_thr, max_thr])

    n = int(np.floor((max_thr - min_thr) / step)) + 1
    return np.linspace(min_thr, max_thr, n)


def parse_dca_report_points(s: str) -> list[float]:
    """
    Parse comma-separated DCA report thresholds.

    Args:
        s: Comma-separated string (e.g., "0.005,0.01,0.02,0.05")

    Returns:
        List of float thresholds between 0 and 1
    """
    pts = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
            if 0.0 < v < 1.0:
                pts.append(v)
        except ValueError:
            logger.warning(f"Invalid DCA report point: {tok}")
            continue
    return pts
