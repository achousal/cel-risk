"""
Calibration wrappers and utilities.

This module provides:
- Calibration metrics (intercept, slope, ECE)
- Advanced calibration assessment metrics (ICI, E50/E90, Spiegelhalter z-test,
  Adaptive ECE, Brier score decomposition)
- sklearn CalibratedClassifierCV wrapper utilities
- OOF (out-of-fold) calibration

Note: Prevalence adjustment functions are in prevalence.py

Supported OOF calibration methods
----------------------------------
isotonic          : Isotonic regression (non-parametric, high variance).
sigmoid           : Alias for logistic_full (Platt scaling, two parameters).
logistic_full     : Fits logit(Y=1) = a + b*logit(p); two-parameter Platt scaling.
logistic_intercept: Fits logit(Y=1) = a + logit(p); intercept-only, lowest variance.
beta              : Fits logit(q) = a*log(p) + b*log(1-p) + c; three parameters.
                    Handles asymmetric miscalibration (Kull et al. 2017).
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.special import expit as _scipy_expit
from scipy.stats import norm as _scipy_norm
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ced_ml.utils.math_utils import logit

from ..data.schema import ModelName

# Suppress convergence warnings to prevent heavy .err files
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)

__all__ = [
    # Original calibration metrics
    "CalibrationMetrics",
    "calibration_intercept_slope",
    "calib_intercept_metric",
    "calib_slope_metric",
    "expected_calibration_error",
    # Advanced calibration assessment metrics
    "CalibrationQuantiles",
    "integrated_calibration_index",
    "calibration_error_quantiles",
    "ici_metric",
    "SpiegelhalterResult",
    "spiegelhalter_z_test",
    "spiegelhalter_z_metric",
    "adaptive_expected_calibration_error",
    "adaptive_ece_metric",
    "BrierDecomposition",
    "brier_score_decomposition",
    "brier_reliability_metric",
    "brier_resolution_metric",
    # Calibration wrapper utilities
    "get_calibrated_estimator_param_name",
    "get_calibrated_cv_param_name",
    "maybe_calibrate_estimator",
    "OOFCalibrator",
    "fit_oof_calibrator",
    "apply_oof_calibrator",
    "OOFCalibratedModel",
]


@dataclass
class CalibrationMetrics:
    """Calibration intercept and slope metrics.

    These indicate how well-calibrated probabilities are:
    - Intercept ~0 indicates probabilities match observed proportions
    - Slope ~1 indicates correct ordering/ranking

    Reference:
        Van Calster et al. (2016). Calibration of risk prediction models.
        Medical Decision Making.

    Attributes:
        intercept: Calibration intercept (ideal: 0)
        slope: Calibration slope (ideal: 1)
    """

    intercept: float
    slope: float


def calibration_intercept_slope(y_true: np.ndarray, p: np.ndarray) -> CalibrationMetrics:
    """
    Compute calibration intercept and slope using logistic regression on logit scale.

    These indicate how well-calibrated probabilities are:
    - Intercept ~0 indicates probabilities match observed proportions
    - Slope ~1 indicates correct ordering/ranking

    Reference:
        Van Calster et al. (2016). Calibration of risk prediction models.
        Medical Decision Making.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities

    Returns:
        CalibrationMetrics dataclass with intercept and slope
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    # Filter valid values
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]

    # Compute log-odds using stable logit function
    log_odds = logit(p)

    # Need both classes for calibration
    if len(np.unique(y)) < 2:
        return CalibrationMetrics(intercept=np.nan, slope=np.nan)

    # Fit logistic regression on log-odds (no regularization)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y)
    return CalibrationMetrics(
        intercept=float(lr.intercept_[0]),
        slope=float(lr.coef_[0][0]),
    )


def calib_intercept_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Compute calibration intercept metric for bootstrap CIs."""
    cal_metrics = calibration_intercept_slope(y, p)
    return float(cal_metrics.intercept)


def calib_slope_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Compute calibration slope metric for bootstrap CIs."""
    cal_metrics = calibration_intercept_slope(y, p)
    return float(cal_metrics.slope)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and observed outcomes
    across probability bins.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping predictions

    Returns:
        ECE value (lower is better calibrated)
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    # Filter valid (remove NaN/inf before converting to int)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int)
    p = p[mask]

    if len(y) == 0:
        return np.nan

    # Create bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (p == 1.0)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_pred = np.mean(p[in_bin])
            avg_true = np.mean(y[in_bin])
            ece += np.abs(avg_pred - avg_true) * prop_in_bin

    return float(ece)


# ---------------------------------------------------------------------------
# LOESS-based calibration assessment helpers (ICI / E50 / E90)
# ---------------------------------------------------------------------------

# Minimum number of unique prediction values required to attempt smoothing.
_MIN_UNIQUE_PREDS_FOR_SMOOTHING = 10


def _loess_calibration_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray | None:
    """
    Compute pointwise |smoothed_calibration(p) - p| using a cubic spline smoother.

    The spline is fitted on (p, y) pairs and evaluated at each p, yielding an
    estimate of E[Y | P=p] -- the calibration curve.  The absolute difference
    between the curve value and p is the pointwise calibration error.

    Intended only as a shared internal helper for ICI and calibration quantile
    functions.  Returns None when smoothing cannot be performed (e.g. too few
    unique prediction values, constant predictions).

    Args:
        y_true: True binary labels, already filtered for finite values, length n.
        y_pred: Predicted probabilities, already filtered for finite values, length n.
            Values must lie in [0, 1].

    Returns:
        Array of pointwise absolute calibration errors (length n), or None if
        smoothing is not possible.
    """
    n_unique = len(np.unique(y_pred))
    if n_unique < _MIN_UNIQUE_PREDS_FOR_SMOOTHING:
        logger.debug(
            "_loess_calibration_errors: only %d unique prediction values "
            "(need >= %d); skipping spline smoothing.",
            n_unique,
            _MIN_UNIQUE_PREDS_FOR_SMOOTHING,
        )
        return None

    # Sort by predicted probability for stable spline fitting.
    sort_idx = np.argsort(y_pred)
    p_sorted = y_pred[sort_idx]
    y_sorted = y_true[sort_idx]

    try:
        # k=3 cubic spline; s controls smoothing (None = interpolating, 0 = interpolating,
        # default None lets scipy choose; we pass s explicitly as n to get a smooth fit).
        # Using s = len(y) as a standard heuristic for a smoothed (not interpolating) fit.
        spline = UnivariateSpline(p_sorted, y_sorted, k=3, s=len(y_sorted), ext=3)
        calibration_curve = np.clip(spline(y_pred), 0.0, 1.0)
    except Exception as exc:
        logger.debug("_loess_calibration_errors: spline fitting failed: %s", exc)
        return None

    return np.abs(calibration_curve - y_pred)


def integrated_calibration_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Integrated Calibration Index (ICI).

    ICI is the weighted mean of pointwise absolute differences between a
    smooth calibration curve (estimated via cubic spline) and the identity
    line.  Lower values indicate better calibration.

    Reference:
        Austin & Steyerberg (2019). The Integrated Calibration Index (ICI) and
        related metrics for quantifying the calibration of logistic regression
        models. Statistics in Medicine.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,).
        y_pred: Predicted probabilities in [0, 1], shape (n_samples,).

    Returns:
        ICI value in [0, 1] (lower is better), or NaN if calibration curve
        cannot be estimated (e.g. too few samples or near-constant predictions).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> p = rng.uniform(0.05, 0.5, n)
        >>> y = rng.binomial(1, p)
        >>> ici = integrated_calibration_index(y, p)
        >>> 0.0 <= ici <= 1.0
        True
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int).astype(float)
    p = p[mask]

    if len(y) == 0:
        return np.nan

    errors = _loess_calibration_errors(y, p)
    if errors is None:
        return np.nan

    return float(np.mean(errors))


@dataclass
class CalibrationQuantiles:
    """
    Pointwise calibration error quantiles derived from a smooth calibration curve.

    Attributes:
        e50: Median (50th percentile) of pointwise absolute calibration errors.
        e90: 90th percentile of pointwise absolute calibration errors.
        ici: Mean of pointwise absolute calibration errors (ICI).
    """

    e50: float
    e90: float
    ici: float


def calibration_error_quantiles(y_true: np.ndarray, y_pred: np.ndarray) -> CalibrationQuantiles:
    """
    Compute E50, E90, and ICI from the pointwise calibration error distribution.

    All three metrics are derived from the same smooth calibration curve
    (cubic spline), computed in a single pass.

    Reference:
        Austin & Steyerberg (2019). The Integrated Calibration Index (ICI) and
        related metrics for quantifying the calibration of logistic regression
        models. Statistics in Medicine.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,).
        y_pred: Predicted probabilities in [0, 1], shape (n_samples,).

    Returns:
        CalibrationQuantiles dataclass with e50, e90, and ici fields.
        All fields are NaN when the calibration curve cannot be estimated.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> p = rng.uniform(0.05, 0.5, n)
        >>> y = rng.binomial(1, p)
        >>> q = calibration_error_quantiles(y, p)
        >>> q.e50 <= q.e90  # E50 <= E90 by construction
        True
    """
    _nan = CalibrationQuantiles(e50=np.nan, e90=np.nan, ici=np.nan)

    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int).astype(float)
    p = p[mask]

    if len(y) == 0:
        return _nan

    errors = _loess_calibration_errors(y, p)
    if errors is None:
        return _nan

    return CalibrationQuantiles(
        e50=float(np.percentile(errors, 50)),
        e90=float(np.percentile(errors, 90)),
        ici=float(np.mean(errors)),
    )


def ici_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return ICI (mean pointwise calibration error) as a scalar.

    Alias for :func:`integrated_calibration_index` with the same signature
    ``(y_true, y_pred) -> float``, compatible with
    :func:`~ced_ml.metrics.bootstrap.stratified_bootstrap_ci`.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted probabilities.

    Returns:
        ICI value in [0, 1] or NaN.
    """
    return integrated_calibration_index(y_true, y_pred)


# ---------------------------------------------------------------------------
# Spiegelhalter's z-test for calibration
# ---------------------------------------------------------------------------


@dataclass
class SpiegelhalterResult:
    """
    Result of Spiegelhalter's z-test for calibration.

    A large absolute z_statistic or small p_value indicates significant
    miscalibration.  The test is two-sided.

    Reference:
        Spiegelhalter (1986). Probabilistic prediction in patient management
        and clinical trials. Statistics in Medicine.

    Attributes:
        z_statistic: Test statistic (standard normal under H0 of calibration).
        p_value: Two-sided p-value under H0.
        is_calibrated: True when p_value > 0.05 (fail to reject H0).
    """

    z_statistic: float
    p_value: float
    is_calibrated: bool


def spiegelhalter_z_test(y_true: np.ndarray, y_pred: np.ndarray) -> SpiegelhalterResult:
    """
    Spiegelhalter's z-test for assessing probability calibration.

    Under the null hypothesis of perfect calibration the test statistic

        Z = sum((y_i - p_i) * (1 - 2*p_i))
            / sqrt(sum(p_i * (1-p_i) * (1 - 2*p_i)**2))

    is asymptotically standard normal.

    Reference:
        Spiegelhalter (1986). Probabilistic prediction in patient management
        and clinical trials. Statistics in Medicine.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,).
        y_pred: Predicted probabilities in [0, 1], shape (n_samples,).

    Returns:
        SpiegelhalterResult with z_statistic, p_value, and is_calibrated flag.
        All numeric fields are NaN when the test cannot be computed (e.g.
        constant predictions, empty arrays, or single class).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> n = 500
        >>> p = rng.uniform(0.05, 0.5, n)
        >>> y = rng.binomial(1, p)
        >>> result = spiegelhalter_z_test(y, p)
        >>> abs(result.z_statistic) < 3  # well-calibrated; should not be extreme
        True
    """
    _nan = SpiegelhalterResult(z_statistic=np.nan, p_value=np.nan, is_calibrated=False)

    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int).astype(float)
    p = p[mask]

    if len(y) == 0:
        return _nan

    if len(np.unique(y)) < 2:
        logger.debug("spiegelhalter_z_test: only one class present; returning NaN.")
        return _nan

    weights = 1.0 - 2.0 * p
    numerator = float(np.sum((y - p) * weights))
    variance = float(np.sum(p * (1.0 - p) * weights**2))

    if variance <= 0.0:
        logger.debug(
            "spiegelhalter_z_test: variance of test statistic is non-positive "
            "(predictions may be near-constant); returning NaN."
        )
        return _nan

    z = numerator / np.sqrt(variance)
    p_value = float(2.0 * _scipy_norm.sf(abs(z)))

    return SpiegelhalterResult(
        z_statistic=float(z),
        p_value=p_value,
        is_calibrated=p_value > 0.05,
    )


def spiegelhalter_z_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spiegelhalter z-statistic as a scalar for bootstrap CI computation.

    Wraps :func:`spiegelhalter_z_test` to expose the z_statistic with the
    standard ``(y_true, y_pred) -> float`` signature required by
    :func:`~ced_ml.metrics.bootstrap.stratified_bootstrap_ci`.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Spiegelhalter z-statistic, or NaN if the test cannot be computed.
    """
    return spiegelhalter_z_test(y_true, y_pred).z_statistic


# ---------------------------------------------------------------------------
# Adaptive Expected Calibration Error
# ---------------------------------------------------------------------------


def adaptive_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    min_events_per_bin: int = 5,
) -> float:
    """
    Compute Adaptive Expected Calibration Error (Adaptive ECE).

    Uses quantile-based (equal-frequency) bins by default, with a merging
    step that absorbs bins containing fewer than ``min_events_per_bin``
    positive events into the adjacent bin.  This is more robust than uniform
    binning when the positive class is rare (0.34% prevalence in this study).

    The existing :func:`expected_calibration_error` is kept unchanged for
    backward compatibility; use this function for improved imbalance handling.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,).
        y_pred: Predicted probabilities in [0, 1], shape (n_samples,).
        n_bins: Initial number of bins before merging (default 10).
        strategy: Binning strategy.  Currently only ``"quantile"`` is
            supported (equal-frequency bins).
        min_events_per_bin: Bins with fewer positive events are merged into
            the next bin.  Default 5.

    Returns:
        Adaptive ECE value in [0, 1] (lower is better), or NaN for empty
        inputs or invalid arguments.

    Raises:
        ValueError: If ``strategy`` is not ``"quantile"``.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 1000
        >>> p = rng.uniform(0.001, 0.05, n)
        >>> y = rng.binomial(1, p)
        >>> aece = adaptive_expected_calibration_error(y, p)
        >>> 0.0 <= aece <= 1.0
        True
    """
    if strategy != "quantile":
        raise ValueError(f"strategy must be 'quantile', got '{strategy}'")

    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int)
    p = p[mask]

    if len(y) == 0:
        return np.nan

    # Build equal-frequency quantile bin edges.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(p, quantiles))

    # Assign each prediction to a bin index.
    bin_ids = np.digitize(p, bin_edges[1:-1])  # 0-indexed bins

    n_raw_bins = len(bin_edges) - 1

    # Collect per-bin statistics.
    bin_counts = np.zeros(n_raw_bins, dtype=int)
    bin_pos_counts = np.zeros(n_raw_bins, dtype=int)
    bin_sum_pred = np.zeros(n_raw_bins, dtype=float)
    bin_sum_true = np.zeros(n_raw_bins, dtype=float)

    for i in range(n_raw_bins):
        idx = bin_ids == i
        bin_counts[i] = np.sum(idx)
        bin_pos_counts[i] = int(np.sum(y[idx]))
        bin_sum_pred[i] = float(np.sum(p[idx]))
        bin_sum_true[i] = float(np.sum(y[idx]))

    # Merge bins that have too few positive events into the next bin.
    merged_counts: list[int] = []
    merged_pos: list[int] = []
    merged_sum_pred: list[float] = []
    merged_sum_true: list[float] = []

    acc_count = 0
    acc_pos = 0
    acc_sum_pred = 0.0
    acc_sum_true = 0.0

    for i in range(n_raw_bins):
        acc_count += bin_counts[i]
        acc_pos += bin_pos_counts[i]
        acc_sum_pred += bin_sum_pred[i]
        acc_sum_true += bin_sum_true[i]

        # Flush the accumulator when this bin meets the minimum-events threshold
        # OR when we reach the last bin (force flush to avoid leftover data).
        if acc_pos >= min_events_per_bin or i == n_raw_bins - 1:
            merged_counts.append(acc_count)
            merged_pos.append(acc_pos)
            merged_sum_pred.append(acc_sum_pred)
            merged_sum_true.append(acc_sum_true)
            acc_count = 0
            acc_pos = 0
            acc_sum_pred = 0.0
            acc_sum_true = 0.0

    n_total = len(y)
    ece = 0.0
    for cnt, _pos, s_pred, s_true in zip(
        merged_counts, merged_pos, merged_sum_pred, merged_sum_true, strict=True
    ):
        if cnt == 0:
            continue
        avg_pred = s_pred / cnt
        avg_true = s_true / cnt
        weight = cnt / n_total
        ece += np.abs(avg_pred - avg_true) * weight

    return float(ece)


def adaptive_ece_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Adaptive ECE as a scalar.

    Alias for :func:`adaptive_expected_calibration_error` with default
    parameters and the standard ``(y_true, y_pred) -> float`` signature,
    compatible with
    :func:`~ced_ml.metrics.bootstrap.stratified_bootstrap_ci`.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Adaptive ECE value in [0, 1] or NaN.
    """
    return adaptive_expected_calibration_error(y_true, y_pred)


# ---------------------------------------------------------------------------
# Brier Score Decomposition (Murphy 1973)
# ---------------------------------------------------------------------------


@dataclass
class BrierDecomposition:
    """
    Murphy (1973) decomposition of the Brier score.

    Brier = reliability - resolution + uncertainty

    Attributes:
        reliability: Calibration component (lower is better).  Measures how
            close average predicted probabilities are to observed frequencies
            within each bin.
        resolution: Discrimination component (higher is better).  Measures
            how much the bin-level observed frequencies deviate from the
            overall event rate.
        uncertainty: Baseline uncertainty of the outcome (= o_bar * (1-o_bar)
            where o_bar is the overall event rate).  Independent of the model.
        brier_score: Total Brier score.  Satisfies the identity
            ``reliability - resolution + uncertainty == brier_score``
            up to floating-point precision.

    Reference:
        Murphy (1973). A new vector partition of the probability score.
        Journal of Applied Meteorology.
    """

    reliability: float
    resolution: float
    uncertainty: float
    brier_score: float


def brier_score_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """
    Compute Murphy (1973) decomposition of the Brier score.

    The Brier score is partitioned as:

        Brier = reliability - resolution + uncertainty

    where
        reliability = (1/N) * sum_k n_k * (o_k - p_k_bar)**2
        resolution  = (1/N) * sum_k n_k * (o_k - o_bar)**2
        uncertainty = o_bar * (1 - o_bar)

    and k indexes uniform-width probability bins, n_k is the bin count,
    o_k is the mean observed event rate in bin k, p_k_bar is the mean
    predicted probability in bin k, and o_bar is the overall event rate.

    The ``brier_score`` field stores ``reliability - resolution + uncertainty``
    (the bin-consistent value), which satisfies the identity exactly.  This
    equals the per-sample MSE only in the limit of infinitely fine bins; the
    two values will differ by a small within-bin variance term when bins are
    coarse.

    Reference:
        Murphy (1973). A new vector partition of the probability score.
        Journal of Applied Meteorology.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,).
        y_pred: Predicted probabilities in [0, 1], shape (n_samples,).
        n_bins: Number of uniform-width probability bins (default 10).

    Returns:
        BrierDecomposition dataclass.  All fields are NaN for empty inputs.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 500)
        >>> y = rng.binomial(1, p)
        >>> decomp = brier_score_decomposition(y, p)
        >>> abs(decomp.reliability - decomp.resolution + decomp.uncertainty - decomp.brier_score) < 1e-10
        True
    """
    _nan = BrierDecomposition(
        reliability=np.nan,
        resolution=np.nan,
        uncertainty=np.nan,
        brier_score=np.nan,
    )

    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int).astype(float)
    p = p[mask]

    n = len(y)
    if n == 0:
        return _nan

    # Overall event rate.
    o_bar = float(np.mean(y))
    uncertainty = o_bar * (1.0 - o_bar)

    # Bin-level statistics using uniform-width bins.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.searchsorted(bin_edges[1:-1], p, side="right")  # 0-indexed

    reliability = 0.0
    resolution = 0.0

    for k in range(n_bins):
        idx = bin_ids == k
        n_k = np.sum(idx)
        if n_k == 0:
            continue
        o_k = float(np.mean(y[idx]))
        p_k_bar = float(np.mean(p[idx]))
        w = n_k / n
        reliability += w * (o_k - p_k_bar) ** 2
        resolution += w * (o_k - o_bar) ** 2

    # brier_score is defined as REL - RES + UNC to satisfy the identity exactly.
    # This is the bin-consistent (Murphy 1973) Brier score, which differs from
    # the per-sample MSE by the within-bin prediction variance.
    brier = reliability - resolution + uncertainty

    return BrierDecomposition(
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        brier_score=float(brier),
    )


def brier_reliability_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Brier reliability (calibration) component as a scalar.

    Wraps :func:`brier_score_decomposition` with the standard
    ``(y_true, y_pred) -> float`` signature for use with
    :func:`~ced_ml.metrics.bootstrap.stratified_bootstrap_ci`.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Reliability component of Brier decomposition (lower is better), or NaN.
    """
    return brier_score_decomposition(y_true, y_pred).reliability


def brier_resolution_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Brier resolution (discrimination) component as a scalar.

    Wraps :func:`brier_score_decomposition` with the standard
    ``(y_true, y_pred) -> float`` signature for use with
    :func:`~ced_ml.metrics.bootstrap.stratified_bootstrap_ci`.

    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted probabilities.

    Returns:
        Resolution component of Brier decomposition (higher is better), or NaN.
    """
    return brier_score_decomposition(y_true, y_pred).resolution


def get_calibrated_estimator_param_name() -> str:
    """
    Detect parameter name for base estimator in CalibratedClassifierCV.

    Different sklearn versions use different names ('estimator' vs 'base_estimator').

    Returns:
        'estimator' or 'base_estimator'
    """
    tmp = CalibratedClassifierCV(LinearSVC())
    params = tmp.get_params().keys()
    if "estimator" in params:
        return "estimator"
    if "base_estimator" in params:
        return "base_estimator"
    raise ValueError("Could not determine base estimator parameter name")


def get_calibrated_cv_param_name() -> str:
    """
    Detect CV parameter name in CalibratedClassifierCV.

    Returns:
        'cv' (standard name)
    """
    tmp = CalibratedClassifierCV(LinearSVC())
    params = tmp.get_params().keys()
    if "cv" in params:
        return "cv"
    raise ValueError("Could not determine cv parameter name")


def maybe_calibrate_estimator(
    estimator, model_name: str, calibrate: bool, method: str = "sigmoid", cv: int = 3
):
    """
    Optional calibration wrapper for LR/RF (SVM is already calibrated).

    This is applied consistently in CV and final training when enabled.

    Args:
        estimator: Base sklearn estimator
        model_name: Model name (e.g., 'RF', 'LR_EN', 'LinSVM_cal')
        calibrate: Whether to apply calibration
        method: Calibration method ('sigmoid' or 'isotonic')
        cv: Number of CV folds for calibration

    Returns:
        Calibrated or original estimator
    """
    if not calibrate:
        return estimator

    # Don't calibrate SVM (already calibrated)
    if model_name == ModelName.LinSVM_cal:
        return estimator

    # Don't double-calibrate
    if isinstance(estimator, CalibratedClassifierCV):
        return estimator

    try:
        kwargs = {"method": str(method), "cv": int(cv)}
        param_name = get_calibrated_estimator_param_name()
        kwargs[param_name] = estimator
        return CalibratedClassifierCV(**kwargs)
    except Exception as e:
        logger.warning(
            f"Failed to calibrate {model_name} with method={method}, cv={cv}: {e}. "
            "Returning original estimator without calibration."
        )
        return estimator


_VALID_OOF_METHODS = frozenset(
    {"isotonic", "sigmoid", "logistic_full", "logistic_intercept", "beta"}
)

# "sigmoid" is an alias for "logistic_full".
_SIGMOID_ALIAS = "logistic_full"


def _beta_nll(params: np.ndarray, log_p: np.ndarray, log1mp: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for beta calibration.

    Model: logit(q) = a*log(p) + b*log(1-p) + c
    Ref: Kull et al. (2017) "Beta calibration: a well-founded and easily
         implemented improvement on logistic calibration for binary classifiers."

    Args:
        params: [a, b, c] - three free parameters.
        log_p: log(p) for each sample (clipped).
        log1mp: log(1-p) for each sample (clipped).
        y: Binary labels (0/1).

    Returns:
        Negative log-likelihood (scalar).
    """
    a, b, c = params
    logit_q = a * log_p + b * log1mp + c
    # Stable log-likelihood via log-sigmoid
    # log P(y=1) = log sigmoid(logit_q); log P(y=0) = log sigmoid(-logit_q)
    log_prob_pos = -np.logaddexp(0.0, -logit_q)
    log_prob_neg = -np.logaddexp(0.0, logit_q)
    nll = -np.sum(y * log_prob_pos + (1.0 - y) * log_prob_neg)
    return float(nll)


class OOFCalibrator:
    """
    Post-hoc calibrator fitted on pooled out-of-fold predictions.

    This calibrator is fit once on genuinely held-out OOF predictions
    after the CV loop completes, avoiding the optimistic bias introduced
    by per-fold CalibratedClassifierCV.

    Supported methods
    -----------------
    isotonic          : Isotonic regression (non-parametric, high variance).
    sigmoid           : Alias for logistic_full.
    logistic_full     : Two-parameter Platt scaling: logit(Y=1) = a + b*logit(p).
    logistic_intercept: Intercept-only recalibration: logit(Y=1) = a + logit(p).
                        Lowest-variance parametric option for small calibration sets.
    beta              : Three-parameter beta calibration: logit(q) = a*log(p) + b*log(1-p) + c.
                        Handles asymmetric miscalibration (Kull et al. 2017).

    Attributes:
        method: Calibration method (normalised; "sigmoid" stored as "logistic_full").
        calibrator_: Fitted object (sklearn estimator or dict of scipy params).
        is_fitted: Whether the calibrator has been fitted.
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize OOF calibrator.

        Args:
            method: Calibration method. One of "isotonic", "sigmoid" (alias for
                    "logistic_full"), "logistic_full", "logistic_intercept", "beta".
        """
        if method not in _VALID_OOF_METHODS:
            raise ValueError(f"method must be one of {sorted(_VALID_OOF_METHODS)}, got '{method}'")
        # Normalise alias so downstream code branches on canonical names.
        self.method = _SIGMOID_ALIAS if method == "sigmoid" else method
        self.calibrator_ = None
        self.is_fitted = False

    def fit(self, oof_preds: np.ndarray, y_true: np.ndarray) -> "OOFCalibrator":
        """
        Fit the calibrator on pooled OOF predictions.

        Args:
            oof_preds: Raw (uncalibrated) OOF predictions, shape (n_samples,).
            y_true: True binary labels, shape (n_samples,).

        Returns:
            self (fitted calibrator).

        Raises:
            ValueError: If inputs have mismatched shapes or insufficient data.
        """
        oof_preds = np.asarray(oof_preds).ravel()
        y_true = np.asarray(y_true).ravel()

        if len(oof_preds) != len(y_true):
            raise ValueError(
                f"oof_preds and y_true must have same length, "
                f"got {len(oof_preds)} and {len(y_true)}"
            )

        # Filter NaN values
        mask = np.isfinite(oof_preds) & np.isfinite(y_true)
        oof_clean = oof_preds[mask]
        y_clean = y_true[mask].astype(int)

        if len(oof_clean) < 50:
            raise ValueError(
                f"Need at least 50 valid samples for calibration, got {len(oof_clean)}"
            )

        if len(np.unique(y_clean)) < 2:
            raise ValueError("Need both classes present for calibration")

        n_positive = int(y_clean.sum())
        if self.method == "isotonic" and n_positive < 30:
            logger.warning(
                f"Isotonic calibration has only {n_positive} positive samples "
                "(recommended >= 30). Consider using 'logistic_intercept' or "
                "'logistic_full' for small calibration sets."
            )

        # Compute pre-calibration metrics
        from sklearn.metrics import brier_score_loss, roc_auc_score

        brier_pre = brier_score_loss(y_clean, oof_clean)
        auroc_pre = roc_auc_score(y_clean, oof_clean)

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator_.fit(oof_clean, y_clean)

            # Count bins for isotonic
            n_bins = len(np.unique(self.calibrator_.f_))
            logger.info(f"OOF calibration ({self.method}, {n_bins} bins)")

        elif self.method == "logistic_full":
            # Two-parameter Platt scaling: logit(Y=1) = a + b*logit(p)
            # No regularization; logit already clips to avoid log(0).
            log_odds = logit(oof_clean)
            self.calibrator_ = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(log_odds.reshape(-1, 1), y_clean)
            logger.info(f"OOF calibration ({self.method})")

        elif self.method == "logistic_intercept":
            # Intercept-only recalibration: logit(Y=1) = a + logit(p)
            # logit(p) enters as a fixed offset (coefficient forced to 1).
            # Lowest-variance parametric option; preferred when n_cal is small.
            log_odds = logit(oof_clean)

            def _intercept_nll(a: np.ndarray) -> float:
                logit_q = float(a[0]) + log_odds
                log_prob_pos = -np.logaddexp(0.0, -logit_q)
                log_prob_neg = -np.logaddexp(0.0, logit_q)
                return -float(np.sum(y_clean * log_prob_pos + (1.0 - y_clean) * log_prob_neg))

            result = minimize(
                _intercept_nll,
                x0=np.array([0.0]),
                method="L-BFGS-B",
            )
            if not result.success:
                logger.warning(
                    f"logistic_intercept optimisation did not converge: {result.message}"
                )
            self.calibrator_ = {"intercept": float(result.x[0])}
            logger.info(
                f"OOF calibration ({self.method}, "
                f"intercept={self.calibrator_['intercept']:.4f})"
            )

        elif self.method == "beta":
            # Beta calibration: logit(q) = a*log(p) + b*log(1-p) + c
            # Ref: Kull et al. (2017) ECML-PKDD.
            from ced_ml.utils.math_utils import EPSILON_LOGIT

            p_clip = np.clip(oof_clean, EPSILON_LOGIT, 1.0 - EPSILON_LOGIT)
            log_p = np.log(p_clip)
            log1mp = np.log(1.0 - p_clip)

            result = minimize(
                _beta_nll,
                x0=np.array([1.0, 1.0, 0.0]),
                args=(log_p, log1mp, y_clean),
                method="L-BFGS-B",
            )
            if not result.success:
                logger.warning(f"beta calibration optimisation did not converge: {result.message}")
            a, b, c = result.x
            self.calibrator_ = {"a": float(a), "b": float(b), "c": float(c)}
            logger.info(f"OOF calibration ({self.method}, a={a:.4f}, b={b:.4f}, c={c:.4f})")

        # Mark as fitted before computing post-calibration metrics
        self.is_fitted = True

        # Compute post-calibration metrics
        calibrated_preds = self.transform(oof_clean)
        brier_post = brier_score_loss(y_clean, calibrated_preds)
        auroc_post = roc_auc_score(y_clean, calibrated_preds)

        brier_improvement = (brier_pre - brier_post) / brier_pre * 100
        auroc_change = auroc_post - auroc_pre

        logger.info(f"  Pre-calibration:  Brier={brier_pre:.3f}, AUROC={auroc_pre:.3f}")
        logger.info(
            f"  Post-calibration: Brier={brier_post:.3f}, AUROC={auroc_post:.3f} "
            f"(In-sample Brier improvement: {brier_improvement:+.1f}%; use holdout for unbiased estimate)"
        )

        _rationale = {
            "isotonic": ("Isotonic regression maps predictions to empirical prevalence bins"),
            "logistic_full": (
                "Two-parameter Platt scaling fits logistic transform (slope + intercept)"
            ),
            "logistic_intercept": (
                "Intercept-only recalibration shifts log-odds with minimal variance"
            ),
            "beta": ("Beta calibration fits asymmetric log-odds model (Kull et al. 2017)"),
        }
        logger.info(f"  Rationale: {_rationale.get(self.method, self.method)}")

        # Warn if calibration degrades discrimination
        if auroc_change < -0.01:
            logger.warning(
                f"Calibration degraded discrimination: AUROC dropped "
                f"{auroc_pre:.3f} -> {auroc_post:.3f} (check data quality)"
            )

        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predictions.

        Args:
            predictions: Raw predictions to calibrate, shape (n_samples,).

        Returns:
            Calibrated predictions, shape (n_samples,).

        Raises:
            RuntimeError: If calibrator is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calling transform")

        predictions = np.asarray(predictions).ravel()

        if self.method == "isotonic":
            calibrated = self.calibrator_.predict(predictions)

        elif self.method == "logistic_full":
            # Two-parameter Platt: use sklearn predict_proba on logit-transformed input.
            log_odds = logit(predictions)
            calibrated = self.calibrator_.predict_proba(log_odds.reshape(-1, 1))[:, 1]

        elif self.method == "logistic_intercept":
            # Intercept-only: apply stored intercept shift on logit scale.
            log_odds = logit(predictions)
            logit_q = self.calibrator_["intercept"] + log_odds
            calibrated = _scipy_expit(logit_q)

        elif self.method == "beta":
            # Beta calibration: logit(q) = a*log(p) + b*log(1-p) + c
            from ced_ml.utils.math_utils import EPSILON_LOGIT

            p_clip = np.clip(predictions, EPSILON_LOGIT, 1.0 - EPSILON_LOGIT)
            logit_q = (
                self.calibrator_["a"] * np.log(p_clip)
                + self.calibrator_["b"] * np.log(1.0 - p_clip)
                + self.calibrator_["c"]
            )
            calibrated = _scipy_expit(logit_q)

        else:
            raise RuntimeError(f"Unknown method '{self.method}' in transform")

        return np.clip(calibrated, 0.0, 1.0)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"OOFCalibrator(method='{self.method}', {status})"


def fit_oof_calibrator(
    oof_preds: np.ndarray, y_true: np.ndarray, method: str = "isotonic"
) -> OOFCalibrator:
    """
    Fit an OOF calibrator on pooled out-of-fold predictions.

    This is a convenience function that creates and fits an OOFCalibrator.

    Args:
        oof_preds: Raw (uncalibrated) OOF predictions, shape (n_samples,).
        y_true: True binary labels, shape (n_samples,).
        method: Calibration method. One of "isotonic", "sigmoid" (alias for
                "logistic_full"), "logistic_full", "logistic_intercept", "beta".

    Returns:
        Fitted OOFCalibrator instance.
    """
    calibrator = OOFCalibrator(method=method)
    calibrator.fit(oof_preds, y_true)
    return calibrator


def apply_oof_calibrator(calibrator: OOFCalibrator, predictions: np.ndarray) -> np.ndarray:
    """
    Apply a fitted OOF calibrator to predictions.

    This is a convenience function that wraps OOFCalibrator.transform().

    Args:
        calibrator: Fitted OOFCalibrator instance.
        predictions: Raw predictions to calibrate, shape (n_samples,).

    Returns:
        Calibrated predictions, shape (n_samples,).
    """
    return calibrator.transform(predictions)


class OOFCalibratedModel(BaseEstimator):
    """
    Wrapper that applies OOF calibration to a base model's predictions.

    This wrapper is used when calibration strategy is "oof_posthoc".
    It combines a base sklearn model with a fitted OOFCalibrator.

    Inherits from BaseEstimator to support sklearn clone() and grid search.

    Attributes:
        base_model: The underlying sklearn model.
        calibrator: Fitted OOFCalibrator instance.
    """

    def __init__(self, base_model=None, calibrator: OOFCalibrator | None = None):
        """
        Initialize OOF calibrated model wrapper.

        Args:
            base_model: Sklearn model with predict_proba method.
            calibrator: Fitted OOFCalibrator instance.

        Note: Parameters can be None to support sklearn clone() which
        creates instances with default parameters before setting attributes.
        """
        # Validation only when both parameters are provided
        if base_model is not None and calibrator is not None:
            if not hasattr(base_model, "predict_proba"):
                raise ValueError("base_model must have predict_proba method")
            if not calibrator.is_fitted:
                raise ValueError("calibrator must be fitted")
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        """
        Generate calibrated probability predictions.

        Args:
            X: Input features.

        Returns:
            Array of shape (n_samples, 2) with calibrated probabilities.
        """
        raw_proba = self.base_model.predict_proba(X)
        calibrated = self.calibrator.transform(raw_proba[:, 1])
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        """
        Generate binary predictions using default threshold of 0.5.

        Args:
            X: Input features.

        Returns:
            Array of binary predictions.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def __repr__(self) -> str:
        return f"OOFCalibratedModel(base_model={type(self.base_model).__name__}, calibrator={self.calibrator})"
