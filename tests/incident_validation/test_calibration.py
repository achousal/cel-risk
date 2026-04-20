"""Unit tests for ivlib.calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivlib.calibration import compute_calibration_metrics, fit_winner_calibrator


_OOF_COLUMNS = ["fold", "strategy", "weight_scheme", "df_idx", "y_true", "y_prob"]

_METRIC_KEYS = {
    "intercept", "slope", "ece", "adaptive_ece", "ici",
    "spiegelhalter_z", "spiegelhalter_p",
    "brier", "brier_reliability", "brier_resolution", "brier_uncertainty",
    "n", "n_pos",
}


def test_fit_winner_calibrator_empty_oof_returns_none():
    empty = pd.DataFrame(columns=_OOF_COLUMNS)
    cal, dev = fit_winner_calibrator(empty, "incident_only", "log")
    assert cal is None
    assert list(dev.columns) == _OOF_COLUMNS
    assert len(dev) == 0


def test_fit_winner_calibrator_no_matching_rows_returns_none():
    # Rows exist, but none match the requested winner combo.
    df = pd.DataFrame({
        "fold": [0, 1],
        "strategy": ["prevalent_only", "prevalent_only"],
        "weight_scheme": ["none", "none"],
        "df_idx": [100, 101],
        "y_true": [0, 1],
        "y_prob": [0.2, 0.7],
    })
    cal, dev = fit_winner_calibrator(df, "incident_only", "log")
    assert cal is None
    assert list(dev.columns) == _OOF_COLUMNS
    assert len(dev) == 0


def test_fit_winner_calibrator_happy_path():
    # 200 synthetic OOF rows for the winner combo. Probs mildly miscalibrated
    # (compressed toward 0.5) so the calibrator has signal to fit.
    try:
        from ced_ml.models.calibration import OOFCalibrator  # noqa: F401
    except ImportError:
        pytest.skip("ced_ml.models.calibration.OOFCalibrator not available")

    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    base = rng.uniform(0.2, 0.8, size=n)
    # Miscalibrate: shrink toward 0.5.
    y_prob = 0.5 + 0.5 * (base - 0.5)

    df = pd.DataFrame({
        "fold": rng.integers(0, 5, size=n),
        "strategy": ["incident_only"] * n,
        "weight_scheme": ["log"] * n,
        "df_idx": np.arange(n),
        "y_true": y_true,
        "y_prob": y_prob,
    })

    cal, dev = fit_winner_calibrator(df, "incident_only", "log", method="logistic_full")
    assert cal is not None
    assert len(dev) == n
    assert set(dev.columns) >= set(_OOF_COLUMNS)


def test_fit_winner_calibrator_dedupes_by_df_idx():
    # Two rows share df_idx -> should dedupe to one (keeps first by (df_idx, fold)).
    df = pd.DataFrame({
        "fold": [0, 1],
        "strategy": ["incident_only", "incident_only"],
        "weight_scheme": ["log", "log"],
        "df_idx": [42, 42],
        "y_true": [0, 1],
        "y_prob": [0.3, 0.6],
    })
    # Fit can fail inside ced_ml with only one row; we just want the dedupe behavior.
    _, dev = fit_winner_calibrator(df, "incident_only", "log")
    assert len(dev) == 1
    assert int(dev["df_idx"].iloc[0]) == 42


def test_compute_calibration_metrics_schema():
    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.uniform(0.05, 0.95, size=n)
    out = compute_calibration_metrics(y_true, y_prob)
    assert set(out.keys()) == _METRIC_KEYS
    assert out["n"] == n
    assert out["n_pos"] == int(y_true.sum())


def test_compute_calibration_metrics_perfect_calibration():
    # 100 samples, balanced 50/50, predictions jittered around 0.5 so LOESS
    # in ICI has multiple x-values (constant predictions yield NaN ICI).
    rng = np.random.default_rng(0)
    n = 100
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    y_prob = np.full(n, 0.5) + rng.normal(0.0, 1e-3, size=n)
    out = compute_calibration_metrics(y_true, y_prob)
    # Brier score for ~0.5 prediction on balanced binary: ~0.25.
    np.testing.assert_allclose(out["brier"], 0.25, atol=1e-3)
    # Spiegelhalter should not flag miscalibration at p ≈ 0.5.
    assert out["spiegelhalter_p"] > 0.01
