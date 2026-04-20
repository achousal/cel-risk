"""Winner OOFCalibrator fit and ced_ml calibration metric bundle."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OOF_COLUMNS = ["fold", "strategy", "weight_scheme", "df_idx", "y_true", "y_prob"]


def fit_winner_calibrator(
    oof_df: pd.DataFrame,
    winner_strategy: str,
    winner_weight: str,
    method: str = "logistic_full",
) -> Tuple[object, pd.DataFrame]:
    """Subset oof_df to winner combo, dedupe by df_idx, fit OOFCalibrator."""
    empty = pd.DataFrame(columns=_OOF_COLUMNS)

    if oof_df is None or oof_df.empty:
        logger.warning("Empty OOF DataFrame; proceeding uncalibrated.")
        return None, empty

    w = oof_df[
        (oof_df["strategy"] == winner_strategy)
        & (oof_df["weight_scheme"] == winner_weight)
    ].copy()

    if w.empty:
        logger.warning(
            "No OOF rows for winner combo (strategy=%s, weight=%s); "
            "proceeding uncalibrated.",
            winner_strategy, winner_weight,
        )
        return None, empty

    w = (
        w.sort_values(["df_idx", "fold"])
        .drop_duplicates(subset=["df_idx"], keep="first")
        .reset_index(drop=True)
    )
    dev_oof_df = w

    try:
        from ced_ml.models.calibration import OOFCalibrator
        calibrator = OOFCalibrator(method=method)
        calibrator.fit(w["y_prob"].to_numpy(), w["y_true"].to_numpy())
        logger.info(
            "OOFCalibrator fitted on %d dev-OOF predictions (winner=%s+%s, method=%s)",
            len(w), winner_strategy, winner_weight, method,
        )
        return calibrator, dev_oof_df
    except Exception as exc:
        logger.warning("OOFCalibrator fit failed: %s. Proceeding uncalibrated.", exc)
        return None, dev_oof_df


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Full cel-risk calibration bundle over ced_ml."""
    from ced_ml.models.calibration import (
        calibration_intercept_slope,
        expected_calibration_error,
        integrated_calibration_index,
        spiegelhalter_z_test,
        adaptive_expected_calibration_error,
        brier_score_decomposition,
    )
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    cis = calibration_intercept_slope(y_true, y_prob)
    sp = spiegelhalter_z_test(y_true, y_prob)
    brier = brier_score_decomposition(y_true, y_prob)
    return {
        "intercept": float(cis.intercept),
        "slope": float(cis.slope),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "adaptive_ece": float(adaptive_expected_calibration_error(y_true, y_prob)),
        "ici": float(integrated_calibration_index(y_true, y_prob)),
        "spiegelhalter_z": float(sp.z_statistic),
        "spiegelhalter_p": float(sp.p_value),
        "brier": float(brier.brier_score),
        "brier_reliability": float(brier.reliability),
        "brier_resolution": float(brier.resolution),
        "brier_uncertainty": float(brier.uncertainty),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
