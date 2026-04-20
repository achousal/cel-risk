"""Class weighting schemes for the 3x4 strategy x weight factorial."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

WEIGHT_SCHEMES: list[str] = ["none", "balanced", "sqrt", "log"]


def compute_class_weight(scheme: str, y: np.ndarray) -> dict | str | None:
    """Compute sklearn-compatible class_weight for a given scheme."""
    if scheme == "none":
        return None
    if scheme == "balanced":
        return "balanced"

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return None

    ratio = n_neg / n_pos

    if scheme == "sqrt":
        w1 = np.sqrt(ratio)
    elif scheme == "log":
        w1 = np.log(ratio)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    return {0: 1.0, 1: max(float(w1), 1.0)}


def class_weight_to_sample_weight(class_weight, y: np.ndarray) -> np.ndarray:
    # Tree models (XGB) take sample_weight at fit() rather than class_weight at init.
    y = np.asarray(y)
    sw = np.ones(len(y), dtype=float)
    if class_weight is None:
        return sw
    if class_weight == "balanced":
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        n = len(y)
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        sw[y == 1] = w_pos
        sw[y == 0] = w_neg
        return sw
    if isinstance(class_weight, dict):
        sw[y == 1] = float(class_weight.get(1, 1.0))
        sw[y == 0] = float(class_weight.get(0, 1.0))
        return sw
    return sw
