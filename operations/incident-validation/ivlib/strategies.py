"""Training strategies for the 3x4 strategy x weight factorial."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

STRATEGIES: list[str] = ["incident_only", "incident_prevalent", "prevalent_only"]
STRATEGY_TAGS: dict[str, str] = {
    "incident_only": "IO",
    "incident_prevalent": "IP",
    "prevalent_only": "PO",
}


def get_training_indices(
    strategy: str,
    fold_train_incident: np.ndarray,
    fold_train_controls: np.ndarray,
    prevalent_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, binary labels) for a given training strategy."""
    if strategy == "incident_only":
        idx = np.concatenate([fold_train_incident, fold_train_controls])
        y = np.concatenate([
            np.ones(len(fold_train_incident)),
            np.zeros(len(fold_train_controls)),
        ])
    elif strategy == "incident_prevalent":
        idx = np.concatenate([fold_train_incident, prevalent_idx, fold_train_controls])
        y = np.concatenate([
            np.ones(len(fold_train_incident)),
            np.ones(len(prevalent_idx)),
            np.zeros(len(fold_train_controls)),
        ])
    elif strategy == "prevalent_only":
        idx = np.concatenate([prevalent_idx, fold_train_controls])
        y = np.concatenate([
            np.ones(len(prevalent_idx)),
            np.zeros(len(fold_train_controls)),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return idx.astype(int), y.astype(int)
