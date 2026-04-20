"""Unit tests for ivlib.strategies."""

from __future__ import annotations

import numpy as np
import pytest

from ivlib.strategies import STRATEGIES, STRATEGY_TAGS, get_training_indices


def test_strategies_constants():
    assert len(STRATEGIES) == 3
    assert set(STRATEGIES) == {"incident_only", "incident_prevalent", "prevalent_only"}
    assert STRATEGY_TAGS["incident_only"] == "IO"
    assert STRATEGY_TAGS["incident_prevalent"] == "IP"
    assert STRATEGY_TAGS["prevalent_only"] == "PO"


def test_get_training_indices_incident_only():
    fold_train_incident = np.array([1, 2, 3])
    fold_train_controls = np.array([10, 11, 12, 13])
    prevalent_idx = np.array([50, 51])

    idx, y = get_training_indices(
        "incident_only", fold_train_incident, fold_train_controls, prevalent_idx,
    )

    # Only incident + controls are used; prevalent is ignored.
    assert len(idx) == len(fold_train_incident) + len(fold_train_controls)
    assert int(y.sum()) == len(fold_train_incident)
    np.testing.assert_array_equal(
        idx, np.concatenate([fold_train_incident, fold_train_controls]).astype(int),
    )
    assert y.dtype.kind == "i"


def test_get_training_indices_incident_prevalent():
    fold_train_incident = np.array([1, 2, 3])
    fold_train_controls = np.array([10, 11, 12, 13])
    prevalent_idx = np.array([50, 51])

    idx, y = get_training_indices(
        "incident_prevalent", fold_train_incident, fold_train_controls, prevalent_idx,
    )

    assert len(idx) == (
        len(fold_train_incident) + len(prevalent_idx) + len(fold_train_controls)
    )
    assert int(y.sum()) == len(fold_train_incident) + len(prevalent_idx)
    # Positives come first (incident then prevalent), controls are negatives.
    n_pos = len(fold_train_incident) + len(prevalent_idx)
    np.testing.assert_array_equal(y[:n_pos], np.ones(n_pos, dtype=int))
    np.testing.assert_array_equal(y[n_pos:], np.zeros(len(fold_train_controls), dtype=int))


def test_get_training_indices_prevalent_only():
    fold_train_incident = np.array([1, 2, 3])
    fold_train_controls = np.array([10, 11, 12, 13])
    prevalent_idx = np.array([50, 51])

    idx, y = get_training_indices(
        "prevalent_only", fold_train_incident, fold_train_controls, prevalent_idx,
    )

    assert len(idx) == len(prevalent_idx) + len(fold_train_controls)
    assert int(y.sum()) == len(prevalent_idx)
    # No incident samples routed through.
    assert not np.isin(fold_train_incident, idx).any()


def test_get_training_indices_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_training_indices(
            "not_a_strategy",
            np.array([1]),
            np.array([2]),
            np.array([3]),
        )
