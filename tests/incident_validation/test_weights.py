"""Unit tests for ivlib.weights."""

from __future__ import annotations

import numpy as np
import pytest

from ivlib.weights import (
    WEIGHT_SCHEMES,
    class_weight_to_sample_weight,
    compute_class_weight,
)


def test_weight_schemes_list():
    assert WEIGHT_SCHEMES == ["none", "balanced", "sqrt", "log"]


def test_compute_class_weight_none():
    y = np.array([0, 0, 1, 1])
    assert compute_class_weight("none", y) is None


def test_compute_class_weight_balanced():
    y = np.array([0, 0, 1, 1])
    assert compute_class_weight("balanced", y) == "balanced"


@pytest.mark.parametrize("scheme,expected_fn", [
    ("sqrt", np.sqrt),
    ("log", np.log),
])
def test_compute_class_weight_sqrt_log(scheme, expected_fn):
    # Imbalanced: 100 negatives, 1 positive -> ratio = 100.
    y = np.array([0] * 100 + [1])
    cw = compute_class_weight(scheme, y)

    assert isinstance(cw, dict)
    assert cw[0] == 1.0
    # Both sqrt(100)=10 and log(100)~=4.6 are >> 1, so the floor does not kick in.
    expected = expected_fn(100.0)
    np.testing.assert_allclose(cw[1], expected)
    assert cw[1] > 1.0


def test_compute_class_weight_floor_applies_when_ratio_small():
    # 2 negatives, 2 positives -> ratio=1, log(1)=0, sqrt(1)=1.
    # Floor of max(., 1.0) should push w1 to exactly 1.0.
    y = np.array([0, 0, 1, 1])
    cw_log = compute_class_weight("log", y)
    cw_sqrt = compute_class_weight("sqrt", y)
    np.testing.assert_allclose(cw_log[1], 1.0)
    np.testing.assert_allclose(cw_sqrt[1], 1.0)


def test_compute_class_weight_no_positives_returns_none():
    y = np.zeros(5, dtype=int)
    assert compute_class_weight("sqrt", y) is None


def test_class_weight_to_sample_weight_none_returns_ones():
    y = np.array([0, 1, 0, 1, 1])
    sw = class_weight_to_sample_weight(None, y)
    assert sw.shape == (len(y),)
    np.testing.assert_allclose(sw, np.ones(len(y)))


def test_class_weight_to_sample_weight_balanced():
    # Heavy imbalance: 10 negatives, 2 positives.
    y = np.array([0] * 10 + [1] * 2)
    sw = class_weight_to_sample_weight("balanced", y)
    assert sw.shape == (len(y),)
    w_pos = sw[y == 1][0]
    w_neg = sw[y == 0][0]
    # With n_neg >> n_pos, positives get larger sample weight than negatives.
    assert w_pos > w_neg
    # Closed form: w_pos = n / (2*n_pos), w_neg = n / (2*n_neg).
    np.testing.assert_allclose(w_pos, len(y) / (2 * 2))
    np.testing.assert_allclose(w_neg, len(y) / (2 * 10))


def test_class_weight_to_sample_weight_dict():
    y = np.array([0, 1, 0, 1])
    cw = {0: 0.7, 1: 3.3}
    sw = class_weight_to_sample_weight(cw, y)
    np.testing.assert_allclose(sw[y == 0], 0.7)
    np.testing.assert_allclose(sw[y == 1], 3.3)
