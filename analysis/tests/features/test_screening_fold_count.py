"""Test that screening runs once per outer CV fold."""

from unittest.mock import patch

import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from ced_ml.features.kbest import ScreeningTransformer


@pytest.fixture
def toy_data(toy_data_screening):
    """Alias for toy_data_screening from conftest for backward compatibility."""
    return toy_data_screening


def _count_screening_fits(X, y, protein_cols, n_splits, n_repeats=1):
    """Simulate outer CV loop counting ScreeningTransformer.fit calls."""
    screener = ScreeningTransformer(
        method="mannwhitney",
        top_n=10,
        protein_cols=protein_cols,
    )
    pipe = Pipeline(
        [
            ("screen", screener),
            ("clf", LogisticRegression(max_iter=100, random_state=0)),
        ]
    )

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=0,
    )

    fit_count = 0
    original_fit = ScreeningTransformer.fit

    def counting_fit(self, X_in, y_in=None):
        nonlocal fit_count
        fit_count += 1
        return original_fit(self, X_in, y_in)

    with patch.object(ScreeningTransformer, "fit", counting_fit):
        for train_idx, _test_idx in rskf.split(X, y):
            fold_pipe = clone(pipe)
            fold_pipe.fit(X.iloc[train_idx], y[train_idx])

    return fit_count


def test_3_folds_gives_3_screening_calls(toy_data):
    X, y, protein_cols = toy_data
    count = _count_screening_fits(X, y, protein_cols, n_splits=3)
    assert count == 3, f"Expected 3 screening calls for 3 folds, got {count}"


def test_5_folds_gives_5_screening_calls(toy_data):
    X, y, protein_cols = toy_data
    count = _count_screening_fits(X, y, protein_cols, n_splits=5)
    assert count == 5, f"Expected 5 screening calls for 5 folds, got {count}"


def test_fold_count_scales_with_repeats(toy_data):
    X, y, protein_cols = toy_data
    count = _count_screening_fits(X, y, protein_cols, n_splits=3, n_repeats=2)
    assert count == 6, f"Expected 6 screening calls for 3 folds x 2 repeats, got {count}"
