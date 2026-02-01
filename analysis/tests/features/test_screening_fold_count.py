"""Test that screening runs once per outer CV fold."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.kbest import ScreeningTransformer
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


@pytest.fixture
def toy_data():
    """Small dataset with clear signal for screening."""
    rng = np.random.default_rng(42)
    n = 200
    n_proteins = 30
    y = np.array([0] * 170 + [1] * 30)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, n_proteins)),
        columns=[f"prot_{i}_resid" for i in range(n_proteins)],
    )
    # Inject signal in first 5 proteins
    X.iloc[170:, :5] += 2.0
    protein_cols = list(X.columns)
    return X, y, protein_cols


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
