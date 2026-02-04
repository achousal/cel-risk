"""
Shared fixtures for models tests.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


@pytest.fixture
def toy_oof_predictions(toy_data):
    """Generate toy OOF predictions from multiple 'base models'.

    Simulates OOF predictions from 3 different models with different
    characteristics (different seeds, slightly different predictions).
    """
    X, y = toy_data
    n_samples = len(y)

    # Simulate 3 base models
    oof_dict = {}

    # Model 1: Good predictions
    rng = np.random.default_rng(42)
    lr1 = LogisticRegression(C=1.0, random_state=42)
    oof1 = cross_val_predict(lr1, X, y, cv=3, method="predict_proba")[:, 1]
    # Add repeat dimension
    oof_dict["LR_EN"] = np.vstack([oof1, oof1 + rng.normal(0, 0.02, n_samples)])

    # Model 2: Different model
    rng = np.random.default_rng(43)
    lr2 = LogisticRegression(C=0.1, random_state=43)
    oof2 = cross_val_predict(lr2, X, y, cv=3, method="predict_proba")[:, 1]
    oof_dict["RF"] = np.vstack([oof2, oof2 + rng.normal(0, 0.02, n_samples)])

    # Model 3: Another different model
    rng = np.random.default_rng(44)
    lr3 = LogisticRegression(C=10.0, random_state=44)
    oof3 = cross_val_predict(lr3, X, y, cv=3, method="predict_proba")[:, 1]
    oof_dict["XGBoost"] = np.vstack([oof3, oof3 + rng.normal(0, 0.02, n_samples)])

    return oof_dict, y


@pytest.fixture
def toy_test_predictions(toy_data):
    """Generate toy test set predictions from multiple models."""
    X, y = toy_data

    # Use a subset as "test set"
    test_idx = np.arange(50, 100)
    X_test = X[test_idx]
    y_test = y[test_idx]

    preds_dict = {}

    # Fit on "train" and predict on "test"
    train_idx = np.concatenate([np.arange(0, 50), np.arange(100, 200)])
    X_train = X[train_idx]
    y_train = y[train_idx]

    for i, model_name in enumerate(["LR_EN", "RF", "XGBoost"]):
        lr = LogisticRegression(C=1.0, random_state=42 + i)
        lr.fit(X_train, y_train)
        preds_dict[model_name] = lr.predict_proba(X_test)[:, 1]

    return preds_dict, y_test, test_idx
