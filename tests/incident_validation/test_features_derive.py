"""Unit tests for ivlib.features derivation helpers.

Ground truth: see conftest.py docstring. Under strict rules
(min_fold_stability=4/5, require_sign_consistent=True) the core is
p0..p4 (5 proteins). Relaxing sign_consistent adds p5..p7 and p9
(4 more -> 9 total).
"""

from __future__ import annotations

import numpy as np
import pytest

from ivlib.features import (
    derive_core_panel,
    derive_per_strategy_panels,
    jaccard_overlap,
)


REQUIRED_COLUMNS = {
    "protein",
    "weight_scheme",
    "mean_coef",
    "std_coef",
    "fold_stability",
    "sign_consistent",
    "bootstrap_freq",
}


def test_jaccard_overlap_identity():
    A = {"a": 1, "b": 1, "c": 1}  # any 3-set
    sets = {"A": {"x", "y", "z"}, "B": {"x", "y", "z"}}
    df = jaccard_overlap(sets)
    row = df[(df["a"] == "A") & (df["b"] == "A")].iloc[0]
    np.testing.assert_allclose(row["jaccard"], 1.0)


def test_jaccard_overlap_disjoint():
    sets = {"A": {"x", "y"}, "B": {"p", "q"}}
    df = jaccard_overlap(sets)
    row = df[(df["a"] == "A") & (df["b"] == "B")].iloc[0]
    np.testing.assert_allclose(row["jaccard"], 0.0)


def test_jaccard_overlap_partial():
    # A = {x,y,z}, B = {y,z,w}  -> intersection 2, union 4 -> 0.5
    sets = {"A": {"x", "y", "z"}, "B": {"y", "z", "w"}}
    df = jaccard_overlap(sets)
    row = df[(df["a"] == "A") & (df["b"] == "B")].iloc[0]
    np.testing.assert_allclose(row["jaccard"], 0.5)
    assert int(row["n_inter"]) == 2


def _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel):
    selection_freq = {p: 0.5 for p in synthetic_panel}
    return derive_per_strategy_panels(
        synthetic_cv_results,
        synthetic_fold_coefs,
        synthetic_panel,
        selection_freq,
        n_folds=5,
    )


def test_derive_per_strategy_panels_shape(
    synthetic_cv_results, synthetic_fold_coefs, synthetic_panel,
):
    panels = _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel)
    assert set(panels.keys()) == {
        "incident_only", "incident_prevalent", "prevalent_only",
    }
    for strategy, df in panels.items():
        assert REQUIRED_COLUMNS.issubset(df.columns), strategy
        assert len(df) == len(synthetic_panel)


def test_derive_per_strategy_panels_picks_winning_weight(
    synthetic_cv_results, synthetic_fold_coefs, synthetic_panel,
):
    panels = _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel)
    # Only 'log' is present in cv_results, so it must be selected for every strategy.
    for df in panels.values():
        assert (df["weight_scheme"] == "log").all()


def test_derive_per_strategy_panels_sign_consistency(
    synthetic_cv_results, synthetic_fold_coefs, synthetic_panel,
):
    panels = _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel)
    df = panels["incident_only"].set_index("protein")

    # p0: +5e-3 in all folds -> sign_consistent True, stability 1.0.
    assert bool(df.loc["p000_resid", "sign_consistent"]) is True
    np.testing.assert_allclose(df.loc["p000_resid", "fold_stability"], 1.0)

    # p9: +5e-3 x3, -5e-3 x2 -> non-zero in all 5 folds but mixed signs.
    assert bool(df.loc["p009_resid", "sign_consistent"]) is False
    np.testing.assert_allclose(df.loc["p009_resid", "fold_stability"], 1.0)

    # p10: all zeros -> sign_consistent False (no non-zero folds), stability 0.
    assert bool(df.loc["p010_resid", "sign_consistent"]) is False
    np.testing.assert_allclose(df.loc["p010_resid", "fold_stability"], 0.0)


def test_derive_core_panel_threshold(
    synthetic_cv_results, synthetic_fold_coefs, synthetic_panel,
):
    panels = _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel)
    core = derive_core_panel(
        panels,
        winner_strategy="incident_only",
        min_fold_stability=4 / 5,
        require_sign_consistent=True,
    )
    # Expected ground truth under strict rules: p0..p4 = 5 proteins
    # (p5..p7 have a zero fold which breaks sign_consistent).
    expected = {f"p{i:03d}_resid" for i in range(0, 5)}
    assert set(core["protein"].tolist()) == expected
    assert len(core) == 5


def test_derive_core_panel_fallback_empty_sign(
    synthetic_cv_results, synthetic_fold_coefs, synthetic_panel,
):
    panels = _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel)
    strict = derive_core_panel(
        panels, winner_strategy="incident_only",
        min_fold_stability=4 / 5, require_sign_consistent=True,
    )
    relaxed = derive_core_panel(
        panels, winner_strategy="incident_only",
        min_fold_stability=4 / 5, require_sign_consistent=False,
    )
    assert len(relaxed) > len(strict)
    # Relaxed: p0..p4 (strict) plus p5..p7 (stability 0.8) plus p9 (stability 1.0).
    expected_relaxed = (
        set(strict["protein"])
        | {f"p{i:03d}_resid" for i in range(5, 8)}
        | {"p009_resid"}
    )
    assert set(relaxed["protein"]) == expected_relaxed


def test_derive_core_panel_unknown_winner_raises(
    synthetic_cv_results, synthetic_fold_coefs, synthetic_panel,
):
    panels = _panels(synthetic_cv_results, synthetic_fold_coefs, synthetic_panel)
    with pytest.raises(KeyError, match="Winner strategy"):
        derive_core_panel(panels, winner_strategy="not_a_strategy")
