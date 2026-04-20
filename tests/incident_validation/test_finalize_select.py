"""Unit tests for ivlib.finalize.select_winner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivlib.finalize import select_winner


def test_select_winner_simple(synthetic_cv_results):
    strategy, weight, summary = select_winner(synthetic_cv_results)
    assert strategy == "incident_only"
    assert weight == "log"
    # Summary columns.
    assert {
        "strategy", "weight_scheme",
        "mean_auprc", "std_auprc", "mean_auroc", "std_auroc", "mean_nonzero",
    }.issubset(summary.columns)


def test_select_winner_single_combo():
    # Only one combo -> it wins by default.
    rows = [{
        "fold": f,
        "strategy": "incident_only",
        "weight_scheme": "none",
        "auprc": 0.5 + 0.01 * f,
        "auroc": 0.6,
        "best_params_json": "{}",
        "best_inner_auprc": 0.4,
        "n_nonzero_coefs": 5,
    } for f in range(5)]
    df = pd.DataFrame(rows)
    strategy, weight, _ = select_winner(df)
    assert strategy == "incident_only"
    assert weight == "none"


def test_select_winner_parsimony_tiebreak():
    # Two combos tie within 0.02 on mean AUPRC. Runner has strictly lower
    # mean_nonzero. Bootstrap CI on Delta spans zero -> parsimony picks runner.
    rng = np.random.default_rng(0)
    n_folds = 5

    # Top combo: mean AUPRC ~= 0.705, high nonzero (20).
    top_auprc = np.array([0.70, 0.71, 0.69, 0.72, 0.705])
    # Runner combo: mean AUPRC ~= 0.700 (within 0.02), low nonzero (5).
    # Paired auprc values that, when bootstrapped, produce a CI spanning 0.
    runner_auprc = np.array([0.705, 0.70, 0.695, 0.715, 0.685])

    rows = []
    for f in range(n_folds):
        rows.append({
            "fold": f,
            "strategy": "incident_prevalent",
            "weight_scheme": "balanced",
            "auprc": float(top_auprc[f]),
            "auroc": 0.75,
            "best_params_json": "{}",
            "best_inner_auprc": 0.68,
            "n_nonzero_coefs": 20,
        })
        rows.append({
            "fold": f,
            "strategy": "incident_only",
            "weight_scheme": "log",
            "auprc": float(runner_auprc[f]),
            "auroc": 0.74,
            "best_params_json": "{}",
            "best_inner_auprc": 0.68,
            "n_nonzero_coefs": 5,
        })
    df = pd.DataFrame(rows)

    # Precondition: top is IP+balanced; runner (within 0.02) is IO+log with fewer
    # nonzero coefs. Tie-break should flip winner to the parsimonious combo.
    mean_top = top_auprc.mean()
    mean_runner = runner_auprc.mean()
    assert 0.0 < (mean_top - mean_runner) < 0.02

    strategy, weight, summary = select_winner(df)
    assert (strategy, weight) == ("incident_only", "log")
    # Summary still ranks top by mean_auprc first.
    assert summary.iloc[0]["strategy"] == "incident_prevalent"
