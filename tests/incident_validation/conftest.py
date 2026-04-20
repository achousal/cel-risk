"""Shared fixtures for ivlib unit tests.

Puts operations/incident-validation on sys.path so tests can import ivlib
without requiring installation.

Synthetic fixture design (documented ground truth):

synthetic_panel: 40 proteins, named p000_resid ... p039_resid.

synthetic_fold_coefs: 15 entries = 3 strategies x 5 folds, all with
weight_scheme='log'. Coefficient layout PER STRATEGY is identical
(makes assertions deterministic across strategies):

  p0..p4   -> +5e-3 in all 5 folds         (stability=1.0, sign_consistent=True)
  p5..p7   -> +5e-3 in folds 0..3, 0.0 in fold 4  (stability=0.8; see note)
  p8       -> +5e-3 in folds 0..2, 0.0 in folds 3..4 (stability=0.6)
  p9       -> +5e-3 in folds 0..2, -5e-3 in folds 3..4 (stability=1.0, mixed)
  p10..p39 -> 0.0 everywhere                (stability=0.0)

Note on sign_consistent: ivlib.features treats zero folds as breaking
sign consistency (pos_frac must equal 1.0 across ALL folds, and zero
folds contribute to the denominator). So p5..p7 have sign_consistent=
False even though the non-zero folds all agree on sign.

Expected core for derive_core_panel(min_fold_stability=4/5,
require_sign_consistent=True): p0..p4 (5 proteins).

Expected set if require_sign_consistent=False: p0..p4 + p5..p7 + p9
= 9 proteins (all with stability >= 0.8).

synthetic_cv_results: 15 rows matching fold_coefs. Designed so that
the per-strategy best weight_scheme is 'log' for all three strategies
(trivially true -- only 'log' present), and the overall winner via
select_winner is ('incident_only', 'log').
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root -> operations/incident-validation on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_IV_DIR = _REPO_ROOT / "operations" / "incident-validation"
if str(_IV_DIR) not in sys.path:
    sys.path.insert(0, str(_IV_DIR))


N_PROTEINS = 40
N_FOLDS = 5


@pytest.fixture
def synthetic_panel() -> list[str]:
    return [f"p{i:03d}_resid" for i in range(N_PROTEINS)]


def _build_strategy_matrix() -> np.ndarray:
    """Build a (N_FOLDS, N_PROTEINS) coef matrix encoding the documented
    ground-truth layout. Identical across strategies."""
    mat = np.zeros((N_FOLDS, N_PROTEINS), dtype=float)
    # p0..p4: +5e-3 in all folds.
    for p in range(0, 5):
        mat[:, p] = 5e-3
    # p5..p7: +5e-3 in folds 0..3, 0.0 in fold 4.
    for p in range(5, 8):
        mat[0:4, p] = 5e-3
        mat[4, p] = 0.0
    # p8: +5e-3 in folds 0..2, 0.0 in folds 3..4 (stability=0.6).
    mat[0:3, 8] = 5e-3
    # p9: +5e-3 in folds 0..2, -5e-3 in folds 3..4 (stability=1.0, mixed signs).
    mat[0:3, 9] = 5e-3
    mat[3:5, 9] = -5e-3
    # p10..p39: zeros already.
    return mat


@pytest.fixture
def synthetic_fold_coefs(synthetic_panel) -> list[dict]:
    mat = _build_strategy_matrix()
    strategies = ["incident_only", "incident_prevalent", "prevalent_only"]
    entries: list[dict] = []
    for strategy in strategies:
        for fold in range(N_FOLDS):
            entries.append({
                "fold": fold,
                "strategy": strategy,
                "weight_scheme": "log",
                "coefs": mat[fold].copy(),
            })
    return entries


@pytest.fixture
def synthetic_cv_results() -> pd.DataFrame:
    """Make incident_only+log the clear AUPRC winner, and the only weight
    scheme present per strategy is 'log' (matches fold_coefs)."""
    rows = []
    # AUPRC means (across folds): IO+log=0.80, IP+log=0.60, PO+log=0.55.
    auprc_means = {
        "incident_only": 0.80,
        "incident_prevalent": 0.60,
        "prevalent_only": 0.55,
    }
    auroc_means = {
        "incident_only": 0.85,
        "incident_prevalent": 0.70,
        "prevalent_only": 0.65,
    }
    rng = np.random.default_rng(7)
    for strategy in ["incident_only", "incident_prevalent", "prevalent_only"]:
        for fold in range(N_FOLDS):
            jitter = rng.normal(0.0, 0.005)
            rows.append({
                "fold": fold,
                "strategy": strategy,
                "weight_scheme": "log",
                "auprc": auprc_means[strategy] + jitter,
                "auroc": auroc_means[strategy] + jitter,
                "best_params_json": "{}",
                "best_inner_auprc": auprc_means[strategy] + jitter - 0.02,
                "n_nonzero_coefs": 8,
            })
    return pd.DataFrame(rows)
