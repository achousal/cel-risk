"""Feature selection and per-strategy/core panel derivation for incident validation."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ced_ml.data.schema import INCIDENT_LABEL, TARGET_COL
from ced_ml.features.corr_prune import prune_correlated_proteins
from ced_ml.features.stability import bootstrap_stability_selection

logger = logging.getLogger(__name__)


def run_feature_selection(cfg: Any, data: dict) -> dict:
    df = data["df"]
    protein_cols = data["protein_cols"]
    dev_ic_idx = np.concatenate([data["dev_incident_idx"], data["dev_control_idx"]])

    X_fs = df.loc[dev_ic_idx]
    y_fs = (X_fs[TARGET_COL] == INCIDENT_LABEL).astype(int).to_numpy()

    logger.info("=== Feature Selection ===")
    logger.info(
        "Bootstrap stability: %d resamples, top %d per resample, threshold %.0f%%",
        cfg.n_bootstrap,
        cfg.bootstrap_top_k,
        cfg.stability_threshold * 100,
    )

    stable_proteins, selection_freq, bootstrap_log = bootstrap_stability_selection(
        X=X_fs,
        y=y_fs,
        protein_cols=protein_cols,
        screen_method=cfg.screen_method,
        n_bootstrap=cfg.n_bootstrap,
        top_k=cfg.bootstrap_top_k,
        stability_threshold=cfg.stability_threshold,
        seed=cfg.split_seed,
    )

    logger.info("Stable proteins (pre-prune): %d", len(stable_proteins))

    logger.info(
        "Correlation pruning: |r| > %.2f (%s), keep more stable feature",
        cfg.corr_threshold,
        cfg.corr_method,
    )
    prune_map, pruned_proteins = prune_correlated_proteins(
        df=X_fs,
        y=y_fs,
        proteins=stable_proteins,
        selection_freq=selection_freq,
        corr_threshold=cfg.corr_threshold,
        corr_method=cfg.corr_method,
        tiebreak_method="freq_then_univariate",
    )

    logger.info("Final panel after pruning: %d proteins", len(pruned_proteins))

    return {
        "stable_proteins": stable_proteins,
        "pruned_proteins": pruned_proteins,
        "selection_freq": selection_freq,
        "bootstrap_log": bootstrap_log,
        "prune_map": prune_map,
    }


def _fold_coefs_to_matrix(
    fold_coefs: list[dict],
    protein_panel: list[str],
    strategy: str,
    weight_scheme: str,
    n_folds: int,
) -> np.ndarray:
    mat = np.zeros((n_folds, len(protein_panel)), dtype=float)
    for entry in fold_coefs:
        if entry["strategy"] != strategy or entry["weight_scheme"] != weight_scheme:
            continue
        fold_i = int(entry["fold"])
        if 0 <= fold_i < n_folds:
            mat[fold_i, :] = np.asarray(entry["coefs"], dtype=float)
    return mat


class warnings_ignore_all_nan:
    # Suppress RuntimeWarning: Mean of empty slice from nanmean on all-NaN cols.

    def __enter__(self):
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cm.__exit__(exc_type, exc_val, exc_tb)


def derive_per_strategy_panels(
    cv_results: pd.DataFrame,
    fold_coefs: list[dict],
    protein_panel: list[str],
    selection_freq: dict,
    n_folds: int,
    nonzero_tol: float = 1e-8,
) -> dict[str, pd.DataFrame]:
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])["auprc"]
        .mean()
        .reset_index()
    )
    panels: dict[str, pd.DataFrame] = {}
    for strategy in sorted(cv_results["strategy"].unique()):
        s_rows = summary[summary["strategy"] == strategy]
        if s_rows.empty:
            continue
        best_weight = s_rows.sort_values("auprc", ascending=False).iloc[0]["weight_scheme"]
        mat = _fold_coefs_to_matrix(
            fold_coefs, protein_panel, strategy, best_weight, n_folds,
        )
        nz_mask = np.abs(mat) > nonzero_tol
        fold_stability = nz_mask.mean(axis=0)
        with np.errstate(invalid="ignore"):
            signs = np.sign(mat)
            signs[~nz_mask] = 0
            row_signs_nz = np.where(nz_mask, signs, np.nan)
            # Sign-consistent requires all non-zero folds agree; zero non-zero folds -> False.
            with warnings_ignore_all_nan():
                pos_frac = np.nanmean(row_signs_nz == 1, axis=0)
                neg_frac = np.nanmean(row_signs_nz == -1, axis=0)
            pos_frac = np.nan_to_num(pos_frac, nan=0.0)
            neg_frac = np.nan_to_num(neg_frac, nan=0.0)
            sign_consistent = (
                (nz_mask.any(axis=0))
                & ((pos_frac == 1.0) | (neg_frac == 1.0))
            )
        mean_coef = mat.mean(axis=0)
        std_coef = mat.std(axis=0, ddof=0)
        panels[strategy] = pd.DataFrame({
            "protein": protein_panel,
            "weight_scheme": best_weight,
            "mean_coef": mean_coef,
            "std_coef": std_coef,
            "fold_stability": fold_stability,
            "sign_consistent": sign_consistent,
            "bootstrap_freq": [selection_freq.get(p, 0.0) for p in protein_panel],
        }).sort_values(
            ["fold_stability", "mean_coef"],
            ascending=[False, False],
            key=lambda s: s.abs() if s.name == "mean_coef" else s,
        ).reset_index(drop=True)
    return panels


def derive_core_panel(
    per_strategy_panels: dict[str, pd.DataFrame],
    winner_strategy: str,
    min_fold_stability: float = 4 / 5,
    require_sign_consistent: bool = True,
) -> pd.DataFrame:
    if winner_strategy not in per_strategy_panels:
        raise KeyError(f"Winner strategy {winner_strategy!r} not in per-strategy panels")
    panel = per_strategy_panels[winner_strategy].copy()
    mask = panel["fold_stability"] >= (min_fold_stability - 1e-12)
    if require_sign_consistent:
        mask &= panel["sign_consistent"].astype(bool)
    core = panel[mask].copy()
    core = core.sort_values(
        "mean_coef", key=lambda s: s.abs(), ascending=False,
    ).reset_index(drop=True)
    return core


def jaccard_overlap(sets: dict[str, set[str]]) -> pd.DataFrame:
    keys = sorted(sets.keys())
    rows = []
    for a in keys:
        for b in keys:
            A, B = sets[a], sets[b]
            union = A | B
            j = (len(A & B) / len(union)) if union else float("nan")
            rows.append({"a": a, "b": b, "jaccard": j, "n_a": len(A), "n_b": len(B), "n_inter": len(A & B)})
    return pd.DataFrame(rows)
