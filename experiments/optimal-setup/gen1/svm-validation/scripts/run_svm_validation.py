#!/usr/bin/env python3
"""
SVM incident validation + winner-conditional feature ranking.

Two phases:
  Phase 1A — Recipe selection
    12 strategy-weight combos (3 strategies x 4 weights), 5-fold CV with
    Optuna-tuned C, AUPRC as primary. Uses Wald-based bootstrap stability
    panel (shared, weight-agnostic) to keep comparison fair.

  Phase 1B — Winner-conditional manifest
    Conditional on the winning (strategy, weight_scheme):
      1. Redefine positive class to match the winning strategy
      2. Apply winning class weights to LinearSVC
      3. Model-based bootstrap stability: fit weighted SVM per resample,
         keep top_k by |coef|
      4. Correlation prune survivors
      5. Rank by: selection frequency -> mean |coef| -> RRA rank tiebreak
      6. Output: winner_order.csv (ordered list for downstream sweep)

  Final evaluation on locked test set with the winning recipe.

Usage:
  cd cel-risk
  python experiments/optimal-setup/svm-validation/scripts/run_svm_validation.py
  python experiments/optimal-setup/svm-validation/scripts/run_svm_validation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import (
    CONTROL_LABEL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    PROTEIN_SUFFIX,
    TARGET_COL,
)
from ced_ml.features.corr_prune import prune_correlated_proteins
from ced_ml.features.stability import bootstrap_stability_selection

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logging.getLogger().handlers[0].stream = open(
    sys.stderr.fileno(), "w", buffering=1, closefd=False
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Config:
    data_path: Path = Path("data/Celiac_dataset_proteomics_w_demo.parquet")
    output_dir: Path = Path("results/svm_validation")
    rra_path: Path = Path(
        "results/experiments/rra_universe_sensitivity/rra_significance_corrected.csv"
    )

    # Splitting
    test_frac: float = 0.20
    n_outer_folds: int = 5
    split_seed: int = 42

    # Phase 1A: Wald-based bootstrap stability (shared panel for recipe comparison)
    n_bootstrap: int = 100
    bootstrap_top_k: int = 200
    stability_threshold: float = 0.70
    screen_method: str = "wald"

    # Phase 1B: Model-based bootstrap stability (winner-conditional)
    n_bootstrap_model: int = 100
    model_top_k: int = 200
    model_stability_threshold: float = 0.70

    # Correlation pruning
    corr_threshold: float = 0.85
    corr_method: str = "spearman"

    # Optuna
    n_optuna_trials: int = 50
    n_inner_folds: int = 3

    # SVM
    max_iter: int = 5000
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5

    # Evaluation
    n_bootstrap_ci: int = 2000
    ci_seed: int = 99

    # Strategies and weights
    strategies: list[str] = field(
        default_factory=lambda: [
            "incident_only", "incident_prevalent", "prevalent_only",
        ]
    )
    weight_schemes: list[str] = field(
        default_factory=lambda: ["none", "balanced", "sqrt", "log"]
    )

    smoke: bool = False

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        self.rra_path = Path(self.rra_path)
        if self.smoke:
            self.n_bootstrap = 10
            self.bootstrap_top_k = 50
            self.n_optuna_trials = 5
            self.n_bootstrap_ci = 100
            self.calibration_cv = 2
            self.n_bootstrap_model = 10
            self.model_top_k = 50


# ============================================================================
# Data preparation
# ============================================================================


def load_and_split(cfg: Config) -> dict:
    """Load data, define groups, create locked test/dev split."""
    logger.info("Loading data from %s", cfg.data_path)
    df = read_proteomics_file(str(cfg.data_path))
    logger.info("Loaded %d samples, %d columns", len(df), len(df.columns))

    protein_cols = [c for c in df.columns if c.endswith(PROTEIN_SUFFIX)]
    logger.info("Found %d protein columns", len(protein_cols))

    incident_idx = df.index[df[TARGET_COL] == INCIDENT_LABEL].to_numpy()
    prevalent_idx = df.index[df[TARGET_COL] == PREVALENT_LABEL].to_numpy()
    control_idx = df.index[df[TARGET_COL] == CONTROL_LABEL].to_numpy()

    logger.info(
        "Groups: %d incident, %d prevalent, %d controls",
        len(incident_idx), len(prevalent_idx), len(control_idx),
    )

    # Locked test set: stratified split of incident + controls by sex
    ic_idx = np.concatenate([incident_idx, control_idx])
    ic_labels = np.concatenate([
        np.ones(len(incident_idx)), np.zeros(len(control_idx)),
    ]).astype(int)

    if "sex" in df.columns:
        sex_vals = df.loc[ic_idx, "sex"].values
        strat_key = [f"{l}_{s}" for l, s in zip(ic_labels, sex_vals)]
    else:
        strat_key = ic_labels

    dev_pos, test_pos = train_test_split(
        np.arange(len(ic_idx)),
        test_size=cfg.test_frac,
        stratify=strat_key,
        random_state=cfg.split_seed,
    )

    dev_ic_idx = ic_idx[dev_pos]
    test_ic_idx = ic_idx[test_pos]

    dev_incident_idx = np.array([i for i in dev_ic_idx if i in set(incident_idx)])
    dev_control_idx = np.array([i for i in dev_ic_idx if i in set(control_idx)])
    test_incident_idx = np.array([i for i in test_ic_idx if i in set(incident_idx)])
    test_control_idx = np.array([i for i in test_ic_idx if i in set(control_idx)])

    logger.info(
        "Dev: %d incident, %d controls. Test: %d incident, %d controls. Prevalent: %d.",
        len(dev_incident_idx), len(dev_control_idx),
        len(test_incident_idx), len(test_control_idx),
        len(prevalent_idx),
    )

    return {
        "df": df,
        "protein_cols": protein_cols,
        "incident_idx": incident_idx,
        "prevalent_idx": prevalent_idx,
        "control_idx": control_idx,
        "dev_incident_idx": dev_incident_idx,
        "dev_control_idx": dev_control_idx,
        "test_incident_idx": test_incident_idx,
        "test_control_idx": test_control_idx,
    }


# ============================================================================
# Phase 1A: Shared feature selection (Wald-based, weight-agnostic)
# ============================================================================


def select_features_shared(cfg: Config, data: dict) -> dict:
    """Wald-based bootstrap stability on dev incident + controls.

    Weight-agnostic: same panel for all 12 strategy-weight combos so the
    recipe comparison is fair.
    """
    logger.info("=== Phase 1A: Shared Feature Selection (Wald) ===")
    df = data["df"]
    protein_cols = data["protein_cols"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]

    fs_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    fs_labels = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    X_fs = df.loc[fs_idx, protein_cols]
    y_fs = pd.Series(fs_labels, index=fs_idx)

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

    prune_map, pruned_proteins = prune_correlated_proteins(
        df=X_fs,
        y=y_fs,
        proteins=stable_proteins,
        selection_freq=selection_freq,
        corr_threshold=cfg.corr_threshold,
        corr_method=cfg.corr_method,
    )

    logger.info("Shared panel: %d proteins", len(pruned_proteins))

    return {
        "stable_proteins": stable_proteins,
        "pruned_proteins": pruned_proteins,
        "selection_freq": selection_freq,
        "prune_map": prune_map,
        "bootstrap_log": bootstrap_log,
    }


# ============================================================================
# Class weighting
# ============================================================================


def compute_class_weight(scheme: str, y: np.ndarray) -> dict | str | None:
    """Compute sklearn-compatible class_weight for a given scheme."""
    if scheme == "none":
        return None
    if scheme == "balanced":
        return "balanced"

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return None

    ratio = n_neg / n_pos
    if scheme == "sqrt":
        w1 = np.sqrt(ratio)
    elif scheme == "log":
        w1 = np.log(ratio)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    return {0: 1.0, 1: float(w1)}


# ============================================================================
# SVM helpers
# ============================================================================


def _build_svm(
    C: float, class_weight, cfg: Config, seed: int,
) -> CalibratedClassifierCV:
    """Build a calibrated LinearSVC estimator."""
    base_svm = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=cfg.max_iter,
        dual="auto",
        random_state=seed,
    )
    return CalibratedClassifierCV(
        base_svm,
        method=cfg.calibration_method,
        cv=cfg.calibration_cv,
    )


def _build_base_svm(
    C: float, class_weight, max_iter: int, seed: int,
) -> LinearSVC:
    """Build an uncalibrated LinearSVC (for feature ranking in Phase 1B)."""
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        dual="auto",
        random_state=seed,
    )


def _extract_calibrated_coefs(model: CalibratedClassifierCV) -> np.ndarray:
    """Mean coefs across calibration folds."""
    coefs_list = [cc.estimator.coef_.ravel() for cc in model.calibrated_classifiers_]
    return np.mean(coefs_list, axis=0)


# ============================================================================
# Training strategy helpers
# ============================================================================


def get_training_data(
    strategy: str,
    incident_idx: np.ndarray,
    control_idx: np.ndarray,
    prevalent_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, binary labels) for a given training strategy."""
    if strategy == "incident_only":
        idx = np.concatenate([incident_idx, control_idx])
        y = np.concatenate([
            np.ones(len(incident_idx)), np.zeros(len(control_idx)),
        ])
    elif strategy == "incident_prevalent":
        idx = np.concatenate([incident_idx, prevalent_idx, control_idx])
        y = np.concatenate([
            np.ones(len(incident_idx)),
            np.ones(len(prevalent_idx)),
            np.zeros(len(control_idx)),
        ])
    elif strategy == "prevalent_only":
        idx = np.concatenate([prevalent_idx, control_idx])
        y = np.concatenate([
            np.ones(len(prevalent_idx)), np.zeros(len(control_idx)),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return idx.astype(int), y.astype(int)


# ============================================================================
# Phase 1A: Optuna + CV
# ============================================================================


def _optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight,
    cfg: Config,
    seed: int,
) -> float:
    """Optuna objective: mean AUPRC across inner CV folds."""
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)

    inner_cv = StratifiedKFold(
        n_splits=cfg.n_inner_folds, shuffle=True, random_state=seed,
    )
    auprcs = []

    for train_ix, val_ix in inner_cv.split(X_train, y_train):
        X_tr, X_va = X_train[train_ix], X_train[val_ix]
        y_tr, y_va = y_train[train_ix], y_train[val_ix]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = _build_svm(C, class_weight, cfg, seed)
        try:
            model.fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_va_s)[:, 1]
            auprcs.append(average_precision_score(y_va, y_prob))
        except Exception:
            auprcs.append(0.0)

    return float(np.mean(auprcs))


def tune_and_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight,
    cfg: Config,
    fold_seed: int,
) -> dict:
    """Tune C via inner CV, evaluate on outer validation fold."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=fold_seed),
    )
    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_train, y_train, class_weight, cfg, fold_seed,
        ),
        n_trials=cfg.n_optuna_trials,
        show_progress_bar=False,
    )

    best_C = study.best_params["C"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = _build_svm(best_C, class_weight, cfg, fold_seed)
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_val_s)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    auroc = roc_auc_score(y_val, y_prob)
    mean_coefs = _extract_calibrated_coefs(model)

    return {
        "auprc": auprc,
        "auroc": auroc,
        "best_C": best_C,
        "best_inner_auprc": study.best_value,
        "n_nonzero_coefs": int(np.sum(mean_coefs != 0)),
        "coefs": mean_coefs.copy(),
    }


def run_cv(cfg: Config, data: dict, features: dict) -> tuple[pd.DataFrame, list]:
    """Phase 1A: 5-fold CV for all (strategy x weight) combinations."""
    logger.info("=== Phase 1A: Cross-Validation (12 combos) ===")

    df = data["df"]
    protein_panel = features["pruned_proteins"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    logger.info("Panel: %d proteins", len(protein_panel))

    dev_ic_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    dev_ic_labels = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    outer_cv = StratifiedKFold(
        n_splits=cfg.n_outer_folds, shuffle=True, random_state=cfg.split_seed,
    )

    all_results = []
    fold_coefs = []
    total = len(cfg.strategies) * len(cfg.weight_schemes) * cfg.n_outer_folds
    done = 0

    for strategy in cfg.strategies:
        for weight_scheme in cfg.weight_schemes:
            for fold_i, (train_pos, val_pos) in enumerate(
                outer_cv.split(dev_ic_idx, dev_ic_labels)
            ):
                fold_train_ic = dev_ic_idx[train_pos]
                fold_val_ic = dev_ic_idx[val_pos]

                incident_set = set(dev_incident_idx)
                control_set = set(dev_control_idx)

                fold_train_incident = np.array(
                    [i for i in fold_train_ic if i in incident_set]
                )
                fold_train_controls = np.array(
                    [i for i in fold_train_ic if i in control_set]
                )

                train_idx, y_train = get_training_data(
                    strategy, fold_train_incident, fold_train_controls,
                    data["prevalent_idx"],
                )

                val_incident = np.array(
                    [i for i in fold_val_ic if i in incident_set]
                )
                val_controls = np.array(
                    [i for i in fold_val_ic if i in control_set]
                )
                val_idx = np.concatenate([val_incident, val_controls])
                y_val = np.concatenate([
                    np.ones(len(val_incident)),
                    np.zeros(len(val_controls)),
                ]).astype(int)

                X_train = df.loc[train_idx, protein_panel].to_numpy(dtype=float)
                X_val = df.loc[val_idx, protein_panel].to_numpy(dtype=float)

                cw = compute_class_weight(weight_scheme, y_train)
                fold_seed = cfg.split_seed + fold_i

                result = tune_and_evaluate_fold(
                    X_train, y_train, X_val, y_val, cw, cfg, fold_seed,
                )

                coefs = result.pop("coefs")
                fold_coefs.append({
                    "fold": fold_i,
                    "strategy": strategy,
                    "weight_scheme": weight_scheme,
                    "coefs": coefs,
                })

                result["fold"] = fold_i
                result["strategy"] = strategy
                result["weight_scheme"] = weight_scheme
                result["n_train"] = len(y_train)
                result["n_train_pos"] = int(y_train.sum())
                result["n_val"] = len(y_val)
                result["n_val_pos"] = int(y_val.sum())
                all_results.append(result)

                done += 1
                if done % 5 == 0:
                    logger.info("  Progress: %d/%d fold evaluations", done, total)

            mean_auprc = np.mean([
                r["auprc"] for r in all_results
                if r["strategy"] == strategy and r["weight_scheme"] == weight_scheme
            ])
            logger.info(
                "  %s + %s: mean AUPRC=%.4f", strategy, weight_scheme, mean_auprc,
            )

    return pd.DataFrame(all_results), fold_coefs


def select_winner(cv_results: pd.DataFrame) -> tuple[pd.DataFrame, str, str, float]:
    """Pick winning (strategy, weight_scheme) by mean AUPRC."""
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])
        .agg(
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            median_C=("best_C", "median"),
        )
        .reset_index()
        .sort_values("mean_auprc", ascending=False)
    )
    best = summary.iloc[0]
    return summary, best["strategy"], best["weight_scheme"], float(best["median_C"])


# ============================================================================
# Phase 1B: Winner-conditional model-based feature ranking
# ============================================================================


def build_winner_manifest(
    cfg: Config,
    data: dict,
    winner_strategy: str,
    winner_weight: str,
    winner_C: float,
) -> dict:
    """Model-based bootstrap stability conditional on the winning recipe.

    1. Define positive class per winning strategy.
    2. Apply winning class weights to LinearSVC.
    3. Per bootstrap resample: fit weighted SVM, keep top_k by |coef|.
    4. Correlation prune survivors.
    5. Rank by: selection frequency -> mean |coef| -> RRA rank tiebreak.
    6. Output ordered list (winner_order.csv).
    """
    logger.info("=== Phase 1B: Winner-Conditional Feature Ranking ===")
    logger.info("  Strategy: %s, Weight: %s, C: %.6f", winner_strategy, winner_weight, winner_C)

    df = data["df"]
    protein_cols = data["protein_cols"]

    # 1. Define training data per winning strategy (dev set only)
    train_idx, y_train = get_training_data(
        winner_strategy,
        data["dev_incident_idx"],
        data["dev_control_idx"],
        data["prevalent_idx"],
    )

    X_all = df.loc[train_idx, protein_cols].to_numpy(dtype=float)
    y_all = y_train

    cw = compute_class_weight(winner_weight, y_all)

    logger.info("  Training set: %d samples (%d positive)", len(y_all), int(y_all.sum()))

    # 2. Model-based bootstrap stability
    case_idx = np.where(y_all == 1)[0]
    ctrl_idx = np.where(y_all == 0)[0]
    n_cases = len(case_idx)
    n_ctrls = len(ctrl_idx)
    rng = np.random.default_rng(cfg.split_seed + 1000)

    counts: dict[str, int] = {}
    coef_accum: dict[str, list[float]] = {}
    log_rows = []

    for b in range(cfg.n_bootstrap_model):
        boot_cases = rng.choice(case_idx, size=n_cases, replace=True)
        boot_ctrls = rng.choice(ctrl_idx, size=n_ctrls, replace=True)
        boot_idx = np.concatenate([boot_cases, boot_ctrls])

        X_boot = X_all[boot_idx]
        y_boot = y_all[boot_idx]

        scaler = StandardScaler()
        X_boot_s = scaler.fit_transform(X_boot)

        svm = _build_base_svm(winner_C, cw, cfg.max_iter, cfg.split_seed + b)
        try:
            svm.fit(X_boot_s, y_boot)
        except Exception as e:
            logger.warning("  Bootstrap %d: SVM failed (%s), skipping", b, e)
            continue

        coefs = np.abs(svm.coef_.ravel())
        top_k_idx = np.argsort(coefs)[::-1][: cfg.model_top_k]

        for idx in top_k_idx:
            prot = protein_cols[idx]
            counts[prot] = counts.get(prot, 0) + 1
            coef_accum.setdefault(prot, []).append(float(coefs[idx]))

        log_rows.append({
            "resample": b,
            "n_nonzero": int(np.sum(coefs > 0)),
            "top_protein": protein_cols[np.argmax(coefs)],
            "top_coef": float(coefs.max()),
        })

        if (b + 1) % 25 == 0:
            logger.info("  Bootstrap %d/%d complete", b + 1, cfg.n_bootstrap_model)

    n_completed = len(log_rows)
    selection_freq = {p: c / n_completed for p, c in counts.items()}
    mean_abs_coef = {p: float(np.mean(cs)) for p, cs in coef_accum.items()}

    stable_proteins = sorted(
        [p for p, f in selection_freq.items() if f >= cfg.model_stability_threshold],
        key=lambda p: (-selection_freq[p], p),
    )

    logger.info(
        "  Model-based stability: %d proteins >= %.0f%% (%d ever selected)",
        len(stable_proteins),
        cfg.model_stability_threshold * 100,
        len(counts),
    )

    # 3. Correlation prune
    fs_idx = train_idx
    X_fs = df.loc[fs_idx, protein_cols]
    y_fs = pd.Series(y_all, index=fs_idx)

    prune_map, pruned_proteins = prune_correlated_proteins(
        df=X_fs,
        y=y_fs,
        proteins=stable_proteins,
        selection_freq=selection_freq,
        corr_threshold=cfg.corr_threshold,
        corr_method=cfg.corr_method,
    )

    logger.info("  After corr prune: %d proteins", len(pruned_proteins))

    # 4. Load RRA ranks for tiebreaking
    rra_rank = {}
    if cfg.rra_path.exists():
        rra_df = pd.read_csv(cfg.rra_path)
        for rank_i, row in rra_df.iterrows():
            rra_rank[row["protein"]] = rank_i
        logger.info("  Loaded RRA ranks for %d proteins", len(rra_rank))
    else:
        logger.warning("  RRA file not found at %s; no tiebreak available", cfg.rra_path)

    # 5. Rank: selection frequency -> mean |coef| -> RRA rank
    ranking_data = []
    for p in pruned_proteins:
        ranking_data.append({
            "protein": p,
            "selection_freq": selection_freq.get(p, 0.0),
            "mean_abs_coef": mean_abs_coef.get(p, 0.0),
            "rra_rank": rra_rank.get(p, 9999),
        })

    ranking_df = pd.DataFrame(ranking_data).sort_values(
        ["selection_freq", "mean_abs_coef"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Break remaining ties by RRA rank
    ranking_df = ranking_df.sort_values(
        ["selection_freq", "mean_abs_coef", "rra_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    logger.info("  Winner order: %d proteins ranked", len(ranking_df))
    logger.info("  Top 5: %s", ", ".join(ranking_df["protein"].head(5).tolist()))

    return {
        "ranking_df": ranking_df,
        "selection_freq": selection_freq,
        "mean_abs_coef": mean_abs_coef,
        "prune_map": prune_map,
        "stable_proteins": stable_proteins,
        "bootstrap_log": pd.DataFrame(log_rows),
        "manifest_meta": {
            "strategy": winner_strategy,
            "weight_scheme": winner_weight,
            "median_C": winner_C,
            "split_seed": cfg.split_seed,
            "stability_threshold": cfg.model_stability_threshold,
            "corr_threshold": cfg.corr_threshold,
            "n_bootstrap": cfg.n_bootstrap_model,
            "top_k": cfg.model_top_k,
            "ranking_rule": "selection_freq desc -> mean_abs_coef desc -> rra_rank asc",
        },
    }


# ============================================================================
# Final refit + locked test evaluation
# ============================================================================


def final_refit_and_test(
    cfg: Config,
    data: dict,
    shared_features: dict,
    winner_strategy: str,
    winner_weight: str,
    winner_C: float,
) -> dict:
    """Refit winning config on full dev set, evaluate on locked test set.

    Uses the shared (Wald-based) panel for apples-to-apples comparison
    with Phase 1A CV results.
    """
    logger.info("=== Final Model: Locked Test Evaluation ===")

    protein_panel = shared_features["pruned_proteins"]
    df = data["df"]

    train_idx, y_train = get_training_data(
        winner_strategy,
        data["dev_incident_idx"],
        data["dev_control_idx"],
        data["prevalent_idx"],
    )

    X_train = df.loc[train_idx, protein_panel].to_numpy(dtype=float)
    cw = compute_class_weight(winner_weight, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    final_model = _build_svm(winner_C, cw, cfg, cfg.split_seed)
    final_model.fit(X_train_s, y_train)

    mean_coefs = _extract_calibrated_coefs(final_model)
    logger.info(
        "Final model: %d/%d non-zero coefs",
        int(np.sum(mean_coefs != 0)), len(protein_panel),
    )

    # Locked test
    test_idx = np.concatenate([data["test_incident_idx"], data["test_control_idx"]])
    y_test = np.concatenate([
        np.ones(len(data["test_incident_idx"])),
        np.zeros(len(data["test_control_idx"])),
    ]).astype(int)

    X_test_s = scaler.transform(
        df.loc[test_idx, protein_panel].to_numpy(dtype=float)
    )
    y_prob = final_model.predict_proba(X_test_s)[:, 1]

    test_auprc = average_precision_score(y_test, y_prob)
    test_auroc = roc_auc_score(y_test, y_prob)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    y_pred = (y_prob >= best_threshold).astype(int)
    sensitivity = recall_score(y_test, y_pred)
    specificity = float(np.mean(y_pred[y_test == 0] == 0))
    precision = precision_score(y_test, y_pred, zero_division=0)

    logger.info("  AUPRC (primary): %.4f", test_auprc)
    logger.info("  AUROC:           %.4f", test_auroc)

    # Bootstrap CIs
    logger.info("  Computing %d bootstrap CIs...", cfg.n_bootstrap_ci)
    rng = np.random.default_rng(cfg.ci_seed)
    boot_auprc, boot_auroc = [], []
    for _ in range(cfg.n_bootstrap_ci):
        ix = rng.choice(len(y_test), size=len(y_test), replace=True)
        if len(np.unique(y_test[ix])) < 2:
            continue
        boot_auprc.append(average_precision_score(y_test[ix], y_prob[ix]))
        boot_auroc.append(roc_auc_score(y_test[ix], y_prob[ix]))

    auprc_ci = (np.percentile(boot_auprc, 2.5), np.percentile(boot_auprc, 97.5))
    auroc_ci = (np.percentile(boot_auroc, 2.5), np.percentile(boot_auroc, 97.5))

    logger.info("  AUPRC 95%% CI: [%.4f, %.4f]", *auprc_ci)
    logger.info("  AUROC 95%% CI: [%.4f, %.4f]", *auroc_ci)

    coef_df = pd.DataFrame({
        "protein": protein_panel,
        "coefficient": mean_coefs,
        "abs_coef": np.abs(mean_coefs),
        "stability_freq": [
            shared_features["selection_freq"].get(p, 0.0) for p in protein_panel
        ],
    }).sort_values("abs_coef", ascending=False)

    return {
        "test_auprc": test_auprc,
        "test_auroc": test_auroc,
        "auprc_ci": auprc_ci,
        "auroc_ci": auroc_ci,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "threshold": best_threshold,
        "coef_df": coef_df,
        "test_predictions": pd.DataFrame({
            "idx": test_idx, "y_true": y_test, "y_prob": y_prob,
        }),
    }


# ============================================================================
# Reporting
# ============================================================================


def save_results(
    cfg: Config,
    data: dict,
    shared_features: dict,
    cv_results: pd.DataFrame,
    summary: pd.DataFrame,
    winner_strategy: str,
    winner_weight: str,
    winner_C: float,
    final: dict,
    manifest: dict,
    fold_coefs: list,
):
    """Save all outputs to disk."""
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Config
    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in cfg.__dict__.items()
    }
    with open(out / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Phase 1A outputs
    cv_results.to_csv(out / "cv_results.csv", index=False)
    summary.to_csv(out / "strategy_comparison.csv", index=False)
    final["coef_df"].to_csv(out / "feature_coefficients.csv", index=False)
    final["test_predictions"].to_csv(out / "test_predictions.csv", index=False)

    pd.DataFrame({"protein": shared_features["pruned_proteins"]}).to_csv(
        out / "shared_panel.csv", index=False,
    )

    if shared_features["prune_map"]:
        pd.DataFrame([
            {"removed": k, "kept": v}
            for k, v in shared_features["prune_map"].items()
        ]).to_csv(out / "corr_prune_map.csv", index=False)

    if shared_features["bootstrap_log"] is not None:
        shared_features["bootstrap_log"].to_csv(out / "bootstrap_log.csv", index=False)

    pd.DataFrame([
        {"protein": p, "stability_freq": f}
        for p, f in sorted(
            shared_features["selection_freq"].items(), key=lambda x: -x[1],
        )
    ]).to_csv(out / "feature_consistency.csv", index=False)

    # Fold coefficients
    combos_dir = out / "combos"
    combos_dir.mkdir(exist_ok=True)
    combo_buffers: dict[str, list] = {}
    for fc in fold_coefs:
        tag = f"{fc['strategy']}_{fc['weight_scheme']}"
        combo_buffers.setdefault(tag, []).append({
            "fold": fc["fold"],
            **{p: c for p, c in zip(shared_features["pruned_proteins"], fc["coefs"])},
        })
    for tag, rows in combo_buffers.items():
        pd.DataFrame(rows).to_csv(combos_dir / f"coefs_{tag}.csv", index=False)

    # Phase 1B outputs: winner manifest
    manifest_dir = out / "manifest"
    manifest_dir.mkdir(exist_ok=True)

    manifest["ranking_df"].to_csv(manifest_dir / "winner_order.csv", index=False)
    manifest["bootstrap_log"].to_csv(manifest_dir / "model_bootstrap_log.csv", index=False)

    with open(manifest_dir / "manifest_meta.json", "w") as f:
        json.dump(manifest["manifest_meta"], f, indent=2)

    if manifest["prune_map"]:
        pd.DataFrame([
            {"removed": k, "kept": v}
            for k, v in manifest["prune_map"].items()
        ]).to_csv(manifest_dir / "model_corr_prune_map.csv", index=False)

    # Also copy winner_order.csv to top level for easy sweep consumption
    manifest["ranking_df"][["protein", "rank"]].to_csv(
        out / "winner_order.csv", index=False,
    )

    # Paired comparison with LR results
    lr_path = Path("results/incident_validation/strategy_comparison.csv")
    if lr_path.exists():
        lr = pd.read_csv(lr_path).rename(columns={
            "mean_auprc": "lr_mean_auprc", "std_auprc": "lr_std_auprc",
            "mean_auroc": "lr_mean_auroc", "std_auroc": "lr_std_auroc",
        })
        svm = summary.rename(columns={
            "mean_auprc": "svm_mean_auprc", "std_auprc": "svm_std_auprc",
            "mean_auroc": "svm_mean_auroc", "std_auroc": "svm_std_auroc",
        })
        paired = svm.merge(
            lr[["strategy", "weight_scheme", "lr_mean_auprc",
                "lr_std_auprc", "lr_mean_auroc", "lr_std_auroc"]],
            on=["strategy", "weight_scheme"], how="left",
        )
        paired["delta_auprc"] = paired["svm_mean_auprc"] - paired["lr_mean_auprc"]
        paired["delta_auroc"] = paired["svm_mean_auroc"] - paired["lr_mean_auroc"]
        paired.to_csv(out / "paired_comparison.csv", index=False)

    # Summary report
    _write_report(cfg, data, shared_features, summary, winner_strategy,
                  winner_weight, winner_C, final, manifest, out)

    logger.info("All results saved to %s", out)


def _write_report(cfg, data, shared_features, summary, winner_strategy,
                  winner_weight, winner_C, final, manifest, out):
    """Write human-readable summary report."""
    ranking_df = manifest["ranking_df"]
    meta = manifest["manifest_meta"]

    lines = [
        "# SVM Incident Validation Report",
        "",
        "## Model",
        "- LinearSVC + CalibratedClassifierCV (sigmoid, cv=5)",
        f"- max_iter: {cfg.max_iter}",
        "",
        "## Dataset",
        f"- Total samples: {len(data['df'])}",
        f"- Dev incident: {len(data['dev_incident_idx'])}",
        f"- Dev controls: {len(data['dev_control_idx'])}",
        f"- Prevalent (training only): {len(data['prevalent_idx'])}",
        f"- Test incident: {len(data['test_incident_idx'])}",
        f"- Test controls: {len(data['test_control_idx'])}",
        "",
        "## Phase 1A: Recipe Selection",
        "",
        "### Shared Feature Selection (Wald-based)",
        f"- Method: Bootstrap stability ({cfg.screen_method} statistic)",
        f"- Resamples: {cfg.n_bootstrap}, top {cfg.bootstrap_top_k} per resample",
        f"- Stability threshold: {cfg.stability_threshold*100:.0f}%",
        f"- Stable proteins (pre-prune): {len(shared_features['stable_proteins'])}",
        f"- Correlation threshold: |r| > {cfg.corr_threshold}",
        f"- **Shared panel: {len(shared_features['pruned_proteins'])} proteins**",
        "",
        "### Strategy Comparison (5-fold CV, mean AUPRC)",
        "",
        summary.to_markdown(index=False),
        "",
        f"**Winner: {winner_strategy} + {winner_weight}** "
        f"(mean AUPRC = {summary.iloc[0]['mean_auprc']:.4f}, median C = {winner_C:.6f})",
        "",
        "## Locked Test Set Results",
        "",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
        f"| **AUPRC** (primary) | {final['test_auprc']:.4f} "
        f"| [{final['auprc_ci'][0]:.4f}, {final['auprc_ci'][1]:.4f}] |",
        f"| AUROC | {final['test_auroc']:.4f} "
        f"| [{final['auroc_ci'][0]:.4f}, {final['auroc_ci'][1]:.4f}] |",
        f"| Sensitivity | {final['sensitivity']:.4f} | - |",
        f"| Specificity | {final['specificity']:.4f} | - |",
        f"| Precision | {final['precision']:.4f} | - |",
        "",
        f"Threshold: {final['threshold']:.4f} (Youden's J)",
        "",
        "## Phase 1B: Winner-Conditional Feature Ranking",
        "",
        f"- Strategy: {meta['strategy']}",
        f"- Weight scheme: {meta['weight_scheme']}",
        f"- Median C: {meta['median_C']:.6f}",
        f"- Model-based bootstrap resamples: {meta['n_bootstrap']}",
        f"- Stability threshold: {meta['stability_threshold']*100:.0f}%",
        f"- Correlation threshold: |r| > {meta['corr_threshold']}",
        f"- Ranking rule: {meta['ranking_rule']}",
        f"- **Winner-ranked panel: {len(ranking_df)} proteins**",
        "",
        "### Top 20 (winner_order.csv)",
        "",
        ranking_df.head(20).to_markdown(index=False),
        "",
        "### Top Features (final model, shared panel)",
        "",
        final["coef_df"].head(20).to_markdown(index=False),
        "",
        "## Next Step",
        "",
        "Use `winner_order.csv` prefixes for the panel-size sweep:",
        "```",
        f"strategy: {winner_strategy}",
        f"weight_scheme: {winner_weight}",
        "model: LinSVM_cal",
        "panel_sizes: 4..N (prefixes of winner_order.csv)",
        "tune: C only",
        "```",
        "",
        "## Hyperparameters",
        f"- Optuna trials: {cfg.n_optuna_trials}",
        f"- Inner CV folds: {cfg.n_inner_folds}",
        f"- Calibration: {cfg.calibration_method}, cv={cfg.calibration_cv}",
        f"- Bootstrap CIs: {cfg.n_bootstrap_ci}",
        "",
    ]

    with open(out / "summary_report.md", "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="SVM incident validation")
    parser.add_argument("--smoke", action="store_true", help="Fast sanity check")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(smoke=args.smoke)
    if args.data_path:
        cfg.data_path = Path(args.data_path)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("SVM Incident Validation (Phase 1A + 1B)")
    logger.info("  Model: LinearSVC + CalibratedClassifierCV")
    logger.info("  Smoke: %s", cfg.smoke)
    logger.info("  Output: %s", cfg.output_dir)
    logger.info("=" * 60)

    t0 = time.time()

    # Data
    data = load_and_split(cfg)

    # Phase 1A: shared panel + 12-way CV
    shared_features = select_features_shared(cfg, data)
    cv_results, fold_coefs = run_cv(cfg, data, shared_features)
    summary, winner_strategy, winner_weight, winner_C = select_winner(cv_results)

    logger.info("Winner: %s + %s (C=%.6f)", winner_strategy, winner_weight, winner_C)

    # Phase 1B: winner-conditional manifest
    manifest = build_winner_manifest(cfg, data, winner_strategy, winner_weight, winner_C)

    # Final evaluation on locked test
    final = final_refit_and_test(
        cfg, data, shared_features, winner_strategy, winner_weight, winner_C,
    )

    # Save everything
    save_results(
        cfg, data, shared_features, cv_results, summary,
        winner_strategy, winner_weight, winner_C,
        final, manifest, fold_coefs,
    )

    elapsed = time.time() - t0
    logger.info("Total runtime: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
