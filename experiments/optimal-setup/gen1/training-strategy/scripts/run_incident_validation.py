#!/usr/bin/env python3
"""
Incident validation pipeline: 3 training strategies × 4 class-weight schemes.

Workflow:
  1. Load data → define groups (incident / prevalent / control)
  2. Create locked 20% test set (incident + controls, stratified by sex)
  3. Bootstrap stability feature selection on dev incident + controls (Wald, 100 resamples)
  4. Correlation pruning (|r| > 0.85, keep more stable feature)
  5. 5-fold CV on dev set (folds defined on incident + controls)
     - For each (strategy × weight_scheme):
       inner 3-fold Optuna tunes C + l1_ratio → AUPRC
       outer fold evaluates on incident + controls
  6. Select best (strategy, weight_scheme) by mean AUPRC
  7. Final refit on full dev set with winning config
  8. Evaluate on locked test set

Usage:
  cd cel-risk
  python analysis/scripts/run_incident_validation.py
  python analysis/scripts/run_incident_validation.py --smoke   # fast sanity check
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --- ced_ml imports (data loading, schema, feature selection) ---
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import (
    CONTROL_LABEL,
    ID_COL,
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
# Force unbuffered stderr so LSF bpeek shows live progress
import functools
logging.getLogger().handlers[0].stream = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Config:
    data_path: Path = Path("data/Celiac_dataset_proteomics_w_demo.parquet")
    output_dir: Path = Path("results/incident_validation")

    # Splitting
    test_frac: float = 0.20
    n_outer_folds: int = 5
    split_seed: int = 42

    # Bootstrap stability
    n_bootstrap: int = 100
    bootstrap_top_k: int = 200
    stability_threshold: float = 0.70
    screen_method: str = "wald"

    # Correlation pruning
    corr_threshold: float = 0.85
    corr_method: str = "spearman"

    # Optuna
    n_optuna_trials: int = 50
    n_inner_folds: int = 3

    # Elastic net
    max_iter: int = 2_000
    solver: str = "saga"

    # Evaluation
    n_bootstrap_ci: int = 2_000
    ci_seed: int = 99

    # Strategies and weights
    strategies: list[str] = field(
        default_factory=lambda: ["incident_only", "incident_prevalent", "prevalent_only"]
    )
    weight_schemes: list[str] = field(
        default_factory=lambda: ["none", "balanced", "sqrt", "log"]
    )

    smoke: bool = False

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        if self.smoke:
            self.n_bootstrap = 10
            self.bootstrap_top_k = 50
            self.n_optuna_trials = 5
            self.n_bootstrap_ci = 100


# ============================================================================
# Data preparation
# ============================================================================


def load_and_split(cfg: Config) -> dict:
    """Load data, define groups, create locked test/dev split."""
    logger.info("Loading data from %s", cfg.data_path)
    df = read_proteomics_file(str(cfg.data_path))
    logger.info("Loaded %d samples, %d columns", len(df), len(df.columns))

    # Identify protein columns
    protein_cols = [c for c in df.columns if c.endswith(PROTEIN_SUFFIX)]
    logger.info("Found %d protein columns", len(protein_cols))

    # Define groups
    incident_mask = df[TARGET_COL] == INCIDENT_LABEL
    prevalent_mask = df[TARGET_COL] == PREVALENT_LABEL
    control_mask = df[TARGET_COL] == CONTROL_LABEL

    n_inc = incident_mask.sum()
    n_prev = prevalent_mask.sum()
    n_ctrl = control_mask.sum()
    logger.info("Groups: %d incident, %d prevalent, %d controls", n_inc, n_prev, n_ctrl)

    # Locked test set: 20% of incident + 20% of controls, stratified by sex
    ic_df = df[incident_mask | control_mask].copy()
    ic_df["_binary"] = (ic_df[TARGET_COL] == INCIDENT_LABEL).astype(int)

    # Stratification key: binary outcome × sex
    sex_col = "sex"
    if sex_col in ic_df.columns:
        ic_df["_strat"] = ic_df["_binary"].astype(str) + "_" + ic_df[sex_col].astype(str)
    else:
        logger.warning("'sex' column not found; stratifying by outcome only")
        ic_df["_strat"] = ic_df["_binary"].astype(str)

    from sklearn.model_selection import train_test_split

    dev_idx, test_idx = train_test_split(
        ic_df.index,
        test_size=cfg.test_frac,
        stratify=ic_df["_strat"],
        random_state=cfg.split_seed,
    )

    # Build index sets
    test_set = set(test_idx)
    dev_incident_idx = np.array([i for i in df.index[incident_mask] if i not in test_set])
    dev_control_idx = np.array([i for i in df.index[control_mask] if i not in test_set])
    test_incident_idx = np.array([i for i in df.index[incident_mask] if i in test_set])
    test_control_idx = np.array([i for i in df.index[control_mask] if i in test_set])
    prevalent_idx = np.array(df.index[prevalent_mask])

    logger.info(
        "Test set: %d incident + %d controls = %d",
        len(test_incident_idx), len(test_control_idx),
        len(test_incident_idx) + len(test_control_idx),
    )
    logger.info(
        "Dev set: %d incident + %d controls + %d prevalent",
        len(dev_incident_idx), len(dev_control_idx), len(prevalent_idx),
    )

    return {
        "df": df,
        "protein_cols": protein_cols,
        "dev_incident_idx": dev_incident_idx,
        "dev_control_idx": dev_control_idx,
        "test_incident_idx": test_incident_idx,
        "test_control_idx": test_control_idx,
        "prevalent_idx": prevalent_idx,
    }


# ============================================================================
# Feature selection
# ============================================================================


def run_feature_selection(cfg: Config, data: dict) -> dict:
    """Bootstrap stability on dev incident + controls, then correlation prune."""
    df = data["df"]
    protein_cols = data["protein_cols"]
    dev_ic_idx = np.concatenate([data["dev_incident_idx"], data["dev_control_idx"]])

    X_fs = df.loc[dev_ic_idx]
    y_fs = (X_fs[TARGET_COL] == INCIDENT_LABEL).astype(int).to_numpy()

    logger.info("=== Feature Selection ===")
    logger.info(
        "Bootstrap stability: %d resamples, top %d per resample, threshold %.0f%%",
        cfg.n_bootstrap, cfg.bootstrap_top_k, cfg.stability_threshold * 100,
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

    # Correlation pruning
    logger.info(
        "Correlation pruning: |r| > %.2f (%s), keep more stable feature",
        cfg.corr_threshold, cfg.corr_method,
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


# ============================================================================
# Class weighting
# ============================================================================


def compute_class_weight(scheme: str, y: np.ndarray) -> dict | str | None:
    """Compute sklearn-compatible class_weight for a given scheme.

    Returns None, "balanced", or {0: w0, 1: w1}.
    """
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
# Training (Optuna + Elastic Net)
# ============================================================================


def _downsample_controls(
    X: np.ndarray, y: np.ndarray, max_ratio: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample controls to max_ratio:1 for faster inner CV fitting."""
    case_mask = y == 1
    ctrl_mask = y == 0
    n_cases = int(case_mask.sum())
    n_ctrls = int(ctrl_mask.sum())
    target_ctrls = n_cases * max_ratio

    if max_ratio <= 0 or n_ctrls <= target_ctrls:
        return X, y

    ctrl_idx = np.where(ctrl_mask)[0]
    keep_ctrl = rng.choice(ctrl_idx, size=target_ctrls, replace=False)
    keep = np.sort(np.concatenate([np.where(case_mask)[0], keep_ctrl]))
    return X[keep], y[keep]


def _optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight,
    n_inner_folds: int,
    max_iter: int,
    solver: str,
    seed: int,
    inner_ctrl_ratio: int = 0,
) -> float:
    """Optuna objective: mean AUPRC across inner CV folds."""
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)

    rng = np.random.default_rng(seed + trial.number)

    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    auprcs = []

    for train_ix, val_ix in inner_cv.split(X_train, y_train):
        X_tr, X_va = X_train[train_ix], X_train[val_ix]
        y_tr, y_va = y_train[train_ix], y_train[val_ix]

        # Downsample controls in inner training fold for speed
        X_tr_ds, y_tr_ds = _downsample_controls(X_tr, y_tr, inner_ctrl_ratio, rng)

        # Scale inside inner fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_ds)
        X_va_s = scaler.transform(X_va)

        model = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=seed,
        )

        try:
            model.fit(X_tr_s, y_tr_ds)
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
    """Tune hyperparams via inner CV, evaluate on outer validation fold."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=fold_seed),
    )
    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_train, y_train, class_weight,
            cfg.n_inner_folds, cfg.max_iter, cfg.solver, fold_seed,
        ),
        n_trials=cfg.n_optuna_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    best_C = best["C"]
    best_l1 = best["l1_ratio"]

    # Refit on full outer training fold
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        C=best_C,
        l1_ratio=best_l1,
        class_weight=class_weight,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        random_state=fold_seed,
    )
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_val_s)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    auroc = roc_auc_score(y_val, y_prob)

    return {
        "auprc": auprc,
        "auroc": auroc,
        "best_C": best_C,
        "best_l1_ratio": best_l1,
        "best_inner_auprc": study.best_value,
        "n_nonzero_coefs": int(np.sum(model.coef_ != 0)),
        "coefs": model.coef_.ravel().copy(),
    }


def get_training_indices(
    strategy: str,
    fold_train_incident: np.ndarray,
    fold_train_controls: np.ndarray,
    prevalent_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, binary labels) for a given training strategy."""
    if strategy == "incident_only":
        idx = np.concatenate([fold_train_incident, fold_train_controls])
        y = np.concatenate([np.ones(len(fold_train_incident)), np.zeros(len(fold_train_controls))])
    elif strategy == "incident_prevalent":
        idx = np.concatenate([fold_train_incident, prevalent_idx, fold_train_controls])
        y = np.concatenate([
            np.ones(len(fold_train_incident)),
            np.ones(len(prevalent_idx)),
            np.zeros(len(fold_train_controls)),
        ])
    elif strategy == "prevalent_only":
        idx = np.concatenate([prevalent_idx, fold_train_controls])
        y = np.concatenate([np.ones(len(prevalent_idx)), np.zeros(len(fold_train_controls))])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return idx.astype(int), y.astype(int)


# ============================================================================
# Cross-validation loop
# ============================================================================


def run_cv(cfg: Config, data: dict, features: dict) -> pd.DataFrame:
    """Run 5-fold CV for all (strategy × weight) combinations."""
    df = data["df"]
    protein_panel = features["pruned_proteins"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    logger.info("=== Cross-Validation ===")
    logger.info("Panel: %d proteins", len(protein_panel))
    logger.info("Strategies: %s", cfg.strategies)
    logger.info("Weight schemes: %s", cfg.weight_schemes)

    # CV folds defined on incident + controls
    dev_ic_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    dev_ic_labels = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    outer_cv = StratifiedKFold(
        n_splits=cfg.n_outer_folds, shuffle=True, random_state=cfg.split_seed,
    )

    all_results = []
    total = len(cfg.strategies) * len(cfg.weight_schemes) * cfg.n_outer_folds
    done = 0

    for strategy in cfg.strategies:
        for weight_scheme in cfg.weight_schemes:
            fold_results = []

            for fold_i, (train_pos, val_pos) in enumerate(
                outer_cv.split(dev_ic_idx, dev_ic_labels)
            ):
                # Map positional indices back to DataFrame indices
                fold_train_ic = dev_ic_idx[train_pos]
                fold_val_ic = dev_ic_idx[val_pos]

                # Separate incident and controls in training fold
                fold_train_incident = np.array(
                    [i for i in fold_train_ic if i in set(dev_incident_idx)]
                )
                fold_train_controls = np.array(
                    [i for i in fold_train_ic if i in set(dev_control_idx)]
                )

                # Training data depends on strategy
                train_idx, y_train = get_training_indices(
                    strategy, fold_train_incident, fold_train_controls, prevalent_idx,
                )

                # Validation: always incident + controls
                val_idx = fold_val_ic
                y_val = np.concatenate([
                    np.ones(int(np.isin(fold_val_ic, dev_incident_idx).sum())),
                    np.zeros(int(np.isin(fold_val_ic, dev_control_idx).sum())),
                ])
                # Reorder val_idx to match y_val (incident first, then controls)
                val_incident = np.array([i for i in fold_val_ic if i in set(dev_incident_idx)])
                val_controls = np.array([i for i in fold_val_ic if i in set(dev_control_idx)])
                val_idx = np.concatenate([val_incident, val_controls])
                y_val = np.concatenate([
                    np.ones(len(val_incident)),
                    np.zeros(len(val_controls)),
                ]).astype(int)

                # Extract features
                X_train = df.loc[train_idx, protein_panel].to_numpy(dtype=float)
                X_val = df.loc[val_idx, protein_panel].to_numpy(dtype=float)

                # Compute class weight
                cw = compute_class_weight(weight_scheme, y_train)

                # Tune and evaluate
                fold_seed = cfg.split_seed + fold_i
                result = tune_and_evaluate_fold(
                    X_train, y_train, X_val, y_val, cw, cfg, fold_seed,
                )
                result["fold"] = fold_i
                result["strategy"] = strategy
                result["weight_scheme"] = weight_scheme
                result["n_train"] = len(y_train)
                result["n_train_pos"] = int(y_train.sum())
                result["n_val"] = len(y_val)
                result["n_val_pos"] = int(y_val.sum())
                fold_results.append(result)

                done += 1
                if done % 5 == 0:
                    logger.info("  Progress: %d/%d fold evaluations", done, total)

            # Summarize this (strategy, weight) combo
            mean_auprc = np.mean([r["auprc"] for r in fold_results])
            mean_auroc = np.mean([r["auroc"] for r in fold_results])
            logger.info(
                "  %s + %s: mean AUPRC=%.4f, mean AUROC=%.4f",
                strategy, weight_scheme, mean_auprc, mean_auroc,
            )

            all_results.extend(fold_results)

    # Extract per-fold coefficients before DataFrame conversion (arrays can't go in flat CSV)
    fold_coefs = []
    for r in all_results:
        coefs = r.pop("coefs")
        fold_coefs.append({
            "fold": r["fold"],
            "strategy": r["strategy"],
            "weight_scheme": r["weight_scheme"],
            "coefs": coefs,
        })

    return pd.DataFrame(all_results), fold_coefs


# ============================================================================
# Final refit and test evaluation
# ============================================================================


def final_refit_and_test(
    cfg: Config,
    data: dict,
    features: dict,
    cv_results: pd.DataFrame,
) -> dict:
    """Refit winning config on full dev set, evaluate on locked test set."""
    logger.info("=== Final Model Selection ===")

    # Find best (strategy, weight_scheme) by mean AUPRC
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])
        .agg(
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            median_C=("best_C", "median"),
            median_l1=("best_l1_ratio", "median"),
        )
        .reset_index()
        .sort_values("mean_auprc", ascending=False)
    )

    logger.info("\n%s", summary.to_string(index=False))

    best_row = summary.iloc[0]
    best_strategy = best_row["strategy"]
    best_weight = best_row["weight_scheme"]
    best_C = float(best_row["median_C"])
    best_l1 = float(best_row["median_l1"])

    logger.info(
        "\nBest: %s + %s (mean AUPRC=%.4f)",
        best_strategy, best_weight, best_row["mean_auprc"],
    )
    logger.info("  Median hyperparams: C=%.6f, l1_ratio=%.4f", best_C, best_l1)

    # Re-run feature selection on full dev set (same as before, for reproducibility)
    protein_panel = features["pruned_proteins"]
    df = data["df"]

    # Build full dev training set
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    train_idx, y_train = get_training_indices(
        best_strategy, dev_incident_idx, dev_control_idx, prevalent_idx,
    )

    X_train = df.loc[train_idx, protein_panel].to_numpy(dtype=float)
    cw = compute_class_weight(best_weight, y_train)

    # Scale and fit final model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    final_model = LogisticRegression(
        C=best_C,
        l1_ratio=best_l1,
        class_weight=cw,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        random_state=cfg.split_seed,
    )
    final_model.fit(X_train_s, y_train)

    logger.info(
        "Final model: %d/%d non-zero coefficients",
        int(np.sum(final_model.coef_ != 0)), len(protein_panel),
    )

    # === Locked test evaluation ===
    logger.info("=== Locked Test Evaluation ===")

    test_incident_idx = data["test_incident_idx"]
    test_control_idx = data["test_control_idx"]
    test_idx = np.concatenate([test_incident_idx, test_control_idx])
    y_test = np.concatenate([
        np.ones(len(test_incident_idx)),
        np.zeros(len(test_control_idx)),
    ]).astype(int)

    X_test = df.loc[test_idx, protein_panel].to_numpy(dtype=float)
    X_test_s = scaler.transform(X_test)
    y_prob = final_model.predict_proba(X_test_s)[:, 1]

    # Metrics
    test_auprc = average_precision_score(y_test, y_prob)
    test_auroc = roc_auc_score(y_test, y_prob)

    # Threshold at Youden's J
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    best_thresh_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_thresh_idx]

    y_pred = (y_prob >= best_threshold).astype(int)
    sensitivity = recall_score(y_test, y_pred)
    specificity = float(np.mean(y_pred[y_test == 0] == 0))
    precision = precision_score(y_test, y_pred, zero_division=0)

    logger.info("  AUPRC (primary): %.4f", test_auprc)
    logger.info("  AUROC:           %.4f", test_auroc)
    logger.info("  Threshold (Youden J): %.4f", best_threshold)
    logger.info("  Sensitivity:     %.4f", sensitivity)
    logger.info("  Specificity:     %.4f", specificity)
    logger.info("  Precision:       %.4f", precision)

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

    # Feature coefficients
    coef_df = pd.DataFrame({
        "protein": protein_panel,
        "coefficient": final_model.coef_.ravel(),
        "abs_coef": np.abs(final_model.coef_.ravel()),
        "stability_freq": [features["selection_freq"].get(p, 0.0) for p in protein_panel],
    }).sort_values("abs_coef", ascending=False)

    nonzero_df = coef_df[coef_df["coefficient"] != 0].copy()

    return {
        "summary": summary,
        "best_strategy": best_strategy,
        "best_weight": best_weight,
        "best_C": best_C,
        "best_l1_ratio": best_l1,
        "test_auprc": test_auprc,
        "test_auroc": test_auroc,
        "auprc_ci": auprc_ci,
        "auroc_ci": auroc_ci,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "threshold": best_threshold,
        "coef_df": coef_df,
        "nonzero_features": nonzero_df,
        "final_model": final_model,
        "scaler": scaler,
    }


# ============================================================================
# Reporting
# ============================================================================


def save_results(cfg: Config, data: dict, features: dict, cv_results: pd.DataFrame, final: dict, fold_coefs: list):
    """Save all outputs to disk."""
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. CV results
    cv_results.to_csv(out / "cv_results.csv", index=False)
    logger.info("Saved: %s", out / "cv_results.csv")

    # Per-fold coefficients (one row per protein per fold per combo)
    protein_panel = features["pruned_proteins"]
    coef_rows = []
    for entry in fold_coefs:
        for i, p in enumerate(protein_panel):
            coef_rows.append({
                "fold": entry["fold"],
                "strategy": entry["strategy"],
                "weight_scheme": entry["weight_scheme"],
                "protein": p,
                "coefficient": entry["coefs"][i],
            })
    pd.DataFrame(coef_rows).to_csv(out / "fold_coefficients.csv", index=False)
    logger.info("Saved: %s", out / "fold_coefficients.csv")

    # 2. Strategy comparison summary
    final["summary"].to_csv(out / "strategy_comparison.csv", index=False)
    logger.info("Saved: %s", out / "strategy_comparison.csv")

    # 3. Feature panel
    panel_df = pd.DataFrame({
        "protein": features["pruned_proteins"],
        "stability_freq": [
            features["selection_freq"].get(p, 0.0) for p in features["pruned_proteins"]
        ],
    })
    panel_df.to_csv(out / "feature_panel.csv", index=False)
    logger.info("Saved: %s", out / "feature_panel.csv")

    # 4. Bootstrap stability log
    features["bootstrap_log"].to_csv(out / "bootstrap_log.csv", index=False)

    # 5. Correlation prune map
    pd.DataFrame(features["prune_map"]).to_csv(out / "corr_prune_map.csv", index=False)

    # 6. Feature coefficients
    final["coef_df"].to_csv(out / "feature_coefficients.csv", index=False)
    logger.info("Saved: %s", out / "feature_coefficients.csv")

    # 7. Test predictions
    test_incident_idx = data["test_incident_idx"]
    test_control_idx = data["test_control_idx"]
    test_idx = np.concatenate([test_incident_idx, test_control_idx])
    y_test = np.concatenate([
        np.ones(len(test_incident_idx)),
        np.zeros(len(test_control_idx)),
    ]).astype(int)

    X_test = data["df"].loc[test_idx, features["pruned_proteins"]].to_numpy(dtype=float)
    X_test_s = final["scaler"].transform(X_test)
    y_prob = final["final_model"].predict_proba(X_test_s)[:, 1]

    preds_df = pd.DataFrame({
        "eid": data["df"].loc[test_idx, ID_COL].values if ID_COL in data["df"].columns else test_idx,
        "y_true": y_test,
        "y_prob": y_prob,
        "y_pred": (y_prob >= final["threshold"]).astype(int),
    })
    preds_df.to_csv(out / "test_predictions.csv", index=False)

    # 8. Summary report
    report = _build_report(cfg, data, features, cv_results, final)
    (out / "summary_report.md").write_text(report)
    logger.info("Saved: %s", out / "summary_report.md")

    # 9. Config snapshot
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()}
    (out / "config.json").write_text(json.dumps(config_dict, indent=2))

    logger.info("All outputs saved to %s", out)


def _build_report(
    cfg: Config, data: dict, features: dict, cv_results: pd.DataFrame, final: dict,
) -> str:
    """Build markdown summary report."""
    lines = [
        "# Incident Validation Report",
        "",
        "## Dataset",
        f"- Total samples: {len(data['df'])}",
        f"- Dev incident: {len(data['dev_incident_idx'])}",
        f"- Dev controls: {len(data['dev_control_idx'])}",
        f"- Prevalent (training only): {len(data['prevalent_idx'])}",
        f"- Test incident: {len(data['test_incident_idx'])}",
        f"- Test controls: {len(data['test_control_idx'])}",
        "",
        "## Feature Selection",
        f"- Method: Bootstrap stability ({cfg.screen_method} statistic)",
        f"- Resamples: {cfg.n_bootstrap}, top {cfg.bootstrap_top_k} per resample",
        f"- Stability threshold: {cfg.stability_threshold:.0%}",
        f"- Stable proteins (pre-prune): {len(features['stable_proteins'])}",
        f"- Correlation threshold: |r| > {cfg.corr_threshold}",
        f"- **Final panel: {len(features['pruned_proteins'])} proteins**",
        "",
        "## Strategy Comparison (5-fold CV, mean AUPRC)",
        "",
        final["summary"].to_markdown(index=False),
        "",
        f"**Best: {final['best_strategy']} + {final['best_weight']}** "
        f"(mean AUPRC = {final['summary'].iloc[0]['mean_auprc']:.4f})",
        "",
        "## Locked Test Set Results",
        "",
        f"| Metric | Value | 95% CI |",
        f"|--------|-------|--------|",
        f"| **AUPRC** (primary) | {final['test_auprc']:.4f} | [{final['auprc_ci'][0]:.4f}, {final['auprc_ci'][1]:.4f}] |",
        f"| AUROC | {final['test_auroc']:.4f} | [{final['auroc_ci'][0]:.4f}, {final['auroc_ci'][1]:.4f}] |",
        f"| Sensitivity | {final['sensitivity']:.4f} | - |",
        f"| Specificity | {final['specificity']:.4f} | - |",
        f"| Precision | {final['precision']:.4f} | - |",
        "",
        f"Threshold: {final['threshold']:.4f} (Youden's J)",
        "",
        "## Final Model",
        f"- C = {final['best_C']:.6f}",
        f"- l1_ratio = {final['best_l1_ratio']:.4f}",
        f"- Non-zero features: {len(final['nonzero_features'])}/{len(features['pruned_proteins'])}",
        "",
        "### Top Features (by |coefficient|)",
        "",
        final["nonzero_features"].head(20).to_markdown(index=False),
        "",
        "## Hyperparameters",
        f"- Optuna trials: {cfg.n_optuna_trials}",
        f"- Inner CV folds: {cfg.n_inner_folds}",
        f"- Solver: {cfg.solver}, max_iter: {cfg.max_iter}",
        f"- Bootstrap CIs: {cfg.n_bootstrap_ci}",
        "",
    ]

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def _save_features(cfg: Config, data: dict, features: dict):
    """Save feature selection artifacts to output dir for downstream jobs."""
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Feature panel
    panel_df = pd.DataFrame({
        "protein": features["pruned_proteins"],
        "stability_freq": [features["selection_freq"].get(p, 0.0) for p in features["pruned_proteins"]],
    })
    panel_df.to_csv(out / "feature_panel.csv", index=False)

    # Bootstrap log
    features["bootstrap_log"].to_csv(out / "bootstrap_log.csv", index=False)

    # Prune map
    prune_df = pd.DataFrame(features["prune_map"])
    prune_df.to_csv(out / "corr_prune_map.csv", index=False)

    # Config
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()}
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    logger.info("Feature selection artifacts saved to %s", out)


def _load_features(cfg: Config) -> dict:
    """Load pre-computed feature selection artifacts from output dir."""
    out = cfg.output_dir
    panel_df = pd.read_csv(out / "feature_panel.csv")
    pruned_proteins = panel_df["protein"].tolist()
    selection_freq = dict(zip(panel_df["protein"], panel_df["stability_freq"]))

    # Bootstrap log and prune map are optional for CV but load for completeness
    bootstrap_log = pd.read_csv(out / "bootstrap_log.csv") if (out / "bootstrap_log.csv").exists() else pd.DataFrame()
    prune_map = pd.read_csv(out / "corr_prune_map.csv").to_dict("records") if (out / "corr_prune_map.csv").exists() else []

    logger.info("Loaded feature panel: %d proteins from %s", len(pruned_proteins), out / "feature_panel.csv")

    return {
        "stable_proteins": pruned_proteins,  # approximate; full list not saved
        "pruned_proteins": pruned_proteins,
        "selection_freq": selection_freq,
        "bootstrap_log": bootstrap_log,
        "prune_map": prune_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Incident validation pipeline")
    parser.add_argument("--smoke", action="store_true", help="Reduced params for quick test")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--data-path", type=str, default=None, help="Override data path")

    # Parallel mode flags
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "features", "cv", "aggregate"],
        help="Pipeline phase: all (sequential), features (step 1-2), cv (step 3), aggregate (step 4-5)",
    )
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy for --phase=cv")
    parser.add_argument("--weight-scheme", type=str, default=None, help="Single weight scheme for --phase=cv")

    args = parser.parse_args()

    cfg = Config(smoke=args.smoke)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.data_path:
        cfg.data_path = Path(args.data_path)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("INCIDENT VALIDATION PIPELINE (phase=%s)", args.phase)
    logger.info("=" * 60)
    if cfg.smoke:
        logger.info("*** SMOKE TEST MODE ***")

    if args.phase == "all":
        # Original sequential mode
        data = load_and_split(cfg)
        features = run_feature_selection(cfg, data)
        cv_results, fold_coefs = run_cv(cfg, data, features)
        final = final_refit_and_test(cfg, data, features, cv_results)
        save_results(cfg, data, features, cv_results, final, fold_coefs)

    elif args.phase == "features":
        # Step 1-2: Load data + feature selection, save artifacts
        data = load_and_split(cfg)
        features = run_feature_selection(cfg, data)
        _save_features(cfg, data, features)

    elif args.phase == "cv":
        # Step 3: Run CV for a single (strategy, weight) combo
        if not args.strategy or not args.weight_scheme:
            parser.error("--phase=cv requires --strategy and --weight-scheme")

        cfg.strategies = [args.strategy]
        cfg.weight_schemes = [args.weight_scheme]

        data = load_and_split(cfg)
        features = _load_features(cfg)
        cv_results, fold_coefs = run_cv(cfg, data, features)

        # Save per-combo results to a unique file
        combo_tag = f"{args.strategy}_{args.weight_scheme}"
        combo_dir = cfg.output_dir / "combos"
        combo_dir.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(combo_dir / f"cv_{combo_tag}.csv", index=False)

        # Save fold coefficients
        protein_panel = features["pruned_proteins"]
        coef_rows = []
        for entry in fold_coefs:
            for i, p in enumerate(protein_panel):
                coef_rows.append({
                    "fold": entry["fold"],
                    "strategy": entry["strategy"],
                    "weight_scheme": entry["weight_scheme"],
                    "protein": p,
                    "coefficient": entry["coefs"][i],
                })
        pd.DataFrame(coef_rows).to_csv(combo_dir / f"coefs_{combo_tag}.csv", index=False)

        logger.info("Saved combo results to %s", combo_dir)

    elif args.phase == "aggregate":
        # Step 4-5: Merge combo results, final refit, test eval, save
        combo_dir = cfg.output_dir / "combos"
        if not combo_dir.exists():
            raise FileNotFoundError(f"No combo results at {combo_dir}")

        cv_parts = sorted(combo_dir.glob("cv_*.csv"))
        coef_parts = sorted(combo_dir.glob("coefs_*.csv"))

        if not cv_parts:
            raise FileNotFoundError(f"No cv_*.csv files in {combo_dir}")

        cv_results = pd.concat([pd.read_csv(f) for f in cv_parts], ignore_index=True)
        logger.info("Merged %d combo files → %d CV rows", len(cv_parts), len(cv_results))

        # Reconstruct fold_coefs list from CSVs
        fold_coefs_df = pd.concat([pd.read_csv(f) for f in coef_parts], ignore_index=True)

        data = load_and_split(cfg)
        features = _load_features(cfg)
        final = final_refit_and_test(cfg, data, features, cv_results)

        # Convert fold_coefs_df back to list format for save_results
        fold_coefs_list = []
        for (fold, strat, wt), grp in fold_coefs_df.groupby(["fold", "strategy", "weight_scheme"]):
            grp_sorted = grp.set_index("protein").loc[features["pruned_proteins"]]
            fold_coefs_list.append({
                "fold": fold,
                "strategy": strat,
                "weight_scheme": wt,
                "coefs": grp_sorted["coefficient"].values,
            })

        save_results(cfg, data, features, cv_results, final, fold_coefs_list)

    elapsed = time.time() - t0
    logger.info("Phase '%s' complete in %.1f minutes", args.phase, elapsed / 60)


if __name__ == "__main__":
    main()
