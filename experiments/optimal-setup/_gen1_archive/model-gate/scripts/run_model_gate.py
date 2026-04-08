#!/usr/bin/env python3
"""
Phase A: Model Gate — 4 models x 3 strategies x 4 weights = 48 combos.

Shared-recipe experiment. Decides whether calibrated Linear SVM earns a
dedicated optimization branch.

Each combo: 5-fold outer CV, 3-fold inner Optuna tuning, AUPRC primary.
Feature selection: Wald-based bootstrap stability (shared, model-agnostic).

Usage:
  cd cel-risk
  python experiments/optimal-setup/model-gate/scripts/run_model_gate.py
  python experiments/optimal-setup/model-gate/scripts/run_model_gate.py --smoke
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
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
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

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

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
# Configuration (loads shared_recipe.yaml)
# ============================================================================

RECIPE_PATH = Path(__file__).resolve().parents[2] / "shared_recipe.yaml"


def _load_recipe() -> dict:
    with open(RECIPE_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    data_path: Path = Path("data/Celiac_dataset_proteomics_w_demo.parquet")
    output_dir: Path = Path("results/model_gate")

    # From shared_recipe.yaml
    split_seed: int = 42
    test_frac: float = 0.20
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    n_optuna_trials: int = 50
    screen_method: str = "wald"
    n_bootstrap: int = 100
    bootstrap_top_k: int = 200
    stability_threshold: float = 0.70
    corr_threshold: float = 0.85
    corr_method: str = "spearman"
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5
    svm_max_iter: int = 5000
    n_bootstrap_ci: int = 2000
    ci_seed: int = 99
    primary_metric: str = "auprc"

    # Experiment axes
    models: list[str] = field(
        default_factory=lambda: ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]
    )
    strategies: list[str] = field(
        default_factory=lambda: [
            "incident_only", "incident_prevalent", "prevalent_only",
        ]
    )
    weight_schemes: list[str] = field(
        default_factory=lambda: ["none", "balanced", "sqrt", "log"]
    )

    recipe_overrides: dict = field(default_factory=dict)
    smoke: bool = False

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        if self.smoke:
            self.n_bootstrap = 10
            self.bootstrap_top_k = 50
            self.n_optuna_trials = 5
            self.n_bootstrap_ci = 100
            self.calibration_cv = 2
            self.models = ["LR_EN", "LinSVM_cal"]
            self.strategies = ["incident_only"]
            self.weight_schemes = ["none", "log"]

    @classmethod
    def from_recipe(cls, smoke: bool = False, **overrides) -> Config:
        """Load from shared_recipe.yaml with optional overrides."""
        recipe = _load_recipe()
        cfg = cls(
            split_seed=recipe["split"]["split_seed"],
            test_frac=recipe["split"]["test_frac"],
            n_outer_folds=recipe["split"]["n_outer_folds"],
            n_inner_folds=recipe["split"]["n_inner_folds"],
            n_optuna_trials=recipe["tuning"]["n_optuna_trials"],
            primary_metric=recipe["tuning"]["primary_metric"],
            screen_method=recipe["features"]["screen_method"],
            n_bootstrap=recipe["features"]["n_bootstrap"],
            bootstrap_top_k=recipe["features"]["bootstrap_top_k"],
            stability_threshold=recipe["features"]["stability_threshold"],
            corr_threshold=recipe["features"]["corr_threshold"],
            corr_method=recipe["features"]["corr_method"],
            calibration_method=recipe["calibration"]["method"],
            calibration_cv=recipe["calibration"]["cv"],
            svm_max_iter=recipe["svm"]["max_iter"],
            n_bootstrap_ci=recipe["evaluation"]["n_bootstrap_ci"],
            ci_seed=recipe["evaluation"]["ci_seed"],
            recipe_overrides=overrides,
            smoke=smoke,
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ============================================================================
# Data
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

    incident_set = set(incident_idx)
    control_set = set(control_idx)

    return {
        "df": df,
        "protein_cols": protein_cols,
        "prevalent_idx": prevalent_idx,
        "dev_incident_idx": np.array([i for i in dev_ic_idx if i in incident_set]),
        "dev_control_idx": np.array([i for i in dev_ic_idx if i in control_set]),
        "test_incident_idx": np.array([i for i in test_ic_idx if i in incident_set]),
        "test_control_idx": np.array([i for i in test_ic_idx if i in control_set]),
    }


def select_features_shared(cfg: Config, data: dict) -> dict:
    """Wald-based bootstrap stability (shared panel, model-agnostic)."""
    logger.info("=== Shared Feature Selection (Wald) ===")
    df = data["df"]
    protein_cols = data["protein_cols"]

    fs_idx = np.concatenate([data["dev_incident_idx"], data["dev_control_idx"]])
    fs_labels = np.concatenate([
        np.ones(len(data["dev_incident_idx"])),
        np.zeros(len(data["dev_control_idx"])),
    ]).astype(int)

    X_fs = df.loc[fs_idx, protein_cols]
    y_fs = pd.Series(fs_labels, index=fs_idx)

    stable_proteins, selection_freq, bootstrap_log = bootstrap_stability_selection(
        X=X_fs, y=y_fs, protein_cols=protein_cols,
        screen_method=cfg.screen_method,
        n_bootstrap=cfg.n_bootstrap,
        top_k=cfg.bootstrap_top_k,
        stability_threshold=cfg.stability_threshold,
        seed=cfg.split_seed,
    )
    logger.info("Stable proteins (pre-prune): %d", len(stable_proteins))

    prune_map, pruned_proteins = prune_correlated_proteins(
        df=X_fs, y=y_fs, proteins=stable_proteins,
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
# Class weighting + training strategy
# ============================================================================


def compute_class_weight(scheme: str, y: np.ndarray) -> dict | str | None:
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
        raise ValueError(f"Unknown: {scheme}")
    return {0: 1.0, 1: float(w1)}


def get_training_data(
    strategy: str,
    incident_idx: np.ndarray,
    control_idx: np.ndarray,
    prevalent_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "incident_only":
        idx = np.concatenate([incident_idx, control_idx])
        y = np.concatenate([np.ones(len(incident_idx)), np.zeros(len(control_idx))])
    elif strategy == "incident_prevalent":
        idx = np.concatenate([incident_idx, prevalent_idx, control_idx])
        y = np.concatenate([
            np.ones(len(incident_idx)), np.ones(len(prevalent_idx)),
            np.zeros(len(control_idx)),
        ])
    elif strategy == "prevalent_only":
        idx = np.concatenate([prevalent_idx, control_idx])
        y = np.concatenate([np.ones(len(prevalent_idx)), np.zeros(len(control_idx))])
    else:
        raise ValueError(f"Unknown: {strategy}")
    return idx.astype(int), y.astype(int)


# ============================================================================
# Model builders
# ============================================================================


def _build_model(
    model_name: str, params: dict, class_weight, cfg: Config, seed: int,
):
    """Build a model with given hyperparameters."""
    if model_name == "LR_EN":
        return LogisticRegression(
            C=params["C"], l1_ratio=params["l1_ratio"],
            penalty="elasticnet", solver="saga",
            class_weight=class_weight,
            max_iter=2000, random_state=seed,
        )
    elif model_name == "LinSVM_cal":
        base = LinearSVC(
            C=params["C"], class_weight=class_weight,
            max_iter=cfg.svm_max_iter, dual="auto", random_state=seed,
        )
        return CalibratedClassifierCV(
            base, method=cfg.calibration_method, cv=cfg.calibration_cv,
        )
    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=class_weight,
            random_state=seed, n_jobs=1,
        )
    elif model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        # XGBoost uses scale_pos_weight instead of class_weight
        spw = 1.0
        if class_weight == "balanced":
            # Approximate balanced weight
            spw = 1.0  # Will be set per-fold
        elif isinstance(class_weight, dict) and 1 in class_weight:
            spw = class_weight[1]
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Suggest hyperparameters per model."""
    if model_name == "LR_EN":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        }
    elif model_name == "LinSVM_cal":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        }
    elif model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Optuna + CV
# ============================================================================


def _optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight,
    cfg: Config,
    seed: int,
) -> float:
    """Mean AUPRC across inner CV folds."""
    params = _suggest_params(trial, model_name)

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

        # For XGBoost balanced: compute scale_pos_weight from training fold
        cw = class_weight
        if model_name == "XGBoost" and class_weight == "balanced":
            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
            cw = {0: 1.0, 1: spw}

        model = _build_model(model_name, params, cw, cfg, seed)
        try:
            model.fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_va_s)[:, 1]
            auprcs.append(average_precision_score(y_va, y_prob))
        except Exception:
            auprcs.append(0.0)

    return float(np.mean(auprcs))


def tune_and_evaluate_fold(
    model_name: str,
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
            trial, model_name, X_train, y_train, class_weight, cfg, fold_seed,
        ),
        n_trials=cfg.n_optuna_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # For XGBoost balanced: compute from full outer training fold
    cw = class_weight
    if model_name == "XGBoost" and class_weight == "balanced":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        cw = {0: 1.0, 1: max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0}

    model = _build_model(model_name, best_params, cw, cfg, fold_seed)
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_val_s)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    auroc = roc_auc_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)

    return {
        "auprc": auprc,
        "auroc": auroc,
        "brier": brier,
        "best_params": json.dumps(best_params),
        "best_inner_auprc": study.best_value,
    }


def run_cv(cfg: Config, data: dict, features: dict) -> pd.DataFrame:
    """Run 5-fold CV for all model x strategy x weight combos."""
    logger.info("=== Phase A: Cross-Validation ===")

    df = data["df"]
    panel = features["pruned_proteins"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    n_combos = len(cfg.models) * len(cfg.strategies) * len(cfg.weight_schemes)
    total = n_combos * cfg.n_outer_folds
    logger.info(
        "%d models x %d strategies x %d weights = %d combos, %d total folds",
        len(cfg.models), len(cfg.strategies), len(cfg.weight_schemes),
        n_combos, total,
    )

    dev_ic_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    dev_ic_labels = np.concatenate([
        np.ones(len(dev_incident_idx)), np.zeros(len(dev_control_idx)),
    ]).astype(int)

    outer_cv = StratifiedKFold(
        n_splits=cfg.n_outer_folds, shuffle=True, random_state=cfg.split_seed,
    )

    incident_set = set(dev_incident_idx)
    control_set = set(dev_control_idx)

    all_results = []
    done = 0

    for model_name in cfg.models:
        for strategy in cfg.strategies:
            for weight_scheme in cfg.weight_schemes:
                combo_results = []

                for fold_i, (train_pos, val_pos) in enumerate(
                    outer_cv.split(dev_ic_idx, dev_ic_labels)
                ):
                    fold_train_ic = dev_ic_idx[train_pos]
                    fold_val_ic = dev_ic_idx[val_pos]

                    fold_train_incident = np.array(
                        [i for i in fold_train_ic if i in incident_set]
                    )
                    fold_train_controls = np.array(
                        [i for i in fold_train_ic if i in control_set]
                    )

                    train_idx, y_train = get_training_data(
                        strategy, fold_train_incident,
                        fold_train_controls, prevalent_idx,
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

                    X_train = df.loc[train_idx, panel].to_numpy(dtype=float)
                    X_val = df.loc[val_idx, panel].to_numpy(dtype=float)

                    cw = compute_class_weight(weight_scheme, y_train)
                    fold_seed = cfg.split_seed + fold_i

                    result = tune_and_evaluate_fold(
                        model_name, X_train, y_train,
                        X_val, y_val, cw, cfg, fold_seed,
                    )

                    result["fold"] = fold_i
                    result["model"] = model_name
                    result["strategy"] = strategy
                    result["weight_scheme"] = weight_scheme
                    result["n_train"] = len(y_train)
                    result["n_train_pos"] = int(y_train.sum())
                    result["n_val"] = len(y_val)
                    result["n_val_pos"] = int(y_val.sum())
                    combo_results.append(result)

                    done += 1
                    if done % 10 == 0:
                        logger.info("  Progress: %d/%d folds", done, total)

                mean_auprc = np.mean([r["auprc"] for r in combo_results])
                mean_auroc = np.mean([r["auroc"] for r in combo_results])
                logger.info(
                    "  %s / %s / %s: AUPRC=%.4f AUROC=%.4f",
                    model_name, strategy, weight_scheme,
                    mean_auprc, mean_auroc,
                )
                all_results.extend(combo_results)

    return pd.DataFrame(all_results)


# ============================================================================
# Decision
# ============================================================================


def make_branch_decision(
    cv_results: pd.DataFrame, cfg: Config,
) -> tuple[pd.DataFrame, dict]:
    """Summarize results and decide which models earn a dedicated branch."""
    logger.info("=== Branch Decision ===")

    summary = (
        cv_results.groupby(["model", "strategy", "weight_scheme"])
        .agg(
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
        )
        .reset_index()
        .sort_values("mean_auprc", ascending=False)
    )

    # Best config overall
    best = summary.iloc[0]
    winner_model = best["model"]
    winner_auprc = best["mean_auprc"]
    winner_std = best["std_auprc"]

    # CI overlap rule: model earns branch if its best config's
    # mean - 1.96*SE overlaps winner's mean + 1.96*SE
    # Approximate SE from std / sqrt(n_folds)
    se_winner = winner_std / np.sqrt(cfg.n_outer_folds)
    winner_lower = winner_auprc - 1.96 * se_winner

    branch_models = []
    for model_name in cfg.models:
        model_best = summary[summary["model"] == model_name].iloc[0]
        se_m = model_best["std_auprc"] / np.sqrt(cfg.n_outer_folds)
        model_upper = model_best["mean_auprc"] + 1.96 * se_m

        if model_upper >= winner_lower:
            branch_models.append(model_name)

    # Best config per model (for Pareto analysis)
    model_bests = (
        summary.groupby("model")
        .apply(lambda g: g.sort_values("mean_auprc", ascending=False).iloc[0])
        .reset_index(drop=True)
    )

    logger.info("\nModel comparison (best config per model):")
    logger.info("\n%s", model_bests[
        ["model", "strategy", "weight_scheme",
         "mean_auprc", "std_auprc", "mean_auroc", "mean_brier"]
    ].to_string(index=False))

    logger.info("\nWinner: %s (AUPRC=%.4f +/- %.4f)", winner_model, winner_auprc, winner_std)
    logger.info("Branch-eligible models: %s", branch_models)

    decision = {
        "winner": winner_model,
        "winner_strategy": best["strategy"],
        "winner_weight": best["weight_scheme"],
        "winner_auprc": float(winner_auprc),
        "winner_std": float(winner_std),
        "branch_models": branch_models,
        "decision_rule": "CI overlap (mean +/- 1.96*SE)",
        "n_folds": cfg.n_outer_folds,
        "primary_metric": cfg.primary_metric,
    }

    return summary, decision


# ============================================================================
# Pareto analysis (AUROC vs Brier)
# ============================================================================


def pareto_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto front across all configs (maximize AUROC, minimize Brier)."""
    df = summary.sort_values("mean_auroc", ascending=False).copy()

    pareto = []
    min_brier = float("inf")
    for _, row in df.iterrows():
        if row["mean_brier"] < min_brier:
            pareto.append(True)
            min_brier = row["mean_brier"]
        else:
            pareto.append(False)
    df["pareto"] = pareto

    return df


# ============================================================================
# Save
# ============================================================================


def save_results(
    cfg: Config,
    data: dict,
    features: dict,
    cv_results: pd.DataFrame,
    summary: pd.DataFrame,
    decision: dict,
):
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Config + provenance
    provenance = {
        "recipe_mode": "shared",
        "recipe_source": str(RECIPE_PATH),
        "recipe_overrides": cfg.recipe_overrides,
        "comparison_class": "fair",
    }
    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in cfg.__dict__.items()
    }
    cfg_dict["provenance"] = provenance
    with open(out / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # CV results
    cv_results.to_csv(out / "cv_results.csv", index=False)

    # Summary
    summary.to_csv(out / "model_comparison.csv", index=False)

    # Pareto
    pareto_df = pareto_analysis(summary)
    pareto_df.to_csv(out / "pareto_analysis.csv", index=False)

    # Decision
    with open(out / "branch_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # Shared panel
    pd.DataFrame({"protein": features["pruned_proteins"]}).to_csv(
        out / "shared_panel.csv", index=False,
    )

    if features["bootstrap_log"] is not None:
        features["bootstrap_log"].to_csv(out / "bootstrap_log.csv", index=False)

    # Report
    _write_report(cfg, data, features, summary, decision, pareto_df, out)

    logger.info("All results saved to %s", out)


def _write_report(cfg, data, features, summary, decision, pareto_df, out):
    pareto_front = pareto_df[pareto_df["pareto"]]

    # Best per model
    model_bests = (
        summary.groupby("model")
        .apply(lambda g: g.sort_values("mean_auprc", ascending=False).iloc[0])
        .reset_index(drop=True)
    )

    lines = [
        "# Phase A: Model Gate Report",
        "",
        "## Provenance",
        "- Recipe mode: shared",
        f"- Recipe source: {RECIPE_PATH}",
        f"- Overrides: {cfg.recipe_overrides}",
        "",
        "## Dataset",
        f"- Dev incident: {len(data['dev_incident_idx'])}",
        f"- Dev controls: {len(data['dev_control_idx'])}",
        f"- Prevalent: {len(data['prevalent_idx'])}",
        f"- Test incident: {len(data['test_incident_idx'])} (locked)",
        f"- Test controls: {len(data['test_control_idx'])} (locked)",
        "",
        f"## Shared Feature Panel: {len(features['pruned_proteins'])} proteins",
        "",
        "## Best Config Per Model",
        "",
        model_bests[
            ["model", "strategy", "weight_scheme",
             "mean_auprc", "std_auprc", "mean_auroc", "std_auroc", "mean_brier"]
        ].to_markdown(index=False),
        "",
        "## Full Ranking (top 20)",
        "",
        summary.head(20)[
            ["model", "strategy", "weight_scheme",
             "mean_auprc", "std_auprc", "mean_auroc", "mean_brier"]
        ].to_markdown(index=False),
        "",
        "## Pareto Front (AUROC vs Brier)",
        "",
        pareto_front[
            ["model", "strategy", "weight_scheme",
             "mean_auroc", "mean_brier", "mean_auprc"]
        ].to_markdown(index=False),
        "",
        "## Branch Decision",
        "",
        f"- **Winner:** {decision['winner']} "
        f"({decision['winner_strategy']} + {decision['winner_weight']}, "
        f"AUPRC = {decision['winner_auprc']:.4f} +/- {decision['winner_std']:.4f})",
        f"- **Branch-eligible models:** {', '.join(decision['branch_models'])}",
        f"- **Decision rule:** {decision['decision_rule']}",
        "",
        "## Next Steps",
        "",
        f"If {decision['winner']} is branch-eligible:",
        "1. Phase B: Build winner-native feature ranking",
        "2. Phase B: Panel-size sweep under winning recipe",
        "3. Phase D: Holdout confirmation",
        "",
    ]

    with open(out / "summary_report.md", "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Phase A: Model Gate")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_recipe(smoke=args.smoke)
    if args.data_path:
        cfg.data_path = Path(args.data_path)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Phase A: Model Gate")
    logger.info("  Models: %s", cfg.models)
    logger.info("  Strategies: %s", cfg.strategies)
    logger.info("  Weights: %s", cfg.weight_schemes)
    logger.info("  Smoke: %s", cfg.smoke)
    logger.info("  Recipe: %s", RECIPE_PATH)
    logger.info("=" * 60)

    t0 = time.time()

    data = load_and_split(cfg)
    features = select_features_shared(cfg, data)
    cv_results = run_cv(cfg, data, features)
    summary, decision = make_branch_decision(cv_results, cfg)

    save_results(cfg, data, features, cv_results, summary, decision)

    elapsed = time.time() - t0
    logger.info("Total runtime: %.1f min (%.1f h)", elapsed / 60, elapsed / 3600)


if __name__ == "__main__":
    main()
