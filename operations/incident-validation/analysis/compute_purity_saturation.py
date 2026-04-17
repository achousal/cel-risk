#!/usr/bin/env python3
"""Purity-ranked vs stability-ranked saturation curves for LR_EN, SVM_L1, SVM_L2.

For each model and each protein ordering (purity-rank / stability-rank), evaluates
AUPRC vs panel size using the same protocol as compute_saturation.py:
  - incident_only strategy
  - model-specific best weight: LR_EN→log, SVM_L1→log, SVM_L2→none
  - 5-fold outer CV, 3-fold inner Optuna (20 trials)
  - median-C refit on full dev set for hold-out test AUPRC

Outputs (under --out)
---------------------
    saturation_all_models.csv            -- numeric results, all models × orderings
    features_purity.csv                  -- long-format (ordering, panel_size, rank, protein)
    features_stability.csv
    fig_purity_saturation.{pdf,png}      -- 2×3 grid: rows=CV/test, cols=model

Usage
-----
    python operations/incident-validation/analysis/compute_purity_saturation.py
    python operations/incident-validation/analysis/compute_purity_saturation.py \\
        --panel-sizes 5 10 28 --out /tmp/sat_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CEL_ROOT / "analysis" / "src"))

from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import (
    CONTROL_LABEL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_PATH = CEL_ROOT / "data" / "Celiac_dataset_proteomics_w_demo.parquet"
NOISE_SCORES_PATH = (
    CEL_ROOT / "operations/incident-validation/analysis/out/prevalent_noise_scores.csv"
)
STABILITY_PANEL_PATH = (
    CEL_ROOT / "results/incident-validation/lr/SVM_L1/feature_coefficients.csv"
)

PANEL_SIZES = [5, 8, 10, 15, 20, 25, 28, 40, 60, 80, 100, 134]
TEST_FRAC = 0.20
SPLIT_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20
N_BOOTSTRAP_CI = 1000
CI_SEED = 123
CALIBRATION_CV = 5

# Best weight per model from incident-validation report
BEST_WEIGHT = {"LR_EN": "log", "SVM_L1": "log", "SVM_L2": "none"}

COLORS = {
    "LR_EN":  ("#4C78A8", "#A8D1F5"),
    "SVM_L1": ("#E7298A", "#F5A8D0"),
    "SVM_L2": ("#1B9E77", "#A0DFC8"),
}
ORDERING_STYLE = {"purity": ("s--", 2.0), "stability": ("o-", 1.5)}


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def _log_weight(y: np.ndarray) -> dict | None:
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return None
    return {0: 1.0, 1: max(float(np.log(n_neg / n_pos)), 1.0)}


def _sample_weight(class_weight: dict | None, y: np.ndarray) -> np.ndarray | None:
    if class_weight is None:
        return None
    sw = np.ones(len(y), dtype=float)
    sw[y == 1] = float(class_weight.get(1, 1.0))
    sw[y == 0] = float(class_weight.get(0, 1.0))
    return sw


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    weight_scheme: str  # "log" or "none"

    def suggest_params(self, trial: optuna.Trial) -> dict:
        raise NotImplementedError

    def build(self, params: dict, class_weight: dict | None, seed: int):
        raise NotImplementedError

    def fit(self, model, X: np.ndarray, y: np.ndarray, class_weight: dict | None):
        model.fit(X, y)

    def class_weight(self, y: np.ndarray) -> dict | None:
        return _log_weight(y) if self.weight_scheme == "log" else None


@dataclass
class LRENSpec(ModelSpec):
    name: str = "LR_EN"
    weight_scheme: str = "log"

    def suggest_params(self, trial):
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0),
        }

    def build(self, params, class_weight, seed):
        return LogisticRegression(
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            penalty="elasticnet",
            solver="saga",
            max_iter=2000,
            class_weight=class_weight,
            random_state=seed,
        )


@dataclass
class SVMSpec(ModelSpec):
    penalty: str = "l2"

    def suggest_params(self, trial):
        return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}

    def build(self, params, class_weight, seed):
        if self.penalty == "l1":
            base = LinearSVC(
                penalty="l1",
                dual=False,
                C=params["C"],
                class_weight=class_weight,
                max_iter=5000,
                random_state=seed,
            )
        else:
            base = LinearSVC(
                penalty="l2",
                C=params["C"],
                class_weight=class_weight,
                max_iter=5000,
                random_state=seed,
            )
        return CalibratedClassifierCV(base, method="sigmoid", cv=CALIBRATION_CV)

    def fit(self, model, X, y, class_weight):
        # CalibratedClassifierCV handles sample weights via fit_params
        sw = _sample_weight(class_weight, y)
        if sw is not None:
            model.fit(X, y, sample_weight=sw)
        else:
            model.fit(X, y)


MODEL_SPECS: dict[str, ModelSpec] = {
    "LR_EN":  LRENSpec(),
    "SVM_L1": SVMSpec(name="SVM_L1", weight_scheme="log",  penalty="l1"),
    "SVM_L2": SVMSpec(name="SVM_L2", weight_scheme="none", penalty="l2"),
}


# ---------------------------------------------------------------------------
# Protein orderings
# ---------------------------------------------------------------------------

def load_purity_ranked(path: Path) -> list[str]:
    df = pd.read_csv(path).sort_values("purity_rank", ascending=True)
    return [p.lower() + "_resid" for p in df["protein"].tolist()]


def load_stability_ranked(path: Path) -> list[str]:
    """Return proteins sorted by stability_freq desc, ties broken by abs_coef desc."""
    df = pd.read_csv(path).sort_values(["stability_freq", "abs_coef"], ascending=[False, False])
    return df["protein"].tolist()


def save_features_table(proteins: list[str], panel_sizes: list[int], out_path: Path) -> None:
    rows = []
    for n in panel_sizes:
        for rank, protein in enumerate(proteins[:n], start=1):
            rows.append({"panel_size": n, "rank": rank, "protein": protein})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Data split (identical to compute_saturation.py)
# ---------------------------------------------------------------------------

def load_and_split(data_path: Path) -> dict:
    log.info("Loading: %s", data_path)
    df = read_proteomics_file(str(data_path))

    incident_mask  = df[TARGET_COL] == INCIDENT_LABEL
    control_mask   = df[TARGET_COL] == CONTROL_LABEL
    prevalent_mask = df[TARGET_COL] == PREVALENT_LABEL

    log.info(
        "Groups: %d incident, %d prevalent, %d controls",
        incident_mask.sum(), prevalent_mask.sum(), control_mask.sum(),
    )

    ic_df = df[incident_mask | control_mask].copy()
    ic_df["_binary"] = (ic_df[TARGET_COL] == INCIDENT_LABEL).astype(int)
    sex_col = "sex"
    if sex_col in ic_df.columns:
        ic_df["_strat"] = ic_df["_binary"].astype(str) + "_" + ic_df[sex_col].astype(str)
    else:
        log.warning("'sex' column not found; stratifying by outcome only")
        ic_df["_strat"] = ic_df["_binary"].astype(str)

    dev_idx, test_idx = train_test_split(
        ic_df.index, test_size=TEST_FRAC, stratify=ic_df["_strat"], random_state=SPLIT_SEED,
    )
    test_set = set(test_idx)

    return {
        "df": df,
        "dev_incident_idx": np.array([i for i in df.index[incident_mask] if i not in test_set]),
        "dev_control_idx":  np.array([i for i in df.index[control_mask]  if i not in test_set]),
        "test_incident_idx": np.array([i for i in df.index[incident_mask] if i in test_set]),
        "test_control_idx":  np.array([i for i in df.index[control_mask]  if i in test_set]),
    }


# ---------------------------------------------------------------------------
# Inner Optuna objective
# ---------------------------------------------------------------------------

def _inner_objective(
    trial: optuna.Trial,
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> float:
    params = spec.suggest_params(trial)
    cw = spec.class_weight(y_train)
    inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    scores = []

    for tr_ix, va_ix in inner_cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr_ix], X_train[va_ix]
        y_tr, y_va = y_train[tr_ix], y_train[va_ix]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        model = spec.build(params, cw, seed)
        try:
            spec.fit(model, X_tr_s, y_tr, cw)
            scores.append(average_precision_score(y_va, model.predict_proba(X_va_s)[:, 1]))
        except Exception:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Single panel-size evaluation
# ---------------------------------------------------------------------------

def evaluate_panel(
    panel_size: int,
    proteins: list[str],
    spec: ModelSpec,
    data: dict,
) -> dict:
    df = data["df"]
    dev_inc  = data["dev_incident_idx"]
    dev_ctl  = data["dev_control_idx"]
    test_inc = data["test_incident_idx"]
    test_ctl = data["test_control_idx"]

    selected = [p for p in proteins[:panel_size] if p in df.columns]
    n_used = len(selected)
    if n_used == 0:
        raise ValueError(f"No proteins available for panel_size={panel_size}")
    if n_used < panel_size:
        log.warning("Panel %d: only %d/%d proteins in DataFrame", panel_size, n_used, panel_size)

    dev_ic_idx    = np.concatenate([dev_inc, dev_ctl])
    dev_ic_labels = np.concatenate([np.ones(len(dev_inc)), np.zeros(len(dev_ctl))]).astype(int)
    dev_inc_set   = set(dev_inc.tolist())
    dev_ctl_set   = set(dev_ctl.tolist())

    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold_auprcs, fold_params = [], []

    for fold_i, (train_pos, val_pos) in enumerate(outer_cv.split(dev_ic_idx, dev_ic_labels)):
        tr_ic = dev_ic_idx[train_pos]
        va_ic = dev_ic_idx[val_pos]

        tr_inc = np.array([i for i in tr_ic if i in dev_inc_set])
        tr_ctl = np.array([i for i in tr_ic if i in dev_ctl_set])
        va_inc = np.array([i for i in va_ic if i in dev_inc_set])
        va_ctl = np.array([i for i in va_ic if i in dev_ctl_set])

        train_idx = np.concatenate([tr_inc, tr_ctl]).astype(int)
        y_train   = np.concatenate([np.ones(len(tr_inc)), np.zeros(len(tr_ctl))]).astype(int)
        val_idx   = np.concatenate([va_inc, va_ctl]).astype(int)
        y_val     = np.concatenate([np.ones(len(va_inc)), np.zeros(len(va_ctl))]).astype(int)

        X_train = df.loc[train_idx, selected].to_numpy(dtype=float)
        X_val   = df.loc[val_idx,   selected].to_numpy(dtype=float)

        fold_seed = SPLIT_SEED + fold_i
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=fold_seed),
        )
        study.optimize(
            lambda t, _sp=spec, _X=X_train, _y=y_train, _s=fold_seed: (
                _inner_objective(t, _sp, _X, _y, _s)
            ),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )

        best = study.best_params
        cw = spec.class_weight(y_train)
        scaler = StandardScaler()
        model = spec.build(best, cw, fold_seed)
        spec.fit(model, scaler.fit_transform(X_train), y_train, cw)
        y_prob = model.predict_proba(scaler.transform(X_val))[:, 1]
        fold_auprcs.append(average_precision_score(y_val, y_prob))
        fold_params.append(best)

    mean_cv = float(np.mean(fold_auprcs))
    std_cv  = float(np.std(fold_auprcs, ddof=1))

    # Median hyperparams for final refit
    median_params = {
        k: float(np.median([p[k] for p in fold_params]))
        for k in fold_params[0]
    }

    dev_idx_arr = np.concatenate([dev_inc, dev_ctl]).astype(int)
    y_dev       = np.concatenate([np.ones(len(dev_inc)), np.zeros(len(dev_ctl))]).astype(int)
    test_idx_arr = np.concatenate([test_inc, test_ctl]).astype(int)
    y_test       = np.concatenate([np.ones(len(test_inc)), np.zeros(len(test_ctl))]).astype(int)

    X_dev  = df.loc[dev_idx_arr,  selected].to_numpy(dtype=float)
    X_test = df.loc[test_idx_arr, selected].to_numpy(dtype=float)

    cw_dev = spec.class_weight(y_dev)
    scaler_f = StandardScaler()
    final = spec.build(median_params, cw_dev, SPLIT_SEED)
    spec.fit(final, scaler_f.fit_transform(X_dev), y_dev, cw_dev)
    y_prob_test = final.predict_proba(scaler_f.transform(X_test))[:, 1]
    test_auprc = float(average_precision_score(y_test, y_prob_test))

    rng = np.random.default_rng(CI_SEED)
    boots = []
    for _ in range(N_BOOTSTRAP_CI):
        ix = rng.choice(len(y_test), len(y_test), replace=True)
        if len(np.unique(y_test[ix])) > 1:
            boots.append(average_precision_score(y_test[ix], y_prob_test[ix]))

    return {
        "panel_size": panel_size,
        "n_features_used": n_used,
        "mean_cv_auprc": mean_cv,
        "std_cv_auprc": std_cv,
        "test_auprc": test_auprc,
        "test_auprc_lo": float(np.percentile(boots, 2.5)) if boots else np.nan,
        "test_auprc_hi": float(np.percentile(boots, 97.5)) if boots else np.nan,
    }


# ---------------------------------------------------------------------------
# Run one (model × ordering) combination
# ---------------------------------------------------------------------------

def run_combination(
    model_name: str,
    ordering: str,
    proteins: list[str],
    panel_sizes: list[int],
    data: dict,
) -> pd.DataFrame:
    spec = MODEL_SPECS[model_name]
    rows = []
    for n in panel_sizes:
        effective = min(n, len(proteins))
        log.info("=== %s | %s | N=%d (eff=%d) ===", model_name, ordering, n, effective)
        row = evaluate_panel(effective, proteins, spec, data)
        row["panel_size"] = n
        row["model"] = model_name
        row["ordering"] = ordering
        rows.append(row)
        log.info(
            "  CV %.4f±%.4f | Test %.4f [%.4f, %.4f]",
            row["mean_cv_auprc"], row["std_cv_auprc"],
            row["test_auprc"], row["test_auprc_lo"], row["test_auprc_hi"],
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_all(results: pd.DataFrame, out_dir: Path) -> None:
    models = ["LR_EN", "SVM_L1", "SVM_L2"]
    metrics = [
        ("mean_cv_auprc", "std_cv_auprc",    None,            None,            "CV AUPRC (mean ± 1 SD)"),
        ("test_auprc",    None,               "test_auprc_lo", "test_auprc_hi", "Test AUPRC (95% CI)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    fig.patch.set_facecolor("white")

    for row_i, (metric, std_col, lo_col, hi_col, ylabel) in enumerate(metrics):
        for col_i, model in enumerate(models):
            ax = axes[row_i][col_i]
            ax.set_facecolor("white")
            color, light = COLORS[model]

            for ordering in ("stability", "purity"):
                sub = results[(results["model"] == model) & (results["ordering"] == ordering)]
                if sub.empty:
                    continue
                sub = sub.sort_values("panel_size")
                sizes = sub["panel_size"].values
                vals  = sub[metric].values
                marker, lw = ORDERING_STYLE[ordering]
                label = f"{ordering} rank"

                if std_col and std_col in sub.columns:
                    sd = sub[std_col].values
                    ax.fill_between(sizes, vals - sd, vals + sd, color=color, alpha=0.12, linewidth=0)
                elif lo_col and hi_col:
                    ax.fill_between(
                        sizes, sub[lo_col].values, sub[hi_col].values,
                        color=color, alpha=0.12, linewidth=0,
                    )

                mk = marker[0]
                ls = marker[1:] if len(marker) > 1 else "-"
                ax.plot(sizes, vals, linestyle=ls, marker=mk, color=color,
                        linewidth=lw, markersize=5, alpha=(0.9 if ordering == "purity" else 0.55),
                        label=label)

            ax.axvline(28, color="#888888", linestyle=":", linewidth=1.0, alpha=0.6)
            ax.set_xlabel("Panel size", fontsize=9)
            ax.set_ylabel(ylabel if col_i == 0 else "", fontsize=9)
            if row_i == 0:
                ax.set_title(model, fontsize=11, fontweight="bold")
            ax.legend(fontsize=7.5, frameon=True, framealpha=0.85)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Purity-ranked vs stability-ranked panel ordering — incident_only, model-best weight",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_purity_saturation.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved %s", p)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=CEL_ROOT / "operations/incident-validation/analysis/out",
    )
    parser.add_argument(
        "--models", nargs="+", default=["LR_EN", "SVM_L1", "SVM_L2"],
        choices=["LR_EN", "SVM_L1", "SVM_L2"],
    )
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=PANEL_SIZES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log.info("CEL_ROOT: %s", CEL_ROOT)
    log.info("Models: %s", args.models)
    log.info("Panel sizes: %s", args.panel_sizes)

    purity_proteins    = load_purity_ranked(NOISE_SCORES_PATH)
    stability_proteins = load_stability_ranked(STABILITY_PANEL_PATH)

    log.info("Purity top-5:    %s", purity_proteins[:5])
    log.info("Stability top-5: %s", stability_proteins[:5])

    save_features_table(purity_proteins,    args.panel_sizes, args.out / "features_purity.csv")
    save_features_table(stability_proteins, args.panel_sizes, args.out / "features_stability.csv")

    data = load_and_split(DATA_PATH)

    all_results = []
    for model_name in args.models:
        for ordering, proteins in [("purity", purity_proteins), ("stability", stability_proteins)]:
            df = run_combination(model_name, ordering, proteins, args.panel_sizes, data)
            all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)

    col_order = ["model", "ordering", "panel_size", "n_features_used",
                 "mean_cv_auprc", "std_cv_auprc",
                 "test_auprc", "test_auprc_lo", "test_auprc_hi"]
    results = results[col_order].sort_values(["model", "ordering", "panel_size"])

    out_csv = args.out / "saturation_all_models.csv"
    results.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    plot_all(results, args.out)

    # Summary: delta test AUPRC (purity − stability) per model × panel size
    pivot = results.pivot_table(
        index=["model", "panel_size"], columns="ordering", values="test_auprc"
    ).reset_index()
    pivot["delta"] = pivot["purity"] - pivot["stability"]
    print("\n=== Delta test AUPRC: purity − stability (positive = purity wins) ===")
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    for model_name in args.models:
        sub = pivot[pivot["model"] == model_name]
        best = sub.loc[sub["delta"].idxmax()]
        print(
            f"\n{model_name}: largest gain at N={int(best['panel_size'])}, "
            f"delta={best['delta']:+.4f} "
            f"(purity={best['purity']:.4f}, stability={best['stability']:.4f})"
        )


if __name__ == "__main__":
    main()
