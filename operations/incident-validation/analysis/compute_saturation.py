#!/usr/bin/env python3
"""
Saturation curve: CV AUPRC vs. panel size for LR_EN (incident_only, log weighting).

For each panel size N in PANEL_SIZES:
  1. Take top-N proteins by stability_freq from feature_panel.csv
  2. 5-fold outer CV, inner 3-fold Optuna (20 trials), log class weighting
  3. Record mean/std CV AUPRC and test AUPRC (median-C refit on full dev set)

Outputs:
  out/saturation_results.csv
  out/fig8_saturation.{png,pdf}

Usage:
  cd /path/to/cel-risk
  python operations/incident-validation/analysis/compute_saturation.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CEL_ROOT / "analysis" / "src"))

from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import (
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    PROTEIN_SUFFIX,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = CEL_ROOT / "data" / "Celiac_dataset_proteomics_w_demo.parquet"
FEATURE_PANEL_PATH = CEL_ROOT / "results" / "incident-validation" / "lr" / "LR_EN" / "feature_panel.csv"
OUT_DIR = Path(__file__).resolve().parent / "out"

PANEL_SIZES = [5, 8, 10, 15, 20, 25, 28, 40, 60, 80, 100, 134]
TEST_FRAC = 0.20
SPLIT_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20
N_BOOTSTRAP_CI = 1000
CI_SEED = 123

# LR_EN non-zero cutoff from full-panel run and SVM L1 reference
VLINE_LR_EN = 28
VLINE_SVM_L1 = 96

# Plot colors
COLOR_CV = "#4C78A8"
COLOR_TEST = "#E7298A"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence Optuna noise
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
# Suppress sklearn 1.8 FutureWarning about 'penalty' argument deprecation
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class weighting (log scheme, matches main pipeline)
# ---------------------------------------------------------------------------

def compute_class_weight_log(y: np.ndarray) -> dict:
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return {0: 1.0, 1: 1.0}
    w1 = np.log(n_neg / n_pos)
    return {0: 1.0, 1: max(float(w1), 1.0)}


# ---------------------------------------------------------------------------
# Data loading and splitting (mirrors main pipeline exactly)
# ---------------------------------------------------------------------------

def load_and_split(data_path: Path) -> dict:
    """Load data and create the locked 20% test split identical to main pipeline."""
    logger.info("Loading data: %s", data_path)
    df = read_proteomics_file(str(data_path))
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    incident_mask = df[TARGET_COL] == INCIDENT_LABEL
    control_mask = df[TARGET_COL] == CONTROL_LABEL
    prevalent_mask = df[TARGET_COL] == PREVALENT_LABEL

    logger.info(
        "Groups: %d incident, %d prevalent, %d controls",
        incident_mask.sum(), prevalent_mask.sum(), control_mask.sum(),
    )

    # Stratify by sex + outcome, identical to main pipeline
    ic_df = df[incident_mask | control_mask].copy()
    ic_df["_binary"] = (ic_df[TARGET_COL] == INCIDENT_LABEL).astype(int)
    sex_col = "sex"
    if sex_col in ic_df.columns:
        ic_df["_strat"] = ic_df["_binary"].astype(str) + "_" + ic_df[sex_col].astype(str)
    else:
        logger.warning("'sex' column not found; stratifying by outcome only")
        ic_df["_strat"] = ic_df["_binary"].astype(str)

    dev_idx, test_idx = train_test_split(
        ic_df.index,
        test_size=TEST_FRAC,
        stratify=ic_df["_strat"],
        random_state=SPLIT_SEED,
    )

    test_set = set(test_idx)
    dev_incident_idx = np.array([i for i in df.index[incident_mask] if i not in test_set])
    dev_control_idx  = np.array([i for i in df.index[control_mask]  if i not in test_set])
    test_incident_idx = np.array([i for i in df.index[incident_mask] if i in test_set])
    test_control_idx  = np.array([i for i in df.index[control_mask]  if i in test_set])
    prevalent_idx = np.array(df.index[prevalent_mask])

    logger.info(
        "Dev: %d incident + %d controls | Test: %d incident + %d controls",
        len(dev_incident_idx), len(dev_control_idx),
        len(test_incident_idx), len(test_control_idx),
    )

    return {
        "df": df,
        "dev_incident_idx": dev_incident_idx,
        "dev_control_idx": dev_control_idx,
        "test_incident_idx": test_incident_idx,
        "test_control_idx": test_control_idx,
        "prevalent_idx": prevalent_idx,
    }


# ---------------------------------------------------------------------------
# Feature ranking
# ---------------------------------------------------------------------------

def load_ranked_features(panel_path: Path) -> list[str]:
    """Return proteins sorted by stability_freq descending."""
    panel_df = pd.read_csv(panel_path)
    panel_df = panel_df.sort_values("stability_freq", ascending=False)
    return panel_df["protein"].tolist()


# ---------------------------------------------------------------------------
# Optuna inner CV
# ---------------------------------------------------------------------------

def _optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: dict,
    seed: int,
) -> float:
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.01, 1.0)

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

        model = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            penalty="elasticnet",
            solver="saga",
            max_iter=2000,
            class_weight=class_weight,
            random_state=seed,
        )
        try:
            model.fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_va_s)[:, 1]
            scores.append(average_precision_score(y_va, y_prob))
        except Exception:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Single panel-size evaluation
# ---------------------------------------------------------------------------

def evaluate_panel_size(
    panel_size: int,
    proteins: list[str],
    data: dict,
) -> dict:
    """Run 5-fold CV + test evaluation for a given top-N protein subset."""
    df = data["df"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx  = data["dev_control_idx"]
    test_incident_idx = data["test_incident_idx"]
    test_control_idx  = data["test_control_idx"]

    # Subset proteins to those present in DataFrame
    selected = [p for p in proteins[:panel_size] if p in df.columns]
    n_used = len(selected)
    if n_used < panel_size:
        logger.warning(
            "Panel size %d: only %d/%d proteins found in DataFrame",
            panel_size, n_used, panel_size,
        )
    if n_used == 0:
        raise ValueError(f"No proteins available for panel size {panel_size}")

    # --- 5-fold CV on dev set ---
    dev_ic_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    dev_ic_labels = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold_auprcs = []
    fold_best_C = []
    fold_best_l1 = []

    for fold_i, (train_pos, val_pos) in enumerate(outer_cv.split(dev_ic_idx, dev_ic_labels)):
        fold_train_ic = dev_ic_idx[train_pos]
        fold_val_ic   = dev_ic_idx[val_pos]

        dev_incident_set = set(dev_incident_idx.tolist())
        dev_control_set  = set(dev_control_idx.tolist())

        fold_train_incident = np.array([i for i in fold_train_ic if i in dev_incident_set])
        fold_train_controls = np.array([i for i in fold_train_ic if i in dev_control_set])
        val_incident = np.array([i for i in fold_val_ic if i in dev_incident_set])
        val_controls = np.array([i for i in fold_val_ic if i in dev_control_set])

        # incident_only strategy
        train_idx = np.concatenate([fold_train_incident, fold_train_controls]).astype(int)
        y_train = np.concatenate([
            np.ones(len(fold_train_incident)),
            np.zeros(len(fold_train_controls)),
        ]).astype(int)

        val_idx = np.concatenate([val_incident, val_controls]).astype(int)
        y_val = np.concatenate([
            np.ones(len(val_incident)),
            np.zeros(len(val_controls)),
        ]).astype(int)

        X_train = df.loc[train_idx, selected].to_numpy(dtype=float)
        X_val   = df.loc[val_idx,   selected].to_numpy(dtype=float)

        cw = compute_class_weight_log(y_train)
        fold_seed = SPLIT_SEED + fold_i

        # Inner Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=fold_seed),
        )
        study.optimize(
            lambda trial, _X=X_train, _y=y_train, _cw=cw, _s=fold_seed: (
                _optuna_objective(trial, _X, _y, _cw, _s)
            ),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )

        best_params = study.best_params

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        model = LogisticRegression(
            C=best_params["C"],
            l1_ratio=best_params["l1_ratio"],
            penalty="elasticnet",
            solver="saga",
            max_iter=2000,
            class_weight=cw,
            random_state=fold_seed,
        )
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_val_s)[:, 1]
        fold_auprcs.append(average_precision_score(y_val, y_prob))
        fold_best_C.append(best_params["C"])
        fold_best_l1.append(best_params["l1_ratio"])

    mean_cv_auprc = float(np.mean(fold_auprcs))
    std_cv_auprc  = float(np.std(fold_auprcs, ddof=1))

    # --- Test evaluation: median-C refit on full dev set ---
    median_C       = float(np.median(fold_best_C))
    median_l1ratio = float(np.median(fold_best_l1))

    dev_idx = np.concatenate([dev_incident_idx, dev_control_idx]).astype(int)
    y_dev = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    X_dev = df.loc[dev_idx, selected].to_numpy(dtype=float)

    test_idx_arr = np.concatenate([test_incident_idx, test_control_idx]).astype(int)
    y_test = np.concatenate([
        np.ones(len(test_incident_idx)),
        np.zeros(len(test_control_idx)),
    ]).astype(int)

    X_test = df.loc[test_idx_arr, selected].to_numpy(dtype=float)

    cw_dev = compute_class_weight_log(y_dev)
    scaler_final = StandardScaler()
    X_dev_s  = scaler_final.fit_transform(X_dev)
    X_test_s = scaler_final.transform(X_test)

    final_model = LogisticRegression(
        C=median_C,
        l1_ratio=median_l1ratio,
        penalty="elasticnet",
        solver="saga",
        max_iter=2000,
        class_weight=cw_dev,
        random_state=SPLIT_SEED,
    )
    final_model.fit(X_dev_s, y_dev)
    y_prob_test = final_model.predict_proba(X_test_s)[:, 1]
    test_auprc = float(average_precision_score(y_test, y_prob_test))

    # Bootstrap CI for test AUPRC
    rng = np.random.default_rng(CI_SEED)
    boot_auprcs = []
    for _ in range(N_BOOTSTRAP_CI):
        ix = rng.choice(len(y_test), size=len(y_test), replace=True)
        if len(np.unique(y_test[ix])) < 2:
            continue
        boot_auprcs.append(average_precision_score(y_test[ix], y_prob_test[ix]))
    test_auprc_lo = float(np.percentile(boot_auprcs, 2.5))
    test_auprc_hi = float(np.percentile(boot_auprcs, 97.5))

    return {
        "panel_size": panel_size,
        "n_features_used": n_used,
        "mean_cv_auprc": mean_cv_auprc,
        "std_cv_auprc": std_cv_auprc,
        "test_auprc": test_auprc,
        "test_auprc_lo": test_auprc_lo,
        "test_auprc_hi": test_auprc_hi,
        "median_C": median_C,
        "median_l1_ratio": median_l1ratio,
        "features_list": ";".join(selected),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_saturation(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate fig8_saturation (CV + test AUPRC vs panel size)."""
    sizes     = results_df["panel_size"].values
    cv_mean   = results_df["mean_cv_auprc"].values
    cv_std    = results_df["std_cv_auprc"].values
    test_mean = results_df["test_auprc"].values
    test_lo   = results_df["test_auprc_lo"].values
    test_hi   = results_df["test_auprc_hi"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # CV ribbon + line
    ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std,
                    color=COLOR_CV, alpha=0.18, linewidth=0)
    ax.plot(sizes, cv_mean, "o-", color=COLOR_CV, linewidth=2.0,
            markersize=5, label="CV AUPRC (mean ± 1 SD)")

    # Test bootstrap CI ribbon + line
    ax.fill_between(sizes, test_lo, test_hi,
                    color=COLOR_TEST, alpha=0.18, linewidth=0)
    ax.plot(sizes, test_mean, "s--", color=COLOR_TEST, linewidth=2.0,
            markersize=5, label="Test AUPRC (median-C refit, 95% CI)")

    # Reference lines
    ymin_ax = ax.get_ylim()[0]
    ax.axvline(VLINE_LR_EN, color="#888888", linestyle=":", linewidth=1.3,
               label=f"LR_EN non-zero features (N={VLINE_LR_EN})")
    ax.axvline(VLINE_SVM_L1, color="#AAAAAA", linestyle="-.", linewidth=1.3,
               label=f"SVM L1 non-zero features (N={VLINE_SVM_L1})")

    # Annotate knee heuristically: largest CV AUPRC improvement per additional feature
    # Compute pairwise slopes (mean_cv_auprc[i] - mean_cv_auprc[i-1]) / (size[i] - size[i-1])
    if len(sizes) > 2:
        slopes = np.diff(cv_mean) / np.diff(sizes.astype(float))
        # Knee = last index where slope > 10% of max slope
        max_slope = slopes.max()
        knee_candidates = np.where(slopes > 0.1 * max_slope)[0]
        if len(knee_candidates) > 0:
            knee_idx = knee_candidates[-1] + 1  # +1 because diff shifts index
            ax.annotate(
                f"knee ~N={sizes[knee_idx]}",
                xy=(sizes[knee_idx], cv_mean[knee_idx]),
                xytext=(sizes[knee_idx] + max(sizes) * 0.04, cv_mean[knee_idx] - 0.015),
                fontsize=8,
                color="#444444",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9),
            )

    ax.set_xlabel("Panel size (number of proteins)", fontsize=11)
    ax.set_ylabel("AUPRC", fontsize=11)
    ax.set_title(
        "Saturation curve: LR_EN — performance vs. panel size\n"
        "(incident_only strategy, log class weighting)",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fpath = out_dir / f"fig8_saturation.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", fpath)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("CEL_ROOT: %s", CEL_ROOT)
    logger.info("Data:     %s", DATA_PATH)
    logger.info("Panel:    %s", FEATURE_PANEL_PATH)
    logger.info("Output:   %s", OUT_DIR)
    logger.info("Panel sizes to test: %s", PANEL_SIZES)

    # Load data
    data = load_and_split(DATA_PATH)

    # Load ranked feature list
    ranked_proteins = load_ranked_features(FEATURE_PANEL_PATH)
    logger.info("Ranked panel: %d proteins (sorted by stability_freq desc)", len(ranked_proteins))
    max_available = min(max(PANEL_SIZES), len(ranked_proteins))
    logger.info("Max panel size available: %d", max_available)

    results = []
    for panel_size in PANEL_SIZES:
        effective_size = min(panel_size, max_available)
        logger.info(
            "=== Panel size: %d (effective: %d) ===",
            panel_size, effective_size,
        )
        row = evaluate_panel_size(effective_size, ranked_proteins, data)
        # Keep the requested panel_size in the output so the table aligns
        row["panel_size"] = panel_size
        results.append(row)
        logger.info(
            "  CV AUPRC: %.4f +/- %.4f | Test AUPRC: %.4f [%.4f, %.4f]",
            row["mean_cv_auprc"], row["std_cv_auprc"],
            row["test_auprc"], row["test_auprc_lo"], row["test_auprc_hi"],
        )

    # Save results
    results_df = pd.DataFrame(results, columns=[
        "panel_size", "n_features_used",
        "mean_cv_auprc", "std_cv_auprc",
        "test_auprc", "test_auprc_lo", "test_auprc_hi",
        "median_C", "median_l1_ratio",
        "features_list",
    ])
    out_csv = OUT_DIR / "saturation_results.csv"
    if out_csv.exists():
        logger.warning("Output CSV already exists; overwriting: %s", out_csv)
    results_df.to_csv(out_csv, index=False)
    logger.info("Saved results: %s", out_csv)

    # Plot
    plot_saturation(results_df, OUT_DIR)

    logger.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
