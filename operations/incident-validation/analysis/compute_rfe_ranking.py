#!/usr/bin/env python3
"""Pre-compute RFE protein ranking from incident-only SVM_L1 model.

Runs backward elimination on the purity-universe proteins using the same data
split, model config, and class weights as compute_purity_saturation.py
(incident_only, SVM_L1 L1-penalty, log class weight, SPLIT_SEED=42).

At each elimination step:
  1. Fit CalibratedClassifierCV(LinearSVC(L1, C=best_C)) on incident-only dev set.
  2. Average |coef| across calibrated classifiers (matches rfe_importance.py).
  3. Remove protein with smallest importance.

Hyperparameter: C tuned once on the full protein set via 3-fold CV (Optuna,
same N_INNER_FOLDS=3, N_OPTUNA_TRIALS=20 as compute_purity_saturation.py).

Output
------
    {out}/rfe_protein_ranking.csv
        rfe_rank  -- 1 = most important (added first in forward sweep)
        protein   -- protein column name (e.g. "tgm2_resid")

Usage
-----
    python compute_rfe_ranking.py
    python compute_rfe_ranking.py --out path/to/out
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CEL_ROOT / "analysis" / "src"))

from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import CONTROL_LABEL, INCIDENT_LABEL, TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
log = logging.getLogger(__name__)

DATA_PATH = CEL_ROOT / "data" / "Celiac_dataset_proteomics_w_demo.parquet"
NOISE_SCORES_PATH = (
    CEL_ROOT / "operations/incident-validation/analysis/out/prevalent_noise_scores.csv"
)

TEST_FRAC = 0.20
SPLIT_SEED = 42
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20
CALIBRATION_CV = 5


# ---------------------------------------------------------------------------
# Model helpers (match compute_purity_saturation.py SVMSpec L1 + log weight)
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


def _build_model(C: float, class_weight: dict | None, seed: int) -> CalibratedClassifierCV:
    base = LinearSVC(
        penalty="l1", dual=False, C=C,
        class_weight=class_weight, max_iter=5000, random_state=seed,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=CALIBRATION_CV)


def _fit_model(
    model: CalibratedClassifierCV,
    X: np.ndarray,
    y: np.ndarray,
    cw: dict | None,
) -> None:
    sw = _sample_weight(cw, y)
    if sw is not None:
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)


def _extract_coefs(model: CalibratedClassifierCV, n_features: int) -> np.ndarray:
    """Average |coef| across calibrated classifiers (matches rfe_importance.py logic)."""
    coefs_list = []
    for cc in model.calibrated_classifiers_:
        est = getattr(cc, "estimator", None)
        if est is not None and hasattr(est, "coef_"):
            coefs_list.append(est.coef_.ravel())
    if not coefs_list:
        return np.zeros(n_features)
    return np.abs(np.mean(np.vstack(coefs_list), axis=0))


# ---------------------------------------------------------------------------
# Data loading (identical split to compute_purity_saturation.py)
# ---------------------------------------------------------------------------

def load_incident_dev(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return full df, dev_idx (label index), y_dev. Incident-only split."""
    log.info("Loading %s", data_path)
    df = read_proteomics_file(str(data_path))

    incident_mask = df[TARGET_COL] == INCIDENT_LABEL
    control_mask  = df[TARGET_COL] == CONTROL_LABEL

    ic_df = df[incident_mask | control_mask].copy()
    ic_df["_binary"] = (ic_df[TARGET_COL] == INCIDENT_LABEL).astype(int)

    sex_col = "sex"
    if sex_col in ic_df.columns:
        ic_df["_strat"] = ic_df["_binary"].astype(str) + "_" + ic_df[sex_col].astype(str)
    else:
        log.warning("'sex' column not found; stratifying by outcome only")
        ic_df["_strat"] = ic_df["_binary"].astype(str)

    dev_idx_full, _ = train_test_split(
        ic_df.index, test_size=TEST_FRAC, stratify=ic_df["_strat"], random_state=SPLIT_SEED,
    )
    dev_df = df.loc[dev_idx_full]
    dev_inc = dev_df[dev_df[TARGET_COL] == INCIDENT_LABEL].index
    dev_ctl = dev_df[dev_df[TARGET_COL] == CONTROL_LABEL].index

    dev_idx = np.concatenate([dev_inc, dev_ctl]).astype(int)
    y_dev   = np.concatenate([np.ones(len(dev_inc)), np.zeros(len(dev_ctl))]).astype(int)

    log.info("Dev set: %d incident, %d controls", len(dev_inc), len(dev_ctl))
    return df, dev_idx, y_dev


# ---------------------------------------------------------------------------
# C tuning
# ---------------------------------------------------------------------------

def tune_C(X_dev: np.ndarray, y_dev: np.ndarray, seed: int) -> float:
    """Tune LinearSVC L1 C on dev set via 3-fold CV. Returns best C."""

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        cw = _log_weight(y_dev)
        cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
        scores = []
        for tr_ix, va_ix in cv.split(X_dev, y_dev):
            X_tr, X_va = X_dev[tr_ix], X_dev[va_ix]
            y_tr, y_va = y_dev[tr_ix], y_dev[va_ix]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue
            sc = StandardScaler()
            model = _build_model(C, cw, seed)
            try:
                _fit_model(model, sc.fit_transform(X_tr), y_tr, cw)
                scores.append(
                    average_precision_score(y_va, model.predict_proba(sc.transform(X_va))[:, 1])
                )
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_C = study.best_params["C"]
    log.info("Tuned C=%.5f (best CV AUPRC=%.4f)", best_C, study.best_value)
    return best_C


# ---------------------------------------------------------------------------
# Backward elimination
# ---------------------------------------------------------------------------

def rfe_elimination(
    proteins: list[str],
    df: pd.DataFrame,
    dev_idx: np.ndarray,
    y_dev: np.ndarray,
    best_C: float,
    seed: int,
) -> list[str]:
    """Backward elimination; returns proteins in addition order (most important first)."""
    current = list(proteins)
    eliminated: list[str] = []  # proteins in order removed (worst → less worst → best)

    cw = _log_weight(y_dev)

    step = 0
    while len(current) > 1:
        step += 1
        X = df.loc[dev_idx, current].to_numpy(dtype=float)
        sc = StandardScaler()
        model = _build_model(best_C, cw, seed)
        _fit_model(model, sc.fit_transform(X), y_dev, cw)

        importances = _extract_coefs(model, len(current))
        protein_imp = dict(zip(current, importances, strict=True))

        if importances.max() == 0:
            rng = np.random.default_rng(seed + step)
            worst = str(rng.choice(current))
            log.warning("Step %d: all-zero importances, tie-breaking randomly → %s", step, worst)
        else:
            worst = min(protein_imp, key=protein_imp.get)

        eliminated.append(worst)
        current.remove(worst)

        if step % 5 == 0 or len(current) <= 5:
            log.info(
                "Step %d/%d: removed %s (|coef|=%.5f), %d remaining",
                step, len(proteins) - 1, worst, protein_imp[worst], len(current),
            )

    eliminated.append(current[0])

    # Reversed: last eliminated = rank 1 (most important), first eliminated = rank N
    return list(reversed(eliminated))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=CEL_ROOT / "operations/incident-validation/analysis/out",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    noise_df = pd.read_csv(NOISE_SCORES_PATH).sort_values("purity_rank", ascending=True)
    proteins = [p.lower() + "_resid" for p in noise_df["protein"].tolist()]
    log.info("Protein universe: %d proteins", len(proteins))
    log.info("Top-5 (purity order): %s", proteins[:5])

    df, dev_idx, y_dev = load_incident_dev(DATA_PATH)

    available = [p for p in proteins if p in df.columns]
    missing = set(proteins) - set(available)
    if missing:
        log.warning("Proteins missing from DataFrame (skipped): %s", sorted(missing))
    log.info("Available: %d / %d proteins", len(available), len(proteins))

    log.info("Tuning C on %d-protein dev set (%d trials)...", len(available), N_OPTUNA_TRIALS)
    X_full = df.loc[dev_idx, available].to_numpy(dtype=float)
    best_C = tune_C(X_full, y_dev, SPLIT_SEED)

    log.info("Starting backward elimination (%d steps)...", len(available) - 1)
    addition_order = rfe_elimination(available, df, dev_idx, y_dev, best_C, SPLIT_SEED)

    out_path = args.out / "rfe_protein_ranking.csv"
    rank_df = pd.DataFrame({
        "rfe_rank": range(1, len(addition_order) + 1),
        "protein": addition_order,
    })
    rank_df.to_csv(out_path, index=False)
    log.info("Saved %s", out_path)

    print(f"\n=== RFE protein ranking (C={best_C:.5f}, incident-only SVM_L1) ===")
    print(rank_df.to_string(index=False))


if __name__ == "__main__":
    main()
