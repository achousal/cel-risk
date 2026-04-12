#!/usr/bin/env python3
"""
Post-hoc SHAP computation and OOF predictions for incident validation models.

Re-derives the exact dev/test split deterministically (seed=42), refits the
winning model config on the full dev set, then:
  - Saves X_test.csv
  - Computes SHAP values via shap.LinearExplainer
  - Saves shap_values.csv and shap_expected_value.txt
  - Computes OOF predictions (5-fold CV on dev set)
  - Saves oof_predictions.csv

Models: LR_EN, SVM_L1, SVM_L2
Output written to results/{model_dir}/ alongside existing training artifacts.

Usage:
    cd /Users/andreschousal/Projects/Chowell_Lab/cel-risk
    python experiments/optimal-setup/incident-validation/analysis/compute_shap_oof.py
    python experiments/optimal-setup/incident-validation/analysis/compute_shap_oof.py --models LR_EN SVM_L1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_OUTPUT_DIRS = {
    "LR_EN": "results/incident-validation/lr/LR_EN",
    "SVM_L1": "results/incident-validation/lr/SVM_L1",
    "SVM_L2": "results/incident-validation/lr/SVM_L2",
}

VALID_MODELS = tuple(MODEL_OUTPUT_DIRS.keys())

# ---------------------------------------------------------------------------
# Helpers copied from training script
# ---------------------------------------------------------------------------


def compute_class_weight(scheme: str, y: np.ndarray):
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

    return {0: 1.0, 1: max(float(w1), 1.0)}


def get_training_indices(
    strategy: str,
    dev_incident_idx: np.ndarray,
    dev_control_idx: np.ndarray,
    prevalent_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, binary labels) for a given training strategy."""
    if strategy == "incident_only":
        idx = np.concatenate([dev_incident_idx, dev_control_idx])
        y = np.concatenate([
            np.ones(len(dev_incident_idx)),
            np.zeros(len(dev_control_idx)),
        ])
    elif strategy == "incident_prevalent":
        idx = np.concatenate([dev_incident_idx, prevalent_idx, dev_control_idx])
        y = np.concatenate([
            np.ones(len(dev_incident_idx)),
            np.ones(len(prevalent_idx)),
            np.zeros(len(dev_control_idx)),
        ])
    elif strategy == "prevalent_only":
        idx = np.concatenate([prevalent_idx, dev_control_idx])
        y = np.concatenate([
            np.ones(len(prevalent_idx)),
            np.zeros(len(dev_control_idx)),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return idx.astype(int), y.astype(int)


# ---------------------------------------------------------------------------
# Data loading and split re-derivation
# ---------------------------------------------------------------------------


def load_and_split(data_path: Path, test_frac: float = 0.20, split_seed: int = 42) -> dict:
    """
    Re-derive the exact dev/test split used during training.
    Uses the same logic: incident+control pool, sex-stratified, seed=42.
    """
    logger.info("Loading data from %s", data_path)
    df = read_proteomics_file(str(data_path))
    logger.info("Loaded %d samples, %d columns", len(df), len(df.columns))

    incident_mask = df[TARGET_COL] == INCIDENT_LABEL
    prevalent_mask = df[TARGET_COL] == PREVALENT_LABEL
    control_mask = df[TARGET_COL] == CONTROL_LABEL

    logger.info(
        "Groups: %d incident, %d prevalent, %d controls",
        incident_mask.sum(), prevalent_mask.sum(), control_mask.sum(),
    )

    # Pool incident + controls for the split
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
        test_size=test_frac,
        stratify=ic_df["_strat"],
        random_state=split_seed,
    )

    test_set = set(test_idx)
    dev_incident_idx = np.array([i for i in df.index[incident_mask] if i not in test_set])
    dev_control_idx = np.array([i for i in df.index[control_mask] if i not in test_set])
    test_incident_idx = np.array([i for i in df.index[incident_mask] if i in test_set])
    test_control_idx = np.array([i for i in df.index[control_mask] if i in test_set])
    prevalent_idx = np.array(df.index[prevalent_mask])

    logger.info(
        "Dev:  %d incident + %d controls | Test: %d incident + %d controls | Prevalent: %d",
        len(dev_incident_idx), len(dev_control_idx),
        len(test_incident_idx), len(test_control_idx),
        len(prevalent_idx),
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
# Model builders (mirror training script)
# ---------------------------------------------------------------------------


def build_lr_en(C: float, l1_ratio: float, class_weight, max_iter: int = 2000, seed: int = 42):
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        penalty="elasticnet",
        class_weight=class_weight,
        solver="saga",
        max_iter=max_iter,
        random_state=seed,
    )


def build_svm(penalty: str, C: float, class_weight, calibration_cv: int = 5, max_iter: int = 2000, seed: int = 42):
    if penalty == "l1":
        base = LinearSVC(
            penalty="l1",
            dual=False,
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
        )
    else:
        base = LinearSVC(
            penalty="l2",
            dual=True,
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
        )
    return CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv)


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------


def shap_explainer_for_lr(model: LogisticRegression, X_background: np.ndarray):
    """LinearExplainer for a LogisticRegression model."""
    return shap.LinearExplainer(model, X_background)


def shap_explainer_for_svm(cal_model: CalibratedClassifierCV, X_background: np.ndarray, n_features: int):
    """
    LinearExplainer for a CalibratedClassifierCV wrapping LinearSVC.
    Averages coefficients across calibration folds and constructs a surrogate
    SGDClassifier so shap.LinearExplainer can inspect coef_ / intercept_.
    """
    mean_coef = np.mean(
        [cc.estimator.coef_.ravel() for cc in cal_model.calibrated_classifiers_],
        axis=0,
    )
    mean_intercept = np.mean(
        [cc.estimator.intercept_[0] for cc in cal_model.calibrated_classifiers_]
    )

    surrogate = SGDClassifier()
    surrogate.coef_ = mean_coef.reshape(1, -1)
    surrogate.intercept_ = np.array([mean_intercept])
    surrogate.classes_ = np.array([0, 1])

    return shap.LinearExplainer(surrogate, X_background)


# ---------------------------------------------------------------------------
# Per-model best param extraction
# ---------------------------------------------------------------------------


def read_best_params_lr_en(cv_results_path: Path, best_strategy: str, best_weight: str) -> dict:
    """Read cv_results.csv and return median C and l1_ratio for the winning config."""
    cv = pd.read_csv(cv_results_path)
    mask = (cv["strategy"] == best_strategy) & (cv["weight_scheme"] == best_weight)
    subset = cv[mask]
    if subset.empty:
        raise ValueError(
            f"No cv_results rows for strategy={best_strategy}, weight={best_weight}"
        )
    return {
        "C": float(subset["best_C"].median()),
        "l1_ratio": float(subset["best_l1_ratio"].median()),
    }


def read_best_params_svm(cv_results_path: Path, best_strategy: str, best_weight: str) -> dict:
    """Read cv_results.csv and return median C for the winning config."""
    cv = pd.read_csv(cv_results_path)
    mask = (cv["strategy"] == best_strategy) & (cv["weight_scheme"] == best_weight)
    subset = cv[mask]
    if subset.empty:
        raise ValueError(
            f"No cv_results rows for strategy={best_strategy}, weight={best_weight}"
        )
    return {"C": float(subset["best_C"].median())}


# ---------------------------------------------------------------------------
# Main per-model processing
# ---------------------------------------------------------------------------


def process_model(model_id: str, data: dict, cel_root: Path) -> None:
    out_dir = cel_root / MODEL_OUTPUT_DIRS[model_id]
    logger.info("=" * 60)
    logger.info("Processing model: %s  ->  %s", model_id, out_dir)

    # -- Read saved artifacts --
    feature_panel = pd.read_csv(out_dir / "feature_panel.csv")["protein"].tolist()
    logger.info("Feature panel: %d proteins", len(feature_panel))

    strategy_comparison = pd.read_csv(out_dir / "strategy_comparison.csv")
    best_row = strategy_comparison.sort_values("mean_auprc", ascending=False).iloc[0]
    best_strategy = best_row["strategy"]
    best_weight = best_row["weight_scheme"]
    logger.info(
        "Best config: strategy=%s, weight=%s, mean_auprc=%.4f",
        best_strategy, best_weight, best_row["mean_auprc"],
    )

    cv_results_path = out_dir / "cv_results.csv"

    # -- Re-derive split components --
    df = data["df"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    test_incident_idx = data["test_incident_idx"]
    test_control_idx = data["test_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    # Build dev training set (same as final_refit_and_test in training script)
    train_idx, y_train = get_training_indices(
        best_strategy, dev_incident_idx, dev_control_idx, prevalent_idx
    )
    X_train_raw = df.loc[train_idx, feature_panel].to_numpy(dtype=float)
    cw = compute_class_weight(best_weight, y_train)

    # Build test set
    test_idx = np.concatenate([test_incident_idx, test_control_idx])
    y_test = np.concatenate([
        np.ones(len(test_incident_idx)),
        np.zeros(len(test_control_idx)),
    ]).astype(int)
    X_test_raw = df.loc[test_idx, feature_panel].to_numpy(dtype=float)

    # Scaler fit on dev
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # -- Build and fit final model --
    if model_id == "LR_EN":
        best_params = read_best_params_lr_en(cv_results_path, best_strategy, best_weight)
        logger.info("Best params: C=%.6f, l1_ratio=%.4f", best_params["C"], best_params["l1_ratio"])
        model = build_lr_en(
            C=best_params["C"],
            l1_ratio=best_params["l1_ratio"],
            class_weight=cw,
        )
    else:
        penalty = "l1" if model_id == "SVM_L1" else "l2"
        best_params = read_best_params_svm(cv_results_path, best_strategy, best_weight)
        logger.info("Best params: C=%.6f", best_params["C"])
        model = build_svm(
            penalty=penalty,
            C=best_params["C"],
            class_weight=cw,
        )

    logger.info("Fitting final model on dev set (%d samples)...", len(y_train))
    model.fit(X_train_scaled, y_train)
    logger.info("Fitting done.")

    # -- Save X_test --
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_panel)
    X_test_df.insert(0, "eid", (
        df.loc[test_idx, ID_COL].values if ID_COL in df.columns else test_idx
    ))
    X_test_df.insert(1, "y_true", y_test)
    X_test_df.to_csv(out_dir / "X_test.csv", index=False)
    logger.info("Saved X_test.csv (%d rows)", len(X_test_df))

    # -- Compute SHAP values --
    logger.info("Computing SHAP values...")
    if model_id == "LR_EN":
        explainer = shap_explainer_for_lr(model, X_train_scaled)
    else:
        explainer = shap_explainer_for_svm(model, X_train_scaled, len(feature_panel))

    shap_vals = explainer.shap_values(X_test_scaled)

    # shap.LinearExplainer may return a list (one array per class) for binary classifiers
    if isinstance(shap_vals, list):
        # Use class-1 (positive class) SHAP values
        shap_arr = shap_vals[1]
    else:
        shap_arr = shap_vals

    shap_df = pd.DataFrame(shap_arr, columns=feature_panel)
    shap_df.to_csv(out_dir / "shap_values.csv", index=False)
    logger.info("Saved shap_values.csv (%s)", shap_df.shape)

    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )
    (out_dir / "shap_expected_value.txt").write_text(str(float(expected_value)) + "\n")
    logger.info("Saved shap_expected_value.txt (%.6f)", float(expected_value))

    # -- OOF predictions (5-fold CV on dev set) --
    logger.info("Computing OOF predictions (5-fold CV on dev set)...")

    # OOF CV uses incident+controls in dev only (same outer CV logic as training)
    dev_ic_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    dev_ic_labels = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_eids: list[np.ndarray] = []
    oof_y_true: list[np.ndarray] = []
    oof_y_prob: list[np.ndarray] = []

    for fold_i, (train_pos, val_pos) in enumerate(outer_cv.split(dev_ic_idx, dev_ic_labels)):
        fold_train_ic = dev_ic_idx[train_pos]
        fold_val_ic = dev_ic_idx[val_pos]

        dev_incident_set = set(dev_incident_idx)
        dev_control_set = set(dev_control_idx)

        fold_train_incident = np.array([i for i in fold_train_ic if i in dev_incident_set])
        fold_train_controls = np.array([i for i in fold_train_ic if i in dev_control_set])

        fold_train_idx, fold_y_train = get_training_indices(
            best_strategy, fold_train_incident, fold_train_controls, prevalent_idx
        )

        val_incident = np.array([i for i in fold_val_ic if i in dev_incident_set])
        val_controls = np.array([i for i in fold_val_ic if i in dev_control_set])
        fold_val_idx = np.concatenate([val_incident, val_controls])
        fold_y_val = np.concatenate([
            np.ones(len(val_incident)),
            np.zeros(len(val_controls)),
        ]).astype(int)

        X_fold_train = df.loc[fold_train_idx, feature_panel].to_numpy(dtype=float)
        X_fold_val = df.loc[fold_val_idx, feature_panel].to_numpy(dtype=float)

        fold_cw = compute_class_weight(best_weight, fold_y_train)
        fold_seed = 42 + fold_i

        fold_scaler = StandardScaler()
        X_fold_train_s = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_s = fold_scaler.transform(X_fold_val)

        if model_id == "LR_EN":
            fold_model = build_lr_en(
                C=best_params["C"],
                l1_ratio=best_params["l1_ratio"],
                class_weight=fold_cw,
                seed=fold_seed,
            )
        else:
            penalty = "l1" if model_id == "SVM_L1" else "l2"
            fold_model = build_svm(
                penalty=penalty,
                C=best_params["C"],
                class_weight=fold_cw,
                seed=fold_seed,
            )

        fold_model.fit(X_fold_train_s, fold_y_train)
        fold_probs = fold_model.predict_proba(X_fold_val_s)[:, 1]

        fold_eids = (
            df.loc[fold_val_idx, ID_COL].values
            if ID_COL in df.columns
            else fold_val_idx
        )
        oof_eids.append(fold_eids)
        oof_y_true.append(fold_y_val)
        oof_y_prob.append(fold_probs)

        logger.info("  Fold %d: %d val samples, %d pos", fold_i, len(fold_y_val), int(fold_y_val.sum()))

    oof_df = pd.DataFrame({
        "eid": np.concatenate(oof_eids),
        "y_true": np.concatenate(oof_y_true).astype(int),
        "y_prob": np.concatenate(oof_y_prob),
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    logger.info("Saved oof_predictions.csv (%d rows, %d positive)", len(oof_df), oof_df["y_true"].sum())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc SHAP + OOF computation for incident validation models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(VALID_MODELS),
        default=list(VALID_MODELS),
        help="Which models to process (default: all three).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override path to the proteomics parquet file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data path: prefer CLI arg, fall back to local, then HPC path
    local_data = CEL_ROOT / "data" / "Celiac_dataset_proteomics_w_demo.parquet"
    hpc_data = Path("/sc/arion/projects/vascbrain/andres/cel-risk/data/Celiac_dataset_proteomics_w_demo.parquet")

    if args.data is not None:
        data_path = args.data
    elif local_data.exists():
        data_path = local_data
    elif hpc_data.exists():
        data_path = hpc_data
    else:
        logger.error(
            "Data file not found at %s or %s. Use --data to specify the path.",
            local_data, hpc_data,
        )
        sys.exit(1)

    logger.info("Data path: %s", data_path)

    # Load and split once -- shared across all models (same split seed)
    data = load_and_split(data_path, test_frac=0.20, split_seed=42)

    for model_id in args.models:
        out_dir = CEL_ROOT / MODEL_OUTPUT_DIRS[model_id]
        if not out_dir.exists():
            logger.warning("Output dir not found for %s (%s) -- skipping.", model_id, out_dir)
            continue
        process_model(model_id, data, CEL_ROOT)

    logger.info("All done.")


if __name__ == "__main__":
    main()
