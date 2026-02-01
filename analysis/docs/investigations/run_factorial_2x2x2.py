#!/usr/bin/env python3
"""
2x2x2 factorial experiment: n_cases x ratio x prevalent_frac.

Standalone script -- handles sampling, training, evaluation without ced CLI.

Design:
    Factors (8 cells):
        n_cases        in {50, 149}
        ratio          in {1, 5}   (controls per case)
        prevalent_frac in {0.5, 1.0}

    Guardrails:
        1. Fixed TEST set (seed=42, stratified, never changes)
        2. Paired seeds -- same shuffled pools; cells take prefixes (nesting)
        3. Frozen features (25-protein panel) + frozen hyperparams

Usage:
    # Step 1: tune hyperparams on baseline cell
    python run_factorial_2x2x2.py \\
        --data-path data/Celiac_dataset_proteomics_w_demo.parquet \\
        --panel-path data/fixed_panel.csv \\
        --output-dir results/factorial_2x2x2 \\
        --tune-baseline

    # Step 2: run full experiment with frozen hyperparams
    python run_factorial_2x2x2.py \\
        --data-path data/Celiac_dataset_proteomics_w_demo.parquet \\
        --panel-path data/fixed_panel.csv \\
        --output-dir results/factorial_2x2x2 \\
        --hyperparams-path results/factorial_2x2x2/frozen_hyperparams.yaml \\
        --n-seeds 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Add ced_ml to path for filter/schema reuse
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "analysis" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ced_ml.data.filters import apply_row_filters  # noqa: E402
from ced_ml.data.schema import (  # noqa: E402
    CONTROL_LABEL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Factorial design constants
# ---------------------------------------------------------------------------
N_CASES_LEVELS = [50, 149]
RATIO_LEVELS = [1, 5]
PREV_FRAC_LEVELS = [0.5, 1.0]

BASELINE_CELL = {"n_cases": 149, "ratio": 5, "prevalent_frac": 0.5}

TEST_SEED = 42
TEST_SIZE = 0.25  # fraction of incident cases held out for test


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_filter(data_path: Path) -> pd.DataFrame:
    """Load data and apply standard row filters."""
    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path)
    df, stats = apply_row_filters(df)
    logger.info(
        "Filtered: %d -> %d rows (removed %d uncertain controls, %d missing meta)",
        stats["n_in"],
        stats["n_out"],
        stats["n_removed_uncertain_controls"],
        stats["n_removed_dropna_meta_num"],
    )
    return df


def load_panel(panel_path: Path) -> list[str]:
    """Load fixed protein panel (one column name per line)."""
    panel = pd.read_csv(panel_path, header=None).iloc[:, 0].tolist()
    logger.info("Loaded %d-protein panel from %s", len(panel), panel_path)
    return panel


def create_fixed_test_set(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a single fixed TEST split (Guardrail 1).

    Returns:
        (train_pool_idx, test_idx) -- positional indices into df.
    """
    y = (df[TARGET_COL] == INCIDENT_LABEL).astype(int)
    # Exclude prevalent from test entirely -- they only appear in train pools
    mask_not_prev = df[TARGET_COL] != PREVALENT_LABEL
    idx_not_prev = np.where(mask_not_prev.values)[0]
    y_not_prev = y.values[idx_not_prev]

    train_idx, test_idx = train_test_split(
        idx_not_prev,
        test_size=TEST_SIZE,
        stratify=y_not_prev,
        random_state=TEST_SEED,
    )
    # Full train pool includes prevalent cases + train split of incident/controls
    prev_idx = np.where((df[TARGET_COL] == PREVALENT_LABEL).values)[0]
    train_pool_idx = np.sort(np.concatenate([train_idx, prev_idx]))

    logger.info(
        "Fixed TEST: %d samples (%d incident). Train pool: %d samples (%d prevalent).",
        len(test_idx),
        int(y.values[test_idx].sum()),
        len(train_pool_idx),
        len(prev_idx),
    )
    return train_pool_idx, test_idx


def build_pools(
    df: pd.DataFrame, pool_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split train pool into incident / prevalent / control index arrays."""
    labels = df[TARGET_COL].values
    I_pool = pool_idx[labels[pool_idx] == INCIDENT_LABEL]
    P_pool = pool_idx[labels[pool_idx] == PREVALENT_LABEL]
    C_pool = pool_idx[labels[pool_idx] == CONTROL_LABEL]
    logger.info(
        "Pools: %d incident, %d prevalent, %d controls",
        len(I_pool), len(P_pool), len(C_pool),
    )
    return I_pool, P_pool, C_pool


def sample_cell(
    I_pool: np.ndarray,
    P_pool: np.ndarray,
    C_pool: np.ndarray,
    n_cases: int,
    ratio: int,
    prevalent_frac: float,
    seed: int,
) -> np.ndarray:
    """
    Sample a training set for one factorial cell (Guardrail 2).

    Takes prefixes from seed-shuffled pools to guarantee nesting.
    """
    rng = np.random.RandomState(seed)

    I_shuf = rng.permutation(I_pool)
    P_shuf = rng.permutation(P_pool)
    C_shuf = rng.permutation(C_pool)

    n_prev = int(round(prevalent_frac * n_cases))
    n_controls = ratio * n_cases

    I = I_shuf[:n_cases]
    P = P_shuf[:n_prev]
    C = C_shuf[:n_controls]

    return np.sort(np.concatenate([I, P, C]))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _default_hyperparams() -> dict:
    """Sensible defaults (used if no frozen hyperparams provided)."""
    return {
        "LR_EN": {
            "C": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 2000,
            "solver": "saga",
            "penalty": "elasticnet",
        },
        "XGBoost": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        },
    }


def build_model(model_name: str, hyperparams: dict) -> Pipeline:
    """Build a sklearn Pipeline with StandardScaler + classifier."""
    hp = hyperparams.get(model_name, {})
    if model_name == "LR_EN":
        clf = LogisticRegression(**hp)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**hp)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# Baseline tuning (Guardrail 3)
# ---------------------------------------------------------------------------

def tune_baseline(
    df: pd.DataFrame,
    panel: list[str],
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
) -> dict:
    """Tune hyperparams on baseline cell via Optuna, save to YAML."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    I_pool, P_pool, C_pool = build_pools(df, train_pool_idx)
    train_idx = sample_cell(
        I_pool, P_pool, C_pool,
        n_cases=BASELINE_CELL["n_cases"],
        ratio=BASELINE_CELL["ratio"],
        prevalent_frac=BASELINE_CELL["prevalent_frac"],
        seed=0,
    )

    X_train = df.iloc[train_idx][panel].values
    y_train = (df.iloc[train_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values
    X_test = df.iloc[test_idx][panel].values
    y_test = (df.iloc[test_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values

    tuned = {}

    # -- LR_EN --
    def lr_objective(trial):
        C = trial.suggest_float("C", 1e-4, 100, log=True)
        l1 = trial.suggest_float("l1_ratio", 0.0, 1.0)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, l1_ratio=l1, penalty="elasticnet",
                solver="saga", max_iter=2000,
            )),
        ])
        pipe.fit(X_train, y_train)
        return roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    study_lr = optuna.create_study(direction="maximize")
    study_lr.optimize(lr_objective, n_trials=50, show_progress_bar=True)
    best_lr = study_lr.best_params
    best_lr.update({"penalty": "elasticnet", "solver": "saga", "max_iter": 2000})
    tuned["LR_EN"] = best_lr
    logger.info("LR_EN best AUROC=%.4f, params=%s", study_lr.best_value, best_lr)

    # -- XGBoost --
    from xgboost import XGBClassifier

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(**params)),
        ])
        pipe.fit(X_train, y_train)
        return roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
    best_xgb = study_xgb.best_params
    best_xgb.update({"use_label_encoder": False, "eval_metric": "logloss"})
    tuned["XGBoost"] = best_xgb
    logger.info("XGBoost best AUROC=%.4f, params=%s", study_xgb.best_value, best_xgb)

    # Save
    out_path = output_dir / "frozen_hyperparams.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(tuned, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved frozen hyperparams to %s", out_path)
    return tuned


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration_slope_intercept(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """Compute calibration slope and intercept via logistic regression on logit(p)."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    from sklearn.linear_model import LogisticRegression as LR
    cal = LR(penalty=None, solver="lbfgs", max_iter=1000)
    cal.fit(logit_p.reshape(-1, 1), y_true)
    slope = float(cal.coef_[0, 0])
    intercept = float(cal.intercept_[0])
    return slope, intercept


def sensitivity_at_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.95
) -> float:
    """Sensitivity at a fixed specificity threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    # Find highest tpr where spec >= target
    mask = spec >= target_spec
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all factorial metrics."""
    auroc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    sens95 = sensitivity_at_specificity(y_true, y_prob, 0.95)
    return {
        "AUROC": auroc,
        "PRAUC": prauc,
        "Brier": brier,
        "cal_slope": slope,
        "cal_intercept": intercept,
        "sens_at_spec95": sens95,
    }


def compute_score_distributions(
    pipe: Pipeline,
    df: pd.DataFrame,
    panel: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Compute mean predicted probability per case-type group.

    Returns dict with mean_prob_incident (test), mean_prob_prevalent (train),
    mean_prob_control (test), and score_gap (incident - prevalent).
    """
    labels_train = df.iloc[train_idx][TARGET_COL].values
    labels_test = df.iloc[test_idx][TARGET_COL].values

    prob_train = pipe.predict_proba(df.iloc[train_idx][panel].values)[:, 1]
    prob_test = pipe.predict_proba(df.iloc[test_idx][panel].values)[:, 1]

    mean_incident = float(prob_test[labels_test == INCIDENT_LABEL].mean())
    mean_control = float(prob_test[labels_test == CONTROL_LABEL].mean())

    prev_mask = labels_train == PREVALENT_LABEL
    mean_prevalent = float(prob_train[prev_mask].mean()) if prev_mask.any() else np.nan

    score_gap = mean_incident - mean_prevalent if not np.isnan(mean_prevalent) else np.nan

    return {
        "mean_prob_incident": round(mean_incident, 6),
        "mean_prob_prevalent": round(mean_prevalent, 6),
        "mean_prob_control": round(mean_control, 6),
        "score_gap": round(score_gap, 6) if not np.isnan(score_gap) else np.nan,
    }


def extract_feature_importances(
    pipe: Pipeline,
    panel: list[str],
    model_name: str,
) -> dict[str, float]:
    """Extract feature importances from a fitted pipeline.

    Returns dict mapping feature name to importance (absolute coefficient
    for LR, feature_importances_ for tree models).
    """
    clf = pipe.named_steps["clf"]
    if model_name == "LR_EN":
        importances = np.abs(clf.coef_[0])
    elif model_name == "XGBoost":
        importances = clf.feature_importances_
    else:
        return {}
    return dict(zip(panel, importances.tolist()))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    df: pd.DataFrame,
    panel: list[str],
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    hyperparams: dict,
    n_seeds: int,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all factorial cells across seeds and models.

    Returns:
        (results_df, feature_importances_df)
    """
    I_pool, P_pool, C_pool = build_pools(df, train_pool_idx)

    X_test = df.iloc[test_idx][panel].values
    y_test = (df.iloc[test_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values

    cells = list(product(N_CASES_LEVELS, RATIO_LEVELS, PREV_FRAC_LEVELS))
    total = n_seeds * len(cells) * len(models)
    logger.info(
        "Running %d seeds x %d cells x %d models = %d total runs",
        n_seeds, len(cells), len(models), total,
    )

    rows = []
    fi_rows = []
    done = 0
    for seed in range(n_seeds):
        for n_cases, ratio, prev_frac in cells:
            cell_train_idx = sample_cell(
                I_pool, P_pool, C_pool,
                n_cases=n_cases,
                ratio=ratio,
                prevalent_frac=prev_frac,
                seed=seed,
            )
            X_train = df.iloc[cell_train_idx][panel].values
            y_train = (
                df.iloc[cell_train_idx][TARGET_COL] == INCIDENT_LABEL
            ).astype(int).values

            for model_name in models:
                t0 = time.perf_counter()
                pipe = build_model(model_name, hyperparams)
                pipe.fit(X_train, y_train)
                y_prob = pipe.predict_proba(X_test)[:, 1]
                runtime = time.perf_counter() - t0

                metrics = compute_metrics(y_test, y_prob)
                score_dist = compute_score_distributions(
                    pipe, df, panel, cell_train_idx, test_idx,
                )
                row = {
                    "seed": seed,
                    "n_cases": n_cases,
                    "ratio": ratio,
                    "prevalent_frac": prev_frac,
                    "train_N": len(cell_train_idx),
                    "model": model_name,
                    **metrics,
                    **score_dist,
                    "runtime_s": round(runtime, 2),
                }
                rows.append(row)

                # Feature importances
                fi = extract_feature_importances(pipe, panel, model_name)
                for feat, imp in fi.items():
                    fi_rows.append({
                        "seed": seed,
                        "n_cases": n_cases,
                        "ratio": ratio,
                        "prevalent_frac": prev_frac,
                        "model": model_name,
                        "feature": feat,
                        "importance": imp,
                    })

                done += 1
                if done % 16 == 0 or done == total:
                    logger.info("Progress: %d / %d runs complete", done, total)

    return pd.DataFrame(rows), pd.DataFrame(fi_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2x2x2 factorial experiment (n_cases x ratio x prevalent_frac)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path", type=Path, required=True,
        help="Path to proteomics parquet file",
    )
    parser.add_argument(
        "--panel-path", type=Path, required=True,
        help="Path to fixed_panel.csv (one protein per line)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/factorial_2x2x2"),
        help="Output directory (default: results/factorial_2x2x2)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds (default: 10)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["LR_EN", "XGBoost"],
        help="Models to run (default: LR_EN XGBoost)",
    )
    parser.add_argument(
        "--tune-baseline", action="store_true",
        help="Tune hyperparams on baseline cell via Optuna, then exit",
    )
    parser.add_argument(
        "--hyperparams-path", type=Path, default=None,
        help="Path to frozen_hyperparams.yaml (required unless --tune-baseline)",
    )
    args = parser.parse_args()

    # Load data
    df = load_and_filter(args.data_path)
    panel = load_panel(args.panel_path)

    # Validate panel columns exist
    missing = [c for c in panel if c not in df.columns]
    if missing:
        logger.error("Panel columns missing from data: %s", missing)
        sys.exit(1)

    # Fixed test set
    train_pool_idx, test_idx = create_fixed_test_set(df)

    # Save test indices for reproducibility
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "test_indices.csv", test_idx, fmt="%d",
        header="positional index into filtered dataframe",
    )

    if args.tune_baseline:
        tune_baseline(df, panel, train_pool_idx, test_idx, args.output_dir)
        logger.info("Baseline tuning complete. Re-run with --hyperparams-path to run experiment.")
        return

    # Load frozen hyperparams
    if args.hyperparams_path is None:
        logger.warning("No --hyperparams-path provided; using default hyperparams.")
        hyperparams = _default_hyperparams()
    else:
        with open(args.hyperparams_path) as f:
            hyperparams = yaml.safe_load(f)
        logger.info("Loaded frozen hyperparams from %s", args.hyperparams_path)

    # Run experiment
    results, feat_imp = run_experiment(
        df, panel, train_pool_idx, test_idx,
        hyperparams=hyperparams,
        n_seeds=args.n_seeds,
        models=args.models,
    )

    # Save results
    out_csv = args.output_dir / "factorial_results.csv"
    results.to_csv(out_csv, index=False)
    logger.info("Saved %d result rows to %s", len(results), out_csv)

    # Save feature importances
    fi_csv = args.output_dir / "feature_importances.csv"
    feat_imp.to_csv(fi_csv, index=False)
    logger.info("Saved feature importances to %s", fi_csv)

    # Print summary
    print("\n--- Summary ---")
    for model in args.models:
        sub = results[results["model"] == model]
        print(f"\n{model} (n={len(sub)} runs):")
        for metric in ["AUROC", "PRAUC", "Brier", "cal_slope", "sens_at_spec95"]:
            print(f"  {metric}: {sub[metric].mean():.4f} +/- {sub[metric].std():.4f}")


if __name__ == "__main__":
    main()
