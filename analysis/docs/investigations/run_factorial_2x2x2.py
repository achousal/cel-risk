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
        3. Frozen features (25-protein panel)
        4. Cell-specific hyperparameter tuning (each cell gets optimal hyperparams)

Usage:
    # Step 1: tune hyperparams for ALL factorial cells (cell-specific tuning)
    python run_factorial_2x2x2.py \\
        --data-path data/Celiac_dataset_proteomics_w_demo.parquet \\
        --panel-path data/fixed_panel.csv \\
        --output-dir results/factorial_2x2x2 \\
        --tune-cells \\
        --n-trials 50

    # Step 2: run full experiment with cell-specific frozen hyperparams
    python run_factorial_2x2x2.py \\
        --data-path data/Celiac_dataset_proteomics_w_demo.parquet \\
        --panel-path data/fixed_panel.csv \\
        --output-dir results/factorial_2x2x2 \\
        --hyperparams-path results/factorial_2x2x2/cell_hyperparams.yaml \\
        --n-seeds 10
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from functools import partial
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

TEST_SEED = 42
TEST_SIZE = 0.25  # fraction of incident cases held out for test
TUNING_SEED = 0  # fixed seed for reproducible tuning splits
VAL_FRAC = 0.2  # fraction of cell used for validation during tuning


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
        len(I_pool),
        len(P_pool),
        len(C_pool),
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

    I = I_shuf[:n_cases]  # noqa: E741 - I/P/C are standard domain abbreviations
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
# Cell-specific hyperparameter tuning (Guardrail 4)
# ---------------------------------------------------------------------------


def tune_cell(
    df: pd.DataFrame,
    panel: list[str],
    I_pool: np.ndarray,
    P_pool: np.ndarray,
    C_pool: np.ndarray,
    n_cases: int,
    ratio: int,
    prevalent_frac: float,
    test_idx: np.ndarray,
    models: list[str],
    n_trials: int = 50,
) -> dict:
    """Tune hyperparams for a specific factorial cell via train/val split.

    Args:
        df: Full dataframe
        panel: List of feature column names
        I_pool, P_pool, C_pool: Index pools for incident/prevalent/control
        n_cases, ratio, prevalent_frac: Cell configuration
        test_idx: Held-out test indices (never used for tuning)
        models: List of model names to tune
        n_trials: Number of Optuna trials per model

    Returns:
        Dict mapping model name to best hyperparams for this cell
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Sample this cell using tuning seed
    cell_idx = sample_cell(
        I_pool,
        P_pool,
        C_pool,
        n_cases=n_cases,
        ratio=ratio,
        prevalent_frac=prevalent_frac,
        seed=TUNING_SEED,
    )

    # Split cell into train/val (80/20)
    y_cell = (df.iloc[cell_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values
    train_idx, val_idx = train_test_split(
        cell_idx,
        test_size=VAL_FRAC,
        stratify=y_cell,
        random_state=999,
    )

    X_train = df.iloc[train_idx][panel].values
    y_train = (df.iloc[train_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values
    X_val = df.iloc[val_idx][panel].values
    y_val = (df.iloc[val_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values

    # Test set (only for logging, never used in optimization)
    X_test = df.iloc[test_idx][panel].values
    y_test = (df.iloc[test_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values

    logger.info(
        "Tuning cell (n_cases=%d, ratio=%d, prev_frac=%.1f): train=%d, val=%d",
        n_cases,
        ratio,
        prevalent_frac,
        len(train_idx),
        len(val_idx),
    )

    tuned = {}

    # Tune each model
    for model_name in models:
        if model_name == "LR_EN":

            def lr_objective(trial):
                C = trial.suggest_float("C", 1e-4, 100, log=True)
                l1 = trial.suggest_float("l1_ratio", 0.0, 1.0)
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=C,
                                l1_ratio=l1,
                                penalty="elasticnet",
                                solver="saga",
                                max_iter=2000,
                            ),
                        ),
                    ]
                )
                pipe.fit(X_train, y_train)
                return roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])

            study = optuna.create_study(direction="maximize")
            study.optimize(lr_objective, n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            best.update({"penalty": "elasticnet", "solver": "saga", "max_iter": 2000})
            tuned[model_name] = best

            # Log val and test AUROC
            final_pipe = build_model(model_name, {model_name: best})
            final_pipe.fit(X_train, y_train)
            val_auroc = study.best_value
            test_auroc = roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:, 1])
            logger.info(
                "  LR_EN: val=%.4f, test=%.4f (held out)",
                val_auroc,
                test_auroc,
            )

        elif model_name == "XGBoost":
            from xgboost import XGBClassifier

            def xgb_objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-8, 10.0, log=True
                    ),
                    "eval_metric": "logloss",
                }
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", XGBClassifier(**params)),
                    ]
                )
                pipe.fit(X_train, y_train)
                return roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])

            study = optuna.create_study(direction="maximize")
            study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
            best = study.best_params
            best.update({"eval_metric": "logloss"})
            tuned[model_name] = best

            # Log val and test AUROC
            final_pipe = build_model(model_name, {model_name: best})
            final_pipe.fit(X_train, y_train)
            val_auroc = study.best_value
            test_auroc = roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:, 1])
            logger.info(
                "  XGBoost: val=%.4f, test=%.4f (held out)",
                val_auroc,
                test_auroc,
            )

    return tuned


def tune_all_cells(
    df: pd.DataFrame,
    panel: list[str],
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    models: list[str],
    n_trials: int,
    output_dir: Path,
) -> dict:
    """Tune hyperparams for all factorial cells.

    Returns nested dict: {(n_cases, ratio, prev_frac): {model: params}}
    """
    I_pool, P_pool, C_pool = build_pools(df, train_pool_idx)
    cells = list(product(N_CASES_LEVELS, RATIO_LEVELS, PREV_FRAC_LEVELS))

    logger.info(
        "Tuning %d cells × %d models × %d trials = %d total Optuna runs",
        len(cells),
        len(models),
        n_trials,
        len(cells) * len(models) * n_trials,
    )

    all_hyperparams = {}
    for i, (n_cases, ratio, prev_frac) in enumerate(cells, 1):
        logger.info(
            "Cell %d/%d: n_cases=%d, ratio=%d, prev_frac=%.1f",
            i,
            len(cells),
            n_cases,
            ratio,
            prev_frac,
        )
        cell_key = f"n{n_cases}_r{ratio}_p{prev_frac}"

        hyperparams = tune_cell(
            df,
            panel,
            I_pool,
            P_pool,
            C_pool,
            n_cases=n_cases,
            ratio=ratio,
            prevalent_frac=prev_frac,
            test_idx=test_idx,
            models=models,
            n_trials=n_trials,
        )
        all_hyperparams[cell_key] = hyperparams

    # Save to YAML
    out_path = output_dir / "cell_hyperparams.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(all_hyperparams, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved cell-specific hyperparams to %s", out_path)

    return all_hyperparams


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

    score_gap = (
        mean_incident - mean_prevalent if not np.isnan(mean_prevalent) else np.nan
    )

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
    return dict(zip(panel, importances.tolist(), strict=False))


# ---------------------------------------------------------------------------
# Parallelized experiment helpers
# ---------------------------------------------------------------------------


def _run_single_job(
    job_spec: dict,
    df: pd.DataFrame,
    panel: list[str],
    I_pool: np.ndarray,
    P_pool: np.ndarray,
    C_pool: np.ndarray,
    test_idx: np.ndarray,
    hyperparams: dict,
) -> tuple[dict, list[dict]]:
    """Run a single (seed, cell, model) combination.

    Returns:
        (result_row, feature_importance_rows)
    """
    seed = job_spec["seed"]
    n_cases = job_spec["n_cases"]
    ratio = job_spec["ratio"]
    prev_frac = job_spec["prevalent_frac"]
    model_name = job_spec["model"]
    cell_key = f"n{n_cases}_r{ratio}_p{prev_frac}"

    # Sample cell
    cell_train_idx = sample_cell(
        I_pool,
        P_pool,
        C_pool,
        n_cases=n_cases,
        ratio=ratio,
        prevalent_frac=prev_frac,
        seed=seed,
    )
    X_train = df.iloc[cell_train_idx][panel].values
    y_train = (df.iloc[cell_train_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values

    X_test = df.iloc[test_idx][panel].values
    y_test = (df.iloc[test_idx][TARGET_COL] == INCIDENT_LABEL).astype(int).values

    # Get cell-specific hyperparams
    if cell_key in hyperparams:
        model_hyperparams = hyperparams[cell_key]
    else:
        model_hyperparams = hyperparams

    # Train and evaluate
    t0 = time.perf_counter()
    pipe = build_model(model_name, model_hyperparams)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    runtime = time.perf_counter() - t0

    metrics = compute_metrics(y_test, y_prob)
    score_dist = compute_score_distributions(
        pipe,
        df,
        panel,
        cell_train_idx,
        test_idx,
    )

    result_row = {
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

    # Feature importances
    fi = extract_feature_importances(pipe, panel, model_name)
    fi_rows = [
        {
            "seed": seed,
            "n_cases": n_cases,
            "ratio": ratio,
            "prevalent_frac": prev_frac,
            "model": model_name,
            "feature": feat,
            "importance": imp,
        }
        for feat, imp in fi.items()
    ]

    return result_row, fi_rows


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

    Args:
        hyperparams: Nested dict {cell_key: {model: params}} for cell-specific tuning,
                     or flat dict {model: params} for shared hyperparams (legacy)

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
        n_seeds,
        len(cells),
        len(models),
        total,
    )

    rows = []
    fi_rows = []
    done = 0
    for seed in range(n_seeds):
        for n_cases, ratio, prev_frac in cells:
            cell_key = f"n{n_cases}_r{ratio}_p{prev_frac}"

            cell_train_idx = sample_cell(
                I_pool,
                P_pool,
                C_pool,
                n_cases=n_cases,
                ratio=ratio,
                prevalent_frac=prev_frac,
                seed=seed,
            )
            X_train = df.iloc[cell_train_idx][panel].values
            y_train = (
                (df.iloc[cell_train_idx][TARGET_COL] == INCIDENT_LABEL)
                .astype(int)
                .values
            )

            for model_name in models:
                t0 = time.perf_counter()

                # Get cell-specific hyperparams if available, else use shared
                if cell_key in hyperparams:
                    model_hyperparams = hyperparams[cell_key]
                else:
                    model_hyperparams = hyperparams

                pipe = build_model(model_name, model_hyperparams)
                pipe.fit(X_train, y_train)
                y_prob = pipe.predict_proba(X_test)[:, 1]
                runtime = time.perf_counter() - t0

                metrics = compute_metrics(y_test, y_prob)
                score_dist = compute_score_distributions(
                    pipe,
                    df,
                    panel,
                    cell_train_idx,
                    test_idx,
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
                    fi_rows.append(
                        {
                            "seed": seed,
                            "n_cases": n_cases,
                            "ratio": ratio,
                            "prevalent_frac": prev_frac,
                            "model": model_name,
                            "feature": feat,
                            "importance": imp,
                        }
                    )

                done += 1
                if done % 16 == 0 or done == total:
                    logger.info("Progress: %d / %d runs complete", done, total)

    return pd.DataFrame(rows), pd.DataFrame(fi_rows)


def run_experiment_parallel(
    df: pd.DataFrame,
    panel: list[str],
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    hyperparams: dict,
    n_seeds: int,
    models: list[str],
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all factorial cells in parallel across seeds and models.

    Args:
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)

    Returns:
        (results_df, feature_importances_df)
    """
    I_pool, P_pool, C_pool = build_pools(df, train_pool_idx)
    cells = list(product(N_CASES_LEVELS, RATIO_LEVELS, PREV_FRAC_LEVELS))

    # Build all job specs (seed, cell, model combinations)
    job_specs = []
    for seed in range(n_seeds):
        for n_cases, ratio, prev_frac in cells:
            for model_name in models:
                job_specs.append(
                    {
                        "seed": seed,
                        "n_cases": n_cases,
                        "ratio": ratio,
                        "prevalent_frac": prev_frac,
                        "model": model_name,
                    }
                )

    total = len(job_specs)
    logger.info(
        "Running %d seeds × %d cells × %d models = %d total jobs",
        n_seeds,
        len(cells),
        len(models),
        total,
    )

    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, mp.cpu_count())

    logger.info("Using %d parallel workers", n_workers)

    # Prepare worker function with fixed args
    worker_fn = partial(
        _run_single_job,
        df=df,
        panel=panel,
        I_pool=I_pool,
        P_pool=P_pool,
        C_pool=C_pool,
        test_idx=test_idx,
        hyperparams=hyperparams,
    )

    # Run jobs in parallel
    if n_workers == 1:
        # Sequential fallback
        results = [worker_fn(job) for job in job_specs]
    else:
        # Parallel execution
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, job_specs)

    # Unpack results
    rows = [r[0] for r in results]
    fi_rows = [fi for r in results for fi in r[1]]

    logger.info("Completed %d jobs", total)

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
        "--data-path",
        type=Path,
        required=True,
        help="Path to proteomics parquet file",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        required=True,
        help="Path to fixed_panel.csv (one protein per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/factorial_2x2x2"),
        help="Output directory (default: results/factorial_2x2x2)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds (default: 10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LR_EN", "XGBoost"],
        help="Models to run (default: LR_EN XGBoost)",
    )
    parser.add_argument(
        "--tune-cells",
        action="store_true",
        help="Tune hyperparams for ALL factorial cells (cell-specific), then exit",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--hyperparams-path",
        type=Path,
        default=None,
        help="Path to cell_hyperparams.yaml (required unless --tune-cells)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (-1=all CPUs, 1=sequential, default: 1)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution (sets --n-jobs=-1 unless specified)",
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
        args.output_dir / "test_indices.csv",
        test_idx,
        fmt="%d",
        header="positional index into filtered dataframe",
    )

    if args.tune_cells:
        tune_all_cells(
            df,
            panel,
            train_pool_idx,
            test_idx,
            models=args.models,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
        )
        logger.info(
            "Cell-specific tuning complete. Re-run with --hyperparams-path cell_hyperparams.yaml"
        )
        return

    # Load frozen hyperparams
    if args.hyperparams_path is None:
        logger.warning("No --hyperparams-path provided; using default hyperparams.")
        hyperparams = _default_hyperparams()
    else:
        with open(args.hyperparams_path) as f:
            hyperparams = yaml.safe_load(f)
        logger.info("Loaded hyperparams from %s", args.hyperparams_path)

    # Determine parallelization
    if args.parallel and args.n_jobs == 1:
        n_jobs = -1  # --parallel flag overrides default
    else:
        n_jobs = args.n_jobs

    # Run experiment
    if n_jobs == 1:
        logger.info("Running in sequential mode")
        results, feat_imp = run_experiment(
            df,
            panel,
            train_pool_idx,
            test_idx,
            hyperparams=hyperparams,
            n_seeds=args.n_seeds,
            models=args.models,
        )
    else:
        logger.info("Running in parallel mode with n_jobs=%d", n_jobs)
        results, feat_imp = run_experiment_parallel(
            df,
            panel,
            train_pool_idx,
            test_idx,
            hyperparams=hyperparams,
            n_seeds=args.n_seeds,
            models=args.models,
            n_jobs=n_jobs,
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
