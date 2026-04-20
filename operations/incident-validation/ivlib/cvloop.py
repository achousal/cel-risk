"""
Cross-validation loop with Optuna inner tuning.

Extracted from the legacy ``scripts/run_lr.py`` into the ivlib package.
Public surface: :func:`tune_and_evaluate_fold` and :func:`run_cv`.
Both helpers (:func:`_downsample_controls`, :func:`_optuna_objective`) are
module-private.

Discrimination metrics are routed through ``ced_ml.metrics.discrimination``
(``prauc``, ``auroc``) so all CV numbers match the project-canonical wrappers.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ced_ml.metrics.discrimination import auroc as ced_auroc
from ced_ml.metrics.discrimination import prauc as ced_prauc

from .strategies import get_training_indices
from .weights import compute_class_weight

# Silence optuna's per-trial INFO chatter at import time.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# --- Module-private helpers -------------------------------------------------


def _downsample_controls(
    X: np.ndarray,
    y: np.ndarray,
    max_ratio: int,
    rng: np.random.Generator,
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
    model_spec,
    cfg,
    seed: int,
    inner_ctrl_ratio: int = 0,
) -> float:
    """Optuna objective: mean AUPRC across inner CV folds."""
    params = model_spec.suggest_params(trial)
    rng = np.random.default_rng(seed + trial.number)

    inner_cv = StratifiedKFold(
        n_splits=cfg.n_inner_folds, shuffle=True, random_state=seed
    )
    scores: list[float] = []

    for train_ix, val_ix in inner_cv.split(X_train, y_train):
        X_tr, X_va = X_train[train_ix], X_train[val_ix]
        y_tr, y_va = y_train[train_ix], y_train[val_ix]

        if inner_ctrl_ratio > 0:
            X_tr, y_tr = _downsample_controls(X_tr, y_tr, inner_ctrl_ratio, rng)

        # Skip degenerate splits (can happen on tiny minority class).
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = model_spec.build(params, class_weight, cfg, seed)
        try:
            model_spec.fit(model, X_tr_s, y_tr, class_weight)
            y_prob = model.predict_proba(X_va_s)[:, 1]
            scores.append(ced_prauc(y_va, y_prob))
        except Exception:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


# --- Public: per-fold tune + evaluate ---------------------------------------


def tune_and_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight,
    model_spec,
    cfg,
    fold_seed: int,
) -> dict:
    """Tune hyperparams via inner CV, evaluate on the outer validation fold."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=fold_seed),
    )
    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_train, y_train, class_weight, model_spec, cfg, fold_seed,
        ),
        n_trials=cfg.n_optuna_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params

    # Refit on the full outer training fold with the selected params.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = model_spec.build(best_params, class_weight, cfg, fold_seed)
    model_spec.fit(model, X_train_s, y_train, class_weight)

    y_prob = model.predict_proba(X_val_s)[:, 1]
    auprc = ced_prauc(y_val, y_prob)
    auroc = ced_auroc(y_val, y_prob)

    coefs = model_spec.extract_coefs(model)

    result = {
        "auprc": auprc,
        "auroc": auroc,
        "best_params_json": json.dumps(best_params, sort_keys=True, default=str),
        "best_inner_auprc": study.best_value,
        "n_nonzero_coefs": int(np.sum(np.abs(coefs) > 1e-8)),
        "coefs": coefs,
        "val_y_prob": y_prob.astype(float).copy(),
    }
    return result


# --- Public: outer CV loop over (strategy x weight_scheme) ------------------


def run_cv(
    cfg,
    data: dict,
    features: dict,
    model_spec,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Run K-fold CV for all (strategy x weight_scheme) combinations.

    Returns:
        cv_results: DataFrame (one row per fold x combo).
        fold_coefs: list of per-fold coefficient dicts.
        oof_df: DataFrame with dev-set out-of-fold predictions
            (columns: fold, strategy, weight_scheme, df_idx, y_true, y_prob).
            Validation folds cover incident+controls only, so the union across
            folds equals the dev incident+control index set.
    """
    df = data["df"]
    protein_panel = features["pruned_proteins"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    logger.info("=== Cross-Validation (%s) ===", cfg.model)
    logger.info("Panel: %d proteins", len(protein_panel))
    logger.info("Strategies: %s", cfg.strategies)
    logger.info("Weight schemes: %s", cfg.weight_schemes)

    # CV folds are defined on the incident + control pool only.
    dev_ic_idx = np.concatenate([dev_incident_idx, dev_control_idx])
    dev_ic_labels = np.concatenate([
        np.ones(len(dev_incident_idx)),
        np.zeros(len(dev_control_idx)),
    ]).astype(int)

    outer_cv = StratifiedKFold(
        n_splits=cfg.n_outer_folds, shuffle=True, random_state=cfg.split_seed,
    )

    all_results: list[dict] = []
    oof_records: list[dict] = []
    total = len(cfg.strategies) * len(cfg.weight_schemes) * cfg.n_outer_folds
    done = 0

    for strategy in cfg.strategies:
        for weight_scheme in cfg.weight_schemes:
            fold_results: list[dict] = []

            for fold_i, (train_pos, val_pos) in enumerate(
                outer_cv.split(dev_ic_idx, dev_ic_labels)
            ):
                fold_train_ic = dev_ic_idx[train_pos]
                fold_val_ic = dev_ic_idx[val_pos]

                # Separate incident vs controls in the training fold.
                fold_train_incident = np.array(
                    [i for i in fold_train_ic if i in set(dev_incident_idx)]
                )
                fold_train_controls = np.array(
                    [i for i in fold_train_ic if i in set(dev_control_idx)]
                )

                # Training composition depends on the strategy.
                train_idx, y_train = get_training_indices(
                    strategy, fold_train_incident, fold_train_controls, prevalent_idx,
                )

                # Validation: always incident + controls.
                val_incident = np.array(
                    [i for i in fold_val_ic if i in set(dev_incident_idx)]
                )
                val_controls = np.array(
                    [i for i in fold_val_ic if i in set(dev_control_idx)]
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
                    X_train, y_train, X_val, y_val, cw, model_spec, cfg, fold_seed,
                )
                val_prob = result.pop("val_y_prob")
                # Persist OOF rows: val_idx carries absolute df index values.
                for df_i, yt, yp in zip(
                    val_idx.tolist(), y_val.tolist(), val_prob.tolist()
                ):
                    oof_records.append({
                        "fold": fold_i,
                        "strategy": strategy,
                        "weight_scheme": weight_scheme,
                        "df_idx": int(df_i),
                        "y_true": int(yt),
                        "y_prob": float(yp),
                    })
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

            mean_auprc = float(np.mean([r["auprc"] for r in fold_results]))
            mean_auroc = float(np.mean([r["auroc"] for r in fold_results]))
            logger.info(
                "  %s + %s: mean AUPRC=%.4f, mean AUROC=%.4f",
                strategy, weight_scheme, mean_auprc, mean_auroc,
            )

            all_results.extend(fold_results)

    # Extract per-fold coefficients before DataFrame conversion.
    fold_coefs: list[dict] = []
    for r in all_results:
        coefs = r.pop("coefs")
        fold_coefs.append({
            "fold": r["fold"],
            "strategy": r["strategy"],
            "weight_scheme": r["weight_scheme"],
            "coefs": coefs,
        })

    oof_df = pd.DataFrame(oof_records)
    return pd.DataFrame(all_results), fold_coefs, oof_df
