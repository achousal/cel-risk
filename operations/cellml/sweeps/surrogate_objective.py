"""Cheap surrogate objective for pre-sweep calibration.

The full cel-risk pipeline (SLURM job, 30 seeds, ~1500 cells) is too
expensive to call from inside calibration. This module builds a much
smaller objective: stratified subsample of the dataset + single-seed
k-fold CV with logistic regression, returning mean AUROC.

Design constraints:
- Strictly in-process; no SLURM, no Minerva.
- Subsample size set by CalibrationConfig.subsample_rows (Rule 8).
- Deterministic under a fixed seed (calib_seed).
- Objective signature: (params: dict) -> float.
- Parameters the surrogate understands are limited -- unknown keys are
  ignored with a DEBUG log (they will be honored by the real sweep).
- Logistic regression is the surrogate model regardless of which model
  the parent sweep tunes. The goal of calibration is to measure the
  SHAPE of the objective-vs-trial curve, not to reproduce the parent
  sweep's absolute metric. LR is a stable, cheap proxy.

Recognized surrogate parameters (mapped to LR / preprocessing):
  C                         -> LogisticRegression(C=...)
  penalty                   -> LogisticRegression(penalty=...)
  l1_ratio                  -> LogisticRegression(l1_ratio=..., penalty='elasticnet')
  class_weight              -> LogisticRegression(class_weight=...)
  data.downsample_majority_ratio -> majority-class downsample ratio
  data.subsample_rows       -> row cap override (debug only)

Unknown keys are silently tolerated so calibration can run against any
parameter space without per-sweep adapter code. The objective reports
AUROC on out-of-fold predictions averaged over `n_folds` folds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ced_ml.data.fingerprint import SUPPORTED_SUFFIXES

from .calibration_schema import CalibrationConfig
from .sweep_schema import SweepSpec

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[dict], float]

# Columns that are never features regardless of dataset shape. The
# surrogate deliberately uses a broad regex rather than importing
# cel-risk schema constants so it stays isolated from pipeline churn.
_METADATA_HINTS = ("eid", "date", "status", "sex", "age", "CeD_")


def _load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported dataset suffix '{suffix}'. Expected one of {sorted(SUPPORTED_SUFFIXES)}."
    )


def _pick_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    """Select numeric protein-like columns as features.

    Rule: any numeric column that is not the label and does not match a
    metadata hint is a feature. This matches cel-risk's convention where
    proteomics features carry a `_resid` suffix while metadata columns
    are explicit. The surrogate does not need exact feature parity --
    it just needs a reasonable signal-bearing subset.
    """
    feats = []
    for col in df.columns:
        if col == label_col:
            continue
        if any(hint in col for hint in _METADATA_HINTS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feats.append(col)
    if not feats:
        raise ValueError(
            "Surrogate objective found no numeric feature columns. "
            "Check the dataset layout and label_col setting."
        )
    return feats


def _stratified_subsample(
    df: pd.DataFrame,
    label_col: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    per_class_n = max_rows // 2
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    pos_take = min(per_class_n, len(pos))
    neg_take = min(max_rows - pos_take, len(neg))
    sampled = pd.concat(
        [
            pos.sample(n=pos_take, random_state=seed),
            neg.sample(n=neg_take, random_state=seed),
        ]
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _apply_downsample(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the majority class to `ratio * n_minority`."""
    if ratio <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    target_neg = int(round(ratio * len(pos_idx)))
    if target_neg >= len(neg_idx):
        return X, y
    chosen = rng.choice(neg_idx, size=target_neg, replace=False)
    keep = np.concatenate([pos_idx, chosen])
    rng.shuffle(keep)
    return X[keep], y[keep]


def _build_lr(params: dict) -> LogisticRegression:
    """Translate a parameter dict into a LogisticRegression estimator."""
    C = float(params.get("C", 1.0))
    penalty = params.get("penalty", "l2")
    class_weight = params.get("class_weight", None)
    l1_ratio = params.get("l1_ratio", None)
    kwargs: dict[str, Any] = dict(
        C=C,
        max_iter=1000,
        solver="saga" if penalty in ("l1", "elasticnet") else "lbfgs",
        class_weight=class_weight,
    )
    # sklearn >= 1.8 deprecates `penalty=`; map to l1_ratio instead.
    if penalty == "l1":
        kwargs["l1_ratio"] = 1.0
        kwargs["solver"] = "saga"
    elif penalty == "elasticnet":
        kwargs["l1_ratio"] = float(l1_ratio) if l1_ratio is not None else 0.5
        kwargs["solver"] = "saga"
    elif penalty == "l2":
        kwargs["l1_ratio"] = 0.0
    return LogisticRegression(**kwargs)


def build_surrogate_objective(
    spec: SweepSpec,
    config: CalibrationConfig,
    project_root: Path,
    n_folds: int = 3,
) -> tuple[ObjectiveFn, int]:
    """Construct the surrogate objective function and report row count.

    Returns
    -------
    objective_fn
        Callable (params) -> mean out-of-fold AUROC. Direction matches
        spec.metric_direction (higher-is-better for AUROC).
    subsample_rows_used
        Actual row count after subsampling (for provenance / Rule 8).

    Notes
    -----
    The dataset is loaded once, subsampled once, and cached in a closure
    so per-trial cost is bounded by the CV + LR fit. This gives a flat
    wall-time curve, which is what makes calibration cheap.
    """
    if not spec.data_path:
        raise ValueError(
            f"Sweep {spec.id} has no data_path; cannot build surrogate objective."
        )

    data_path = (project_root / spec.data_path).resolve()
    logger.info("Surrogate: loading %s", data_path)
    df = _load_dataset(data_path)

    label_col = spec.label_col
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not in dataset {data_path.name}")

    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)

    df_sub = _stratified_subsample(
        df, label_col, max_rows=config.subsample_rows, seed=config.calib_seed,
    )
    feature_cols = _pick_feature_columns(df_sub, label_col)
    logger.info(
        "Surrogate: subsampled to %d rows, %d features",
        len(df_sub), len(feature_cols),
    )

    X_all = df_sub[feature_cols].to_numpy(dtype=float)
    y_all = df_sub[label_col].to_numpy(dtype=int)
    seed = config.calib_seed

    def objective(params: dict) -> float:
        ratio = params.get("data.downsample_majority_ratio")
        if ratio is not None:
            X, y = _apply_downsample(X_all, y_all, float(ratio), seed=seed)
        else:
            X, y = X_all, y_all

        if len(np.unique(y)) < 2:
            logger.warning("Surrogate: single-class fold after downsample; returning 0.5")
            return 0.5

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in skf.split(X, y):
            pipe = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("lr", _build_lr(params)),
                ]
            )
            pipe.fit(X[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
        return float(roc_auc_score(y, oof))

    return objective, len(df_sub)
