"""Model specifications for the incident-validation factorial.

Each ModelSpec wraps an estimator family with Optuna search space,
construction, coefficient/importance extraction, CV aggregation, and
a custom fit hook (used by XGB for sample_weight dispatch).
"""

from __future__ import annotations

import abc
import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ivlib.weights import class_weight_to_sample_weight

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

if TYPE_CHECKING:
    # Avoid runtime import cycle: Config lives in scripts/run_lr.py.
    from scripts.run_lr import Config  # pragma: no cover

logger = logging.getLogger(__name__)


VALID_MODELS: tuple[str, ...] = ("LR_EN", "SVM_L1", "SVM_L2", "RF", "XGB")

MODEL_OUTPUT_DIRS: dict[str, str] = {
    "LR_EN": "results/incident-validation/lr/LR_EN",
    "SVM_L1": "results/incident-validation/lr/SVM_L1",
    "SVM_L2": "results/incident-validation/lr/SVM_L2",
    "RF": "results/incident-validation/rf/RF",
    "XGB": "results/incident-validation/xgb/XGB",
}


class ModelSpec(abc.ABC):
    """Interface for model-specific build, tune, and coefficient extraction."""

    @abc.abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Register Optuna search space and return sampled params."""

    @abc.abstractmethod
    def build(self, params: dict, class_weight, cfg: "Config", seed: int):
        """Construct a fitted-ready sklearn estimator."""

    @abc.abstractmethod
    def extract_coefs(self, model) -> np.ndarray:
        """Extract coefficient vector from a fitted model."""

    @abc.abstractmethod
    def param_summary(self, params: dict) -> str:
        """Human-readable summary of hyperparams for the report."""

    @abc.abstractmethod
    def aggregate_best_params(self, cv_results: pd.DataFrame) -> dict:
        """Derive final hyperparams from CV results (e.g. median)."""

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """Model name for reports."""

    def fit(self, model, X, y, class_weight):
        """Fit hook. Default: class_weight already baked into init, plain fit."""
        model.fit(X, y)


class LRElasticNetSpec(ModelSpec):
    """ElasticNet logistic regression."""

    def suggest_params(self, trial):
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        }

    def build(self, params, class_weight, cfg, seed):
        return LogisticRegression(
            C=params["C"],
            l1_ratio=params["l1_ratio"],
            penalty="elasticnet",
            class_weight=class_weight,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
            random_state=seed,
        )

    def extract_coefs(self, model):
        return model.coef_.ravel().copy()

    def param_summary(self, params):
        return f"C={params['C']:.6f}, l1_ratio={params['l1_ratio']:.4f}"

    def aggregate_best_params(self, cv_results):
        parsed = [json.loads(s) for s in cv_results["best_params_json"]]
        return {
            "C": float(np.median([p["C"] for p in parsed])),
            "l1_ratio": float(np.median([p["l1_ratio"] for p in parsed])),
        }

    @property
    def display_name(self):
        return "ElasticNet Logistic Regression"


class SVMSpec(ModelSpec):
    """LinearSVC + CalibratedClassifierCV (L1 or L2 penalty)."""

    def __init__(self, penalty: str = "l2"):
        if penalty not in ("l1", "l2"):
            raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")
        self.penalty = penalty

    def suggest_params(self, trial):
        return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}

    def build(self, params, class_weight, cfg, seed):
        if self.penalty == "l1":
            base = LinearSVC(
                penalty="l1",
                dual=False,
                C=params["C"],
                class_weight=class_weight,
                max_iter=cfg.max_iter,
                random_state=seed,
            )
        else:
            base = LinearSVC(
                penalty="l2",
                dual=True,
                C=params["C"],
                class_weight=class_weight,
                max_iter=cfg.max_iter,
                random_state=seed,
            )
        return CalibratedClassifierCV(base, method="sigmoid", cv=cfg.calibration_cv)

    def extract_coefs(self, model):
        return np.mean(
            [cc.estimator.coef_.ravel() for cc in model.calibrated_classifiers_],
            axis=0,
        )

    def param_summary(self, params):
        return f"C={params['C']:.6f}"

    def aggregate_best_params(self, cv_results):
        parsed = [json.loads(s) for s in cv_results["best_params_json"]]
        return {"C": float(np.median([p["C"] for p in parsed]))}

    @property
    def display_name(self):
        return f"LinSVM_cal (penalty={self.penalty})"


class RFSpec(ModelSpec):
    """Random Forest classifier (already outputs probabilities; scale-invariant)."""

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }

    def build(self, params, class_weight, cfg, seed):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=class_weight,
            n_jobs=1,
            random_state=seed,
        )

    def extract_coefs(self, model):
        # Feature importances stand in for linear coefs (same shape).
        return np.asarray(model.feature_importances_, dtype=float).ravel().copy()

    def param_summary(self, params):
        return (
            f"n_est={params['n_estimators']}, max_depth={params['max_depth']}, "
            f"msl={params['min_samples_leaf']}, mf={params['max_features']:.3f}"
        )

    def aggregate_best_params(self, cv_results):
        parsed = [json.loads(s) for s in cv_results["best_params_json"]]
        return {
            "n_estimators": int(np.median([p["n_estimators"] for p in parsed])),
            "max_depth": int(np.median([p["max_depth"] for p in parsed])),
            "min_samples_split": int(np.median([p["min_samples_split"] for p in parsed])),
            "min_samples_leaf": int(np.median([p["min_samples_leaf"] for p in parsed])),
            "max_features": float(np.median([p["max_features"] for p in parsed])),
        }

    @property
    def display_name(self):
        return "Random Forest"


class XGBSpec(ModelSpec):
    """XGBoost classifier. Class imbalance handled via sample_weight in fit(),
    so scale_pos_weight is left at 1 here to stay consistent with the existing
    class_weight dispatch used by linear models."""

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }

    def build(self, params, class_weight, cfg, seed):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )

    def extract_coefs(self, model):
        return np.asarray(model.feature_importances_, dtype=float).ravel().copy()

    def fit(self, model, X, y, class_weight):
        sw = class_weight_to_sample_weight(class_weight, y)
        model.fit(X, y, sample_weight=sw)

    def param_summary(self, params):
        return (
            f"n_est={params['n_estimators']}, max_depth={params['max_depth']}, "
            f"lr={params['learning_rate']:.4f}, subsample={params['subsample']:.2f}"
        )

    def aggregate_best_params(self, cv_results):
        parsed = [json.loads(s) for s in cv_results["best_params_json"]]
        return {
            "n_estimators": int(np.median([p["n_estimators"] for p in parsed])),
            "max_depth": int(np.median([p["max_depth"] for p in parsed])),
            "learning_rate": float(np.median([p["learning_rate"] for p in parsed])),
            "min_child_weight": float(np.median([p["min_child_weight"] for p in parsed])),
            "gamma": float(np.median([p["gamma"] for p in parsed])),
            "subsample": float(np.median([p["subsample"] for p in parsed])),
            "colsample_bytree": float(np.median([p["colsample_bytree"] for p in parsed])),
            "reg_alpha": float(np.median([p["reg_alpha"] for p in parsed])),
            "reg_lambda": float(np.median([p["reg_lambda"] for p in parsed])),
        }

    @property
    def display_name(self):
        return "XGBoost"


def get_model_spec(model_id: str) -> ModelSpec:
    """Factory: model ID -> ModelSpec."""
    if model_id == "LR_EN":
        return LRElasticNetSpec()
    elif model_id == "SVM_L1":
        return SVMSpec(penalty="l1")
    elif model_id == "SVM_L2":
        return SVMSpec(penalty="l2")
    elif model_id == "RF":
        return RFSpec()
    elif model_id == "XGB":
        return XGBSpec()
    else:
        raise ValueError(
            f"Unknown model '{model_id}'. Valid: {', '.join(VALID_MODELS)}"
        )
