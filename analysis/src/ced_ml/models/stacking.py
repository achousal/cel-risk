"""Model stacking ensemble for combining base model predictions.

This module provides a StackingEnsemble class that trains a meta-learner
on out-of-fold (OOF) predictions from multiple base models to improve
overall predictive performance.

Architecture:
    1. Base models are trained independently (via standard training pipeline)
    2. OOF predictions from each base model are collected
    3. Meta-learner (Logistic Regression with L2) is trained on stacked OOF predictions
    4. Final predictions combine base model outputs through the meta-learner

Expected improvement: +2-5% AUROC over best single model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ced_ml.utils.math_utils import logit

logger = logging.getLogger(__name__)


@dataclass
class CalibrationInfo:
    """Container for calibration information from a base model.

    Attributes:
        strategy: Calibration strategy ('none', 'per_fold', 'oof_posthoc')
        method: Calibration method ('isotonic', 'sigmoid', or None)
        oof_calibrator: Optional fitted OOF calibrator object
    """

    strategy: str
    method: str | None = None
    oof_calibrator: Any = None

    @property
    def needs_posthoc_calibration(self) -> bool:
        """Return True if this model needs posthoc calibration."""
        return self.strategy == "oof_posthoc" and self.oof_calibrator is not None


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble that combines base model predictions via a meta-learner.

    The meta-learner is trained on out-of-fold (OOF) predictions from multiple
    base models. This approach prevents information leakage by using predictions
    that were generated when each sample was in the validation fold.

    Attributes:
        base_model_names: List of base model identifiers
        meta_model: Fitted meta-learner (LogisticRegression)
        base_models: Dict mapping model name to loaded model bundle
        scaler: Optional feature scaler for meta-learner input
        classes_: Class labels [0, 1]
        is_fitted_: Whether the ensemble has been fitted

    Example:
        >>> ensemble = StackingEnsemble(
        ...     base_model_names=['LR_EN', 'RF', 'XGBoost'],
        ...     meta_penalty='l2',
        ...     meta_C=1.0
        ... )
        >>> ensemble.fit_from_oof(oof_dict, y_train)
        >>> test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)
    """

    def __init__(
        self,
        base_model_names: list[str] | None = None,
        meta_penalty: str = "l2",
        meta_C: float = 1.0,
        meta_max_iter: int = 1000,
        meta_solver: str = "lbfgs",
        use_probabilities: bool = True,
        scale_meta_features: bool = True,
        calibrate_meta: bool = True,
        calibration_cv: int = 5,
        random_state: int | None = None,
    ):
        """Initialize stacking ensemble.

        Args:
            base_model_names: List of base model identifiers to include
            meta_penalty: Regularization penalty for meta-learner ('l2', 'l1', 'elasticnet', 'none')
            meta_C: Inverse regularization strength for meta-learner
            meta_max_iter: Max iterations for meta-learner convergence
            meta_solver: Solver for logistic regression meta-learner
            use_probabilities: Use probabilities (True) or logits (False) as meta features
            scale_meta_features: Whether to standardize meta-learner input
            calibrate_meta: Whether to calibrate meta-learner predictions
            calibration_cv: CV folds for meta-learner calibration
            random_state: Random seed for reproducibility
        """
        self.base_model_names = base_model_names or []
        self.meta_penalty = meta_penalty
        self.meta_C = meta_C
        self.meta_max_iter = meta_max_iter
        self.meta_solver = meta_solver
        self.use_probabilities = use_probabilities
        self.scale_meta_features = scale_meta_features
        self.calibrate_meta = calibrate_meta
        self.calibration_cv = calibration_cv
        self.random_state = random_state

        # Will be set during fitting
        self.meta_model: LogisticRegression | CalibratedClassifierCV | None = None
        self.scaler: StandardScaler | None = None
        self.base_models: dict[str, Any] = {}
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False
        self._feature_names: list[str] = []

    def _build_meta_estimator(self) -> LogisticRegression:
        """Build the LogisticRegression meta-learner with valid sklearn settings.

        sklearn >= 1.8 deprecates the ``penalty`` parameter. We use
        ``l1_ratio`` and ``C`` instead:
          l1_ratio=0 -> L2, l1_ratio=1 -> L1, l1_ratio=0.5 -> elasticnet,
          C=np.inf -> no penalty.
        """
        penalty = None if self.meta_penalty in (None, "none") else self.meta_penalty
        if penalty not in {"l1", "l2", "elasticnet", None}:
            penalty = "l2"

        # Map penalty string to l1_ratio / C for sklearn >= 1.8
        if penalty is None:
            l1_ratio = 0.0
            C = np.inf
        elif penalty == "l1":
            l1_ratio = 1.0
            C = self.meta_C
        elif penalty == "elasticnet":
            l1_ratio = 0.5
            C = self.meta_C
        else:  # l2
            l1_ratio = 0.0
            C = self.meta_C

        solver = self.meta_solver
        if l1_ratio == 1.0:
            if solver not in {"saga", "liblinear"}:
                solver = "saga"
        elif l1_ratio == 0.5:
            solver = "saga"
        elif C == np.inf:
            if solver == "liblinear":
                solver = "lbfgs"

        return LogisticRegression(
            l1_ratio=l1_ratio,
            C=C,
            max_iter=self.meta_max_iter,
            solver=solver,
            random_state=self.random_state,
            class_weight="balanced",
        )

    def _build_meta_features(
        self,
        oof_dict: dict[str, np.ndarray],
        aggregate_repeats: bool = True,
    ) -> np.ndarray:
        """Build meta-feature matrix from OOF predictions.

        Args:
            oof_dict: Dict mapping model name to OOF predictions
                      Each value is (n_repeats x n_samples) or (n_samples,)
            aggregate_repeats: Whether to average across CV repeats

        Returns:
            Meta-feature matrix (n_samples x n_base_models)
        """
        features = []
        self._feature_names = []

        for model_name in self.base_model_names:
            if model_name not in oof_dict:
                raise ValueError(f"Missing OOF predictions for base model: {model_name}")

            preds = oof_dict[model_name]
            preds = np.asarray(preds)

            # Handle multi-repeat OOF predictions
            if preds.ndim == 2 and aggregate_repeats:
                # Average across repeats (axis 0)
                preds = np.nanmean(preds, axis=0)
            elif preds.ndim == 2:
                # Use first repeat only
                preds = preds[0, :]

            # Convert to logits if requested
            if not self.use_probabilities:
                preds = logit(preds)

            features.append(preds)
            self._feature_names.append(f"oof_{model_name}")

        return np.column_stack(features)

    def fit_from_oof(
        self,
        oof_dict: dict[str, np.ndarray],
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> StackingEnsemble:
        """Fit meta-learner on OOF predictions from base models.

        Args:
            oof_dict: Dict mapping model name to OOF predictions
                      Shape: (n_repeats x n_train) or (n_train,)
            y_train: Training labels
            sample_weight: Optional sample weights

        Returns:
            self (fitted ensemble)
        """
        logger.info(f"Fitting stacking ensemble with {len(self.base_model_names)} base models")

        # Build meta-feature matrix
        X_meta = self._build_meta_features(oof_dict, aggregate_repeats=True)
        y = np.asarray(y_train)

        # Validate shapes
        if X_meta.shape[0] != len(y):
            raise ValueError(
                f"Shape mismatch: X_meta has {X_meta.shape[0]} samples, "
                f"y_train has {len(y)} samples"
            )

        # Check for NaN values (can occur if base model had missing OOF predictions)
        nan_mask = np.isnan(X_meta).any(axis=1)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            logger.warning(f"Dropping {n_nan} samples with NaN meta-features")
            X_meta = X_meta[~nan_mask]
            y = y[~nan_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[~nan_mask]

        # Scale meta-features if requested
        if self.scale_meta_features:
            self.scaler = StandardScaler()
            X_meta = self.scaler.fit_transform(X_meta)

        # Build meta-learner
        base_meta = self._build_meta_estimator()

        # Optionally wrap in calibration
        if self.calibrate_meta and len(y) >= 2 * self.calibration_cv:
            self.meta_model = CalibratedClassifierCV(
                estimator=base_meta,
                method="isotonic",
                cv=self.calibration_cv,
            )
        else:
            self.meta_model = base_meta

        # Fit meta-learner
        logger.info(
            f"Training meta-learner on {X_meta.shape[0]} samples, {X_meta.shape[1]} features"
        )
        self.meta_model.fit(X_meta, y, sample_weight=sample_weight)
        self.is_fitted_ = True

        logger.info("Stacking ensemble fitted successfully")
        return self

    def predict_proba_from_base_preds(
        self,
        preds_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict class probabilities using base model predictions.

        Args:
            preds_dict: Dict mapping model name to predictions (n_samples,)

        Returns:
            Probability matrix (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted. Call fit_from_oof first.")

        # Build meta-features from base model predictions
        features = []
        for model_name in self.base_model_names:
            if model_name not in preds_dict:
                raise ValueError(f"Missing predictions for base model: {model_name}")

            preds = np.asarray(preds_dict[model_name])

            # Handle 2D predictions (take positive class column)
            if preds.ndim == 2:
                preds = preds[:, 1]

            # Convert to logits if needed
            if not self.use_probabilities:
                preds = logit(preds)

            features.append(preds)

        X_meta = np.column_stack(features)

        # Scale if scaler was fitted
        if self.scaler is not None:
            X_meta = self.scaler.transform(X_meta)

        # Predict with meta-learner
        return self.meta_model.predict_proba(X_meta)

    def predict_from_base_preds(
        self,
        preds_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict class labels using base model predictions.

        Args:
            preds_dict: Dict mapping model name to predictions

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba_from_base_preds(preds_dict)
        return (proba[:, 1] >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> StackingEnsemble:
        """Sklearn-compatible fit method (not recommended for stacking).

        For proper stacking, use fit_from_oof() with pre-computed OOF predictions.
        This method is provided for sklearn pipeline compatibility only.

        Args:
            X: Feature matrix (assumed to be stacked OOF predictions)
            y: Target labels

        Returns:
            self
        """
        logger.warning(
            "Using fit() directly assumes X contains stacked OOF predictions. "
            "For proper stacking, use fit_from_oof() instead."
        )
        # Assume X is already the meta-feature matrix
        if self.scale_meta_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        base_meta = self._build_meta_estimator()

        if self.calibrate_meta and len(y) >= 2 * self.calibration_cv:
            self.meta_model = CalibratedClassifierCV(
                estimator=base_meta,
                method="isotonic",
                cv=self.calibration_cv,
            )
        else:
            self.meta_model = base_meta

        self.meta_model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Meta-feature matrix (stacked base model predictions)

        Returns:
            Probability matrix (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.meta_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Meta-feature matrix

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_meta_model_coef(self) -> dict[str, float]:
        """Get meta-learner coefficients for interpretability.

        Returns:
            Dict mapping base model name to coefficient
        """
        if not self.is_fitted_:
            raise RuntimeError("Ensemble not fitted.")

        # Handle calibrated wrapper
        meta = self.meta_model
        if hasattr(meta, "estimator"):
            # CalibratedClassifierCV stores base estimator
            # Try to get average coefficients from calibrated estimators
            if hasattr(meta, "calibrated_classifiers_"):
                coefs = []
                for cc in meta.calibrated_classifiers_:
                    if hasattr(cc, "estimator") and hasattr(cc.estimator, "coef_"):
                        coefs.append(cc.estimator.coef_[0])
                if coefs:
                    avg_coef = np.mean(coefs, axis=0)
                    return dict(zip(self._feature_names, avg_coef, strict=False))

        # Direct access to LogisticRegression coefficients
        if hasattr(meta, "coef_"):
            return dict(zip(self._feature_names, meta.coef_[0], strict=False))

        return {}

    def save(self, path: Path | str) -> None:
        """Save ensemble to disk.

        Args:
            path: Output path for joblib file
        """
        path = Path(path)
        bundle = {
            "ensemble": self,
            "base_model_names": self.base_model_names,
            "meta_penalty": self.meta_penalty,
            "meta_C": self.meta_C,
            "is_fitted": self.is_fitted_,
            "feature_names": self._feature_names,
        }
        joblib.dump(bundle, path)
        logger.info(f"Ensemble saved to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> StackingEnsemble:
        """Load ensemble from disk.

        Args:
            path: Path to saved ensemble file

        Returns:
            Loaded StackingEnsemble instance
        """
        path = Path(path)
        bundle = joblib.load(path)
        ensemble = bundle["ensemble"]
        logger.info(f"Ensemble loaded from: {path}")
        return ensemble


def _find_model_split_dir(
    results_dir: Path,
    model_name: str,
    split_seed: int,
    run_id: str | None = None,
) -> Path:
    """Find the split directory for a model.

    Primary layout: results_dir/run_{run_id}/{model}/splits/split_seed{N}/

    Searches in order of preference:
    1. results_dir/run_{run_id}/{model}/splits/split_seed{N} (explicit run_id)
    2. results_dir/run_*/{model}/splits/split_seed{N} (auto-discover run_id)

    Args:
        results_dir: Root results directory
        model_name: Model name (e.g., 'LR_EN')
        split_seed: Split seed number
        run_id: Optional run_id to target specific run

    Returns:
        Path to the split directory containing model outputs

    Raises:
        FileNotFoundError: If no matching directory found
    """
    # Pattern 1: Explicit run_id
    if run_id is not None:
        candidate = (
            results_dir / f"run_{run_id}" / model_name / "splits" / f"split_seed{split_seed}"
        )
        if candidate.exists():
            return candidate

    # Pattern 2: Auto-discover run directories (prefer most recent)
    run_dirs = sorted(results_dir.glob("run_*"), reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / model_name / "splits" / f"split_seed{split_seed}"
        if candidate.exists():
            logger.debug(f"Auto-discovered run directory: {run_dir.name}")
            return candidate

    searched = [
        f"{results_dir}/run_{run_id or '*'}/{model_name}/splits/split_seed{split_seed}",
    ]
    raise FileNotFoundError(
        f"Could not find split directory for {model_name} seed {split_seed}. "
        f"Searched: {searched}"
    )


def load_base_model_calibration_info(
    results_dir: Path,
    model_name: str,
    split_seed: int,
    run_id: str | None = None,
) -> CalibrationInfo:
    """Load calibration info from a saved model bundle.

    Args:
        results_dir: Root results directory
        model_name: Model name
        split_seed: Split seed
        run_id: Optional run_id

    Returns:
        CalibrationInfo with strategy and optional OOF calibrator
    """
    model_dir = _find_model_split_dir(results_dir, model_name, split_seed, run_id)
    model_path = model_dir / "core" / f"{model_name}__final_model.joblib"

    if not model_path.exists():
        logger.warning(f"Model bundle not found: {model_path}, assuming no calibration")
        return CalibrationInfo(strategy="none")

    bundle = joblib.load(model_path)
    calib_info = bundle.get("calibration", {})

    strategy = calib_info.get("strategy", "none")
    method = calib_info.get("method")
    oof_calibrator = calib_info.get("oof_calibrator")

    logger.debug(
        f"Loaded calibration info for {model_name}: strategy={strategy}, "
        f"method={method}, has_oof_calibrator={oof_calibrator is not None}"
    )

    return CalibrationInfo(
        strategy=strategy,
        method=method,
        oof_calibrator=oof_calibrator,
    )


def apply_calibration_to_predictions(
    predictions: np.ndarray,
    calib_info: CalibrationInfo,
    model_name: str,
) -> np.ndarray:
    """Apply calibration to predictions based on calibration strategy.

    For 'per_fold': predictions are already calibrated, return as-is
    For 'oof_posthoc': apply the OOF calibrator
    For 'none': return as-is

    Args:
        predictions: Raw predictions array
        calib_info: CalibrationInfo from the base model
        model_name: Model name (for logging)

    Returns:
        Calibrated predictions
    """
    if calib_info.needs_posthoc_calibration:
        logger.debug(f"Applying OOF calibrator to {model_name} predictions")
        return calib_info.oof_calibrator.transform(predictions)

    # per_fold or none: predictions are already in correct form
    return predictions


def load_calibration_info_for_models(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    run_id: str | None = None,
) -> dict[str, CalibrationInfo]:
    """Load calibration info for multiple base models.

    Args:
        results_dir: Root results directory
        base_models: List of model names
        split_seed: Split seed
        run_id: Optional run_id

    Returns:
        Dict mapping model name to CalibrationInfo
    """
    calib_dict = {}
    for model_name in base_models:
        calib_dict[model_name] = load_base_model_calibration_info(
            results_dir, model_name, split_seed, run_id
        )
    return calib_dict


def _validate_indices_match(
    reference_idx: np.ndarray,
    reference_model: str,
    current_idx: np.ndarray,
    current_model: str,
    context: str,
) -> None:
    """Validate that indices from two models match exactly.

    Args:
        reference_idx: Indices from the reference (first) model
        reference_model: Name of the reference model
        current_idx: Indices from the current model being checked
        current_model: Name of the current model
        context: Description of the context (e.g., "OOF predictions", "test predictions")

    Raises:
        ValueError: If indices do not match (different length or different values)
    """
    if len(reference_idx) != len(current_idx):
        raise ValueError(
            f"Index length mismatch in {context}: "
            f"{reference_model} has {len(reference_idx)} samples, "
            f"{current_model} has {len(current_idx)} samples. "
            f"Base models must be trained on the same data split."
        )

    if not np.array_equal(reference_idx, current_idx):
        # Find first mismatching position for helpful error message
        mismatch_mask = reference_idx != current_idx
        first_mismatch_pos = np.argmax(mismatch_mask)
        raise ValueError(
            f"Index mismatch in {context}: "
            f"{reference_model} and {current_model} have different sample indices. "
            f"First mismatch at position {first_mismatch_pos}: "
            f"{reference_model}={reference_idx[first_mismatch_pos]}, "
            f"{current_model}={current_idx[first_mismatch_pos]}. "
            f"Base models must be trained on the same data split."
        )


def collect_oof_predictions(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    run_id: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
    """Collect OOF predictions from trained base models.

    Args:
        results_dir: Root results directory
        base_models: List of base model names to collect
        split_seed: Split seed to identify correct subdirectory
        run_id: Optional run_id to target specific run

    Returns:
        oof_dict: Dict mapping model name to OOF predictions
        y_train: Training labels (from first model)
        train_idx: Training indices (from first model)
        category: Category labels (Controls/Incident/Prevalent), or None if not available

    Raises:
        FileNotFoundError: If OOF predictions file not found for any model
        ValueError: If base models have mismatched indices (trained on different splits)
    """
    oof_dict = {}
    y_train = None
    train_idx = None
    category = None
    reference_model = None

    for model_name in base_models:
        # Look for OOF predictions file using flexible path discovery
        model_dir = _find_model_split_dir(results_dir, model_name, split_seed, run_id)

        # Flat preds directory structure
        oof_path = model_dir / "preds" / f"train_oof__{model_name}.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

        # Load OOF predictions
        oof_df = pd.read_csv(oof_path)

        # Extract predictions (may have multiple repeat columns)
        prob_cols = [c for c in oof_df.columns if c.startswith("y_prob")]
        if not prob_cols:
            raise ValueError(f"No probability columns found in {oof_path}")

        # Stack all repeat predictions
        preds = oof_df[prob_cols].values.T  # (n_repeats x n_samples)
        oof_dict[model_name] = preds

        current_idx = oof_df["idx"].values

        # Get labels, indices, and category from first model, validate subsequent models match
        if y_train is None:
            y_train = oof_df["y_true"].values
            train_idx = current_idx
            # Load category if available (Controls/Incident/Prevalent)
            if "category" in oof_df.columns:
                category = oof_df["category"].values
            reference_model = model_name
        else:
            # Validate that this model's indices match the reference model
            _validate_indices_match(
                train_idx, reference_model, current_idx, model_name, "OOF predictions"
            )

        logger.info(f"Loaded OOF predictions for {model_name}: shape {preds.shape}")

    return oof_dict, y_train, train_idx, category


def collect_split_predictions(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    split_name: str = "test",
    run_id: str | None = None,
    calibration_info: dict[str, CalibrationInfo] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
    """Collect validation or test predictions from trained base models.

    Args:
        results_dir: Root results directory
        base_models: List of base model names
        split_seed: Split seed to identify correct subdirectory
        split_name: 'val' or 'test'
        run_id: Optional run_id to target specific run
        calibration_info: Optional dict mapping model name to CalibrationInfo.
                          If provided, applies calibration based on each model's strategy.

    Returns:
        preds_dict: Dict mapping model name to predictions (calibrated if applicable)
        y_true: True labels
        indices: Sample indices
        category: Category labels (Controls/Incident/Prevalent), or None if not available

    Raises:
        FileNotFoundError: If predictions file not found for any model
        ValueError: If base models have mismatched indices (trained on different splits)
    """
    preds_dict = {}
    y_true = None
    indices = None
    category = None
    reference_model = None

    for model_name in base_models:
        model_dir = _find_model_split_dir(results_dir, model_name, split_seed, run_id)

        # Predictions are stored directly in preds/ directory, not in subdirectories
        if split_name == "val":
            pred_path = model_dir / "preds" / f"val_preds__{model_name}.csv"
        else:
            pred_path = model_dir / "preds" / f"test_preds__{model_name}.csv"

        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_path}")

        pred_df = pd.read_csv(pred_path)
        raw_preds = pred_df["y_prob"].values
        current_idx = pred_df["idx"].values

        # Apply calibration if info provided and model has oof_posthoc strategy
        if calibration_info and model_name in calibration_info:
            calib_info = calibration_info[model_name]
            preds_dict[model_name] = apply_calibration_to_predictions(
                raw_preds, calib_info, model_name
            )
        else:
            preds_dict[model_name] = raw_preds

        # Get labels, indices, and category from first model, validate subsequent models match
        if y_true is None:
            y_true = pred_df["y_true"].values
            indices = current_idx
            # Load category if available (Controls/Incident/Prevalent)
            if "category" in pred_df.columns:
                category = pred_df["category"].values
            reference_model = model_name
        else:
            # Validate that this model's indices match the reference model
            _validate_indices_match(
                indices, reference_model, current_idx, model_name, f"{split_name} predictions"
            )

        logger.info(f"Loaded {split_name} predictions for {model_name}")

    return preds_dict, y_true, indices, category


def train_stacking_ensemble(
    results_dir: Path,
    base_models: list[str],
    split_seed: int,
    meta_penalty: str = "l2",
    meta_C: float = 1.0,
    calibrate_meta: bool = True,
    random_state: int = 42,
    run_id: str | None = None,
    apply_base_calibration: bool = True,
) -> tuple[StackingEnsemble, dict[str, Any]]:
    """Train a stacking ensemble from pre-computed base model outputs.

    This is the main entry point for ensemble training. It:
    1. Collects OOF predictions from base models (already calibrated by strategy)
    2. Loads calibration info for base models
    3. Trains the meta-learner on calibrated OOF predictions
    4. Generates ensemble predictions on val/test sets (applying calibration)
    5. Computes ensemble metrics

    Args:
        results_dir: Root results directory containing base model outputs
        base_models: List of base model names to stack
        split_seed: Split seed for identifying model outputs
        meta_penalty: Regularization penalty for meta-learner
        meta_C: Inverse regularization strength
        calibrate_meta: Whether to calibrate meta-learner
        random_state: Random seed
        run_id: Optional run_id to target specific run (auto-discovered if None)
        apply_base_calibration: Whether to apply base model calibrators to val/test predictions.
                                For oof_posthoc strategy, this applies the OOF calibrator.
                                For per_fold strategy, predictions are already calibrated.

    Returns:
        ensemble: Fitted StackingEnsemble
        results: Dict containing predictions and metrics
    """
    logger.info(f"Training stacking ensemble with base models: {base_models}")

    # Collect OOF predictions (already calibrated by respective strategies)
    oof_dict, y_train, train_idx = collect_oof_predictions(
        results_dir, base_models, split_seed, run_id=run_id
    )

    # Load calibration info for base models (needed for val/test predictions)
    calibration_info = None
    if apply_base_calibration:
        calibration_info = load_calibration_info_for_models(
            results_dir, base_models, split_seed, run_id
        )
        # Log calibration strategies
        for model_name, calib_info in calibration_info.items():
            logger.info(
                f"Base model {model_name}: calibration strategy={calib_info.strategy}, "
                f"needs_posthoc={calib_info.needs_posthoc_calibration}"
            )

    # Create and fit ensemble
    ensemble = StackingEnsemble(
        base_model_names=base_models,
        meta_penalty=meta_penalty,
        meta_C=meta_C,
        calibrate_meta=calibrate_meta,
        random_state=random_state,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    # Collect val/test predictions and generate ensemble predictions
    results = {
        "base_models": base_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
        "calibration_strategies": {
            name: info.strategy for name, info in (calibration_info or {}).items()
        },
    }

    # Validation set
    try:
        val_preds_dict, y_val, val_idx = collect_split_predictions(
            results_dir,
            base_models,
            split_seed,
            "val",
            run_id=run_id,
            calibration_info=calibration_info,
        )
        val_proba = ensemble.predict_proba_from_base_preds(val_preds_dict)
        results["val_proba"] = val_proba[:, 1]
        results["y_val"] = y_val
        results["val_idx"] = val_idx
    except FileNotFoundError as e:
        logger.warning(f"Could not load validation predictions: {e}")

    # Test set
    try:
        test_preds_dict, y_test, test_idx = collect_split_predictions(
            results_dir,
            base_models,
            split_seed,
            "test",
            run_id=run_id,
            calibration_info=calibration_info,
        )
        test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)
        results["test_proba"] = test_proba[:, 1]
        results["y_test"] = y_test
        results["test_idx"] = test_idx
    except FileNotFoundError as e:
        logger.warning(f"Could not load test predictions: {e}")

    # Get meta-model coefficients
    results["meta_coef"] = ensemble.get_meta_model_coef()

    return ensemble, results


def save_ensemble_results(
    ensemble: StackingEnsemble,
    results: dict[str, Any],
    output_dir: Path,
    scenario: str = "IncidentOnly",
    git_version: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Save ensemble model and results to disk with comprehensive metadata.

    Args:
        ensemble: Fitted stacking ensemble
        results: Results dict from train_stacking_ensemble
        output_dir: Output directory
        scenario: Scenario name
        git_version: Optional git commit hash for reproducibility
        timestamp: Optional timestamp for run metadata
    """
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble model
    ensemble_path = output_dir / "ENSEMBLE__final_model.joblib"
    ensemble.save(ensemble_path)

    # Save predictions
    if "val_proba" in results and results["val_proba"] is not None:
        val_df = pd.DataFrame(
            {
                "idx": results["val_idx"],
                "y_true": results["y_val"],
                "y_prob": results["val_proba"],
            }
        )
        val_path = output_dir / "val_preds__ENSEMBLE.csv"
        val_df.to_csv(val_path, index=False)
        logger.info(f"Validation predictions saved: {val_path}")

    if "test_proba" in results and results["test_proba"] is not None:
        test_df = pd.DataFrame(
            {
                "idx": results["test_idx"],
                "y_true": results["y_test"],
                "y_prob": results["test_proba"],
            }
        )
        test_path = output_dir / "test_preds__ENSEMBLE.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test predictions saved: {test_path}")

    # Build comprehensive metadata
    now = timestamp or datetime.now().isoformat()

    meta = {
        # Model configuration
        "model": "ENSEMBLE",
        "base_models": results["base_models"],
        "n_base_models": len(results["base_models"]),
        "scenario": scenario,
        # Meta-learner configuration
        "meta_penalty": results["meta_penalty"],
        "meta_C": results["meta_C"],
        "meta_coef": results.get("meta_coef", {}),
        # Split information
        "split_seed": results["split_seed"],
        # Calibration strategies of base models
        "calibration_strategies": results.get("calibration_strategies", {}),
        # Provenance
        "timestamp": now,
        "git_version": git_version,
        # Data summary
        "n_train_samples": len(results.get("y_train", [])),
        "train_prevalence": (
            float(np.mean(results.get("y_train", [])))
            if results.get("y_train", []) is not None
            else None
        ),
        # Prediction summary
        "n_val_samples": len(results.get("y_val", [])) if "y_val" in results else None,
        "val_prevalence": (
            float(np.mean(results.get("y_val", [])))
            if results.get("y_val", []) is not None
            else None
        ),
        "n_test_samples": len(results.get("y_test", [])) if "y_test" in results else None,
        "test_prevalence": (
            float(np.mean(results.get("y_test", [])))
            if results.get("y_test", []) is not None
            else None
        ),
    }

    meta_path = output_dir / "ensemble_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Ensemble metadata saved: {meta_path}")

    # Also save run settings (like train.py does for single models)
    run_settings = {
        "split_seed": results["split_seed"],
        "base_models": results["base_models"],
        "meta_penalty": results["meta_penalty"],
        "meta_C": results["meta_C"],
        "meta_coef": results.get("meta_coef", {}),
        "timestamp": now,
    }
    settings_path = output_dir / "run_settings.json"
    with open(settings_path, "w") as f:
        json.dump(run_settings, f, indent=2, default=str)
    logger.info(f"Run settings saved: {settings_path}")
