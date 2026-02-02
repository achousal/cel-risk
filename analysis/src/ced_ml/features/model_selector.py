"""Model-specific feature selector for hybrid_stability pipeline.

Adds a model-aware feature selection step between KBest and the classifier.
Each model type uses its own inductive bias (L1 coefficients, tree importances)
to further prune features, producing genuinely model-specific stable panels.

Pipeline position:
    MW screen -> KBest F-test -> ModelSpecificSelector -> Classifier

"""

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, SelectorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..data.schema import ModelName

logger = logging.getLogger(__name__)


def _make_lr_selector(random_state: int = 42):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        solver="saga",
        C=1.0,
        l1_ratio=1.0,
        max_iter=2000,  # Higher than registry default (1000) for feature selection stability
        class_weight="balanced",
        random_state=random_state,
    )


def _make_svm_selector(random_state: int = 42):
    from sklearn.svm import LinearSVC

    return LinearSVC(
        penalty="l1",
        dual=False,
        C=1.0,
        max_iter=2000,  # Higher than registry default (1000) for feature selection stability
        class_weight="balanced",
        random_state=random_state,
    )


def _make_rf_selector(random_state: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def _make_xgb_selector(random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
    )


# Selector estimator factories keyed by model name.
# Each returns an unfitted lightweight estimator suitable for feature selection.
_SELECTOR_ESTIMATORS = {
    ModelName.LR_EN: _make_lr_selector,
    ModelName.LR_L1: _make_lr_selector,
    ModelName.LinSVM_cal: _make_svm_selector,
    ModelName.RF: _make_rf_selector,
    ModelName.XGBoost: _make_xgb_selector,
}


class ModelSpecificSelector(SelectorMixin, BaseEstimator):
    """Feature selector that uses a model-specific estimator to prune features.

    Wraps sklearn SelectFromModel with a lightweight estimator chosen based on
    the downstream classifier type. Provides a min_features floor to prevent
    degenerate empty selections (e.g. L1 models zeroing all coefficients).

    Parameters
    ----------
    model_name : str
        Downstream model identifier (LR_EN, LR_L1, LinSVM_cal, RF, XGBoost).
        Determines which lightweight estimator is used for selection.
    threshold : str or float, default="median"
        Feature importance threshold for SelectFromModel.
    max_features : int or None, default=None
        Maximum number of features to select. None means no cap.
    min_features : int, default=10
        Minimum number of features to retain. If the selector produces fewer
        than this, the top-k features by absolute importance are kept instead.
    """

    def __init__(
        self,
        model_name: str = ModelName.LR_EN,
        threshold: str | float = "median",
        max_features: int | None = None,
        min_features: int = 10,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.max_features = max_features
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the selector estimator and compute the feature support mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = check_array(X, accept_sparse=False, ensure_min_features=1)

        # Store feature names if available (pandas DataFrame)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        n_features = X.shape[1]

        # Build and fit selector estimator
        factory = _SELECTOR_ESTIMATORS.get(self.model_name)
        if factory is None:
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. "
                f"Supported: {sorted(_SELECTOR_ESTIMATORS.keys())}"
            )

        estimator = factory(random_state=self.random_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            estimator.fit(X, y)

        self.estimator_ = estimator

        # Extract importances for fallback logic
        self.importances_ = self._get_importances(estimator, n_features)

        # Use SelectFromModel to compute initial mask
        sfm = SelectFromModel(
            estimator,
            threshold=self.threshold,
            max_features=self.max_features,
            prefit=True,
        )
        mask = sfm.get_support()

        # Enforce min_features floor
        n_selected = mask.sum()
        if n_selected < self.min_features and n_features >= self.min_features:
            logger.warning(
                "ModelSpecificSelector(%s): only %d features selected (threshold=%s). "
                "Falling back to top-%d by importance.",
                self.model_name,
                n_selected,
                self.threshold,
                self.min_features,
            )
            top_indices = np.argsort(self.importances_)[::-1][: self.min_features]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_indices] = True
        elif n_selected == 0:
            # Edge case: no features and fewer than min_features total
            logger.warning(
                "ModelSpecificSelector(%s): 0 features selected. " "Keeping all %d features.",
                self.model_name,
                n_features,
            )
            mask = np.ones(n_features, dtype=bool)

        self.support_mask_ = mask
        self.n_features_in_ = n_features

        logger.debug(
            "ModelSpecificSelector(%s): %d -> %d features",
            self.model_name,
            n_features,
            mask.sum(),
        )
        return self

    def _get_support_mask(self):
        """Return the boolean support mask (required by SelectorMixin)."""
        check_is_fitted(self, "support_mask_")
        return self.support_mask_

    @staticmethod
    def _get_importances(estimator, n_features: int) -> np.ndarray:
        """Extract absolute feature importances from fitted estimator."""
        if hasattr(estimator, "feature_importances_"):
            return np.abs(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = coef.ravel()
            return np.abs(coef)
        else:
            # Fallback: uniform importance
            return np.ones(n_features)
