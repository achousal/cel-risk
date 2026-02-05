"""Protein-only feature selection wrapper.

Ensures that feature selection (SelectKBest, ModelSpecificSelector, etc.)
operates exclusively on protein features, while covariate columns
(age, BMI, one-hot encoded demographics) pass through unchanged.

This prevents covariates from competing with proteins during univariate
scoring and keeps the biomarker panel strictly protein-only.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

_PROTEIN_SUFFIX = "_resid"


class ProteinOnlySelector(BaseEstimator, TransformerMixin):
    """Feature selector wrapper that restricts selection to protein columns only.

    Wraps any sklearn selector (e.g. SelectKBest, ModelSpecificSelector) and
    applies it only to protein columns (identified by suffix). Non-protein
    columns (numeric covariates, one-hot encoded categoricals) pass through
    unchanged.

    Requires DataFrame input. Set ``set_output(transform="pandas")`` on the
    preceding ColumnTransformer so this step receives column names.

    Parameters
    ----------
    selector : estimator
        Any sklearn-compatible selector with ``fit(X, y)``, ``get_support()``,
        and ``transform(X)`` methods (e.g. ``SelectKBest``).
    protein_suffix : str, default="_resid"
        Column name suffix used to identify protein features.
    """

    def __init__(self, selector=None, protein_suffix: str = _PROTEIN_SUFFIX):
        self.selector = selector
        self.protein_suffix = protein_suffix

    def fit(self, X, y=None):
        """Fit the wrapped selector on protein columns only.

        Parameters
        ----------
        X : DataFrame
            Feature matrix with named columns.
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self
        """
        if not hasattr(X, "columns"):
            raise TypeError(
                "ProteinOnlySelector requires DataFrame input. "
                "Ensure the preceding ColumnTransformer uses "
                "set_output(transform='pandas')."
            )

        all_cols = list(X.columns)
        self.feature_names_in_ = np.array(all_cols)
        self.protein_cols_ = [c for c in all_cols if c.endswith(self.protein_suffix)]
        self.non_protein_cols_ = [c for c in all_cols if not c.endswith(self.protein_suffix)]

        n_prot = len(self.protein_cols_)
        if n_prot == 0:
            logger.warning(
                "ProteinOnlySelector: no protein columns found (suffix=%s). "
                "Passing all %d features through unchanged.",
                self.protein_suffix,
                len(all_cols),
            )
            self.selected_proteins_ = []
            self.n_features_in_ = len(all_cols)
            return self

        # Fit selector on protein slice only
        X_prot = X[self.protein_cols_]
        self.selector_ = clone(self.selector)
        self.selector_.fit(X_prot, y)

        # Determine which proteins survived selection
        prot_support = self.selector_.get_support()
        self.selected_proteins_ = [
            col for col, keep in zip(self.protein_cols_, prot_support, strict=False) if keep
        ]
        self.n_features_in_ = len(all_cols)

        logger.debug(
            "ProteinOnlySelector: %d proteins -> %d selected, " "%d covariates passed through",
            n_prot,
            len(self.selected_proteins_),
            len(self.non_protein_cols_),
        )
        return self

    def transform(self, X):
        """Transform: keep selected proteins + all non-protein columns.

        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix.

        Returns
        -------
        X_out : DataFrame or ndarray
            Reduced feature matrix.
        """
        check_is_fitted(self, "n_features_in_")

        # No proteins found during fit -> pass through
        if not self.protein_cols_:
            return X

        cols_to_keep = self.selected_proteins_ + self.non_protein_cols_

        if hasattr(X, "columns"):
            return X[cols_to_keep]

        # Fallback for numpy arrays
        mask = self.get_support()
        return X[:, mask]

    def get_support(self, indices=False):
        """Return boolean mask over input features.

        Protein features follow the wrapped selector's support.
        Non-protein features are always True (pass through).

        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices instead of boolean mask.

        Returns
        -------
        support : ndarray
            Boolean mask or integer indices.
        """
        check_is_fitted(self, "n_features_in_")

        selected_set = set(self.selected_proteins_) | set(self.non_protein_cols_)
        mask = np.array([col in selected_set for col in self.feature_names_in_])

        if indices:
            return np.where(mask)[0]
        return mask

    def get_feature_names_out(self, input_features=None):
        """Return output feature names after selection.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        feature_names : ndarray of str
        """
        check_is_fitted(self, "n_features_in_")
        return np.array(self.selected_proteins_ + self.non_protein_cols_)

    def _more_tags(self):
        return {"requires_y": True}
