"""K-best feature selection using univariate statistical tests.

This module provides functions for selecting top-K features based on
univariate F-statistic (ANOVA F-value) between each feature and the target.

Key functions:
- select_kbest_features: Main interface for K-best selection
- compute_f_classif_scores: Compute F-statistics for each feature
- rank_features_by_score: Rank features by score (descending)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


def select_kbest_features(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int,
    protein_cols: list[str],
    missing_strategy: str = "median",
    fallback_to_variance: bool = True,
    log_stats: bool = False,
) -> list[str]:
    """Select top-K protein features using univariate F-test.

    Computes ANOVA F-value for each protein and selects the K features
    with highest scores. Missing values are imputed before scoring.

    Args:
        X: Feature matrix (must contain protein_cols)
        y: Binary target vector (0/1)
        k: Number of features to select (clipped to available proteins)
        protein_cols: List of protein column names to consider
        missing_strategy: Imputation strategy ("median" or "mean")
        fallback_to_variance: If F-test fails, use variance ranking
        log_stats: Whether to log selection statistics

    Returns:
        List of selected protein column names (length <= k)

    Raises:
        ValueError: If no valid protein columns found

    Example:
        >>> X = pd.DataFrame({"P1": [1, 2, 3], "P2": [4, 5, 6]})
        >>> y = np.array([0, 0, 1])
        >>> select_kbest_features(X, y, k=1, protein_cols=["P1", "P2"])
        ['P2']
    """
    # Filter to available proteins
    available_prots = [c for c in protein_cols if c in X.columns]

    if len(available_prots) == 0:
        raise ValueError("No valid protein columns found in X")

    # Clip k to available range
    k_effective = max(1, min(len(available_prots), k))

    if log_stats:
        logger.info(f"K-best selection (k={k_effective})")
        logger.info("  Score function: f_classif (ANOVA F-value)")
        logger.info(f"  Evaluating {len(available_prots)} proteins")

    # Extract and impute protein matrix
    X_proteins = X[available_prots].apply(pd.to_numeric, errors="coerce")
    X_imputed = _impute_proteins(X_proteins, strategy=missing_strategy)

    # Compute F-statistics
    used_fallback = False
    try:
        scores = compute_f_classif_scores(X_imputed.to_numpy(), y)
        selected_indices = rank_features_by_score(scores, k=k_effective)
    except (ValueError, RuntimeError) as e:
        if not fallback_to_variance:
            raise
        # Fallback: rank by variance
        logger.warning(f"F-test failed ({type(e).__name__}): using variance ranking fallback")
        variances = np.nanvar(X_imputed.to_numpy(), axis=0)
        selected_indices = rank_features_by_score(variances, k=k_effective)
        used_fallback = True

    selected_proteins = [available_prots[i] for i in selected_indices]

    if log_stats and not used_fallback:
        selected_scores = scores[selected_indices]
        score_min = float(np.min(selected_scores))
        score_median = float(np.median(selected_scores))
        score_max = float(np.max(selected_scores))

        logger.info(
            f"  Selected protein scores: min={score_min:.2f}, median={score_median:.2f}, max={score_max:.2f}"
        )
        logger.info(
            f"  Rationale: Top-{k_effective} proteins by F-statistic maximize univariate discrimination"
        )

    return selected_proteins


def compute_f_classif_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute ANOVA F-value for each feature vs binary target.

    Uses sklearn.feature_selection.f_classif which computes the ANOVA
    F-statistic for each feature. Higher F-values indicate stronger
    univariate association with the target.

    Args:
        X: Feature matrix (n_samples, n_features), must be numeric
        y: Binary target vector (n_samples,)

    Returns:
        Array of F-statistics (n_features,)

    Raises:
        ValueError: If y has fewer than 2 classes

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 0, 1])
        >>> scores = compute_f_classif_scores(X, y)
        >>> scores.shape
        (2,)
    """
    y = np.asarray(y).astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError("Target y must have at least 2 classes")

    F_scores, _ = f_classif(X, y)
    return np.asarray(F_scores, dtype=float)


def rank_features_by_score(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Rank features by score and return indices of top-K.

    Args:
        scores: Array of feature scores (higher is better)
        k: Number of top features to select

    Returns:
        Indices of top-K features (sorted by score descending)

    Example:
        >>> scores = np.array([0.5, 2.0, 1.0])
        >>> rank_features_by_score(scores, k=2)
        array([1, 2])
    """
    scores = np.asarray(scores, dtype=float)
    k_effective = max(1, min(len(scores), k))

    # Sort descending (highest scores first)
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices[:k_effective]


def extract_selected_proteins_from_kbest(
    fitted_pipeline,
    protein_cols: list[str],
    step_name: str = "sel",
    feature_name_prefix: str = "num__",
) -> list[str]:
    """Extract selected protein names from fitted sklearn SelectKBest step.

    Args:
        fitted_pipeline: Fitted sklearn Pipeline with SelectKBest step
        protein_cols: Original protein column names
        step_name: Name of SelectKBest step in pipeline
        feature_name_prefix: Prefix added by ColumnTransformer

    Returns:
        List of selected protein column names

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.feature_selection import SelectKBest
        >>> pipe = Pipeline([("sel", SelectKBest(k=2))])
        >>> # ... fit pipeline ...
        >>> extract_selected_proteins_from_kbest(pipe, ["P1", "P2", "P3"])
    """
    if not hasattr(fitted_pipeline, "named_steps"):
        return []

    if step_name not in fitted_pipeline.named_steps:
        return []

    # Get feature names from preprocessing step
    pre_step = fitted_pipeline.named_steps.get("pre")
    if pre_step is None:
        return []

    feature_names = _get_feature_names_from_preprocessor(pre_step)

    # Get support mask from SelectKBest
    selector = fitted_pipeline.named_steps[step_name]
    support_mask = selector.get_support()

    selected_features = feature_names[support_mask]

    # Extract proteins (remove prefix)
    protein_set = set()
    for name in selected_features:
        if name.startswith(feature_name_prefix):
            original_name = name[len(feature_name_prefix) :]
            if original_name in protein_cols:
                protein_set.add(original_name)

    return sorted(protein_set)


def _impute_proteins(
    X_proteins: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """Impute missing values in protein matrix.

    Args:
        X_proteins: Protein feature matrix (may contain NaN)
        strategy: "median" or "mean"

    Returns:
        Imputed DataFrame (no missing values)
    """
    if strategy.strip().lower() == "mean":
        fill_values = X_proteins.mean(axis=0, skipna=True)
    else:
        fill_values = X_proteins.median(axis=0, skipna=True)

    return X_proteins.fillna(fill_values)


def _get_feature_names_from_preprocessor(preprocessor) -> np.ndarray:
    """Extract feature names from fitted ColumnTransformer.

    Args:
        preprocessor: Fitted sklearn ColumnTransformer

    Returns:
        Array of feature names
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out()

    # Fallback for older sklearn versions
    if hasattr(preprocessor, "transformers_"):
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if trans == "drop" or cols is None:
                continue
            if hasattr(trans, "get_feature_names_out"):
                trans_names = trans.get_feature_names_out(cols)
            else:
                trans_names = [f"{name}__{c}" for c in cols]
            names.extend(trans_names)
        return np.array(names)

    raise AttributeError("Cannot extract feature names from preprocessor")


def compute_protein_statistics(
    X: pd.DataFrame,
    y: np.ndarray,
    protein: str,
) -> dict | None:
    """Compute univariate statistics for a single protein feature.

    Computes summary statistics separately for cases (y=1) and controls (y=0),
    plus effect size (Cohen's d) and t-test p-value.

    Args:
        X: Feature matrix containing protein column
        y: Binary target (0/1)
        protein: Name of protein column

    Returns:
        Dictionary with keys:
            - protein: column name
            - n_total: total non-missing samples
            - n_case: case samples
            - n_control: control samples
            - mean_case: mean in cases
            - mean_control: mean in controls
            - sd_case: standard deviation in cases
            - sd_control: standard deviation in controls
            - cohens_d: standardized mean difference
            - p_ttest: t-test p-value (Welch's, unequal variance)
        Returns None if insufficient data
    """
    if protein not in X.columns:
        return None

    x = pd.to_numeric(X[protein], errors="coerce")
    y = np.asarray(y).astype(int)

    # Filter to non-missing
    valid_mask = x.notna().to_numpy()
    x_valid = x[valid_mask].to_numpy()
    y_valid = y[valid_mask]

    # Check both classes present with sufficient samples
    unique_classes = np.unique(y_valid)
    if len(unique_classes) < 2:
        return None

    # Need at least 2 samples per class for meaningful statistics
    for cls in unique_classes:
        if np.sum(y_valid == cls) < 2:
            return None

    # Split by class
    x_control = x_valid[y_valid == 0]
    x_case = x_valid[y_valid == 1]

    # Compute statistics
    mean_control = float(np.mean(x_control))
    mean_case = float(np.mean(x_case))
    sd_control = float(np.std(x_control, ddof=1))
    sd_case = float(np.std(x_case, ddof=1))

    # Cohen's d (pooled standard deviation)
    n_control, n_case = len(x_control), len(x_case)
    pooled_sd = np.sqrt(
        ((n_control - 1) * sd_control**2 + (n_case - 1) * sd_case**2)
        / max(1, n_control + n_case - 2)
    )

    cohens_d = (mean_case - mean_control) / pooled_sd if pooled_sd > 0 else np.nan

    # T-test (Welch's, unequal variance)
    try:
        from scipy import stats

        _, p_value = stats.ttest_ind(x_case, x_control, equal_var=False, nan_policy="omit")
        p_ttest = float(p_value)
    except (ValueError, RuntimeError, AttributeError):
        # Graceful degradation: t-test is optional diagnostic statistic
        p_ttest = np.nan

    return {
        "protein": protein,
        "n_total": int(valid_mask.sum()),
        "n_case": int(n_case),
        "n_control": int(n_control),
        "mean_case": mean_case,
        "mean_control": mean_control,
        "sd_case": sd_case,
        "sd_control": sd_control,
        "cohens_d": cohens_d,
        "p_ttest": p_ttest,
    }


def build_kbest_pipeline_step(k: int) -> Any:
    """Build SelectKBest pipeline step for feature selection.

    Creates a sklearn SelectKBest transformer configured with F-test scoring.
    This isolates sklearn implementation details from CLI code.

    Args:
        k: Number of features to select

    Returns:
        Unfitted SelectKBest transformer

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> kbest = build_kbest_pipeline_step(k=100)
        >>> pipe = Pipeline([("sel", kbest), ("clf", classifier)])
    """
    return SelectKBest(score_func=f_classif, k=k)


class ScreeningTransformer:
    """Univariate screening transformer for feature pre-filtering.

    Wraps screen_proteins to work as an sklearn transformer.
    Screens features using Mann-Whitney or F-statistic, keeping top-N.
    """

    def __init__(
        self,
        method: str = "mannwhitney",
        top_n: int = 1000,
        protein_cols: list[str] | None = None,
    ):
        """Initialize screening transformer.

        Args:
            method: "mannwhitney" or "f_classif"
            top_n: Number of top features to keep
            protein_cols: List of protein column names (set during fit)
        """
        self.method = method
        self.top_n = top_n
        self.protein_cols = protein_cols or []
        self.selected_features_ = None
        self.screening_stats_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """Fit screening transformer.

        Args:
            X: Feature matrix (DataFrame or array from ColumnTransformer)
            y: Binary target vector

        Returns:
            self
        """
        if y is None:
            raise ValueError("y must be provided for screening")

        import logging

        from ced_ml.features.screening import screen_proteins

        logger = logging.getLogger(__name__)

        # Store all column names for reconstruction in transform()
        if isinstance(X, pd.DataFrame):
            self.all_feature_names_ = X.columns.tolist()
        else:
            raise TypeError(
                "ScreeningTransformer expects DataFrame input. "
                "It should be placed before ColumnTransformer in the pipeline, "
                "not after. Current pipeline has ColumnTransformer -> ScreeningTransformer "
                "which is incompatible."
            )

        # Determine protein columns if not set
        if not self.protein_cols:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.protein_cols = numeric_cols

        selected_prots, stats, was_cached = screen_proteins(
            X_train=X,
            y_train=y,
            protein_cols=self.protein_cols,
            method=self.method,
            top_n=self.top_n,
        )

        # Use DEBUG logging for cached results to reduce noise in repeated CV folds
        log_fn = logger.debug if was_cached else logger.info
        log_fn(
            f"Screening: {self.method} on {len(self.protein_cols)} proteins "
            f"(N={len(X)}, top_n={self.top_n}) -> {len(selected_prots)} selected"
            + (" [CACHED]" if was_cached else "")
        )

        self.selected_features_ = selected_prots
        self.selected_proteins_ = selected_prots
        self.screening_stats_ = stats

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: keep only selected protein features + all non-protein columns.

        Args:
            X: Feature matrix (must be DataFrame)

        Returns:
            Subset of X with selected proteins + all metadata columns
        """
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ScreeningTransformer.transform() expects DataFrame input. "
                "Ensure ScreeningTransformer is placed before ColumnTransformer."
            )

        # Keep selected proteins + all non-protein columns (metadata)
        selected_proteins = [c for c in self.selected_features_ if c in X.columns]
        non_protein_cols = [c for c in X.columns if c not in self.protein_cols]
        cols_to_keep = selected_proteins + non_protein_cols

        return X[cols_to_keep]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        """Get output feature names (for sklearn compatibility)."""
        if self.selected_features_ is None:
            return np.array([])
        return np.array(self.selected_features_)
