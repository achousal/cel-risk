"""
Split generation for CeliacRiskML pipeline.

This module handles three-way stratified splitting (TRAIN/VAL/TEST) with:
- Control downsampling to manage class imbalance
- Prevalent case enrichment (TRAIN only)
- Temporal split support for prospective validation
- Holdout set creation for final external validation
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ced_ml.data.schema import (
    CONTROL_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Stratification Utilities
# ============================================================================


def age_bins(age: pd.Series, scheme: str) -> pd.Series:
    """
    Bin age values for stratification.

    Args:
        age: Age series (missing values filled with median)
        scheme: Binning scheme ("age3" or "age2")

    Returns:
        Categorical series with age bins

    Raises:
        ValueError: If scheme is unknown
    """
    age = age.fillna(age.median())
    if scheme == "age3":
        return pd.cut(age, bins=[0, 40, 60, 150], labels=["young", "middle", "old"]).astype(str)
    elif scheme == "age2":
        return pd.cut(age, bins=[0, 60, 150], labels=["lt60", "ge60"]).astype(str)
    else:
        raise ValueError(f"Unknown age scheme: {scheme}")


def make_strata(
    df: pd.DataFrame,
    scheme: str,
    sex_col: str = "sex",
    age_col: str = "age",
) -> pd.Series:
    """
    Create stratification labels combining outcome, sex, and age.

    Schemes (ordered by granularity):
    - "outcome+sex+age3": Outcome x Sex x Age3 bins
    - "outcome+sex+age2": Outcome x Sex x Age2 bins
    - "outcome+age3": Outcome x Age3 bins
    - "outcome+sex": Outcome x Sex
    - "outcome": Outcome only

    Args:
        df: DataFrame with TARGET_COL, and optionally sex and age columns
        scheme: Stratification scheme
        sex_col: Column name for sex variable (default "sex")
        age_col: Column name for age variable (default "age")

    Returns:
        Series of stratification labels (strings)

    Raises:
        KeyError: If required columns are missing for the scheme
        ValueError: If scheme is unknown
    """
    outcome = df[TARGET_COL].astype(str).fillna("UnknownOutcome")

    if scheme == "outcome+sex+age3":
        sex = df[sex_col].astype(str).fillna("UnknownSex")
        ageb = age_bins(df[age_col], "age3")
        return (outcome + "_" + sex + "_" + ageb).astype(str)

    if scheme == "outcome+sex+age2":
        sex = df[sex_col].astype(str).fillna("UnknownSex")
        ageb = age_bins(df[age_col], "age2")
        return (outcome + "_" + sex + "_" + ageb).astype(str)

    if scheme == "outcome+age3":
        ageb = age_bins(df[age_col], "age3")
        return (outcome + "_" + ageb).astype(str)

    if scheme == "outcome+sex":
        sex = df[sex_col].astype(str).fillna("UnknownSex")
        return (outcome + "_" + sex).astype(str)

    if scheme == "outcome":
        return outcome.astype(str)

    raise ValueError(f"Unknown stratification scheme: {scheme}")


def collapse_rare_strata(df: pd.DataFrame, strata: pd.Series, min_count: int) -> pd.Series:
    """
    Collapse rare strata (< min_count samples) into outcome-based groups.

    Args:
        df: DataFrame with TARGET_COL
        strata: Stratification labels
        min_count: Minimum samples per stratum

    Returns:
        Series with rare strata collapsed
    """
    vc = strata.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index.tolist())
    if not rare:
        return strata

    outcome = df[TARGET_COL].astype(str).fillna("UnknownOutcome")
    collapsed = strata.copy()
    mask_rare = collapsed.isin(rare)
    collapsed.loc[mask_rare] = outcome.loc[mask_rare] + "_RARE"
    return collapsed.astype(str)


@dataclass
class SplitValidation:
    """Result of split validation.

    Attributes:
        is_valid: Whether the split is valid
        reason: Reason message (empty if valid, error description if invalid)
    """

    is_valid: bool
    reason: str


def validate_strata(strata: pd.Series) -> SplitValidation:
    """Validate that all strata have at least 2 samples (required for splitting).

    Args:
        strata: Stratification labels

    Returns:
        SplitValidation dataclass with validation result
    """
    vc = strata.value_counts(dropna=False)
    minc = int(vc.min()) if len(vc) else 0
    if minc < 2:
        return SplitValidation(is_valid=False, reason=f"min stratum count is {minc} (<2)")
    return SplitValidation(is_valid=True, reason="ok")


def build_working_strata(
    df: pd.DataFrame,
    min_count: int = 2,
    sex_col: str = "sex",
    age_col: str = "age",
) -> tuple[pd.Series, str]:
    """
    Build robust stratification labels by trying schemes from most to least granular.

    Tries schemes in order:
    1. outcome+sex+age3
    2. outcome+sex+age2
    3. outcome+age3
    4. outcome+sex
    5. outcome (fallback)

    Args:
        df: DataFrame with outcome, sex, age columns
        min_count: Minimum samples per stratum
        sex_col: Column name for sex variable (default "sex")
        age_col: Column name for age variable (default "age")

    Returns:
        (strata_series, scheme_name)
    """
    schemes = [
        "outcome+sex+age3",
        "outcome+sex+age2",
        "outcome+age3",
        "outcome+sex",
        "outcome",
    ]
    last_reason: str | None = None

    for sch in schemes:
        try:
            strata = make_strata(df, sch, sex_col=sex_col, age_col=age_col)
            strata = collapse_rare_strata(df, strata, min_count=min_count)
            validation = validate_strata(strata)
            if validation.is_valid:
                return strata, sch
            last_reason = f"{sch}: {validation.reason}"
        except KeyError as e:
            # Missing column required for this scheme, try next
            last_reason = f"{sch}: missing column {e}"
            continue

    # Fallback to outcome-only
    strata = make_strata(df, "outcome", sex_col=sex_col, age_col=age_col)
    strata = collapse_rare_strata(df, strata, min_count=min_count)
    return strata, f"outcome (fallback; last failure: {last_reason})"


# ============================================================================
# Control Downsampling
# ============================================================================


def downsample_controls(
    idx_set: np.ndarray,
    df: pd.DataFrame,
    case_labels: list[str],
    controls_per_case: float | None,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Downsample control samples to achieve target case:control ratio.

    Args:
        idx_set: Indices to downsample
        df: DataFrame with TARGET_COL
        case_labels: List of positive outcome labels (e.g., ["Incident"])
        controls_per_case: Target number of controls per case (None = no downsampling)
        rng: Random state for reproducibility

    Returns:
        Downsampled indices (sorted)

    Example:
        >>> idx = np.array([0, 1, 2, 3, 4])  # 2 cases, 3 controls
        >>> rng = np.random.RandomState(42)
        >>> result = downsample_controls(idx, df, ["Incident"], 1.0, rng)
        >>> # Returns ~2 cases + 2 controls
    """
    if controls_per_case is None or controls_per_case <= 0:
        return np.sort(idx_set.astype(int))

    idx_set = np.asarray(idx_set, dtype=int)
    if idx_set.size == 0:
        return idx_set

    labels = df.loc[idx_set, TARGET_COL].astype(str)
    if case_labels is None:
        case_labels = []
    if isinstance(case_labels, str):
        case_labels = [case_labels]

    idx_cases = idx_set[labels.isin(case_labels)]
    idx_controls = idx_set[labels == CONTROL_LABEL]

    n_cases = int(idx_cases.size)
    n_controls = int(idx_controls.size)

    if n_cases == 0 or n_controls == 0:
        logger.debug(f"Skip control downsample (cases={n_cases}, controls={n_controls})")
        return np.sort(idx_set.astype(int))

    target_controls = int(round(n_cases * float(controls_per_case)))
    if target_controls >= n_controls:
        logger.debug(f"Keep all controls ({n_controls}); target={target_controls}")
        return np.sort(idx_set.astype(int))

    keep_controls = rng.choice(idx_controls, size=target_controls, replace=False)
    kept = np.sort(np.concatenate([idx_cases, keep_controls]).astype(int))
    logger.info(f"Downsample controls: {n_controls} -> {target_controls} (cases={n_cases})")
    return kept


# ============================================================================
# Temporal Ordering
# ============================================================================


def temporal_order_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Sort dataframe indices by temporal column (date or numeric).

    Handles:
    - Datetime columns (parsed with pd.to_datetime)
    - Numeric columns
    - Missing values (filled with min - 1 or -inf)

    Args:
        df: DataFrame to sort
        col: Column name for temporal ordering

    Returns:
        Array of row indices in temporal order

    Raises:
        ValueError: If column not found in dataframe or has no valid sortable values
    """
    if col not in df.columns:
        raise ValueError(f"Temporal column '{col}' not found in dataframe.")

    ser = df[col]
    order_vals = None

    # Try datetime parsing
    try:
        order_vals = pd.to_datetime(ser, errors="coerce")
    except Exception:
        logger.warning(
            f"Failed to parse temporal column '{col}' as datetime; will try numeric parsing.",
            exc_info=True,
        )
        order_vals = None

    # Fallback to numeric
    if order_vals is None or order_vals.isna().all():
        order_vals = pd.to_numeric(ser, errors="coerce")

    # Validate: at least some values must be parseable
    if isinstance(order_vals, pd.Series) and order_vals.isna().all():
        raise ValueError(
            f"Temporal column '{col}' has no valid sortable values. "
            "Column must contain datetime or numeric values."
        )

    # Check for high missingness rate (potential temporal leakage risk)
    if isinstance(order_vals, pd.Series):
        missing_rate = order_vals.isna().sum() / len(order_vals)
        if missing_rate > 0.05:
            import warnings

            warnings.warn(
                f"Temporal column '{col}' has {missing_rate:.1%} missing values. "
                f"Missing values will be placed at the beginning of the timeline. "
                f"If missingness patterns are informative, this may introduce temporal leakage. "
                f"Consider imputing temporal values or dropping samples with missing dates.",
                category=UserWarning,
                stacklevel=2,
            )

    # Fill missing values with min - 1 (or -inf for numeric)
    if isinstance(order_vals, pd.Series):
        fill_value = order_vals.min()
        if isinstance(fill_value, pd.Timestamp) or np.issubdtype(order_vals.dtype, np.datetime64):
            fill_value = (
                fill_value - pd.Timedelta(days=1)
                if pd.notna(fill_value)
                else pd.Timestamp("1970-01-01")
            )
        elif pd.isna(fill_value):
            fill_value = float("-inf")
        order_vals = order_vals.fillna(fill_value)

    # Sort by temporal value, then by original index (stable sort)
    tmp = pd.DataFrame(
        {
            "order_val": order_vals,
            "idx": np.arange(len(df)),
        }
    )
    tmp = tmp.sort_values(["order_val", "idx"], kind="mergesort").reset_index(drop=True)
    return tmp["idx"].to_numpy(dtype=int)


# ============================================================================
# Prevalent Case Enrichment
# ============================================================================


def add_prevalent_to_train(
    train_idx: np.ndarray,
    df: pd.DataFrame,
    prevalent_frac: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Add fraction of prevalent cases to TRAIN set.

    Used for training signal enrichment while keeping VAL/TEST prospective.

    Args:
        train_idx: Current TRAIN indices
        df: DataFrame with TARGET_COL
        prevalent_frac: Fraction of prevalent cases to add (0.0-1.0)
        rng: Random state for sampling

    Returns:
        Updated TRAIN indices with prevalent cases added (sorted)

    Example:
        >>> train_idx = np.array([0, 1, 2])  # Base training set
        >>> # Add 50% of prevalent cases
        >>> result = add_prevalent_to_train(train_idx, df, 0.5, rng)
    """
    idx_prev = df.index[df[TARGET_COL] == PREVALENT_LABEL].to_numpy(dtype=int)

    if prevalent_frac >= 1.0:
        idx_prev_keep = idx_prev
    elif prevalent_frac <= 0.0 or len(idx_prev) == 0:
        idx_prev_keep = np.array([], dtype=int)
    else:
        n_keep = int(round(prevalent_frac * len(idx_prev)))
        n_keep = min(len(idx_prev), max(1, n_keep))
        idx_prev_keep = rng.choice(idx_prev, size=n_keep, replace=False)

    if len(idx_prev_keep) > 0:
        logger.info(f"Added prevalent={len(idx_prev_keep):,} (frac={prevalent_frac:.2f}) to TRAIN")
        return np.sort(np.concatenate([train_idx, idx_prev_keep]).astype(int))

    return train_idx


# ============================================================================
# Three-Way Split Generation
# ============================================================================


def stratified_train_val_test_split(
    indices: np.ndarray,
    y: np.ndarray,
    strata: pd.Series,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform three-way stratified split into TRAIN/VAL/TEST.

    Args:
        indices: Row indices to split
        y: Binary outcome labels
        strata: Stratification labels
        val_size: Validation set fraction (0.0 for no validation)
        test_size: Test set fraction
        random_state: Random seed for reproducibility

    Returns:
        (idx_train, idx_val, idx_test, y_train, y_val, y_test)

    Raises:
        ValueError: If val_size + test_size >= 1.0 (must leave room for training)

    Example:
        >>> indices = np.arange(100)
        >>> y = np.random.randint(0, 2, 100)
        >>> strata = pd.Series(["A"] * 50 + ["B"] * 50)
        >>> idx_tr, idx_val, idx_te, y_tr, y_val, y_te = stratified_train_val_test_split(
        ...     indices, y, strata, val_size=0.25, test_size=0.25, random_state=42
        ... )
    """
    # Validate split sizes leave room for training data
    effective_val = val_size if val_size and val_size > 0 else 0.0
    if effective_val + test_size >= 1.0:
        raise ValueError(
            f"val_size ({effective_val}) + test_size ({test_size}) = {effective_val + test_size} >= 1.0. "
            "Must leave room for training data (sum must be < 1.0)."
        )

    if val_size and val_size > 0:
        # Two-step split: TRAIN vs (VAL+TEST), then VAL vs TEST
        temp_size = float(val_size + test_size)
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            indices,
            y,
            test_size=temp_size,
            random_state=random_state,
            stratify=strata,
        )

        # Re-stratify for VAL vs TEST split
        strata_temp = strata.loc[idx_temp]
        rel_test = float(test_size) / temp_size

        # Use a derived seed to ensure different randomness from first split
        second_seed = random_state ^ 0x5678

        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp,
            y_temp,
            test_size=rel_test,
            random_state=second_seed,
            stratify=strata_temp,
        )
    else:
        # Two-way split: TRAIN vs TEST
        idx_train, idx_test, y_train, y_test = train_test_split(
            indices,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strata,
        )
        idx_val = np.array([], dtype=int)
        y_val = np.array([], dtype=int)

    # Validate minimum cases and controls in each split
    def _validate_split(split_name: str, y_split: np.ndarray) -> None:
        """Validate that split has minimum 2 cases and 2 controls."""
        if len(y_split) == 0:
            return  # Empty split is OK (e.g., no validation set)

        n_cases = int(y_split.sum())
        n_controls = int(len(y_split) - n_cases)

        if n_cases < 2:
            raise ValueError(
                f"{split_name} split has insufficient cases: {n_cases} < 2. "
                "This will cause downstream metric failures. "
                "Try reducing val_size/test_size or using a larger dataset."
            )
        if n_controls < 2:
            raise ValueError(
                f"{split_name} split has insufficient controls: {n_controls} < 2. "
                "This will cause downstream metric failures. "
                "Try reducing val_size/test_size or using a larger dataset."
            )

    _validate_split("TRAIN", y_train)
    _validate_split("VAL", y_val)
    _validate_split("TEST", y_test)

    return idx_train, idx_val, idx_test, y_train, y_val, y_test


def temporal_train_val_test_split(
    indices: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform three-way temporal split (chronological order).

    TRAIN = earliest samples, TEST = latest samples, VAL = middle samples.

    NOTE: The indices array should already be sorted by timestamp;
    this function does NOT perform sorting.

    Args:
        indices: Row indices (already in temporal order)
        y: Binary outcome labels
        val_size: Validation set fraction (0.0 for no validation)
        test_size: Test set fraction

    Returns:
        (idx_train, idx_val, idx_test, y_train, y_val, y_test)

    Raises:
        ValueError: If split produces empty TRAIN set or val_size + test_size >= 1.0

    Example:
        >>> indices = np.arange(100)  # Pre-sorted by date
        >>> y = np.random.randint(0, 2, 100)
        >>> idx_tr, idx_val, idx_te, y_tr, y_val, y_te = temporal_train_val_test_split(
        ...     indices, y, val_size=0.25, test_size=0.25
        ... )
    """
    # Validate split sizes leave room for training data
    effective_val = val_size if val_size and val_size > 0 else 0.0
    if effective_val + test_size >= 1.0:
        raise ValueError(
            f"val_size ({effective_val}) + test_size ({test_size}) = {effective_val + test_size} >= 1.0. "
            "Must leave room for training data (sum must be < 1.0)."
        )

    if len(indices) < 2:
        raise ValueError("Temporal split requires at least 2 samples.")

    n_total = len(indices)
    n_test = int(round(test_size * n_total))
    n_val = int(round(val_size * n_total)) if val_size > 0 else 0

    # Ensure at least 1 sample in test, at least 0 in val, at least 1 in train
    n_test = min(max(1, n_test), max(1, n_total - 1))
    n_val = min(max(0, n_val), max(0, n_total - n_test - 1))
    n_train = n_total - n_test - n_val

    if n_train < 1:
        raise ValueError("Temporal split produced empty TRAIN. Reduce val_size/test_size.")

    idx_train = indices[:n_train]
    idx_val = indices[n_train : n_train + n_val] if n_val > 0 else np.array([], dtype=int)
    idx_test = indices[n_train + n_val :]

    y_train = y[np.isin(indices, idx_train)]
    y_val = y[np.isin(indices, idx_val)] if n_val > 0 else np.array([], dtype=int)
    y_test = y[np.isin(indices, idx_test)]

    # Validate minimum cases and controls in each split
    def _validate_split(split_name: str, y_split: np.ndarray) -> None:
        """Validate that split has minimum 2 cases and 2 controls."""
        if len(y_split) == 0:
            return  # Empty split is OK (e.g., no validation set)

        n_cases = int(y_split.sum())
        n_controls = int(len(y_split) - n_cases)

        if n_cases < 2:
            raise ValueError(
                f"{split_name} split has insufficient cases: {n_cases} < 2. "
                "This will cause downstream metric failures. "
                "Try reducing val_size/test_size or using a larger dataset."
            )
        if n_controls < 2:
            raise ValueError(
                f"{split_name} split has insufficient controls: {n_controls} < 2. "
                "This will cause downstream metric failures. "
                "Try reducing val_size/test_size or using a larger dataset."
            )

    _validate_split("TRAIN", y_train)
    _validate_split("VAL", y_val)
    _validate_split("TEST", y_test)

    return idx_train, idx_val, idx_test, y_train, y_val, y_test


# ============================================================================
# Split Summary Utilities
# ============================================================================


def compute_split_id(indices: np.ndarray) -> str:
    """
    Generate reproducible hash ID for split indices.

    Args:
        indices: Array of indices

    Returns:
        12-character hex hash

    Example:
        >>> idx = np.array([0, 1, 2, 3])
        >>> split_id = compute_split_id(idx)
        >>> assert len(split_id) == 12
    """
    import hashlib

    sorted_idx = np.sort(indices)
    hash_obj = hashlib.md5(sorted_idx.tobytes())
    return hash_obj.hexdigest()[:12]


def summarize_split(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Compute summary statistics for a split.

    Args:
        idx_train, idx_val, idx_test: Index arrays
        y_train, y_val, y_test: Outcome arrays

    Returns:
        Dictionary with counts, prevalence, and split IDs
    """
    summary = {
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "n_train_pos": int(y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "prevalence_train": float(y_train.mean()) if len(y_train) > 0 else 0.0,
        "prevalence_test": float(y_test.mean()) if len(y_test) > 0 else 0.0,
        "split_id_train": compute_split_id(idx_train),
        "split_id_test": compute_split_id(idx_test),
    }

    if len(idx_val) > 0 and len(y_val) > 0:
        summary.update(
            {
                "n_val": int(len(idx_val)),
                "n_val_pos": int(y_val.sum()),
                "prevalence_val": float(y_val.mean()),
                "split_id_val": compute_split_id(idx_val),
            }
        )

    return summary
