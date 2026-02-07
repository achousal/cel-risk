"""
Data I/O utilities for CeliacRiskML pipeline.

This module handles reading proteomics CSV files with schema validation,
dtype coercion, and quality checks.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ced_ml.data.schema import (
    CAT_COLS,
    CED_DATE_COL,
    EXPECTED_N_PROTEINS,
    ID_COL,
    META_NUM_COLS,
    PROTEIN_SUFFIX,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def usecols_for_proteomics(
    numeric_metadata: list[str] | None = None,
    categorical_metadata: list[str] | None = None,
) -> Callable[[str], bool]:
    """
    Create column filter function for pd.read_csv(usecols=...).

    Returns columns needed for modeling:
    - ID column (eid)
    - Target column (CeD_comparison)
    - Date column (CeD_date)
    - Numeric metadata (defaults to schema META_NUM_COLS if not specified)
    - Categorical metadata (defaults to schema CAT_COLS if not specified)
    - Protein features (*_resid columns)

    Args:
        numeric_metadata: Numeric metadata columns to include (default: META_NUM_COLS)
        categorical_metadata: Categorical metadata columns to include (default: CAT_COLS)

    Returns:
        Function that takes column name and returns True if column should be loaded
    """
    num_cols = numeric_metadata if numeric_metadata is not None else META_NUM_COLS
    cat_cols = categorical_metadata if categorical_metadata is not None else CAT_COLS

    def _filter(col: str) -> bool:
        if col in (ID_COL, TARGET_COL, CED_DATE_COL):
            return True
        if col in num_cols:
            return True
        if col in cat_cols:
            return True
        if isinstance(col, str) and col.endswith(PROTEIN_SUFFIX):
            return True
        return False

    return _filter


def read_proteomics_csv(
    filepath: str,
    *,
    usecols: Callable[[str], bool] | None = None,
    low_memory: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """Read proteomics CSV file with schema-aware column filtering.

    Args:
        filepath: Path to CSV file
        usecols: Optional column filter function (default: usecols_for_proteomics())
        low_memory: Whether to use low_memory mode for pd.read_csv (default: False)
        validate: Whether to validate required columns after loading (default: True)

    Returns:
        DataFrame with selected columns

    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If validate=True and required columns are missing

    Example:
        >>> df = read_proteomics_csv("data/celiac_proteomics.csv")
        >>> assert "eid" in df.columns
        >>> assert "CeD_comparison" in df.columns
    """
    from ced_ml.utils.paths import get_project_root

    filepath = Path(filepath)

    # Validate path is within expected boundaries
    try:
        project_root = get_project_root()
        filepath_resolved = filepath.resolve()
        try:
            filepath_resolved.relative_to(project_root)
        except ValueError:
            logger.warning(
                f"Input file path outside project root: {filepath}\n"
                f"Resolved to: {filepath_resolved}\n"
                f"Project root: {project_root}"
            )
    except (ValueError, RuntimeError, OSError):
        # Cannot determine project root, skip validation
        pass

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if usecols is None:
        usecols = usecols_for_proteomics()

    logger.info(f"Reading CSV: {filepath}")
    df = pd.read_csv(filepath, usecols=usecols, low_memory=low_memory)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")

    if validate:
        validate_required_columns(df)

    return df


def read_proteomics_file(
    filepath: str,
    *,
    usecols: Callable[[str], bool] | None = None,
    low_memory: bool = False,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Read proteomics data file (CSV or Parquet) with schema-aware column filtering.

    Automatically detects file format based on extension and calls the appropriate reader.
    If a CSV file is provided but a corresponding Parquet file exists, the Parquet file
    will be used automatically for better performance.

    Args:
        filepath: Path to CSV or Parquet file
        usecols: Optional column filter function (default: usecols_for_proteomics())
        low_memory: Whether to use low_memory mode for pd.read_csv (default: False; ignored for Parquet)
        validate: Whether to validate required columns after loading (default: True)

    Returns:
        DataFrame with selected columns

    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If validate=True and required columns are missing, or if file format is unsupported

    Example:
        >>> df = read_proteomics_file("data/celiac_proteomics.parquet")
        >>> assert "eid" in df.columns
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    # Auto-prefer Parquet if CSV is specified but Parquet exists
    if suffix == ".csv":
        parquet_path = filepath.with_suffix(".parquet")
        if parquet_path.exists():
            logger.info(f"Found Parquet version: {parquet_path}")
            logger.info("Using Parquet for faster loading (53%+ smaller, 10x+ faster)")
            filepath = parquet_path
            suffix = ".parquet"
        else:
            # Warn user about performance
            logger.warning(f"Reading CSV: {filepath}")
            logger.warning(
                "Performance tip: Convert to Parquet for 53%+ size reduction and 10x+ faster loading:"
            )
            logger.warning(f"  ced convert-to-parquet {filepath}")

    if suffix == ".csv":
        return read_proteomics_csv(
            str(filepath),
            usecols=usecols,
            low_memory=low_memory,
            validate=validate,
        )
    elif suffix == ".parquet":
        logger.info(f"Reading Parquet: {filepath}")
        # Read Parquet file
        df = pd.read_parquet(filepath, engine="pyarrow")
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")

        # Apply column filter if provided
        if usecols is not None:
            selected_cols = [col for col in df.columns if usecols(col)]
            df = df[selected_cols]
            logger.info(f"Filtered to {len(selected_cols):,} columns")

        # Validate required columns
        if validate:
            validate_required_columns(df)

        return df
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Expected .csv or .parquet. "
            f"File: {filepath}"
        )


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that required columns are present in DataFrame.

    Required columns:
    - ID_COL (eid)
    - TARGET_COL (CeD_comparison)

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing, or if schema validation fails
    """
    from ced_ml.data.schema import CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL

    required = [ID_COL, TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. " f"Available columns: {list(df.columns)}"
        )
    logger.debug(f"Validated required columns: {required}")

    # Validate target column labels
    valid_labels = {CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL}
    observed_labels = set(df[TARGET_COL].dropna().unique())
    invalid_labels = observed_labels - valid_labels
    if invalid_labels:
        raise ValueError(
            f"Target column '{TARGET_COL}' contains invalid labels: {invalid_labels}. "
            f"Valid labels: {valid_labels}"
        )

    # Check for at least one case and one control
    label_counts = df[TARGET_COL].value_counts()
    if CONTROL_LABEL not in label_counts or label_counts[CONTROL_LABEL] < 1:
        raise ValueError(
            f"Dataset must contain at least 1 control sample ('{CONTROL_LABEL}'). "
            f"Found labels: {label_counts.to_dict()}"
        )
    case_labels = {INCIDENT_LABEL, PREVALENT_LABEL}
    case_count = sum(label_counts.get(label, 0) for label in case_labels)
    if case_count < 1:
        raise ValueError(
            f"Dataset must contain at least 1 case sample ('{INCIDENT_LABEL}' or '{PREVALENT_LABEL}'). "
            f"Found labels: {label_counts.to_dict()}"
        )

    logger.debug(f"Validated target labels: {observed_labels}")

    # Validate metadata column dtypes (common numeric metadata)
    non_numeric_metadata = []
    for col in META_NUM_COLS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_metadata.append(f"{col} ({df[col].dtype})")

    if non_numeric_metadata:
        raise ValueError(
            f"Metadata columns must be numeric but found non-numeric dtypes: {non_numeric_metadata}"
        )

    # Validate protein columns are numeric (sample check on first 10 proteins)
    protein_cols = [col for col in df.columns if col.endswith(PROTEIN_SUFFIX)]
    if protein_cols:
        sample_proteins = protein_cols[:10]
        non_numeric_proteins = []
        for col in sample_proteins:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_proteins.append(f"{col} ({df[col].dtype})")

        if non_numeric_proteins:
            raise ValueError(
                f"Protein columns must be numeric but found non-numeric dtypes in sample: {non_numeric_proteins}"
            )


def validate_binary_outcome(y) -> None:
    """
    Validate that outcome labels are binary (0/1 only).

    Args:
        y: Outcome array (np.ndarray) or pd.Series to validate

    Raises:
        ValueError: If outcome contains values other than 0 or 1
    """
    import numpy as np

    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.floating):
        mask = ~np.isnan(y_arr)
        unique_vals = np.unique(y_arr[mask])
    else:
        unique_vals = np.unique(y_arr)
    valid_vals = {0, 1}
    invalid = [v for v in unique_vals.tolist() if v not in valid_vals]
    if invalid:
        raise ValueError(
            f"Outcome labels must be binary (0/1). Found invalid values: {sorted(invalid)}. "
            f"Unique values in outcome: {sorted(unique_vals.tolist())}"
        )


def coerce_numeric_columns(
    df: pd.DataFrame, columns: list[str], inplace: bool = False
) -> pd.DataFrame:
    """
    Coerce columns to numeric dtype, converting errors to NaN.

    Note: pd.to_numeric() will return int64 if all values are integers,
    and float64 if there are floats or NaN values. This matches pandas behavior.

    Args:
        df: DataFrame to process
        columns: List of column names to coerce
        inplace: Whether to modify df in place (default: False)

    Returns:
        DataFrame with coerced columns (int64 or float64 depending on values)

    Example:
        >>> df = pd.DataFrame({"age": ["25", "30", "invalid"]})
        >>> result = coerce_numeric_columns(df, ["age"])
        >>> assert pd.api.types.is_numeric_dtype(result["age"])
        >>> assert pd.isna(result.loc[2, "age"])
    """
    if not inplace:
        df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping coercion")
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.debug(f"Coerced column '{col}' to numeric")

    return df


def fill_missing_categorical(
    df: pd.DataFrame,
    columns: list[str],
    fill_value: str = "Missing",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Fill missing values in categorical columns with explicit category.

    Strategy: Use fillna() to handle missing values, then convert to object dtype.
    This works consistently across pandas versions.

    Args:
        df: DataFrame to process
        columns: List of categorical column names
        fill_value: Value to use for missing data (default: "Missing")
        inplace: Whether to modify df in place (default: False)

    Returns:
        DataFrame with missing values filled

    Example:
        >>> df = pd.DataFrame({"sex": ["Male", None, "Female"]})
        >>> result = fill_missing_categorical(df, ["sex"])
        >>> assert result.loc[1, "sex"] == "Missing"
    """
    if not inplace:
        df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping missing fill")
            continue
        # Fill missing values directly with fillna, then convert to object dtype
        df[col] = df[col].fillna(fill_value).astype(object)
        logger.debug(f"Filled missing values in '{col}' with '{fill_value}'")

    return df


def identify_protein_columns(df: pd.DataFrame) -> list[str]:
    """Identify protein feature columns (*_resid suffix).

    Args:
        df: DataFrame to scan

    Returns:
        List of protein column names (sorted)

    Raises:
        ValueError: If no protein columns found

    Example:
        >>> df = pd.DataFrame({"age": [25], "APOE_resid": [0.5], "IL6_resid": [1.2]})
        >>> proteins = identify_protein_columns(df)
        >>> assert proteins == ["APOE_resid", "IL6_resid"]
    """
    protein_cols = sorted(
        [c for c in df.columns if isinstance(c, str) and c.endswith(PROTEIN_SUFFIX)]
    )
    if not protein_cols:
        raise ValueError(
            "No protein columns (*_resid) found. Check column naming or usecols filter."
        )
    if len(protein_cols) < 10:
        logger.warning(
            f"Only {len(protein_cols)} protein columns found (expected ~{EXPECTED_N_PROTEINS})"
        )
    logger.info(f"Identified {len(protein_cols):,} protein columns")
    return protein_cols


def get_data_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics for loaded data.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with:
        - n_rows: Number of rows
        - n_cols: Number of columns
        - n_proteins: Number of protein columns
        - outcome_counts: Counts by TARGET_COL
        - missing_metadata: Missing value counts for metadata columns

    Example:
        >>> df = pd.DataFrame({"eid": [1, 2], "CeD_comparison": ["Controls", "Incident"]})
        >>> stats = get_data_stats(df)
        >>> assert stats["n_rows"] == 2
    """
    stats = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }

    # Protein count
    try:
        protein_cols = identify_protein_columns(df)
        stats["n_proteins"] = len(protein_cols)
    except ValueError:
        stats["n_proteins"] = 0

    # Outcome distribution
    if TARGET_COL in df.columns:
        stats["outcome_counts"] = df[TARGET_COL].value_counts().to_dict()

    # Missing metadata
    metadata_cols = META_NUM_COLS + CAT_COLS
    missing = {}
    for col in metadata_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                missing[col] = int(n_missing)
    if missing:
        stats["missing_metadata"] = missing

    return stats


def convert_csv_to_parquet(
    csv_path: str,
    parquet_path: str | None = None,
    compression: str = "snappy",
    usecols: Callable[[str], bool] | None = None,
    validate: bool = True,
) -> Path:
    """
    Convert proteomics CSV file to Parquet format.

    This function reads a CSV file and converts it to Parquet format with
    compression. By default, only columns needed for modeling are included
    (same as read_proteomics_csv).

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file (default: same as csv_path with .parquet extension)
        compression: Compression algorithm (default: 'snappy'; options: 'snappy', 'gzip', 'brotli', 'zstd', 'none')
        usecols: Optional column filter function (default: usecols_for_proteomics())
        validate: Whether to validate required columns after loading (default: True)

    Returns:
        Path to the created Parquet file

    Raises:
        FileNotFoundError: If csv_path does not exist
        ValueError: If validate=True and required columns are missing

    Example:
        >>> parquet_file = convert_csv_to_parquet("data/celiac_proteomics.csv")
        >>> print(f"Converted to: {parquet_file}")
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Default output path: replace .csv with .parquet
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path)

    logger.info(f"Converting CSV to Parquet: {csv_path}")
    logger.info(f"Output: {parquet_path}")
    logger.info(f"Compression: {compression}")

    # Read CSV using existing function (applies column filtering and validation)
    df = read_proteomics_csv(
        str(csv_path),
        usecols=usecols,
        low_memory=False,
        validate=validate,
    )

    # Write to Parquet with compression
    logger.info(f"Writing Parquet file with {compression} compression...")
    df.to_parquet(
        parquet_path,
        compression=compression,
        index=False,
        engine="pyarrow",
    )

    # Report file sizes
    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    compression_ratio = 100 * (1 - parquet_size / csv_size)

    logger.info("Conversion complete:")
    logger.info(f"  CSV size:     {csv_size:,} bytes")
    logger.info(f"  Parquet size: {parquet_size:,} bytes")
    logger.info(f"  Compression:  {compression_ratio:.1f}% reduction")
    logger.info(f"  Saved to:     {parquet_path}")

    return parquet_path
