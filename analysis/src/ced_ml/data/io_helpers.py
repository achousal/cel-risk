"""CSV I/O helpers for centralized read/write patterns.

Provides consistent interfaces for loading and saving predictions, metrics,
and feature reports across the pipeline. This module consolidates repetitive
pd.read_csv/to_csv patterns found in train.py, aggregate_splits.py,
stacking.py, and evaluation/reports.py.

Functions:
    read_predictions: Load predictions with optional column validation
    save_predictions: Save predictions (test/val/train_oof) with consistent format
    read_feature_report: Load feature importance reports
    save_feature_report: Save feature importance reports
    read_metrics: Load metrics CSV with validation
    save_metrics: Save metrics with append mode support
    normalize_protein_names: Strip _resid suffix from protein names
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_protein_names(names: list[str]) -> list[str]:
    """Strip _resid suffix from protein names.

    Args:
        names: List of protein column names (possibly with _resid suffix)

    Returns:
        List of normalized names with _resid stripped
    """
    return [name.replace("_resid", "") if name.endswith("_resid") else name for name in names]


def read_predictions(
    path: str | Path,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load prediction CSV with optional column validation.

    Args:
        path: Path to predictions CSV
        required_cols: Optional list of required columns to validate

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If required columns are missing
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)

    if required_cols is not None:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Predictions CSV missing required columns: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )

    logger.debug(f"Loaded predictions: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def save_predictions(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    append: bool = False,
) -> str:
    """Save predictions CSV with consistent format.

    Args:
        df: Predictions DataFrame (should contain ID, y_true, predictions columns)
        path: Output path
        index: Whether to include index (default False)
        append: Whether to append to existing file (default False)

    Returns:
        Path to saved file (as string)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=index)
    else:
        df.to_csv(path, index=index)

    logger.debug(f"Saved predictions: {path} ({len(df)} rows)")
    return str(path)


def read_feature_report(
    path: str | Path,
) -> pd.DataFrame:
    """Load feature importance/stability report CSV.

    Args:
        path: Path to feature report CSV

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If path does not exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature report not found: {path}")

    df = pd.read_csv(path)
    logger.debug(f"Loaded feature report: {path} ({len(df)} rows)")
    return df


def save_feature_report(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> str:
    """Save feature importance/stability report CSV.

    Args:
        df: Feature report DataFrame (should contain protein, effect_size, p_value, etc.)
        path: Output path
        index: Whether to include index (default False)

    Returns:
        Path to saved file (as string)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=index)
    logger.debug(f"Saved feature report: {path} ({len(df)} rows)")
    return str(path)


def read_metrics(
    path: str | Path,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load metrics CSV with optional column validation.

    Args:
        path: Path to metrics CSV
        required_cols: Optional list of required columns to validate

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If required columns are missing
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    df = pd.read_csv(path)

    if required_cols is not None:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Metrics CSV missing required columns: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )

    logger.debug(f"Loaded metrics: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def save_metrics(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    append: bool = False,
) -> str:
    """Save metrics CSV with append mode support.

    Args:
        df: Metrics DataFrame
        path: Output path
        index: Whether to include index (default False)
        append: Whether to append to existing file (default False)

    Returns:
        Path to saved file (as string)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=index)
    else:
        df.to_csv(path, index=index)

    logger.debug(f"Saved metrics: {path} ({len(df)} rows)")
    return str(path)
