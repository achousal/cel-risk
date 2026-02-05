"""
Tests for data.columns module (column resolution logic).
"""

import pandas as pd
import pytest

from ced_ml.config.schema import ColumnsConfig
from ced_ml.data.columns import resolve_columns
from ced_ml.data.schema import CAT_COLS, META_NUM_COLS


@pytest.fixture
def sample_columns_full():
    """Sample column list with all default metadata."""
    return (
        ["eid", "CeD_comparison", "CeD_date"]
        + META_NUM_COLS
        + CAT_COLS
        + [f"protein_{i}_resid" for i in range(10)]
    )


@pytest.fixture
def sample_columns_protein_only():
    """Sample column list with proteins only (no metadata)."""
    return ["eid", "CeD_comparison"] + [f"protein_{i}_resid" for i in range(10)]


@pytest.fixture
def sample_columns_partial_metadata():
    """Sample column list with some metadata missing."""
    return (
        ["eid", "CeD_comparison"]
        + ["age"]  # only one numeric metadata
        + ["sex"]  # only one categorical metadata
        + [f"protein_{i}_resid" for i in range(10)]
    )


def test_resolve_columns_auto_mode_full_metadata(sample_columns_full):
    """Test auto mode with all default metadata present."""
    config = ColumnsConfig(mode="auto")
    resolved = resolve_columns(sample_columns_full, config)

    assert len(resolved.protein_cols) == 10
    assert resolved.numeric_metadata == sorted(META_NUM_COLS)
    assert resolved.categorical_metadata == sorted(CAT_COLS)
    assert len(resolved.all_feature_cols) == 10 + len(META_NUM_COLS) + len(CAT_COLS)


def test_resolve_columns_auto_mode_protein_only(sample_columns_protein_only):
    """Test auto mode with no metadata (proteins only)."""
    config = ColumnsConfig(mode="auto", warn_missing_defaults=False)
    resolved = resolve_columns(sample_columns_protein_only, config)

    assert len(resolved.protein_cols) == 10
    assert resolved.numeric_metadata == []
    assert resolved.categorical_metadata == []
    assert len(resolved.all_feature_cols) == 10


def test_resolve_columns_auto_mode_partial_metadata(sample_columns_partial_metadata):
    """Test auto mode with partial metadata."""
    config = ColumnsConfig(mode="auto", warn_missing_defaults=False)
    resolved = resolve_columns(sample_columns_partial_metadata, config)

    assert len(resolved.protein_cols) == 10
    assert resolved.numeric_metadata == ["age"]
    assert resolved.categorical_metadata == ["sex"]
    assert len(resolved.all_feature_cols) == 12


def test_resolve_columns_explicit_mode_empty_metadata(sample_columns_protein_only):
    """Test explicit mode with empty metadata lists (protein-only)."""
    config = ColumnsConfig(mode="explicit", numeric_metadata=[], categorical_metadata=[])
    resolved = resolve_columns(sample_columns_protein_only, config)

    assert len(resolved.protein_cols) == 10
    assert resolved.numeric_metadata == []
    assert resolved.categorical_metadata == []


def test_resolve_columns_explicit_mode_custom_metadata(sample_columns_full):
    """Test explicit mode with custom metadata selection."""
    config = ColumnsConfig(mode="explicit", numeric_metadata=["age"], categorical_metadata=["sex"])
    resolved = resolve_columns(sample_columns_full, config)

    assert len(resolved.protein_cols) == 10
    assert resolved.numeric_metadata == ["age"]
    assert resolved.categorical_metadata == ["sex"]
    assert len(resolved.all_feature_cols) == 12


def test_resolve_columns_explicit_mode_requires_specification():
    """Test that explicit mode requires at least one metadata list."""
    config = ColumnsConfig(mode="explicit")
    columns = ["eid", "CeD_comparison", "protein_1_resid"]

    with pytest.raises(ValueError, match="mode='explicit' requires at least one"):
        resolve_columns(columns, config)


def test_resolve_columns_explicit_mode_validates_existence(sample_columns_full):
    """Test that explicit mode validates specified columns exist."""
    config = ColumnsConfig(mode="explicit", numeric_metadata=["age", "nonexistent_column"])

    with pytest.raises(ValueError, match="not found in data"):
        resolve_columns(sample_columns_full, config)


def test_resolve_columns_no_proteins_raises_error():
    """Test that missing protein columns raises an error."""
    config = ColumnsConfig(mode="auto")
    columns = ["eid", "CeD_comparison", "age", "BMI"]

    with pytest.raises(ValueError, match="No protein columns found"):
        resolve_columns(columns, config)


def test_resolve_columns_accepts_dataframe(sample_columns_full):
    """Test that resolve_columns accepts a DataFrame."""
    df = pd.DataFrame(columns=sample_columns_full)
    config = ColumnsConfig(mode="auto")
    resolved = resolve_columns(df, config)

    assert len(resolved.protein_cols) == 10
    assert len(resolved.all_metadata) == len(META_NUM_COLS) + len(CAT_COLS)


def test_resolved_columns_all_metadata_property():
    """Test the all_metadata property."""
    config = ColumnsConfig(mode="explicit", numeric_metadata=["age"], categorical_metadata=["sex"])
    columns = ["eid", "CeD_comparison", "age", "sex", "protein_1_resid"]
    resolved = resolve_columns(columns, config)

    assert resolved.all_metadata == ["age", "sex"]


def test_resolved_columns_all_feature_cols_property():
    """Test the all_feature_cols property."""
    config = ColumnsConfig(mode="explicit", numeric_metadata=["age"], categorical_metadata=["sex"])
    columns = ["eid", "CeD_comparison", "age", "sex", "protein_1_resid"]
    resolved = resolve_columns(columns, config)

    assert set(resolved.all_feature_cols) == {"protein_1_resid", "age", "sex"}
    assert len(resolved.all_feature_cols) == 3
