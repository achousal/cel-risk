"""
Tests for data I/O module.
"""

import numpy as np
import pandas as pd
import pytest

from ced_ml.data.io import (
    coerce_numeric_columns,
    fill_missing_categorical,
    get_data_stats,
    identify_protein_columns,
    read_proteomics_csv,
    usecols_for_proteomics,
    validate_binary_outcome,
    validate_required_columns,
)
from ced_ml.data.schema import CED_DATE_COL, ID_COL, TARGET_COL


class TestUsecolsFilter:
    """Test column filtering function."""

    def test_filter_includes_required_columns(self):
        """Filter should include ID, target, and date columns."""
        filter_fn = usecols_for_proteomics()
        assert filter_fn(ID_COL)
        assert filter_fn(TARGET_COL)
        assert filter_fn(CED_DATE_COL)

    def test_filter_includes_metadata(self):
        """Filter should include numeric and categorical metadata."""
        filter_fn = usecols_for_proteomics()
        assert filter_fn("age")
        assert filter_fn("BMI")
        assert filter_fn("sex")
        assert filter_fn("Genetic ethnic grouping")

    def test_filter_includes_proteins(self):
        """Filter should include protein columns (*_resid)."""
        filter_fn = usecols_for_proteomics()
        assert filter_fn("APOE_resid")
        assert filter_fn("IL6_resid")
        assert filter_fn("TGM2_resid")

    def test_filter_excludes_other_columns(self):
        """Filter should exclude non-proteomics columns."""
        filter_fn = usecols_for_proteomics()
        assert not filter_fn("random_column")
        assert not filter_fn("APOE")  # No _resid suffix
        assert not filter_fn("123")


class TestReadProteomicsCSV:
    """Test CSV reading with proteomics schema."""

    def test_read_valid_csv(self, tmp_path):
        """Should read valid CSV successfully."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "eid": [1, 2, 3],
                "CeD_comparison": ["Controls", "Incident", "Controls"],
                "age": [25, 30, 35],
                "IL6_resid": [0.1, 0.2, 0.3],
                "APOE_resid": [0.4, 0.5, 0.6],
            }
        )
        df.to_csv(csv_path, index=False)

        result = read_proteomics_csv(str(csv_path))
        assert len(result) == 3
        assert "eid" in result.columns
        assert "CeD_comparison" in result.columns
        assert "IL6_resid" in result.columns

    def test_read_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_proteomics_csv("nonexistent.csv")

    def test_column_filtering(self, tmp_path):
        """Should filter columns based on proteomics schema."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "eid": [1, 2],
                "CeD_comparison": ["Controls", "Incident"],
                "age": [25, 30],
                "random_col": ["A", "B"],  # Should be filtered out
                "IL6_resid": [0.1, 0.2],
            }
        )
        df.to_csv(csv_path, index=False)

        result = read_proteomics_csv(str(csv_path))
        assert "eid" in result.columns
        assert "IL6_resid" in result.columns
        assert "random_col" not in result.columns

    def test_validation_enabled_by_default(self, tmp_path):
        """Should validate required columns by default."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"age": [25, 30]})  # Missing required columns
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Required columns missing"):
            read_proteomics_csv(str(csv_path))

    def test_validation_can_be_disabled(self, tmp_path):
        """Should allow disabling validation."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"age": [25, 30]})
        df.to_csv(csv_path, index=False)

        # Should not raise with validate=False
        result = read_proteomics_csv(str(csv_path), validate=False)
        assert "age" in result.columns


class TestValidateRequiredColumns:
    """Test required column validation."""

    def test_valid_dataframe(self):
        """Should pass for DataFrame with required columns."""
        df = pd.DataFrame(
            {
                "eid": [1, 2],
                "CeD_comparison": ["Controls", "Incident"],
            }
        )
        validate_required_columns(df)  # Should not raise

    def test_missing_id_column(self):
        """Should raise for missing ID column."""
        df = pd.DataFrame({"CeD_comparison": ["Controls"]})
        with pytest.raises(ValueError, match="Required columns missing.*eid"):
            validate_required_columns(df)

    def test_missing_target_column(self):
        """Should raise for missing target column."""
        df = pd.DataFrame({"eid": [1, 2]})
        with pytest.raises(ValueError, match="Required columns missing.*CeD_comparison"):
            validate_required_columns(df)

    def test_missing_both_columns(self):
        """Should raise for missing both required columns."""
        df = pd.DataFrame({"age": [25, 30]})
        with pytest.raises(ValueError, match="Required columns missing"):
            validate_required_columns(df)


class TestCoerceNumericColumns:
    """Test numeric dtype coercion."""

    def test_coerce_string_to_numeric(self):
        """Should convert string columns to numeric."""
        df = pd.DataFrame({"age": ["25", "30", "35"]})
        result = coerce_numeric_columns(df, ["age"])
        assert pd.api.types.is_numeric_dtype(result["age"])
        assert result["age"].tolist() == [25, 30, 35]

    def test_coerce_invalid_values_to_nan(self):
        """Should convert invalid values to NaN."""
        df = pd.DataFrame({"age": ["25", "invalid", "35"]})
        result = coerce_numeric_columns(df, ["age"])
        assert pd.isna(result.loc[1, "age"])
        assert result.loc[0, "age"] == 25
        # When NaN is present, dtype becomes float64
        assert result["age"].dtype == np.float64

    def test_inplace_modification(self):
        """Should modify DataFrame in place when inplace=True."""
        df = pd.DataFrame({"age": ["25", "30"]})
        df_id_before = id(df)
        result = coerce_numeric_columns(df, ["age"], inplace=True)
        assert id(result) == df_id_before
        assert pd.api.types.is_numeric_dtype(result["age"])

    def test_copy_by_default(self):
        """Should create copy by default (inplace=False)."""
        df = pd.DataFrame({"age": ["25", "30"]})
        result = coerce_numeric_columns(df, ["age"], inplace=False)
        assert id(result) != id(df)
        # Original unchanged (dtype may be object or StringDtype depending on pandas version)
        assert not pd.api.types.is_numeric_dtype(df["age"])
        assert pd.api.types.is_numeric_dtype(result["age"])

    def test_skip_missing_columns(self):
        """Should skip columns that don't exist."""
        df = pd.DataFrame({"age": ["25"]})
        result = coerce_numeric_columns(df, ["age", "nonexistent"])
        assert "age" in result.columns
        assert "nonexistent" not in result.columns

    def test_multiple_columns(self):
        """Should coerce multiple columns."""
        df = pd.DataFrame(
            {
                "age": ["25", "30"],
                "BMI": ["22.5", "25.0"],
            }
        )
        result = coerce_numeric_columns(df, ["age", "BMI"])
        assert pd.api.types.is_numeric_dtype(result["age"])
        assert pd.api.types.is_numeric_dtype(result["BMI"])


class TestFillMissingCategorical:
    """Test filling missing categorical values."""

    def test_fill_none_values(self):
        """Should fill None with 'Missing'."""
        df = pd.DataFrame({"sex": ["Male", None, "Female"]})
        result = fill_missing_categorical(df, ["sex"])
        assert result.loc[1, "sex"] == "Missing"

    def test_fill_nan_values(self):
        """Should fill NaN with 'Missing'."""
        df = pd.DataFrame({"sex": ["Male", np.nan, "Female"]})
        result = fill_missing_categorical(df, ["sex"])
        assert result.loc[1, "sex"] == "Missing"

    def test_custom_fill_value(self):
        """Should use custom fill value."""
        df = pd.DataFrame({"sex": ["Male", None, "Female"]})
        result = fill_missing_categorical(df, ["sex"], fill_value="Unknown")
        assert result.loc[1, "sex"] == "Unknown"

    def test_inplace_modification(self):
        """Should modify DataFrame in place when inplace=True."""
        df = pd.DataFrame({"sex": ["Male", None]})
        df_id = id(df)
        result = fill_missing_categorical(df, ["sex"], inplace=True)
        assert id(result) == df_id

    def test_multiple_columns(self):
        """Should fill multiple columns."""
        df = pd.DataFrame(
            {
                "sex": ["Male", None],
                "ethnicity": [None, "Asian"],
            }
        )
        result = fill_missing_categorical(df, ["sex", "ethnicity"])
        assert result.loc[1, "sex"] == "Missing"
        assert result.loc[0, "ethnicity"] == "Missing"


class TestIdentifyProteinColumns:
    """Test protein column identification."""

    def test_identify_protein_columns(self):
        """Should identify columns with _resid suffix."""
        df = pd.DataFrame(
            {
                "age": [25],
                "APOE_resid": [0.5],
                "IL6_resid": [1.2],
                "TGM2_resid": [0.8],
                "P1_resid": [0.1],
                "P2_resid": [0.2],
                "P3_resid": [0.3],
                "P4_resid": [0.4],
                "P5_resid": [0.5],
                "P6_resid": [0.6],
                "P7_resid": [0.7],
            }
        )
        proteins = identify_protein_columns(df)
        assert proteins == [
            "APOE_resid",
            "IL6_resid",
            "P1_resid",
            "P2_resid",
            "P3_resid",
            "P4_resid",
            "P5_resid",
            "P6_resid",
            "P7_resid",
            "TGM2_resid",
        ]

    def test_exclude_non_protein_columns(self):
        """Should exclude columns without _resid suffix."""
        df = pd.DataFrame(
            {
                "APOE": [0.5],  # No suffix
                "APOE_resid": [0.5],  # Has suffix
                "P1_resid": [0.1],
                "P2_resid": [0.2],
                "P3_resid": [0.3],
                "P4_resid": [0.4],
                "P5_resid": [0.5],
                "P6_resid": [0.6],
                "P7_resid": [0.7],
                "P8_resid": [0.8],
                "P9_resid": [0.9],
            }
        )
        proteins = identify_protein_columns(df)
        assert proteins == [
            "APOE_resid",
            "P1_resid",
            "P2_resid",
            "P3_resid",
            "P4_resid",
            "P5_resid",
            "P6_resid",
            "P7_resid",
            "P8_resid",
            "P9_resid",
        ]

    def test_no_proteins_raises_error(self):
        """Should raise ValueError if no protein columns found."""
        df = pd.DataFrame({"age": [25], "BMI": [22.5]})
        with pytest.raises(ValueError, match="No protein columns"):
            identify_protein_columns(df)

    def test_sorted_output(self):
        """Should return sorted list of protein names."""
        df = pd.DataFrame(
            {
                "ZZZ_resid": [1.0],
                "AAA_resid": [2.0],
                "MMM_resid": [3.0],
                "P1_resid": [0.1],
                "P2_resid": [0.2],
                "P3_resid": [0.3],
                "P4_resid": [0.4],
                "P5_resid": [0.5],
                "P6_resid": [0.6],
                "P7_resid": [0.7],
            }
        )
        proteins = identify_protein_columns(df)
        assert proteins == [
            "AAA_resid",
            "MMM_resid",
            "P1_resid",
            "P2_resid",
            "P3_resid",
            "P4_resid",
            "P5_resid",
            "P6_resid",
            "P7_resid",
            "ZZZ_resid",
        ]

    def test_warns_on_low_protein_count(self, caplog):
        """Should warn when protein count is below 10."""
        import logging

        # Re-enable propagation on ced_ml logger so caplog can capture
        ced_ml_logger = logging.getLogger("ced_ml")
        orig_propagate = ced_ml_logger.propagate
        ced_ml_logger.propagate = True
        try:
            df = pd.DataFrame({f"P{i}_resid": [0.1] for i in range(5)})
            with caplog.at_level(logging.WARNING, logger="ced_ml.data.io"):
                proteins = identify_protein_columns(df)
        finally:
            ced_ml_logger.propagate = orig_propagate
        assert len(proteins) == 5
        assert "only 5 protein" in caplog.text.lower()


class TestGetDataStats:
    """Test data summary statistics."""

    def test_basic_stats(self):
        """Should compute basic row/column counts."""
        df = pd.DataFrame(
            {
                "eid": [1, 2, 3],
                "CeD_comparison": ["Controls", "Incident", "Controls"],
            }
        )
        stats = get_data_stats(df)
        assert stats["n_rows"] == 3
        assert stats["n_cols"] == 2

    def test_protein_count(self):
        """Should count protein columns."""
        df = pd.DataFrame(
            {
                "eid": [1],
                "CeD_comparison": ["Controls"],
                "IL6_resid": [0.1],
                "APOE_resid": [0.2],
                "P1_resid": [0.1],
                "P2_resid": [0.2],
                "P3_resid": [0.3],
                "P4_resid": [0.4],
                "P5_resid": [0.5],
                "P6_resid": [0.6],
                "P7_resid": [0.7],
                "P8_resid": [0.8],
            }
        )
        stats = get_data_stats(df)
        assert stats["n_proteins"] == 10

    def test_outcome_distribution(self):
        """Should compute outcome counts."""
        df = pd.DataFrame(
            {
                "eid": [1, 2, 3],
                "CeD_comparison": ["Controls", "Incident", "Controls"],
            }
        )
        stats = get_data_stats(df)
        assert stats["outcome_counts"] == {"Controls": 2, "Incident": 1}

    def test_missing_metadata(self):
        """Should track missing metadata counts."""
        df = pd.DataFrame(
            {
                "eid": [1, 2, 3],
                "CeD_comparison": ["Controls", "Incident", "Controls"],
                "age": [25, None, 30],
                "BMI": [22.5, 23.0, None],
            }
        )
        stats = get_data_stats(df)
        assert "missing_metadata" in stats
        assert stats["missing_metadata"]["age"] == 1
        assert stats["missing_metadata"]["BMI"] == 1

    def test_no_missing_metadata_omitted(self):
        """Should omit missing_metadata key if no missing values."""
        df = pd.DataFrame(
            {
                "eid": [1, 2],
                "CeD_comparison": ["Controls", "Incident"],
                "age": [25, 30],
            }
        )
        stats = get_data_stats(df)
        assert "missing_metadata" not in stats


class TestValidateBinaryOutcome:
    """Test binary outcome validation."""

    def test_valid_binary_array(self):
        """Should pass for valid 0/1 array."""
        y = np.array([0, 1, 0, 1, 1, 0])
        validate_binary_outcome(y)  # Should not raise

    def test_valid_binary_series(self):
        """Should pass for valid 0/1 Series."""
        y = pd.Series([0, 1, 0, 1])
        validate_binary_outcome(y)  # Should not raise

    def test_invalid_values_raises_error(self):
        """Should raise for non-binary values."""
        y = np.array([0, 1, 2, 1, 0])
        with pytest.raises(ValueError, match="Outcome labels must be binary"):
            validate_binary_outcome(y)

    def test_negative_values_raises_error(self):
        """Should raise for negative values."""
        y = np.array([0, 1, -1, 1, 0])
        with pytest.raises(ValueError, match="Outcome labels must be binary"):
            validate_binary_outcome(y)

    def test_float_labels_raises_error(self):
        """Should raise for float labels like 0.5."""
        y = np.array([0.0, 1.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="Outcome labels must be binary"):
            validate_binary_outcome(y)

    def test_all_zeros_valid(self):
        """Should pass for all zeros."""
        y = np.array([0, 0, 0, 0])
        validate_binary_outcome(y)  # Should not raise

    def test_all_ones_valid(self):
        """Should pass for all ones."""
        y = np.array([1, 1, 1, 1])
        validate_binary_outcome(y)  # Should not raise

    def test_error_message_shows_invalid_values(self):
        """Error message should show invalid values."""
        y = np.array([0, 1, 3, 5])
        with pytest.raises(ValueError, match="Found invalid values: \\[3, 5\\]"):
            validate_binary_outcome(y)
