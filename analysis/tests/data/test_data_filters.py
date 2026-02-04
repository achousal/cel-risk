"""
Tests for data filtering module.
"""

import pandas as pd
import pytest
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.schema import (
    CED_DATE_COL,
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)


@pytest.fixture
def sample_data(sample_data_filters):
    """Alias for sample_data_filters from conftest for backward compatibility."""
    return sample_data_filters


class TestApplyRowFilters:
    """Test row filtering function."""

    def test_no_filtering(self, sample_data):
        """No filtering should return identical data with reset index."""
        df_out, stats = apply_row_filters(
            sample_data,
            drop_uncertain_controls=False,
            dropna_meta_num=False,
        )

        assert len(df_out) == len(sample_data)
        assert stats["n_in"] == 10
        assert stats["n_out"] == 10
        assert stats["n_removed_uncertain_controls"] == 0
        assert stats["n_removed_dropna_meta_num"] == 0
        assert stats["drop_uncertain_controls"] is False
        assert stats["dropna_meta_num"] is False

    def test_drop_uncertain_controls_only(self, sample_data):
        """Should remove Controls with CeD_date present."""
        df_out, stats = apply_row_filters(
            sample_data,
            drop_uncertain_controls=True,
            dropna_meta_num=False,
        )

        # Should remove rows 1 and 9 (Controls with CeD_date)
        assert len(df_out) == 8
        assert stats["n_in"] == 10
        assert stats["n_out"] == 8
        assert stats["n_removed_uncertain_controls"] == 2
        assert stats["n_removed_dropna_meta_num"] == 0

        # Verify correct rows removed
        remaining_ids = df_out[ID_COL].tolist()
        assert 1 not in remaining_ids  # Uncertain control removed
        assert 9 not in remaining_ids  # Uncertain control removed
        assert 0 in remaining_ids  # Normal control kept
        assert 8 in remaining_ids  # Normal control kept

        # Verify cases with dates are kept (not controls)
        assert 4 in remaining_ids  # Incident with date kept
        assert 6 in remaining_ids  # Prevalent with date kept
        assert 7 in remaining_ids  # Prevalent with date kept

    def test_dropna_meta_num_only(self, sample_data):
        """Should remove rows with missing age or BMI."""
        df_out, stats = apply_row_filters(
            sample_data,
            drop_uncertain_controls=False,
            dropna_meta_num=True,
        )

        # Should remove rows 2 (missing age), 3 (missing BMI), 5 (missing age), 9 (missing age)
        assert len(df_out) == 6
        assert stats["n_in"] == 10
        assert stats["n_out"] == 6
        assert stats["n_removed_uncertain_controls"] == 0
        assert stats["n_removed_dropna_meta_num"] == 4

        # Verify correct rows removed
        remaining_ids = df_out[ID_COL].tolist()
        assert 2 not in remaining_ids  # Missing age
        assert 3 not in remaining_ids  # Missing BMI
        assert 5 not in remaining_ids  # Missing age
        assert 9 not in remaining_ids  # Missing age

        # Verify rows with complete metadata kept
        assert 0 in remaining_ids
        assert 1 in remaining_ids
        assert 4 in remaining_ids
        assert 6 in remaining_ids

    def test_both_filters(self, sample_data):
        """Should apply both filters in sequence."""
        df_out, stats = apply_row_filters(
            sample_data,
            drop_uncertain_controls=True,
            dropna_meta_num=True,
        )

        # Should remove:
        # - Uncertain controls: 1, 9
        # - Then missing metadata: 2, 3, 5
        # Total removed: 5 unique rows (1, 2, 3, 5, 9)
        assert len(df_out) == 5
        assert stats["n_in"] == 10
        assert stats["n_out"] == 5
        assert stats["n_removed_uncertain_controls"] == 2
        assert stats["n_removed_dropna_meta_num"] == 3  # After uncertain controls removed

        # Verify correct rows kept
        remaining_ids = df_out[ID_COL].tolist()
        assert remaining_ids == [0, 4, 6, 7, 8]

    def test_index_reset(self, sample_data):
        """Output should have reset 0-based index."""
        df_out, _ = apply_row_filters(sample_data)

        assert df_out.index.tolist() == list(range(len(df_out)))

    def test_no_mutation_of_input(self, sample_data):
        """Original DataFrame should not be modified."""
        original_len = len(sample_data)
        original_ids = sample_data[ID_COL].tolist()

        apply_row_filters(sample_data)

        assert len(sample_data) == original_len
        assert sample_data[ID_COL].tolist() == original_ids

    def test_missing_ced_date_column(self):
        """Should handle data without CeD_date column."""
        df = pd.DataFrame(
            {
                ID_COL: [0, 1, 2],
                TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL],
                "age": [45, 50, 55],
                "BMI": [22.5, 25.0, 27.5],
            }
        )

        df_out, stats = apply_row_filters(
            df,
            drop_uncertain_controls=True,
            dropna_meta_num=False,
        )

        # No uncertain controls removed (no date column to check)
        assert len(df_out) == 3
        assert stats["n_removed_uncertain_controls"] == 0

    def test_missing_metadata_columns(self):
        """Should handle data without age/BMI columns."""
        df = pd.DataFrame(
            {
                ID_COL: [0, 1, 2],
                TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL],
                CED_DATE_COL: [None, "2020-01-01", "2019-01-01"],
            }
        )

        df_out, stats = apply_row_filters(
            df,
            drop_uncertain_controls=False,
            dropna_meta_num=True,
        )

        # No metadata to drop
        assert len(df_out) == 3
        assert stats["n_removed_dropna_meta_num"] == 0

    def test_partial_metadata_columns(self):
        """Should handle data with only some metadata columns."""
        df = pd.DataFrame(
            {
                ID_COL: [0, 1, 2],
                TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL],
                "age": [45, None, 55],
                # No BMI column
            }
        )

        df_out, stats = apply_row_filters(
            df,
            drop_uncertain_controls=False,
            dropna_meta_num=True,
        )

        # Should drop row 1 (missing age)
        assert len(df_out) == 2
        assert stats["n_removed_dropna_meta_num"] == 1
        assert df_out[ID_COL].tolist() == [0, 2]

    def test_all_rows_filtered(self):
        """Should handle case where all rows are filtered."""
        df = pd.DataFrame(
            {
                ID_COL: [0, 1],
                TARGET_COL: [CONTROL_LABEL, CONTROL_LABEL],
                CED_DATE_COL: ["2020-01-01", "2021-01-01"],
                "age": [None, None],
                "BMI": [None, None],
            }
        )

        df_out, stats = apply_row_filters(df)

        assert len(df_out) == 0
        assert stats["n_in"] == 2
        assert stats["n_out"] == 0
        assert stats["n_removed_uncertain_controls"] == 2
        assert stats["n_removed_dropna_meta_num"] == 0  # Already removed

    def test_no_uncertain_controls(self):
        """Should handle data with no uncertain controls."""
        df = pd.DataFrame(
            {
                ID_COL: [0, 1, 2],
                TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL],
                CED_DATE_COL: [None, "2020-01-01", "2019-01-01"],
                "age": [45, 50, 55],
                "BMI": [22.5, 25.0, 27.5],
            }
        )

        df_out, stats = apply_row_filters(df)

        assert len(df_out) == 3
        assert stats["n_removed_uncertain_controls"] == 0

    def test_stats_keys(self, sample_data):
        """Stats dict should contain all expected keys."""
        _, stats = apply_row_filters(sample_data)

        expected_keys = {
            "n_in",
            "drop_uncertain_controls",
            "dropna_meta_num",
            "meta_num_cols_used",  # Added to support column-resolution metadata persistence
            "n_removed_uncertain_controls",
            "n_removed_dropna_meta_num",
            "n_out",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_consistency(self, sample_data):
        """Stats should be internally consistent."""
        _, stats = apply_row_filters(sample_data)

        # n_out should equal n_in minus removals
        expected_out = (
            stats["n_in"]
            - stats["n_removed_uncertain_controls"]
            - stats["n_removed_dropna_meta_num"]
        )
        assert stats["n_out"] == expected_out

    def test_filter_flags_recorded(self, sample_data):
        """Stats should record which filters were applied."""
        _, stats1 = apply_row_filters(
            sample_data,
            drop_uncertain_controls=True,
            dropna_meta_num=False,
        )
        assert stats1["drop_uncertain_controls"] is True
        assert stats1["dropna_meta_num"] is False

        _, stats2 = apply_row_filters(
            sample_data,
            drop_uncertain_controls=False,
            dropna_meta_num=True,
        )
        assert stats2["drop_uncertain_controls"] is False
        assert stats2["dropna_meta_num"] is True
