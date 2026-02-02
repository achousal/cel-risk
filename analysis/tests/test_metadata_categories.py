"""Tests for enhanced category breakdown in metadata annotations."""

import pandas as pd
from ced_ml.utils.metadata import (
    build_aggregated_metadata,
    build_plot_metadata,
    count_category_breakdown,
)


class TestCountCategoryBreakdown:
    """Test category counting helper function."""

    def test_count_categories(self):
        """Test that category counts are correct."""
        df = pd.DataFrame(
            {"category": ["Controls", "Controls", "Incident", "Prevalent", "Prevalent"]}
        )
        result = count_category_breakdown(df)

        assert result["controls"] == 2
        assert result["incident"] == 1
        assert result["prevalent"] == 2
        assert result["total"] == 5

    def test_missing_categories(self):
        """Test when some categories are absent."""
        df = pd.DataFrame({"category": ["Controls", "Controls", "Controls"]})
        result = count_category_breakdown(df)

        assert result["controls"] == 3
        assert result["incident"] == 0
        assert result["prevalent"] == 0
        assert result["total"] == 3

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"category": []})
        result = count_category_breakdown(df)

        assert result["controls"] == 0
        assert result["incident"] == 0
        assert result["prevalent"] == 0
        assert result["total"] == 0

    def test_missing_category_column(self):
        """Test when category column is missing."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = count_category_breakdown(df)

        assert result == {}

    def test_none_dataframe(self):
        """Test with None DataFrame."""
        result = count_category_breakdown(None)

        assert result == {}


class TestMetadataWithCategories:
    """Test metadata builder with category breakdown."""

    def test_metadata_with_all_categories(self):
        """Test metadata includes category breakdown when provided."""
        meta = build_plot_metadata(
            model="LR_EN",
            scenario="test",
            seed=0,
            train_prev=0.167,
            n_train=894,
            n_train_controls=745,
            n_train_incident=74,
            n_train_prevalent=75,
            n_train_pos=149,
            timestamp=False,
        )

        # Find the train line
        train_line = next((line for line in meta if line.startswith("Train:")), None)
        assert train_line is not None

        # Check that it contains category breakdown
        assert "controls=745" in train_line
        assert "incident=74" in train_line
        assert "prevalent=75" in train_line
        assert "prev=0.167" in train_line

    def test_metadata_without_categories_fallback(self):
        """Test metadata falls back to (pos=X) when categories not provided."""
        meta = build_plot_metadata(
            model="LR_EN",
            scenario="test",
            seed=0,
            train_prev=0.167,
            n_train=894,
            n_train_pos=149,
            timestamp=False,
        )

        # Find the train line
        train_line = next((line for line in meta if line.startswith("Train:")), None)
        assert train_line is not None

        # Should use old format
        assert "pos=149" in train_line
        assert "controls=" not in train_line

    def test_metadata_partial_categories(self):
        """Test metadata with only some categories provided."""
        meta = build_plot_metadata(
            model="LR_EN",
            scenario="test",
            seed=0,
            train_prev=0.167,
            n_train=894,
            n_train_controls=745,
            n_train_incident=149,
            timestamp=False,
        )

        # Find the train line
        train_line = next((line for line in meta if line.startswith("Train:")), None)
        assert train_line is not None

        # Should have controls and incident
        assert "controls=745" in train_line
        assert "incident=149" in train_line

    def test_metadata_val_test_breakdown(self):
        """Test that val/test sets also get category breakdown."""
        meta = build_plot_metadata(
            model="LR_EN",
            scenario="test",
            seed=0,
            train_prev=0.167,
            n_val=222,
            n_val_controls=185,
            n_val_incident=37,
            n_val_pos=37,
            n_test=216,
            n_test_controls=180,
            n_test_incident=36,
            n_test_pos=36,
            timestamp=False,
        )

        # Find the line with Val and Test (they're on same line)
        sample_line = next((line for line in meta if "Val:" in line and "Test:" in line), None)
        assert sample_line is not None

        # Check val breakdown
        assert "Val: n=222 (controls=185, incident=37, prev=0.167)" in sample_line

        # Check test breakdown
        assert "Test: n=216 (controls=180, incident=36, prev=0.167)" in sample_line

    def test_prevalence_calculation(self):
        """Test that prevalence is calculated correctly."""
        # 50 incident + 50 prevalent = 100 positive out of 1000 = 0.100
        meta = build_plot_metadata(
            model="LR_EN",
            scenario="test",
            seed=0,
            train_prev=0.100,
            n_train=1000,
            n_train_controls=900,
            n_train_incident=50,
            n_train_prevalent=50,
            n_train_pos=100,
            timestamp=False,
        )

        train_line = next((line for line in meta if line.startswith("Train:")), None)
        assert train_line is not None
        assert "prev=0.100" in train_line


class TestAggregatedMetadata:
    """Test metadata builder for aggregated plots."""

    def test_aggregated_metadata_basic(self):
        """Test basic aggregated metadata without categories."""
        meta = build_aggregated_metadata(
            n_splits=10,
            split_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            timestamp=False,
        )

        assert len(meta) == 1
        assert "Pooled from 10 splits" in meta[0]
        assert "seeds:" in meta[0]

    def test_aggregated_metadata_with_categories(self):
        """Test aggregated metadata includes category breakdown."""
        sample_cats = {
            "test": {"controls": 1800, "incident": 36, "prevalent": 30, "total": 1866},
            "val": {"controls": 1900, "incident": 38, "prevalent": 32, "total": 1970},
            "train_oof": {
                "controls": 3750,
                "incident": 74,
                "prevalent": 76,
                "total": 3900,
            },
        }

        meta = build_aggregated_metadata(
            n_splits=10,
            split_seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            sample_categories=sample_cats,
            timestamp=False,
        )

        # Should have pooled line + 3 split lines
        assert len(meta) == 4

        # Check for expected lines
        text = "\n".join(meta)
        assert "Pooled from 10 splits" in text
        assert "Train Oof:" in text
        assert "Val:" in text
        assert "Test:" in text
        assert "controls=" in text
        assert "incident=" in text
        assert "prevalent=" in text

    def test_aggregated_metadata_partial_categories(self):
        """Test aggregated metadata with only test split."""
        sample_cats = {
            "test": {"controls": 1800, "incident": 36, "prevalent": 30, "total": 1866},
        }

        meta = build_aggregated_metadata(
            n_splits=5,
            split_seeds=[0, 1, 2, 3, 4],
            sample_categories=sample_cats,
            timestamp=False,
        )

        text = "\n".join(meta)
        assert "Test:" in text
        assert "controls=1800" in text
        assert "incident=36" in text

    def test_aggregated_metadata_with_timestamp(self):
        """Test that timestamp is included when requested."""
        meta = build_aggregated_metadata(
            n_splits=5,
            split_seeds=[0, 1, 2, 3, 4],
            timestamp=True,
        )

        text = "\n".join(meta)
        assert "Generated:" in text
