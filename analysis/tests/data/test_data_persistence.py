"""Tests for split validation and persistence utilities.

Test categories:
    1. Split validation (overlap, coverage, bounds)
    2. File existence checking
    3. Index CSV persistence
    4. Metadata JSON persistence
    5. Holdout persistence
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ced_ml.data.persistence import (
    check_split_files_exist,
    load_split_metadata,
    save_holdout_indices,
    save_holdout_metadata,
    save_split_indices,
    save_split_metadata,
    validate_split_indices,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_outdir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def valid_split():
    """Create valid train/val/test split."""
    rng = np.random.RandomState(42)
    n = 100
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    train_idx = all_idx[:50]
    val_idx = all_idx[50:75]
    test_idx = all_idx[75:]

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
        "total": n,
    }


@pytest.fixture
def valid_labels():
    """Create valid labels for split."""
    return {
        "y_train": np.array([0] * 40 + [1] * 10),
        "y_val": np.array([0] * 20 + [1] * 5),
        "y_test": np.array([0] * 20 + [1] * 5),
    }


# ============================================================================
# Split Validation Tests
# ============================================================================


def test_validate_split_indices_valid_threeway(valid_split):
    """Test validation passes for valid three-way split."""
    is_valid, msg = validate_split_indices(
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        val_idx=valid_split["val"],
        total_samples=valid_split["total"],
    )
    assert is_valid
    assert msg == ""


def test_validate_split_indices_valid_twoway(valid_split):
    """Test validation passes for valid two-way split."""
    is_valid, msg = validate_split_indices(
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        val_idx=None,
        total_samples=valid_split["total"],
    )
    assert is_valid
    assert msg == ""


def test_validate_split_indices_empty_train():
    """Test validation fails for empty TRAIN."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([], dtype=int),
        test_idx=np.array([0, 1, 2]),
    )
    assert not is_valid
    assert "TRAIN set is empty" in msg


def test_validate_split_indices_empty_test():
    """Test validation fails for empty TEST."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0, 1, 2]),
        test_idx=np.array([], dtype=int),
    )
    assert not is_valid
    assert "TEST set is empty" in msg


def test_validate_split_indices_train_test_overlap():
    """Test validation fails for TRAIN/TEST overlap."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0, 1, 2, 3]),
        test_idx=np.array([2, 3, 4, 5]),
    )
    assert not is_valid
    assert "TRAIN/TEST overlap" in msg


def test_validate_split_indices_train_val_overlap():
    """Test validation fails for TRAIN/VAL overlap."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([2, 3, 4]),
        test_idx=np.array([5, 6, 7]),
    )
    assert not is_valid
    assert "TRAIN/VAL overlap" in msg


def test_validate_split_indices_val_test_overlap():
    """Test validation fails for VAL/TEST overlap."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3, 4, 5]),
        test_idx=np.array([5, 6, 7]),
    )
    assert not is_valid
    assert "VAL/TEST overlap" in msg


def test_validate_split_indices_negative_indices():
    """Test validation fails for negative indices."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0, 1, -1]),
        test_idx=np.array([2, 3, 4]),
    )
    assert not is_valid
    assert "negative indices" in msg


def test_validate_split_indices_float_dtype():
    """Test validation fails for non-integer dtype."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0.0, 1.0, 2.0]),
        test_idx=np.array([3, 4, 5]),
    )
    assert not is_valid
    assert "must be integers" in msg


def test_validate_split_indices_out_of_bounds():
    """Test validation fails for indices >= total_samples."""
    is_valid, msg = validate_split_indices(
        train_idx=np.array([0, 1, 2]),
        test_idx=np.array([3, 4, 100]),  # 100 >= 10
        total_samples=10,
    )
    assert not is_valid
    assert "indices >= 10" in msg


# ============================================================================
# File Existence Checking Tests
# ============================================================================


def test_check_split_files_exist_none(temp_outdir):
    """Test returns False when no files exist."""
    exists, paths = check_split_files_exist(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        has_val=False,
    )
    assert not exists
    assert len(paths) == 0


def test_check_split_files_exist_train_only(temp_outdir):
    """Test detects existing train file."""
    train_path = os.path.join(temp_outdir, "train_idx_IncidentOnly_seed42.csv")
    Path(train_path).touch()

    exists, paths = check_split_files_exist(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        has_val=False,
    )
    assert exists
    assert train_path in paths


def test_check_split_files_exist_all_splits(temp_outdir):
    """Test detects all split files including metadata."""
    for fname in [
        "train_idx_IncidentOnly_seed42.csv",
        "test_idx_IncidentOnly_seed42.csv",
        "split_meta_IncidentOnly_seed42.json",
    ]:
        Path(os.path.join(temp_outdir, fname)).touch()

    exists, paths = check_split_files_exist(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        has_val=False,
    )
    assert exists
    assert len(paths) == 3


def test_check_split_files_exist_with_val(temp_outdir):
    """Test detects validation split file."""
    for fname in [
        "train_idx_IncidentOnly_seed42.csv",
        "test_idx_IncidentOnly_seed42.csv",
        "val_idx_IncidentOnly_seed42.csv",
        "split_meta_IncidentOnly_seed42.json",
    ]:
        Path(os.path.join(temp_outdir, fname)).touch()

    exists, paths = check_split_files_exist(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        has_val=True,
    )
    assert exists
    assert len(paths) == 4


# ============================================================================
# Index Persistence Tests
# ============================================================================


def test_save_split_indices_twoway(temp_outdir, valid_split):
    """Test saving two-way split (TRAIN/TEST)."""
    paths = save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        val_idx=None,
    )

    assert "train" in paths
    assert "test" in paths
    assert "val" not in paths

    # Check files exist
    assert os.path.exists(paths["train"])
    assert os.path.exists(paths["test"])

    # Check content
    train_df = np.loadtxt(paths["train"], delimiter=",", skiprows=1, dtype=int)
    test_df = np.loadtxt(paths["test"], delimiter=",", skiprows=1, dtype=int)

    np.testing.assert_array_equal(train_df, np.sort(valid_split["train"]))
    np.testing.assert_array_equal(test_df, np.sort(valid_split["test"]))


def test_save_split_indices_threeway(temp_outdir, valid_split):
    """Test saving three-way split (TRAIN/VAL/TEST)."""
    paths = save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentPlusPrevalent",
        seed=0,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        val_idx=valid_split["val"],
    )

    assert "train" in paths
    assert "val" in paths
    assert "test" in paths

    # Check files exist
    assert os.path.exists(paths["train"])
    assert os.path.exists(paths["val"])
    assert os.path.exists(paths["test"])

    # Check filenames have seed suffix
    assert "_seed0.csv" in paths["train"]
    assert "_seed0.csv" in paths["val"]
    assert "_seed0.csv" in paths["test"]

    # Check content
    val_df = np.loadtxt(paths["val"], delimiter=",", skiprows=1, dtype=int)
    np.testing.assert_array_equal(val_df, np.sort(valid_split["val"]))


def test_save_split_indices_creates_directory(valid_split):
    """Test creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "splits", "nested", "dir")

        paths = save_split_indices(
            outdir=outdir,
            scenario="IncidentOnly",
            seed=42,
            train_idx=valid_split["train"],
            test_idx=valid_split["test"],
        )

        assert os.path.exists(outdir)
        assert os.path.exists(paths["train"])


def test_save_split_indices_raises_on_existing(temp_outdir, valid_split):
    """Test behavior when files exist and overwrite=False.

    Behavior depends on whether indices match:
    - If indices match: skip with warning (no error)
    - If indices differ: raise FileExistsError
    """
    # Save once
    save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
    )

    # Try to save again with SAME indices (should skip with warning, no error)
    paths = save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        overwrite=False,
    )
    assert "train" in paths
    assert "test" in paths

    # Try to save again with DIFFERENT indices (should raise error)
    different_train = valid_split["train"] + 100  # Different indices
    with pytest.raises(FileExistsError, match="DO NOT match"):
        save_split_indices(
            outdir=temp_outdir,
            scenario="IncidentOnly",
            seed=42,
            train_idx=different_train,
            test_idx=valid_split["test"],
            overwrite=False,
        )


def test_save_split_indices_overwrites(temp_outdir, valid_split):
    """Test overwrites existing files when overwrite=True."""
    # Save once
    save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
    )

    # Save again with overwrite
    new_train = np.array([0, 1, 2, 3, 4])
    new_test = np.array([5, 6, 7, 8, 9])

    paths2 = save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=new_train,
        test_idx=new_test,
        overwrite=True,
    )

    # Check content was updated
    train_df = np.loadtxt(paths2["train"], delimiter=",", skiprows=1, dtype=int)
    np.testing.assert_array_equal(train_df, np.sort(new_train))


def test_save_split_indices_validates_indices(temp_outdir):
    """Test raises ValueError for invalid indices."""
    with pytest.raises(ValueError, match="Invalid split indices"):
        save_split_indices(
            outdir=temp_outdir,
            scenario="IncidentOnly",
            seed=42,
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([2, 3, 4]),  # Overlaps with train
        )


def test_save_split_indices_sorts_indices(temp_outdir):
    """Test indices are sorted before saving."""
    unsorted_train = np.array([5, 2, 8, 1, 9])
    unsorted_test = np.array([3, 7, 0, 4, 6])

    paths = save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=unsorted_train,
        test_idx=unsorted_test,
    )

    # Check saved indices are sorted
    train_df = np.loadtxt(paths["train"], delimiter=",", skiprows=1, dtype=int)
    test_df = np.loadtxt(paths["test"], delimiter=",", skiprows=1, dtype=int)

    np.testing.assert_array_equal(train_df, np.array([1, 2, 5, 8, 9]))
    np.testing.assert_array_equal(test_df, np.array([0, 3, 4, 6, 7]))


def test_validate_existing_splits_match(temp_outdir, valid_split):
    """Test validate_existing_splits returns True when splits match."""
    from ced_ml.data.persistence import validate_existing_splits

    # Save splits
    save_split_indices(
        outdir=temp_outdir,
        scenario="TestScenario",
        seed=0,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
    )

    # Validate with same indices
    is_match, msg = validate_existing_splits(
        outdir=temp_outdir,
        scenario="TestScenario",
        seed=0,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
    )

    assert is_match is True
    assert "match" in msg.lower()


def test_validate_existing_splits_differ(temp_outdir, valid_split):
    """Test validate_existing_splits returns False when splits differ."""
    from ced_ml.data.persistence import validate_existing_splits

    # Save splits
    save_split_indices(
        outdir=temp_outdir,
        scenario="TestScenario",
        seed=0,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
    )

    # Validate with different indices
    different_train = valid_split["train"] + 100
    is_match, msg = validate_existing_splits(
        outdir=temp_outdir,
        scenario="TestScenario",
        seed=0,
        train_idx=different_train,
        test_idx=valid_split["test"],
    )

    assert is_match is False
    assert "differ" in msg.lower()


def test_validate_existing_splits_missing(temp_outdir, valid_split):
    """Test validate_existing_splits returns False when files don't exist."""
    from ced_ml.data.persistence import validate_existing_splits

    # Don't save anything, just validate
    is_match, msg = validate_existing_splits(
        outdir=temp_outdir,
        scenario="NonExistent",
        seed=999,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
    )

    assert is_match is False
    assert "do not exist" in msg.lower()


# ============================================================================
# Metadata Persistence Tests
# ============================================================================


def test_save_split_metadata_minimal(temp_outdir, valid_split):
    """Test saves minimal metadata."""
    y_train = np.array([0, 0, 1, 1, 0])
    y_test = np.array([1, 0, 1])

    meta_path = save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=y_train,
        y_test=y_test,
    )

    assert meta_path.endswith("split_meta_IncidentOnly_seed42.json")
    assert os.path.exists(meta_path)

    # Load and check
    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["scenario"] == "IncidentOnly"
    assert meta["seed"] == 42
    assert meta["n_train"] == len(valid_split["train"])
    assert meta["n_test"] == len(valid_split["test"])
    assert meta["n_train_pos"] == 2
    assert meta["n_test_pos"] == 2


def test_save_split_metadata_with_val(temp_outdir, valid_split, valid_labels):
    """Test saving metadata with validation set."""
    meta_path = save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentPlusPrevalent",
        seed=0,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        val_idx=valid_split["val"],
        y_val=valid_labels["y_val"],
    )

    with open(meta_path) as f:
        meta = json.load(f)

    # Check validation fields present
    assert "n_val" in meta
    assert "n_val_pos" in meta
    assert "prevalence_val" in meta
    assert "split_id_val" in meta

    assert meta["n_val"] == 25
    assert meta["n_val_pos"] == 5
    assert abs(meta["prevalence_val"] - 0.2) < 1e-6


def test_save_split_metadata_with_stratification(temp_outdir, valid_split, valid_labels):
    """Test saving metadata with stratification scheme."""
    meta_path = save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        strat_scheme="outcome+sex+age3",
    )

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["stratification_scheme"] == "outcome+sex+age3"


def test_save_split_metadata_with_row_filters(temp_outdir, valid_split, valid_labels):
    """Test saving metadata with row filter stats."""
    row_filter_stats = {
        "n_removed_uncertain_controls": 10,
        "n_removed_dropna_meta_num": 5,
    }

    meta_path = save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        row_filter_stats=row_filter_stats,
    )

    with open(meta_path) as f:
        meta = json.load(f)

    assert "row_filters" in meta
    assert meta["row_filters"]["n_removed_uncertain_controls"] == 10
    assert meta["row_filters"]["n_removed_dropna_meta_num"] == 5


def test_save_split_metadata_temporal(temp_outdir, valid_split, valid_labels):
    """Test saving metadata with temporal split info."""
    meta_path = save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        temporal_split=True,
        temporal_col="CeD_date",
        temporal_train_end="2020-01-01",
        temporal_test_start="2020-01-02",
        temporal_test_end="2021-12-31",
    )

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["temporal_split"] is True
    assert meta["temporal_col"] == "CeD_date"
    assert meta["temporal_train_end_value"] == "2020-01-01"
    assert meta["temporal_test_start_value"] == "2020-01-02"
    assert meta["temporal_test_end_value"] == "2021-12-31"


# ============================================================================
# Holdout Persistence Tests
# ============================================================================


def test_save_holdout_indices(temp_outdir):
    """Test saving holdout indices with scenario-specific naming."""
    holdout_idx = np.array([95, 96, 97, 98, 99])

    holdout_path = save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
    )

    assert os.path.exists(holdout_path)
    # Now includes scenario in filename to prevent overwrites
    assert holdout_path.endswith("HOLDOUT_idx_IncidentOnly.csv")

    # Check content
    holdout_df = np.loadtxt(holdout_path, delimiter=",", skiprows=1, dtype=int)
    np.testing.assert_array_equal(holdout_df, np.sort(holdout_idx))


def test_save_holdout_indices_raises_on_existing(temp_outdir):
    """Test raises FileExistsError for existing holdout file."""
    holdout_idx = np.array([95, 96, 97, 98, 99])

    save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_holdout_indices(
            outdir=temp_outdir,
            scenario="IncidentOnly",
            holdout_idx=holdout_idx,
            overwrite=False,
        )


def test_save_holdout_metadata(temp_outdir):
    """Test saving holdout metadata with scenario-specific naming."""
    holdout_idx = np.array([95, 96, 97, 98, 99])
    y_holdout = np.array([0, 0, 1, 1, 0])

    meta_path = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
        strat_scheme="outcome+sex+age3",
        warning="Test warning message",
    )

    assert os.path.exists(meta_path)
    # Now includes scenario in filename to prevent overwrites
    assert meta_path.endswith("HOLDOUT_meta_IncidentOnly.json")

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["scenario"] == "IncidentOnly"
    assert meta["split_type"] == "holdout"
    assert meta["seed"] == 42
    assert meta["n_holdout"] == 5
    assert meta["n_holdout_pos"] == 2
    assert abs(meta["prevalence_holdout"] - 0.4) < 1e-6
    assert meta["stratification_scheme"] == "outcome+sex+age3"
    assert meta["reverse_causality_warning"] == "Test warning message"
    assert meta["index_space"] == "full"
    assert "NEVER use" in meta["note"]


def test_save_holdout_metadata_temporal(temp_outdir):
    """Test saving holdout metadata with temporal info."""
    holdout_idx = np.array([95, 96, 97, 98, 99])
    y_holdout = np.array([0, 0, 1, 1, 0])

    meta_path = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
        temporal_split=True,
        temporal_col="CeD_date",
        temporal_start="2021-01-01",
        temporal_end="2021-12-31",
    )

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["temporal_split"] is True
    assert meta["temporal_col"] == "CeD_date"
    assert meta["temporal_start_value"] == "2021-01-01"
    assert meta["temporal_end_value"] == "2021-12-31"


def test_save_holdout_indices_with_model_and_seed(temp_outdir):
    """Test holdout indices with model name and split seed in filename."""
    holdout_idx = np.array([95, 96, 97, 98, 99])

    holdout_path = save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        model_name="LR_EN",
        split_seed=42,
    )

    assert os.path.exists(holdout_path)
    # Filename includes scenario, model, and seed
    assert holdout_path.endswith("HOLDOUT_idx_IncidentOnly_LR_EN_seed42.csv")

    # Check content
    holdout_df = np.loadtxt(holdout_path, delimiter=",", skiprows=1, dtype=int)
    np.testing.assert_array_equal(holdout_df, np.sort(holdout_idx))


def test_save_holdout_indices_with_run_id(temp_outdir):
    """Test holdout indices with run_id for unique timestamped files."""
    holdout_idx = np.array([95, 96, 97, 98, 99])

    holdout_path = save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        run_id="20260122_143022",
    )

    assert os.path.exists(holdout_path)
    # Filename includes scenario and run_id
    assert holdout_path.endswith("HOLDOUT_idx_IncidentOnly_20260122_143022.csv")


def test_save_holdout_indices_full_naming(temp_outdir):
    """Test holdout indices with all optional naming parameters."""
    holdout_idx = np.array([95, 96, 97, 98, 99])

    holdout_path = save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentPlusPrevalent",
        holdout_idx=holdout_idx,
        model_name="XGBoost",
        split_seed=0,
        run_id="run001",
    )

    assert os.path.exists(holdout_path)
    # Filename includes all parts
    assert holdout_path.endswith("HOLDOUT_idx_IncidentPlusPrevalent_XGBoost_seed0_run001.csv")


def test_save_holdout_indices_no_overwrite_different_scenarios(temp_outdir):
    """Test that different scenarios do not overwrite each other."""
    holdout_idx = np.array([95, 96, 97, 98, 99])

    # Save for scenario 1
    path1 = save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
    )

    # Save for scenario 2 (should not overwrite)
    path2 = save_holdout_indices(
        outdir=temp_outdir,
        scenario="IncidentPlusPrevalent",
        holdout_idx=holdout_idx,
    )

    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)


def test_save_holdout_metadata_with_model_and_seed(temp_outdir):
    """Test holdout metadata with model name and split seed in filename."""
    holdout_idx = np.array([95, 96, 97, 98, 99])
    y_holdout = np.array([0, 0, 1, 1, 0])

    meta_path = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
        model_name="LR_EN",
        split_seed=42,
    )

    assert os.path.exists(meta_path)
    # Filename includes scenario, model, and seed
    assert meta_path.endswith("HOLDOUT_meta_IncidentOnly_LR_EN_seed42.json")

    with open(meta_path) as f:
        meta = json.load(f)

    # Check that identifiers are stored in metadata for traceability
    assert meta["model_name"] == "LR_EN"
    assert meta["split_seed"] == 42


def test_save_holdout_metadata_with_run_id(temp_outdir):
    """Test holdout metadata with run_id for unique timestamped files."""
    holdout_idx = np.array([95, 96, 97, 98, 99])
    y_holdout = np.array([0, 0, 1, 1, 0])

    meta_path = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
        run_id="20260122_143022",
    )

    assert os.path.exists(meta_path)
    # Filename includes scenario and run_id
    assert meta_path.endswith("HOLDOUT_meta_IncidentOnly_20260122_143022.json")

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["run_id"] == "20260122_143022"


def test_save_holdout_metadata_full_naming(temp_outdir):
    """Test holdout metadata with all optional naming parameters."""
    holdout_idx = np.array([95, 96, 97, 98, 99])
    y_holdout = np.array([0, 0, 1, 1, 0])

    meta_path = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentPlusPrevalent",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
        model_name="XGBoost",
        split_seed=0,
        run_id="run001",
    )

    assert os.path.exists(meta_path)
    # Filename includes all parts
    assert meta_path.endswith("HOLDOUT_meta_IncidentPlusPrevalent_XGBoost_seed0_run001.json")

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["model_name"] == "XGBoost"
    assert meta["split_seed"] == 0
    assert meta["run_id"] == "run001"


def test_save_holdout_metadata_no_overwrite_different_scenarios(temp_outdir):
    """Test that different scenarios do not overwrite each other."""
    holdout_idx = np.array([95, 96, 97, 98, 99])
    y_holdout = np.array([0, 0, 1, 1, 0])

    # Save for scenario 1
    path1 = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
    )

    # Save for scenario 2 (should not overwrite)
    path2 = save_holdout_metadata(
        outdir=temp_outdir,
        scenario="IncidentPlusPrevalent",
        holdout_idx=holdout_idx,
        y_holdout=y_holdout,
    )

    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_split_persistence_workflow(temp_outdir, valid_split, valid_labels):
    """Test complete workflow: save indices + metadata."""
    # Save indices
    idx_paths = save_split_indices(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        val_idx=valid_split["val"],
    )

    # Save metadata
    meta_path = save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        val_idx=valid_split["val"],
        y_val=valid_labels["y_val"],
        strat_scheme="outcome+sex+age3",
    )

    # Verify all files exist
    assert os.path.exists(idx_paths["train"])
    assert os.path.exists(idx_paths["val"])
    assert os.path.exists(idx_paths["test"])
    assert os.path.exists(meta_path)

    # Check file existence detection
    exists, existing = check_split_files_exist(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        has_val=True,
    )
    assert exists
    assert len(existing) == 4  # train, val, test, metadata


# ============================================================================
# Load Split Metadata Tests
# ============================================================================


def test_load_split_metadata_returns_dict(temp_outdir, valid_split, valid_labels):
    """Test loading split metadata returns correct dict."""
    # Save metadata first
    save_split_metadata(
        outdir=temp_outdir,
        scenario="IncidentOnly",
        seed=42,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        row_filter_stats={"meta_num_cols_used": ["age", "BMI"]},
    )

    # Load and verify
    meta = load_split_metadata(temp_outdir, "IncidentOnly", 42)
    assert meta is not None
    assert meta["scenario"] == "IncidentOnly"
    assert meta["seed"] == 42
    assert meta["row_filters"]["meta_num_cols_used"] == ["age", "BMI"]


def test_load_split_metadata_missing_file(temp_outdir):
    """Test loading missing metadata returns None."""
    meta = load_split_metadata(temp_outdir, "NonExistent", 99)
    assert meta is None


def test_load_split_metadata_preserves_row_filter_cols(temp_outdir, valid_split, valid_labels):
    """Test row filter columns are preserved in round-trip."""
    row_filters = {
        "meta_num_cols_used": ["age", "BMI", "custom_col"],
        "n_removed_uncertain_controls": 5,
        "n_removed_dropna_meta_num": 10,
    }

    save_split_metadata(
        outdir=temp_outdir,
        scenario="TestScenario",
        seed=0,
        train_idx=valid_split["train"],
        test_idx=valid_split["test"],
        y_train=valid_labels["y_train"],
        y_test=valid_labels["y_test"],
        row_filter_stats=row_filters,
    )

    meta = load_split_metadata(temp_outdir, "TestScenario", 0)
    assert meta["row_filters"]["meta_num_cols_used"] == ["age", "BMI", "custom_col"]
    assert meta["row_filters"]["n_removed_uncertain_controls"] == 5
    assert meta["row_filters"]["n_removed_dropna_meta_num"] == 10
