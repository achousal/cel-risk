"""
Tests for serialization utilities.

Tests cover:
- Model bundle saving/loading
- Version metadata storage
- Version mismatch warnings
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
from ced_ml.utils.serialization import load_joblib, load_json, save_joblib, save_json
from sklearn.linear_model import LogisticRegression


class TestJoblibSerialization:
    """Test joblib save/load functionality."""

    def test_save_load_basic_object(self, tmp_path):
        """Basic save and load works."""
        obj = {"key": "value", "number": 42}
        path = tmp_path / "test.joblib"

        save_joblib(obj, path)
        loaded = load_joblib(path, check_versions=False)

        assert loaded == obj

    def test_save_load_sklearn_model(self, tmp_path):
        """Save and load sklearn model."""
        model = LogisticRegression(random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)

        path = tmp_path / "model.joblib"
        save_joblib(model, path)
        loaded_model = load_joblib(path, check_versions=False)

        assert isinstance(loaded_model, LogisticRegression)
        np.testing.assert_array_equal(loaded_model.coef_, model.coef_)


class TestModelBundleVersioning:
    """Test version metadata in model bundles."""

    def test_save_load_bundle_with_versions(self, tmp_path):
        """Model bundle with version metadata loads correctly."""
        model = LogisticRegression()
        bundle = {
            "model": model,
            "model_name": "LR_EN",
            "versions": {
                "sklearn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }

        path = tmp_path / "bundle.joblib"
        save_joblib(bundle, path)
        loaded = load_joblib(path, check_versions=True)

        assert "versions" in loaded
        assert loaded["versions"]["sklearn"] == sklearn.__version__

    def test_load_bundle_version_mismatch_warns(self, tmp_path):
        """Loading bundle with version mismatch issues warning."""
        bundle = {
            "model": LogisticRegression(),
            "versions": {
                "sklearn": "0.0.1",  # Fake old version
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }

        path = tmp_path / "old_bundle.joblib"
        save_joblib(bundle, path)

        with pytest.warns(UserWarning, match="Model bundle version mismatch"):
            loaded = load_joblib(path, check_versions=True)

        assert "model" in loaded

    def test_load_bundle_no_version_check(self, tmp_path):
        """Loading with check_versions=False does not warn."""
        bundle = {
            "model": LogisticRegression(),
            "versions": {
                "sklearn": "0.0.1",  # Fake old version
            },
        }

        path = tmp_path / "old_bundle.joblib"
        save_joblib(bundle, path)

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            loaded = load_joblib(path, check_versions=False)

        assert "model" in loaded

    def test_load_non_bundle_no_check(self, tmp_path):
        """Loading non-bundle object with check_versions=True does not error."""
        obj = {"not": "a bundle"}
        path = tmp_path / "obj.joblib"
        save_joblib(obj, path)

        # Should not error or warn (no versions key)
        loaded = load_joblib(path, check_versions=True)
        assert loaded == obj


class TestJSONSerialization:
    """Test JSON save/load functionality."""

    def test_save_load_json(self, tmp_path):
        """Basic JSON save and load."""
        obj = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        path = tmp_path / "test.json"

        save_json(obj, path)
        loaded = load_json(path)

        assert loaded == obj

    def test_save_json_creates_parent_dirs(self, tmp_path):
        """save_json creates parent directories."""
        path = tmp_path / "subdir" / "test.json"
        save_json({"key": "value"}, path)

        assert path.exists()
        loaded = load_json(path)
        assert loaded["key"] == "value"
