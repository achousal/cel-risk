"""Tests for Recursive Feature Elimination module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.rfe import (
    RFEResult,
    aggregate_rfe_results,
    compute_eval_sizes,
    compute_feature_importance,
    detect_knee_point,
    find_recommended_panels,
    save_rfe_results,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestComputeEvalSizes:
    """Tests for compute_eval_sizes function."""

    def test_geometric_strategy(self):
        """Geometric strategy produces powers of 2."""
        sizes = compute_eval_sizes(100, 5, "geometric")
        # Should include 100, 50, 25, 12, 6, 5
        assert 100 in sizes
        assert 50 in sizes
        assert 25 in sizes
        assert 5 in sizes
        assert sizes == sorted(sizes, reverse=True)

    def test_linear_strategy(self):
        """Linear strategy produces all sizes."""
        sizes = compute_eval_sizes(10, 5, "linear")
        assert sizes == [10, 9, 8, 7, 6, 5]

    def test_geometric_default(self):
        """Geometric is the default strategy."""
        default = compute_eval_sizes(100, 5)
        explicit = compute_eval_sizes(100, 5, "geometric")
        assert default == explicit

    def test_min_size_included(self):
        """Min size is always included."""
        sizes = compute_eval_sizes(100, 7, "geometric")
        assert 7 in sizes

    def test_edge_case_max_equals_min(self):
        """Handles max == min gracefully."""
        sizes = compute_eval_sizes(5, 5, "geometric")
        assert sizes == [5]

    def test_small_range(self):
        """Handles small ranges."""
        sizes = compute_eval_sizes(8, 5, "geometric")
        assert 8 in sizes
        assert 5 in sizes

    def test_fine_strategy(self):
        """Fine strategy produces more granular steps."""
        sizes = compute_eval_sizes(100, 5, "fine")
        # Should have more points than geometric
        geometric_sizes = compute_eval_sizes(100, 5, "geometric")
        assert len(sizes) > len(geometric_sizes)
        # Should include 100, 75, 50, 37, 25, etc.
        assert 100 in sizes
        assert 75 in sizes
        assert 50 in sizes
        assert 5 in sizes
        assert sizes == sorted(sizes, reverse=True)

    def test_fine_strategy_intermediate_points(self):
        """Fine strategy includes quarter-step interpolation."""
        sizes = compute_eval_sizes(100, 5, "fine")
        # Should have intermediate points between powers of 2
        # Between 100 and 50, should have 75
        assert 75 in sizes
        # Between 50 and 25, should have 37
        assert 37 in sizes or 38 in sizes  # int(50 * 0.75)

    def test_fine_vs_geometric(self):
        """Fine strategy produces more evaluation points than geometric."""
        geometric = compute_eval_sizes(200, 10, "geometric")
        fine = compute_eval_sizes(200, 10, "fine")
        # Fine should have at least 1.5x as many points
        assert len(fine) >= len(geometric) * 1.5

    def test_invalid_strategy_raises(self):
        """Invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step_strategy"):
            compute_eval_sizes(100, 5, "adaptive")

        with pytest.raises(ValueError, match="Invalid step_strategy"):
            compute_eval_sizes(100, 5, "invalid")

        with pytest.raises(ValueError, match="Must be one of"):
            compute_eval_sizes(100, 5, "foobar")


class TestFindRecommendedPanels:
    """Tests for find_recommended_panels function."""

    def test_basic_recommendations(self):
        """Basic threshold recommendations work."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.88},
            {"size": 25, "auroc_val": 0.85},
            {"size": 10, "auroc_val": 0.75},
        ]
        rec = find_recommended_panels(curve, [0.95, 0.90])
        # 95% of 0.90 = 0.855, smallest meeting this is 25
        # 90% of 0.90 = 0.81, smallest meeting this is 10
        assert "min_size_95pct" in rec
        assert "min_size_90pct" in rec
        # Higher threshold (95%) requires larger panel to maintain AUROC
        assert rec["min_size_95pct"] >= rec["min_size_90pct"]

    def test_knee_point_included(self):
        """Knee point is always included."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.89},
            {"size": 25, "auroc_val": 0.85},
        ]
        rec = find_recommended_panels(curve)
        assert "knee_point" in rec

    def test_empty_curve(self):
        """Empty curve returns empty dict."""
        rec = find_recommended_panels([])
        assert rec == {}

    def test_single_point(self):
        """Single point curve handled."""
        curve = [{"size": 50, "auroc_val": 0.85}]
        rec = find_recommended_panels(curve, [0.95])
        assert "min_size_95pct" in rec
        assert rec["min_size_95pct"] == 50


class TestDetectKneePoint:
    """Tests for detect_knee_point function."""

    def test_clear_knee(self):
        """Detects clear knee point."""
        # Curve with obvious knee at size 25
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 75, "auroc_val": 0.895},
            {"size": 50, "auroc_val": 0.89},
            {"size": 25, "auroc_val": 0.88},  # Knee here
            {"size": 10, "auroc_val": 0.70},
            {"size": 5, "auroc_val": 0.60},
        ]
        knee = detect_knee_point(curve)
        # Knee should be around where the curve bends
        assert knee in [25, 50, 10]  # Reasonable range

    def test_monotonic_decline(self):
        """Handles monotonically declining curve."""
        curve = [
            {"size": 100, "auroc_val": 0.90},
            {"size": 50, "auroc_val": 0.80},
            {"size": 25, "auroc_val": 0.70},
        ]
        knee = detect_knee_point(curve)
        assert knee in [100, 50, 25]

    def test_few_points(self):
        """Handles curves with few points."""
        curve = [
            {"size": 50, "auroc_val": 0.85},
            {"size": 25, "auroc_val": 0.80},
        ]
        knee = detect_knee_point(curve)
        assert knee in [50, 25]

    def test_single_point(self):
        """Handles single point."""
        curve = [{"size": 50, "auroc_val": 0.85}]
        knee = detect_knee_point(curve)
        assert knee == 50


class TestComputeFeatureImportance:
    """Tests for compute_feature_importance function."""

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple fitted pipeline for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"protein_{i}" for i in range(n_features)],
        )
        y = (X["protein_0"] + X["protein_1"] > 0).astype(int)

        pipeline = Pipeline(
            [
                ("pre", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        pipeline.fit(X, y)
        return pipeline, X, y

    def test_linear_model_importance(self, simple_pipeline):
        """Linear model importance extraction works."""
        pipeline, X, y = simple_pipeline
        protein_cols = list(X.columns)

        importance = compute_feature_importance(
            pipeline,
            model_name="LR_EN",
            protein_cols=protein_cols,
            X=X,
            y=y.values,
        )

        assert len(importance) == len(protein_cols)
        assert all(v >= 0 for v in importance.values())  # Absolute values

    def test_importance_nonzero(self, simple_pipeline):
        """At least some features have non-zero importance."""
        pipeline, X, y = simple_pipeline
        protein_cols = list(X.columns)

        importance = compute_feature_importance(
            pipeline,
            model_name="LR_EN",
            protein_cols=protein_cols,
            X=X,
            y=y.values,
        )

        assert sum(importance.values()) > 0


class TestRFEResult:
    """Tests for RFEResult dataclass."""

    def test_default_values(self):
        """Default values are empty."""
        result = RFEResult()
        assert result.curve == []
        assert result.feature_ranking == {}
        assert result.recommended_panels == {}
        assert result.max_auroc == 0.0

    def test_with_values(self):
        """Can initialize with values."""
        result = RFEResult(
            curve=[{"size": 50, "auroc_val": 0.85}],
            feature_ranking={"protein_0": 0},
            recommended_panels={"knee_point": 50},
            max_auroc=0.85,
            model_name="LR_EN",
        )
        assert len(result.curve) == 1
        assert result.model_name == "LR_EN"


class TestSaveRFEResults:
    """Tests for save_rfe_results function."""

    def test_saves_all_artifacts(self):
        """All artifacts are saved."""
        result = RFEResult(
            curve=[
                {
                    "size": 50,
                    "auroc_val": 0.85,
                    "auroc_cv": 0.84,
                    "auroc_cv_std": 0.02,
                    "proteins": ["p1", "p2"],
                },
            ],
            feature_ranking={"protein_0": 0, "protein_1": 1},
            recommended_panels={"knee_point": 50, "min_size_95pct": 25},
            max_auroc=0.85,
            model_name="LR_EN",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_rfe_results(result, tmpdir, "LR_EN", 0)

            # Check all files exist
            assert Path(paths["panel_curve"]).exists()
            assert Path(paths["feature_ranking"]).exists()
            assert Path(paths["recommended_panels"]).exists()

            # Check panel_curve.csv content
            curve_df = pd.read_csv(paths["panel_curve"])
            assert "size" in curve_df.columns
            assert "auroc_val" in curve_df.columns

            # Check recommended_panels.json content
            with open(paths["recommended_panels"]) as f:
                rec = json.load(f)
            assert rec["model"] == "LR_EN"
            assert rec["max_auroc"] == 0.85

    def test_creates_output_directory(self):
        """Creates output directory if not exists."""
        result = RFEResult(
            curve=[
                {
                    "size": 50,
                    "auroc_val": 0.85,
                    "auroc_cv": 0.84,
                    "auroc_cv_std": 0.02,
                    "proteins": [],
                }
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "nested" / "output"
            save_rfe_results(result, str(outdir), "LR_EN", 0)
            assert outdir.exists()


class TestAggregateRFEResults:
    """Tests for aggregate_rfe_results function."""

    def _make_result(self, seed, auroc_offset=0.0):
        """Create a synthetic RFEResult for testing."""
        curve = [
            {
                "size": 100,
                "auroc_cv": 0.80 + auroc_offset,
                "auroc_cv_std": 0.02,
                "auroc_val": 0.85 + auroc_offset,
                "auroc_val_std": 0.03,
                "auroc_val_ci_low": 0.82 + auroc_offset,
                "auroc_val_ci_high": 0.88 + auroc_offset,
                "prauc_cv": 0.40,
                "prauc_val": 0.45 + auroc_offset,
                "brier_cv": 0.07,
                "brier_val": 0.06,
                "sens_at_95spec_cv": 0.50,
                "sens_at_95spec_val": 0.55 + auroc_offset,
                "proteins": [f"prot_{i}" for i in range(100)],
            },
            {
                "size": 50,
                "auroc_cv": 0.78 + auroc_offset,
                "auroc_cv_std": 0.03,
                "auroc_val": 0.83 + auroc_offset,
                "auroc_val_std": 0.04,
                "auroc_val_ci_low": 0.79 + auroc_offset,
                "auroc_val_ci_high": 0.87 + auroc_offset,
                "prauc_cv": 0.38,
                "prauc_val": 0.42 + auroc_offset,
                "brier_cv": 0.08,
                "brier_val": 0.07,
                "sens_at_95spec_cv": 0.45,
                "sens_at_95spec_val": 0.50 + auroc_offset,
                "proteins": [f"prot_{i}" for i in range(50)],
            },
            {
                "size": 10,
                "auroc_cv": 0.70 + auroc_offset,
                "auroc_cv_std": 0.05,
                "auroc_val": 0.72 + auroc_offset,
                "auroc_val_std": 0.06,
                "auroc_val_ci_low": 0.66 + auroc_offset,
                "auroc_val_ci_high": 0.78 + auroc_offset,
                "prauc_cv": 0.30,
                "prauc_val": 0.32 + auroc_offset,
                "brier_cv": 0.10,
                "brier_val": 0.09,
                "sens_at_95spec_cv": 0.30,
                "sens_at_95spec_val": 0.35 + auroc_offset,
                "proteins": [f"prot_{i}" for i in range(10)],
            },
        ]
        ranking = {f"prot_{i}": i + seed for i in range(90)}
        return RFEResult(
            curve=curve,
            feature_ranking=ranking,
            recommended_panels={"min_size_95pct": 50, "knee_point": 50},
            max_auroc=0.85 + auroc_offset,
            model_name="LR_EN",
        )

    def test_single_seed_passthrough(self):
        """Single seed returns the result as-is."""
        r = self._make_result(seed=0)
        agg = aggregate_rfe_results([r])
        assert agg is r

    def test_empty_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            aggregate_rfe_results([])

    def test_two_seeds_mean_auroc(self):
        """Two seeds produce mean AUROC values."""
        r0 = self._make_result(seed=0, auroc_offset=0.0)
        r1 = self._make_result(seed=1, auroc_offset=0.04)
        agg = aggregate_rfe_results([r0, r1])

        # Check curve has 3 sizes
        assert len(agg.curve) == 3
        sizes = [p["size"] for p in agg.curve]
        assert sizes == [100, 50, 10]

        # Mean auroc_val at size 100: (0.85 + 0.89) / 2 = 0.87
        size_100 = next(p for p in agg.curve if p["size"] == 100)
        assert abs(size_100["auroc_val"] - 0.87) < 1e-6

    def test_two_seeds_ci(self):
        """Cross-seed CI computed from percentiles."""
        r0 = self._make_result(seed=0, auroc_offset=0.0)
        r1 = self._make_result(seed=1, auroc_offset=0.04)
        agg = aggregate_rfe_results([r0, r1])

        size_100 = next(p for p in agg.curve if p["size"] == 100)
        # With 2 seeds, percentile CI is just min/max approximately
        assert size_100["auroc_val_ci_low"] <= size_100["auroc_val"]
        assert size_100["auroc_val_ci_high"] >= size_100["auroc_val"]

    def test_feature_ranking_aggregated(self):
        """Feature rankings are averaged and re-ranked."""
        r0 = self._make_result(seed=0)
        r1 = self._make_result(seed=1)
        agg = aggregate_rfe_results([r0, r1])

        # All proteins present
        assert len(agg.feature_ranking) == 90
        # Rankings are 0-indexed integers
        ranks = sorted(agg.feature_ranking.values())
        assert ranks == list(range(90))

    def test_recommended_panels_computed(self):
        """Recommended panels derived from aggregated curve."""
        r0 = self._make_result(seed=0)
        r1 = self._make_result(seed=1, auroc_offset=0.02)
        agg = aggregate_rfe_results([r0, r1])

        assert "knee_point" in agg.recommended_panels
        assert agg.model_name == "LR_EN"

    def test_max_auroc_from_mean(self):
        """max_auroc reflects aggregated mean, not single-seed max."""
        r0 = self._make_result(seed=0, auroc_offset=0.0)
        r1 = self._make_result(seed=1, auroc_offset=0.04)
        agg = aggregate_rfe_results([r0, r1])

        # Mean of 0.85 and 0.89 = 0.87
        assert abs(agg.max_auroc - 0.87) < 1e-6

    def test_n_seeds_in_curve(self):
        """Each curve point records number of seeds."""
        r0 = self._make_result(seed=0)
        r1 = self._make_result(seed=1)
        agg = aggregate_rfe_results([r0, r1])

        for point in agg.curve:
            assert point["n_seeds"] == 2
