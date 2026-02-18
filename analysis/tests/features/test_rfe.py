"""Tests for Recursive Feature Elimination module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ced_ml.features.rfe import (
    RFEResult,
    aggregate_rfe_results,
    compute_eval_sizes,
    compute_feature_importance,
    detect_knee_point,
    find_recommended_panels,
    save_rfe_results,
)


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
        assert result.retention_freq == {}

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
            assert Path(paths["rfe_feature_report"]).exists()
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

            # Check feature report content
            feature_report_df = pd.read_csv(paths["rfe_feature_report"])
            assert "rank" in feature_report_df.columns
            assert "importance_score" in feature_report_df.columns
            assert "retention_freq" in feature_report_df.columns

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

    def test_single_seed_normalized(self):
        """Single-seed aggregation returns normalized output with complete ranking."""
        r = self._make_result(seed=0)
        agg = aggregate_rfe_results([r])
        assert agg is not r
        assert len(agg.feature_ranking) == 100
        assert agg.recommended_panels == r.recommended_panels
        assert agg.max_auroc == r.max_auroc

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
        assert len(agg.feature_ranking) == 100
        # Rankings are 0-indexed integers
        ranks = sorted(agg.feature_ranking.values())
        assert ranks == list(range(100))

    def test_retention_frequency_aggregated(self):
        """Retention frequencies are available and bounded."""
        r0 = self._make_result(seed=0)
        r1 = self._make_result(seed=1)
        agg = aggregate_rfe_results([r0, r1])

        assert agg.retention_freq
        assert all(0.0 <= freq <= 1.0 for freq in agg.retention_freq.values())

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


class TestGetRfeTuneSpace:
    """Tests for get_rfe_tune_space in hyperparams module."""

    def test_known_models(self):
        """All expected models have tune spaces."""
        from ced_ml.models.hyperparams import get_rfe_tune_space

        for model in ("LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"):
            space = get_rfe_tune_space(model)
            assert isinstance(space, dict)
            assert len(space) > 0

    def test_lr_en_params(self):
        """LR_EN has C and l1_ratio."""
        from ced_ml.models.hyperparams import get_rfe_tune_space

        space = get_rfe_tune_space("LR_EN")
        assert "clf__C" in space
        assert "clf__l1_ratio" in space
        assert space["clf__C"]["type"] == "float"
        assert space["clf__C"]["log"] is True

    def test_rf_params(self):
        """RF has max_depth and min_samples_leaf."""
        from ced_ml.models.hyperparams import get_rfe_tune_space

        space = get_rfe_tune_space("RF")
        assert "clf__max_depth" in space
        assert "clf__min_samples_leaf" in space

    def test_unknown_model_raises(self):
        """Unknown model raises ValueError."""
        from ced_ml.models.hyperparams import get_rfe_tune_space

        with pytest.raises(ValueError, match="No RFE tune space"):
            get_rfe_tune_space("UNKNOWN_MODEL")

    def test_returns_copy(self):
        """Returns a copy, not a reference to the module-level dict."""
        from ced_ml.models.hyperparams import RFE_TUNE_SPACES, get_rfe_tune_space

        space = get_rfe_tune_space("LR_EN")
        space["extra_param"] = {"type": "float", "low": 0, "high": 1}
        assert "extra_param" not in RFE_TUNE_SPACES["LR_EN"]


class TestMakeFreshEstimator:
    """Tests for make_fresh_estimator in rfe module."""

    def test_lr_en(self):
        """LR_EN produces a LogisticRegression."""
        from ced_ml.features.rfe import make_fresh_estimator

        est = make_fresh_estimator("LR_EN", random_state=0)
        assert hasattr(est, "predict_proba")
        assert est.get_params()["random_state"] == 0

    def test_rf(self):
        """RF produces a RandomForestClassifier with n_estimators=300."""
        from ced_ml.features.rfe import make_fresh_estimator

        est = make_fresh_estimator("RF", random_state=0)
        assert est.get_params()["n_estimators"] == 300

    def test_xgboost(self):
        """XGBoost produces an XGBClassifier with n_estimators=300."""
        pytest.importorskip("xgboost")
        from ced_ml.features.rfe import make_fresh_estimator

        est = make_fresh_estimator("XGBoost", random_state=0)
        assert est.get_params()["n_estimators"] == 300

    def test_unknown_raises(self):
        """Unknown model raises ValueError."""
        from ced_ml.features.rfe import make_fresh_estimator

        with pytest.raises(ValueError, match="Unknown model"):
            make_fresh_estimator("UNKNOWN")


class TestQuickTuneAtK:
    """Tests for _quick_tune_at_k with synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Create small synthetic binary classification dataset."""
        np.random.seed(42)
        n = 200
        n_features = 10
        X = pd.DataFrame(
            np.random.randn(n, n_features),
            columns=[f"prot_{i}" for i in range(n_features)],
        )
        y = (X["prot_0"] + X["prot_1"] + np.random.randn(n) * 0.3 > 0).astype(int).values
        return X, y

    def test_returns_pipeline_and_params(self, synthetic_data):
        """Returns a fitted pipeline and best_params dict."""
        from ced_ml.features.rfe import _quick_tune_at_k

        X, y = synthetic_data
        feature_cols = list(X.columns)

        pipeline, best_params = _quick_tune_at_k(
            model_name="LR_EN",
            X_train=X,
            y_train=y,
            feature_cols=feature_cols,
            cat_cols=[],
            cv_folds=2,
            n_trials=5,
            n_jobs=1,
            random_state=42,
        )

        assert hasattr(pipeline, "predict_proba")
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        # Pipeline should be fitted (can predict)
        probs = pipeline.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_best_params_have_clf_prefix(self, synthetic_data):
        """Best params keys have clf__ prefix."""
        from ced_ml.features.rfe import _quick_tune_at_k

        X, y = synthetic_data

        _, best_params = _quick_tune_at_k(
            model_name="LR_EN",
            X_train=X,
            y_train=y,
            feature_cols=list(X.columns),
            cat_cols=[],
            cv_folds=2,
            n_trials=3,
            random_state=42,
        )

        for key in best_params:
            assert key.startswith("clf__"), f"Expected clf__ prefix, got {key}"


class TestRunEliminationFallback:
    """Regression tests for RFE evaluation fallback behavior."""

    def test_retune_failure_falls_back_to_baseline_pipeline(self, monkeypatch):
        """Per-k tuning failures should still produce a valid curve point."""
        from ced_ml.features import rfe_engine

        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            rng.normal(size=(120, 2)),
            columns=["prot_0", "prot_1"],
        )
        y = (X["prot_0"] + X["prot_1"] + rng.normal(scale=0.25, size=len(X)) > 0).astype(int)

        X_train = X.iloc[:80].copy()
        X_val = X.iloc[80:].copy()
        y_train = y.iloc[:80].to_numpy()
        y_val = y.iloc[80:].to_numpy()

        base_pipeline = Pipeline(
            [
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )

        def _always_fail_tuning(**kwargs):
            raise RuntimeError("All 10 Optuna trials failed")

        monkeypatch.setattr(rfe_engine, "quick_tune_at_k", _always_fail_tuning)

        curve, feature_ranking, max_auroc, all_best_params = (
            rfe_engine.run_elimination_with_evaluation(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                base_pipeline=base_pipeline,
                model_name="LR_EN",
                current_proteins=["prot_0", "prot_1"],
                cat_cols=[],
                meta_num_cols=[],
                eval_sizes=[2],
                min_size=1,
                cv_folds=0,
                random_state=42,
                n_perm_repeats=1,
                can_retune=True,
                retune_n_trials=10,
                retune_cv_folds=2,
                retune_n_jobs=4,
                rfe_tune_spaces=None,
                min_auroc_frac=0.5,
            )
        )

        assert len(curve) == 1
        assert curve[0]["size"] == 2
        assert 0.0 <= curve[0]["auroc_val"] <= 1.0
        assert feature_ranking == {}
        assert max_auroc >= 0.0
        assert all_best_params == []
