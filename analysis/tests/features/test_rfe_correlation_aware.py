"""Tests for correlation-aware pre-filtering in RFE panel optimization."""

import tempfile

import numpy as np
import pandas as pd

from ced_ml.features.rfe import (
    RFEResult,
    aggregate_rfe_results,
    cluster_correlated_proteins_for_rfe,
    save_rfe_results,
)


def _make_correlated_data(n_samples=200, n_proteins=20, n_correlated_pairs=5, seed=42):
    """Create synthetic data with known correlated protein pairs.

    Returns X_train, y_train, protein_cols where the first n_correlated_pairs*2
    proteins are pairwise correlated (r > 0.9).
    """
    rng = np.random.default_rng(seed)
    protein_cols = [f"prot_{i}_resid" for i in range(n_proteins)]

    X = rng.standard_normal((n_samples, n_proteins))

    # Make pairs correlated: prot_0 ~ prot_1, prot_2 ~ prot_3, etc.
    for i in range(n_correlated_pairs):
        base_idx = i * 2
        partner_idx = base_idx + 1
        X[:, partner_idx] = X[:, base_idx] + rng.normal(0, 0.1, n_samples)

    df = pd.DataFrame(X, columns=protein_cols)
    y = rng.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    return df, y, protein_cols


class TestClusterCorrelatedProteinsForRFE:
    """Tests for the clustering pre-filter function."""

    def test_basic_clustering(self):
        """Correlated pairs should be collapsed into single representatives."""
        X_train, y_train, protein_cols = _make_correlated_data(n_proteins=10, n_correlated_pairs=3)
        reps, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train, y_train, protein_cols, corr_threshold=0.80
        )

        # Should have fewer representatives than input proteins
        assert len(reps) < len(protein_cols)
        # Each correlated pair should map to one cluster with 2 members
        multi_clusters = [v for v in cluster_map.values() if v["cluster_size"] > 1]
        assert len(multi_clusters) >= 1

    def test_cluster_map_structure(self):
        """cluster_map has expected keys."""
        X_train, y_train, protein_cols = _make_correlated_data(n_proteins=10, n_correlated_pairs=2)
        reps, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train, y_train, protein_cols, corr_threshold=0.80
        )

        for rep, info in cluster_map.items():
            assert "cluster_id" in info
            assert "cluster_size" in info
            assert "members" in info
            assert isinstance(info["members"], list)
            assert rep in info["members"]
            assert info["cluster_size"] == len(info["members"])

    def test_all_singletons(self):
        """No correlations above threshold -> all singletons."""
        rng = np.random.default_rng(99)
        n = 50
        protein_cols = [f"prot_{i}_resid" for i in range(5)]
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=protein_cols)
        y = rng.choice([0, 1], size=n, p=[0.9, 0.1])

        reps, cluster_map = cluster_correlated_proteins_for_rfe(
            X, y, protein_cols, corr_threshold=0.99
        )

        assert len(reps) == len(protein_cols)
        for info in cluster_map.values():
            assert info["cluster_size"] == 1

    def test_skip_when_threshold_ge_1(self):
        """Threshold >= 1.0 skips clustering entirely."""
        X_train, y_train, protein_cols = _make_correlated_data()
        reps, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train, y_train, protein_cols, corr_threshold=1.0
        )

        assert reps == list(protein_cols)
        assert cluster_map == {}

    def test_skip_when_fewer_than_2_proteins(self):
        """< 2 proteins skips clustering."""
        X_train, y_train, _ = _make_correlated_data()
        reps, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train, y_train, ["prot_0_resid"], corr_threshold=0.80
        )

        assert reps == ["prot_0_resid"]
        assert cluster_map == {}

    def test_selection_freq_used(self):
        """Representative selection respects selection frequency."""
        X_train, y_train, protein_cols = _make_correlated_data(n_proteins=4, n_correlated_pairs=2)
        # Give partner proteins higher freq than base proteins
        selection_freq = {
            "prot_0_resid": 0.5,
            "prot_1_resid": 0.9,  # partner, higher freq
            "prot_2_resid": 0.5,
            "prot_3_resid": 0.9,  # partner, higher freq
        }
        reps, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train,
            y_train,
            protein_cols,
            selection_freq=selection_freq,
            corr_threshold=0.80,
        )

        # Higher-freq proteins should be selected as representatives
        for rep in reps:
            if rep in cluster_map and cluster_map[rep]["cluster_size"] > 1:
                assert selection_freq.get(rep, 0) >= 0.9


class TestRFEResultClusterMap:
    """Tests for cluster_map field in RFEResult."""

    def test_default_empty(self):
        """Default cluster_map is empty dict."""
        result = RFEResult()
        assert result.cluster_map == {}

    def test_set_cluster_map(self):
        """cluster_map can be set."""
        cmap = {"prot_0": {"cluster_id": 1, "cluster_size": 2, "members": ["prot_0", "prot_1"]}}
        result = RFEResult(cluster_map=cmap)
        assert result.cluster_map == cmap


class TestSaveRFEResultsWithClusters:
    """Tests for save_rfe_results with cluster metadata."""

    def _make_result_with_clusters(self):
        """Create RFEResult with cluster metadata."""
        cluster_map = {
            "prot_0_resid": {
                "cluster_id": 1,
                "cluster_size": 2,
                "members": ["prot_0_resid", "prot_1_resid"],
            },
            "prot_2_resid": {
                "cluster_id": 2,
                "cluster_size": 1,
                "members": ["prot_2_resid"],
            },
        }
        return RFEResult(
            curve=[
                {
                    "size": 2,
                    "auroc_cv": 0.85,
                    "auroc_cv_std": 0.02,
                    "auroc_val": 0.84,
                    "proteins": ["prot_0_resid", "prot_2_resid"],
                },
            ],
            feature_ranking={"prot_2_resid": 0, "prot_0_resid": 1},
            recommended_panels={"95pct": 2},
            max_auroc=0.84,
            model_name="LR_EN",
            cluster_map=cluster_map,
        )

    def test_cluster_mapping_csv_created(self):
        """cluster_mapping.csv is created when cluster_map is present."""
        result = self._make_result_with_clusters()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_rfe_results(result, tmpdir, "LR_EN", 0)

            assert "cluster_mapping" in paths
            cluster_df = pd.read_csv(paths["cluster_mapping"])
            assert "representative" in cluster_df.columns
            assert "member_protein" in cluster_df.columns
            assert "cluster_id" in cluster_df.columns
            assert "is_representative" in cluster_df.columns
            # 2 members in cluster 1 + 1 in cluster 2 = 3 rows
            assert len(cluster_df) == 3

    def test_feature_ranking_has_cluster_columns(self):
        """feature_ranking.csv has cluster metadata columns."""
        result = self._make_result_with_clusters()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_rfe_results(result, tmpdir, "LR_EN", 0)

            ranking_df = pd.read_csv(paths["feature_ranking"])
            assert "cluster_id" in ranking_df.columns
            assert "cluster_size" in ranking_df.columns
            assert "cluster_members" in ranking_df.columns

    def test_panel_curve_has_cluster_columns(self):
        """panel_curve.csv has cluster summary columns."""
        result = self._make_result_with_clusters()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_rfe_results(result, tmpdir, "LR_EN", 0)

            curve_df = pd.read_csv(paths["panel_curve"])
            assert "total_cluster_members" in curve_df.columns
            assert "cluster_members_json" in curve_df.columns
            # 2 + 1 = 3 total members
            assert curve_df["total_cluster_members"].iloc[0] == 3

    def test_no_cluster_columns_when_empty(self):
        """No cluster columns when cluster_map is empty."""
        result = RFEResult(
            curve=[
                {
                    "size": 2,
                    "auroc_cv": 0.85,
                    "auroc_cv_std": 0.02,
                    "auroc_val": 0.84,
                    "proteins": ["prot_0_resid", "prot_2_resid"],
                },
            ],
            feature_ranking={"prot_2_resid": 0, "prot_0_resid": 1},
            recommended_panels={"95pct": 2},
            max_auroc=0.84,
            model_name="LR_EN",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_rfe_results(result, tmpdir, "LR_EN", 0)

            assert "cluster_mapping" not in paths
            ranking_df = pd.read_csv(paths["feature_ranking"])
            assert "cluster_id" not in ranking_df.columns


class TestAggregateRFEResultsWithClusters:
    """Tests for aggregation preserving cluster_map."""

    def test_cluster_map_preserved(self):
        """Aggregated result uses first seed's cluster_map."""
        cluster_map = {
            "prot_0": {"cluster_id": 1, "cluster_size": 2, "members": ["prot_0", "prot_1"]},
        }
        results = []
        for seed in range(3):
            results.append(
                RFEResult(
                    curve=[
                        {
                            "size": 10,
                            "auroc_cv": 0.80 + seed * 0.01,
                            "auroc_cv_std": 0.02,
                            "auroc_val": 0.79 + seed * 0.01,
                            "proteins": ["prot_0"],
                        },
                    ],
                    feature_ranking={"prot_0": 0},
                    recommended_panels={},
                    max_auroc=0.79 + seed * 0.01,
                    model_name="LR_EN",
                    cluster_map=cluster_map,
                )
            )

        agg = aggregate_rfe_results(results)
        assert agg.cluster_map == cluster_map

    def test_empty_cluster_map_preserved(self):
        """Aggregation works when cluster_map is empty."""
        results = []
        for _seed in range(2):
            results.append(
                RFEResult(
                    curve=[
                        {
                            "size": 10,
                            "auroc_cv": 0.80,
                            "auroc_cv_std": 0.02,
                            "auroc_val": 0.79,
                            "proteins": ["prot_0"],
                        },
                    ],
                    feature_ranking={"prot_0": 0},
                    recommended_panels={},
                    max_auroc=0.79,
                    model_name="LR_EN",
                )
            )

        agg = aggregate_rfe_results(results)
        assert agg.cluster_map == {}
