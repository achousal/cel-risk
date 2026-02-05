"""Tests for cross-model consensus panel generation module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.consensus import (
    ConsensusResult,
    build_consensus_panel,
    cluster_and_select_representatives,
    compute_per_model_ranking,
    robust_rank_aggregate,
    save_consensus_results,
)


class TestComputePerModelRanking:
    """Tests for compute_per_model_ranking function."""

    def test_stability_only(self):
        """Ranking with stability frequency only (no OOF or essentiality)."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2", "P3", "P4"],
                "selection_fraction": [0.9, 0.8, 0.7, 0.6],
            }
        )

        result = compute_per_model_ranking(stability_df)

        assert len(result) == 4
        assert result.iloc[0]["protein"] == "P1"
        assert result.iloc[0]["stability_rank"] == 1
        assert result.iloc[0]["final_rank"] == 1
        # No OOF, so oof_importance should be NaN
        assert pd.isna(result.iloc[0]["oof_importance"])

    def test_with_oof_importance(self):
        """Ranking with stability and OOF importance."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2", "P3", "P4"],
                "selection_fraction": [0.9, 0.8, 0.7, 0.6],
            }
        )
        oof_df = pd.DataFrame(
            {
                "feature": ["P1", "P2", "P3", "P4"],
                "mean_importance": [0.5, 0.4, 0.3, 0.2],
            }
        )

        result = compute_per_model_ranking(stability_df, oof_importance_df=oof_df)

        assert len(result) == 4
        # P1 should be top: highest stability AND highest OOF importance
        assert result.iloc[0]["protein"] == "P1"
        assert not pd.isna(result.iloc[0]["oof_importance"])
        assert result.iloc[0]["oof_rank"] == 1

    def test_oof_weight_dominates(self):
        """Higher OOF weight prioritizes OOF importance over stability."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2"],
                "selection_fraction": [0.9, 0.5],
            }
        )
        # P2 has better OOF importance but worse stability
        oof_df = pd.DataFrame(
            {
                "feature": ["P1", "P2"],
                "mean_importance": [0.1, 0.9],
            }
        )

        result = compute_per_model_ranking(
            stability_df,
            oof_importance_df=oof_df,
            oof_weight=0.9,
            essentiality_weight=0.0,
            stability_weight=0.1,
        )

        # With high oof_weight, P2 should be first (better OOF)
        assert result.iloc[0]["protein"] == "P2"

    def test_stability_weight_dominates(self):
        """Higher stability weight prioritizes stability over OOF."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2"],
                "selection_fraction": [0.9, 0.5],
            }
        )
        # P2 has better OOF importance but worse stability
        oof_df = pd.DataFrame(
            {
                "feature": ["P1", "P2"],
                "mean_importance": [0.1, 0.9],
            }
        )

        result = compute_per_model_ranking(
            stability_df,
            oof_importance_df=oof_df,
            oof_weight=0.1,
            essentiality_weight=0.0,
            stability_weight=0.9,
        )

        # With high stability_weight, P1 should be first (better stability)
        assert result.iloc[0]["protein"] == "P1"

    def test_partial_oof_coverage(self):
        """Handles proteins not in OOF importance."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2", "P3"],
                "selection_fraction": [0.9, 0.8, 0.7],
            }
        )
        # OOF only for P1 and P2
        oof_df = pd.DataFrame(
            {
                "feature": ["P1", "P2"],
                "mean_importance": [0.5, 0.4],
            }
        )

        result = compute_per_model_ranking(stability_df, oof_importance_df=oof_df)

        # P3 should have NaN for OOF importance
        p3_row = result[result["protein"] == "P3"].iloc[0]
        assert pd.isna(p3_row["oof_importance"])
        # P3 composite score should still exist (uses stability only)
        assert not pd.isna(p3_row["composite_score"])

    def test_with_essentiality(self):
        """Ranking with stability, OOF importance, and essentiality."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2", "P3"],
                "selection_fraction": [0.9, 0.8, 0.7],
            }
        )
        oof_df = pd.DataFrame(
            {
                "feature": ["P1", "P2", "P3"],
                "mean_importance": [0.5, 0.4, 0.3],
            }
        )
        ess_df = pd.DataFrame(
            {
                "representative": ["P1", "P2", "P3"],
                "mean_delta_auroc": [0.05, 0.04, 0.03],
            }
        )

        result = compute_per_model_ranking(
            stability_df, oof_importance_df=oof_df, essentiality_df=ess_df
        )

        assert len(result) == 3
        assert result.iloc[0]["protein"] == "P1"
        assert not pd.isna(result.iloc[0]["essentiality"])
        assert result.iloc[0]["essentiality_rank"] == 1

    def test_missing_protein_column_raises(self):
        """Missing protein column raises ValueError."""
        stability_df = pd.DataFrame(
            {
                "name": ["P1", "P2"],
                "selection_fraction": [0.9, 0.8],
            }
        )

        with pytest.raises(ValueError, match="protein"):
            compute_per_model_ranking(stability_df)

    def test_custom_stability_column(self):
        """Uses custom stability column name."""
        stability_df = pd.DataFrame(
            {
                "protein": ["P1", "P2"],
                "custom_freq": [0.9, 0.8],
            }
        )

        result = compute_per_model_ranking(stability_df, stability_col="custom_freq")

        assert len(result) == 2
        assert result.iloc[0]["stability_freq"] == 0.9


class TestRobustRankAggregate:
    """Tests for robust_rank_aggregate function."""

    def test_geometric_mean_basic(self):
        """Basic geometric mean aggregation."""
        rankings = {
            "model_A": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 2, 3],
                }
            ),
            "model_B": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 2, 3],
                }
            ),
        }

        result = robust_rank_aggregate(rankings, method="geometric_mean")

        assert len(result) == 3
        # P1 should be ranked first (rank 1 in both)
        assert result.iloc[0]["protein"] == "P1"
        assert result.iloc[0]["consensus_rank"] == 1
        assert result.iloc[0]["n_models_present"] == 2

    def test_geometric_mean_different_rankings(self):
        """Geometric mean with different rankings per model."""
        rankings = {
            "model_A": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 2, 3],
                }
            ),
            "model_B": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [3, 2, 1],
                }
            ),
        }

        result = robust_rank_aggregate(rankings, method="geometric_mean")

        # Geometric mean of reciprocal ranks:
        # P1: gmean(1/1, 1/3) = gmean(1, 0.333) = 0.577
        # P2: gmean(1/2, 1/2) = gmean(0.5, 0.5) = 0.5
        # P3: gmean(1/3, 1/1) = gmean(0.333, 1) = 0.577
        # P1 and P3 tie with higher scores than P2
        # After sorting alphabetically for ties, P1 should be first
        assert result.iloc[0]["protein"] in ["P1", "P3"]
        # P2 has the lowest score
        assert result.iloc[2]["protein"] == "P2"

    def test_missing_protein_penalty(self):
        """Missing proteins get penalized."""
        rankings = {
            "model_A": pd.DataFrame(
                {
                    "protein": ["P1", "P2"],
                    "final_rank": [1, 2],
                }
            ),
            "model_B": pd.DataFrame(
                {
                    "protein": ["P1", "P3"],  # P2 missing, P3 new
                    "final_rank": [1, 2],
                }
            ),
        }

        result = robust_rank_aggregate(rankings, method="geometric_mean")

        # P1 is in both models -> should be ranked first
        assert result.iloc[0]["protein"] == "P1"
        assert result.iloc[0]["n_models_present"] == 2

        # P2 and P3 are each in only 1 model
        p2_row = result[result["protein"] == "P2"].iloc[0]
        p3_row = result[result["protein"] == "P3"].iloc[0]
        assert p2_row["n_models_present"] == 1
        assert p3_row["n_models_present"] == 1

    def test_borda_count(self):
        """Borda count aggregation."""
        rankings = {
            "model_A": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 2, 3],
                }
            ),
            "model_B": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 3, 2],
                }
            ),
        }

        result = robust_rank_aggregate(rankings, method="borda")

        assert len(result) == 3
        # P1 is rank 1 in both -> highest Borda score
        assert result.iloc[0]["protein"] == "P1"

    def test_median_rank(self):
        """Median rank aggregation."""
        rankings = {
            "model_A": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 2, 3],
                }
            ),
            "model_B": pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "final_rank": [1, 3, 2],
                }
            ),
        }

        result = robust_rank_aggregate(rankings, method="median")

        assert len(result) == 3
        # P1 has median rank 1 -> best
        assert result.iloc[0]["protein"] == "P1"

    def test_per_model_columns(self):
        """Result includes per-model rank columns."""
        rankings = {
            "LR_EN": pd.DataFrame(
                {
                    "protein": ["P1", "P2"],
                    "final_rank": [1, 2],
                }
            ),
            "RF": pd.DataFrame(
                {
                    "protein": ["P1", "P2"],
                    "final_rank": [2, 1],
                }
            ),
        }

        result = robust_rank_aggregate(rankings)

        assert "LR_EN_rank" in result.columns
        assert "RF_rank" in result.columns

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        rankings = {
            "model_A": pd.DataFrame(
                {
                    "protein": ["P1"],
                    "final_rank": [1],
                }
            ),
        }

        with pytest.raises(ValueError, match="method must be"):
            robust_rank_aggregate(rankings, method="invalid")

    def test_empty_rankings_raises(self):
        """Empty rankings raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            robust_rank_aggregate({})


class TestClusterAndSelectRepresentatives:
    """Tests for cluster_and_select_representatives function."""

    def test_no_correlation(self):
        """Uncorrelated proteins stay as singletons."""
        # Create uncorrelated data
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "P1": np.random.randn(100),
                "P2": np.random.randn(100),
                "P3": np.random.randn(100),
            }
        )
        proteins = ["P1", "P2", "P3"]
        scores = {"P1": 1.0, "P2": 0.8, "P3": 0.6}

        cluster_df, kept = cluster_and_select_representatives(
            df, proteins, scores, corr_threshold=0.85
        )

        # All should be kept (no high correlations)
        assert len(kept) == 3
        assert set(kept) == {"P1", "P2", "P3"}

    def test_correlated_proteins_clustered(self):
        """Correlated proteins are clustered."""
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame(
            {
                "P1": base,
                "P2": base + np.random.randn(100) * 0.1,  # Highly correlated with P1
                "P3": np.random.randn(100),  # Uncorrelated
            }
        )
        proteins = ["P1", "P2", "P3"]
        scores = {"P1": 1.0, "P2": 0.8, "P3": 0.6}

        cluster_df, kept = cluster_and_select_representatives(
            df, proteins, scores, corr_threshold=0.85
        )

        # P1 and P2 should be clustered, P1 kept (higher score)
        assert len(kept) == 2
        assert "P1" in kept
        assert "P3" in kept
        assert "P2" not in kept

    def test_representative_by_score(self):
        """Representative selected by highest consensus score."""
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame(
            {
                "P1": base,
                "P2": base + np.random.randn(100) * 0.1,
            }
        )
        proteins = ["P1", "P2"]
        # P2 has higher score even though P1 came first
        scores = {"P1": 0.5, "P2": 1.0}

        cluster_df, kept = cluster_and_select_representatives(
            df, proteins, scores, corr_threshold=0.85
        )

        # P2 should be kept (higher score)
        assert len(kept) == 1
        assert "P2" in kept

    def test_invalid_proteins_filtered(self):
        """Invalid proteins (not in df) are filtered."""
        # Use uncorrelated data to ensure P1 and P2 are not clustered
        np.random.seed(123)
        df = pd.DataFrame(
            {
                "P1": np.random.randn(50),
                "P2": np.random.randn(50),
            }
        )
        proteins = ["P1", "P2", "P_INVALID"]
        scores = {"P1": 1.0, "P2": 0.8, "P_INVALID": 0.6}

        cluster_df, kept = cluster_and_select_representatives(
            df, proteins, scores, corr_threshold=0.85
        )

        # Only valid proteins should be in result
        assert "P_INVALID" not in kept
        # P1 and P2 should both be kept (uncorrelated)
        assert set(kept) == {"P1", "P2"}

    def test_empty_proteins_returns_empty(self):
        """Empty protein list returns empty results."""
        df = pd.DataFrame({"P1": [1, 2, 3]})

        cluster_df, kept = cluster_and_select_representatives(df, [], {}, corr_threshold=0.85)

        assert len(kept) == 0
        assert len(cluster_df) == 0


class TestBuildConsensusPanel:
    """Tests for build_consensus_panel function."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_proteins = 20

        # Create training data
        df_train = pd.DataFrame(
            {f"P{i}": np.random.randn(n_samples) for i in range(1, n_proteins + 1)}
        )

        # Create stability data for 2 models
        model_stability = {
            "LR_EN": pd.DataFrame(
                {
                    "protein": [f"P{i}" for i in range(1, 16)],
                    "selection_fraction": [0.9 - i * 0.02 for i in range(15)],
                }
            ),
            "RF": pd.DataFrame(
                {
                    "protein": [f"P{i}" for i in range(1, 16)],
                    "selection_fraction": [0.85 - i * 0.02 for i in range(15)],
                }
            ),
        }

        # OOF importance data
        model_oof = {
            "LR_EN": pd.DataFrame(
                {
                    "feature": [f"P{i}" for i in range(1, 16)],
                    "mean_importance": [0.9 - i * 0.05 for i in range(15)],
                }
            ),
            "RF": pd.DataFrame(
                {
                    "feature": [f"P{i}" for i in range(1, 16)],
                    "mean_importance": [0.85 - i * 0.05 for i in range(15)],
                }
            ),
        }

        return df_train, model_stability, model_oof

    def test_basic_consensus(self, mock_data):
        """Basic consensus panel generation."""
        df_train, model_stability, model_oof = mock_data

        result = build_consensus_panel(
            model_stability=model_stability,
            df_train=df_train,
            model_oof_importance=model_oof,
            stability_threshold=0.75,
            target_size=10,
        )

        assert isinstance(result, ConsensusResult)
        assert len(result.final_panel) <= 10
        assert len(result.consensus_ranking) > 0
        assert len(result.per_model_rankings) > 0
        assert "n_models" in result.metadata

    def test_respects_target_size(self, mock_data):
        """Final panel respects target size."""
        df_train, model_stability, model_oof = mock_data

        result = build_consensus_panel(
            model_stability=model_stability,
            df_train=df_train,
            model_oof_importance=model_oof,
            stability_threshold=0.75,
            target_size=5,
        )

        assert len(result.final_panel) <= 5

    def test_stability_threshold_filtering(self, mock_data):
        """Only proteins above stability threshold are considered."""
        df_train, model_stability, model_oof = mock_data

        # High threshold should filter out most proteins
        result = build_consensus_panel(
            model_stability=model_stability,
            df_train=df_train,
            model_oof_importance=model_oof,
            stability_threshold=0.85,
            target_size=10,
        )

        # Final panel should be smaller due to filtering
        assert len(result.final_panel) < 10

    def test_insufficient_models_raises(self):
        """Raises error if fewer than 2 models."""
        df_train = pd.DataFrame({"P1": [1, 2, 3]})
        model_stability = {
            "LR_EN": pd.DataFrame(
                {
                    "protein": ["P1"],
                    "selection_fraction": [0.9],
                }
            ),
        }

        with pytest.raises(ValueError, match="at least 2 models"):
            build_consensus_panel(
                model_stability=model_stability,
                df_train=df_train,
                stability_threshold=0.75,
            )

    def test_metadata_populated(self, mock_data):
        """Metadata contains expected fields."""
        df_train, model_stability, model_oof = mock_data

        result = build_consensus_panel(
            model_stability=model_stability,
            df_train=df_train,
            model_oof_importance=model_oof,
            stability_threshold=0.75,
        )

        assert "timestamp" in result.metadata
        assert "n_models" in result.metadata
        assert "models" in result.metadata
        assert "parameters" in result.metadata
        assert "results" in result.metadata
        # Check composite_ranking weights in parameters
        assert "composite_ranking" in result.metadata["parameters"]


class TestSaveConsensusResults:
    """Tests for save_consensus_results function."""

    def test_saves_all_artifacts(self):
        """All expected artifacts are saved."""
        result = ConsensusResult(
            final_panel=["P1", "P2", "P3"],
            consensus_ranking=pd.DataFrame(
                {
                    "protein": ["P1", "P2", "P3"],
                    "consensus_score": [1.0, 0.8, 0.6],
                    "consensus_rank": [1, 2, 3],
                }
            ),
            per_model_rankings=pd.DataFrame(
                {
                    "model": ["LR_EN", "LR_EN"],
                    "protein": ["P1", "P2"],
                    "final_rank": [1, 2],
                }
            ),
            correlation_clusters=pd.DataFrame(
                {
                    "protein": ["P1", "P2"],
                    "cluster_id": [1, 2],
                    "kept": [True, True],
                }
            ),
            metadata={"timestamp": "2026-01-27", "n_models": 2},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_consensus_results(result, tmpdir)

            assert "final_panel_txt" in paths
            assert "final_panel_csv" in paths
            assert "consensus_ranking" in paths
            assert "per_model_rankings" in paths
            assert "correlation_clusters" in paths
            assert "metadata" in paths

            # Verify files exist
            for path in paths.values():
                assert Path(path).exists()

    def test_final_panel_txt_format(self):
        """Final panel text file has correct format."""
        result = ConsensusResult(
            final_panel=["PROTEIN_A", "PROTEIN_B", "PROTEIN_C"],
            consensus_ranking=pd.DataFrame(
                {
                    "protein": ["PROTEIN_A", "PROTEIN_B", "PROTEIN_C"],
                    "consensus_score": [1.0, 0.8, 0.6],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_consensus_results(result, tmpdir)

            # Read and verify format
            with open(paths["final_panel_txt"]) as f:
                lines = f.read().strip().split("\n")

            assert lines == ["PROTEIN_A", "PROTEIN_B", "PROTEIN_C"]

    def test_metadata_json_valid(self):
        """Metadata JSON is valid."""
        result = ConsensusResult(
            final_panel=["P1"],
            consensus_ranking=pd.DataFrame(
                {
                    "protein": ["P1"],
                    "consensus_score": [1.0],
                }
            ),
            metadata={"key": "value", "nested": {"a": 1}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_consensus_results(result, tmpdir)

            with open(paths["metadata"]) as f:
                loaded = json.load(f)

            assert loaded["key"] == "value"
            assert loaded["nested"]["a"] == 1
