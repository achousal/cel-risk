"""Tests for RRA permutation significance testing."""

import numpy as np
import pandas as pd

from ced_ml.features.consensus.significance import rra_permutation_test


def _make_rankings(
    n_proteins: int = 20,
    n_models: int = 4,
    n_signal: int = 3,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create synthetic per-model rankings with known signal proteins.

    Signal proteins get consistently low ranks (1..n_signal) across all models.
    Noise proteins get random ranks.
    """
    rng = np.random.default_rng(seed)
    proteins = [f"P{i}" for i in range(n_proteins)]
    rankings = {}

    for m in range(n_models):
        # Signal proteins always ranked 1..n_signal (may shuffle among themselves)
        signal_ranks = rng.permutation(n_signal) + 1
        # Noise proteins get random ranks from n_signal+1..n_proteins
        noise_ranks = rng.permutation(n_proteins - n_signal) + n_signal + 1

        all_ranks = np.concatenate([signal_ranks, noise_ranks])
        rankings[f"Model_{m}"] = pd.DataFrame(
            {
                "protein": proteins,
                "final_rank": all_ranks,
            }
        )

    return rankings


class TestRRAPermutationTest:
    """Tests for rra_permutation_test."""

    def test_signal_proteins_significant(self):
        """Known-signal proteins should have p << 0.05."""
        rankings = _make_rankings(n_proteins=20, n_models=4, n_signal=3)
        result = rra_permutation_test(rankings, n_perms=1000, alpha=0.05, seed=42)

        assert "protein" in result.columns
        assert "observed_rra" in result.columns
        assert "perm_p" in result.columns
        assert "bh_adjusted_p" in result.columns
        assert "significant" in result.columns

        # Signal proteins (P0, P1, P2) should be significant
        signal = result[result["protein"].isin(["P0", "P1", "P2"])]
        assert signal[
            "significant"
        ].all(), f"Signal proteins should be significant: {signal[['protein', 'bh_adjusted_p']]}"

    def test_noise_proteins_not_significant(self):
        """Random-ranked proteins should mostly not be significant."""
        rankings = _make_rankings(n_proteins=30, n_models=4, n_signal=2)
        result = rra_permutation_test(rankings, n_perms=1000, alpha=0.05, seed=42)

        noise = result[~result["protein"].isin(["P0", "P1"])]
        # Allow a few false positives but the majority should not be significant
        noise_sig_rate = noise["significant"].mean()
        assert noise_sig_rate < 0.2, f"Noise significant rate too high: {noise_sig_rate:.2f}"

    def test_output_shape(self):
        """Output should have one row per protein."""
        rankings = _make_rankings(n_proteins=15, n_models=3)
        result = rra_permutation_test(rankings, n_perms=100, seed=42)
        assert len(result) == 15

    def test_sorted_by_rra_descending(self):
        """Output should be sorted by observed_rra descending."""
        rankings = _make_rankings()
        result = rra_permutation_test(rankings, n_perms=100, seed=42)
        rra_vals = result["observed_rra"].values
        assert np.all(rra_vals[:-1] >= rra_vals[1:])

    def test_pvalues_in_valid_range(self):
        """P-values should be in (0, 1]."""
        rankings = _make_rankings()
        result = rra_permutation_test(rankings, n_perms=500, seed=42)
        assert (result["perm_p"] > 0).all()
        assert (result["perm_p"] <= 1).all()
        assert (result["bh_adjusted_p"] > 0).all()
        assert (result["bh_adjusted_p"] <= 1).all()

    def test_reproducible_with_seed(self):
        """Same seed should give same results."""
        rankings = _make_rankings()
        r1 = rra_permutation_test(rankings, n_perms=200, seed=123)
        r2 = rra_permutation_test(rankings, n_perms=200, seed=123)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_differ(self):
        """Different seeds should give (slightly) different p-values."""
        rankings = _make_rankings()
        r1 = rra_permutation_test(rankings, n_perms=200, seed=1)
        r2 = rra_permutation_test(rankings, n_perms=200, seed=2)
        # P-values should differ for at least some proteins
        assert not np.allclose(r1["perm_p"].values, r2["perm_p"].values)

    def test_missing_proteins_handled(self):
        """Models with different protein sets should work."""
        rankings = {
            "M1": pd.DataFrame(
                {
                    "protein": ["A", "B", "C", "D"],
                    "final_rank": [1, 2, 3, 4],
                }
            ),
            "M2": pd.DataFrame(
                {
                    "protein": ["A", "B", "E"],  # C, D missing; E new
                    "final_rank": [1, 2, 3],
                }
            ),
        }
        result = rra_permutation_test(rankings, n_perms=100, seed=42)
        assert len(result) == 5  # A, B, C, D, E
        assert set(result["protein"]) == {"A", "B", "C", "D", "E"}

    def test_all_identical_ranks(self):
        """Perfect agreement: all models rank identically."""
        rankings = {
            f"M{i}": pd.DataFrame(
                {
                    "protein": ["X", "Y", "Z"],
                    "final_rank": [1, 2, 3],
                }
            )
            for i in range(4)
        }
        result = rra_permutation_test(rankings, n_perms=500, seed=42)
        # Top protein should be significant with perfect agreement
        top = result[result["protein"] == "X"]
        assert top["perm_p"].iloc[0] < 0.05

    def test_two_models_minimum(self):
        """Should work with exactly 2 models."""
        rankings = {
            "M1": pd.DataFrame(
                {
                    "protein": ["A", "B", "C"],
                    "final_rank": [1, 2, 3],
                }
            ),
            "M2": pd.DataFrame(
                {
                    "protein": ["A", "B", "C"],
                    "final_rank": [1, 3, 2],
                }
            ),
        }
        result = rra_permutation_test(rankings, n_perms=100, seed=42)
        assert len(result) == 3
