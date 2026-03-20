"""Permutation null for RRA consensus scores.

Tests whether cross-model rank agreement for each protein exceeds
what would be expected by chance. Shuffles per-model final_rank vectors
independently, recomputes geometric mean RRA, and builds a null
distribution for empirical p-values with BH FDR correction.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import gmean

logger = logging.getLogger(__name__)


def rra_permutation_test(
    per_model_rankings: dict[str, pd.DataFrame],
    n_perms: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation null for RRA consensus scores.

    Shuffles per-model final_rank vectors independently,
    recomputes geometric mean RRA, builds null distribution.
    Returns per-protein empirical p-values with BH correction.

    Args:
        per_model_rankings: Dict mapping model_name -> DataFrame with columns
            [protein, final_rank]. Same structure used by
            ``geometric_mean_rank_aggregate()``.
        n_perms: Number of permutations (default 10,000).
        alpha: FDR threshold for BH correction (default 0.05).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            - protein: Protein name
            - observed_rra: Observed geometric-mean RRA consensus score
            - perm_p: Empirical permutation p-value
            - bh_adjusted_p: Benjamini-Hochberg adjusted p-value
            - significant: Boolean, True if bh_adjusted_p < alpha
    """
    from statsmodels.stats.multitest import multipletests

    from .aggregation import geometric_mean_rank_aggregate

    # Compute observed consensus scores
    observed = geometric_mean_rank_aggregate(per_model_rankings, method="geometric_mean")
    observed_scores = dict(zip(observed["protein"], observed["consensus_score"], strict=False))
    proteins = list(observed_scores.keys())
    n_proteins = len(proteins)

    # Build rank arrays for fast permutation (proteins x models)
    model_names = list(per_model_rankings.keys())
    n_models = len(model_names)

    # Precompute: for each model, the rank vector aligned to protein order
    # Missing proteins get max_rank + 1 penalty (same as aggregation.py)
    max_ranks = {}
    rank_matrix = np.empty((n_proteins, n_models), dtype=np.float64)

    # Missing-protein penalty: max rank across all models + 1
    # (consistent with aggregation.py)
    missing_rank = max(float(df["final_rank"].max()) for df in per_model_rankings.values()) + 1

    for j, model_name in enumerate(model_names):
        df = per_model_rankings[model_name]
        max_ranks[model_name] = float(df["final_rank"].max())
        lookup = dict(zip(df["protein"], df["final_rank"], strict=False))
        for i, protein in enumerate(proteins):
            rank_matrix[i, j] = lookup.get(protein, missing_rank)

    # Precompute max_rank array for normalization
    max_rank_arr = np.array([max_ranks[m] for m in model_names], dtype=np.float64)

    def _compute_rra_scores(rmat: np.ndarray) -> np.ndarray:
        """Geometric mean of normalized reciprocal ranks."""
        # rmat: (n_proteins, n_models)
        # normalized reciprocal: max_rank[j] / rank[i, j]
        nrr = max_rank_arr[np.newaxis, :] / rmat
        return gmean(nrr, axis=1)

    observed_arr = _compute_rra_scores(rank_matrix)

    # Permutation loop
    rng = np.random.default_rng(seed)
    count_ge = np.zeros(n_proteins, dtype=np.int64)

    logger.info(
        f"Running RRA permutation test: {n_perms} permutations, "
        f"{n_proteins} proteins, {n_models} models"
    )

    for _b in range(n_perms):
        perm_matrix = rank_matrix.copy()
        for j in range(n_models):
            rng.shuffle(perm_matrix[:, j])
        perm_scores = _compute_rra_scores(perm_matrix)
        count_ge += (perm_scores >= observed_arr).astype(np.int64)

    # Empirical p-values (Phipson & Smyth 2010 correction)
    perm_p = (1 + count_ge) / (1 + n_perms)

    # BH correction
    reject, bh_p, _, _ = multipletests(perm_p, alpha=alpha, method="fdr_bh")

    result = pd.DataFrame(
        {
            "protein": proteins,
            "observed_rra": observed_arr,
            "perm_p": perm_p,
            "bh_adjusted_p": bh_p,
            "significant": reject,
        }
    )

    result = result.sort_values("observed_rra", ascending=False).reset_index(drop=True)

    n_sig = result["significant"].sum()
    logger.info(
        f"RRA permutation test complete: {n_sig}/{n_proteins} proteins significant "
        f"at BH-adjusted alpha={alpha}"
    )

    return result
