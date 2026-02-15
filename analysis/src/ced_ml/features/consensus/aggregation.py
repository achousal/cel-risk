"""Cross-model rank aggregation for consensus panel generation.

Implements geometric mean rank aggregation with rank normalization for
comparability across models with different protein sets.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import gmean

logger = logging.getLogger(__name__)


def geometric_mean_rank_aggregate(
    per_model_rankings: dict[str, pd.DataFrame],
    method: str = "geometric_mean",
) -> pd.DataFrame:
    """Aggregate rankings across models using geometric mean of normalized reciprocal ranks.

    This method computes a consensus ranking by taking the geometric mean of
    reciprocal ranks, normalized by each model's list length. This is NOT the
    formal Robust Rank Aggregation (RRA) method from Kolde et al. (2012), which
    uses beta-model p-values. See ADR-004 for rationale.

    Rank normalization ensures that ranks are comparable across models with
    different numbers of proteins. For each model, ranks are normalized by dividing
    by the maximum rank (list length) before taking reciprocals.

    Args:
        per_model_rankings: Dict mapping model_name -> DataFrame with columns
            [protein, final_rank]. Each model may have different protein sets.
        method: Aggregation method:
            - "geometric_mean": Geometric mean of normalized reciprocal ranks (default).
              Penalizes proteins missing from some models.
            - "borda": Borda count (sum of (N - rank + 1) across models).
            - "median": Median rank across models.

    Returns:
        DataFrame with columns:
            - protein: Protein name
            - consensus_score: Aggregated score (higher = better)
            - consensus_rank: Rank by consensus score
            - n_models_present: Number of models with this protein
            - rank_std: Standard deviation of ranks across models
            - rank_cv: Coefficient of variation of ranks (std/mean)
            - agreement_strength: Fraction of models agreeing (n_models_present / n_models)
            - per-model columns: {model}_rank for each model

    Raises:
        ValueError: If method is not recognized or no valid rankings provided.
    """
    _validate_aggregation_method(method)
    _validate_input_rankings(per_model_rankings)

    all_proteins = _collect_all_proteins(per_model_rankings)
    model_names = list(per_model_rankings.keys())
    n_models = len(model_names)

    # Precompute rank lookups for O(1) access
    max_ranks, rank_lookups = _build_rank_lookups(per_model_rankings)
    missing_rank = max(max_ranks.values()) + 1

    # Build result rows
    rows = []
    for protein in sorted(all_proteins):
        row = _compute_protein_consensus(
            protein,
            model_names,
            rank_lookups,
            max_ranks,
            missing_rank,
            n_models,
            method,
        )
        rows.append(row)

    # Create DataFrame and sort by consensus score
    result = pd.DataFrame(rows)
    result = result.sort_values("consensus_score", ascending=False)
    result["consensus_rank"] = range(1, len(result) + 1)

    # Reorder columns
    col_order = [
        "protein",
        "consensus_score",
        "consensus_rank",
        "n_models_present",
        "agreement_strength",
        "presence_fraction",
        "rank_std",
        "rank_cv",
    ] + [f"{m}_rank" for m in model_names]
    result = result[col_order]

    _warn_low_presence_in_top_results(result)

    return result


def explicit_shap_normalized_aggregate(
    per_model_rankings: dict[str, pd.DataFrame],
    score_col: str = "oof_importance",
) -> pd.DataFrame:
    """Aggregate SHAP scores across models with explicit per-model normalization.

    Implements the core normalization used in Nature Medicine
    (s41591-024-03398-5) for cross-model SHAP aggregation:
    1) Normalize each model's SHAP magnitude to [0, 1] via min-max on ``|score|``.
    2) Restore sign after magnitude normalization.
    3) Average normalized signed SHAP across models.

    Missing proteins in a model receive a contribution of 0.0 (explicit penalty
    for model absence).

    Args:
        per_model_rankings: Dict mapping model_name -> DataFrame with columns:
            ``protein`` and ``score_col``.
        score_col: Column containing SHAP-derived per-protein scores.

    Returns:
        DataFrame with columns:
            - protein: Protein name
            - shap_signed_score: Mean normalized signed SHAP across models
            - shap_magnitude_score: Mean normalized magnitude (absolute value)
            - shap_n_models_present: Number of models with a non-null SHAP score
            - shap_presence_fraction: Fraction of models where protein is present
    """
    if not per_model_rankings:
        raise ValueError("per_model_rankings cannot be empty")

    all_proteins = _collect_all_proteins(per_model_rankings)
    model_names = list(per_model_rankings.keys())
    n_models = len(model_names)

    # Initialize dense matrix with explicit 0.0 penalty for missing proteins.
    norm_signed = pd.DataFrame(0.0, index=sorted(all_proteins), columns=model_names, dtype=float)
    presence = pd.DataFrame(False, index=sorted(all_proteins), columns=model_names, dtype=bool)

    for model_name, df in per_model_rankings.items():
        if "protein" not in df.columns:
            raise ValueError(f"Model '{model_name}' is missing required column 'protein'")
        if score_col not in df.columns:
            raise ValueError(
                f"Model '{model_name}' is missing required SHAP score column '{score_col}'"
            )

        model_scores = (
            df[["protein", score_col]]
            .dropna(subset=[score_col])
            .drop_duplicates(subset=["protein"])
            .set_index("protein")[score_col]
            .astype(float)
        )
        if model_scores.empty:
            continue

        abs_scores = model_scores.abs()
        min_abs = float(abs_scores.min())
        max_abs = float(abs_scores.max())

        if np.isclose(max_abs, min_abs):
            # Degenerate case: all proteins have same magnitude in this model.
            # Keep them equally weighted.
            norm_abs = pd.Series(1.0, index=model_scores.index, dtype=float)
        else:
            norm_abs = (abs_scores - min_abs) / (max_abs - min_abs)

        signed_norm = np.sign(model_scores) * norm_abs

        common_idx = norm_signed.index.intersection(signed_norm.index)
        norm_signed.loc[common_idx, model_name] = signed_norm.loc[common_idx]
        presence.loc[common_idx, model_name] = True

    signed_mean = norm_signed.mean(axis=1)
    magnitude_mean = signed_mean.abs()
    n_present = presence.sum(axis=1).astype(int)
    presence_fraction = n_present / n_models

    out = pd.DataFrame(
        {
            "protein": signed_mean.index,
            "shap_signed_score": signed_mean.values,
            "shap_magnitude_score": magnitude_mean.values,
            "shap_n_models_present": n_present.values,
            "shap_presence_fraction": presence_fraction.values,
        }
    )
    out = out.sort_values("shap_magnitude_score", ascending=False, ignore_index=True)
    return out


def _validate_aggregation_method(method: str) -> None:
    """Validate aggregation method."""
    valid_methods = {"geometric_mean", "borda", "median"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")


def _validate_input_rankings(per_model_rankings: dict[str, pd.DataFrame]) -> None:
    """Validate input rankings are non-empty and have required columns."""
    if not per_model_rankings:
        raise ValueError("per_model_rankings cannot be empty")

    for model_name, df in per_model_rankings.items():
        if "protein" not in df.columns or "final_rank" not in df.columns:
            raise ValueError(
                f"Model '{model_name}' ranking must have 'protein' and 'final_rank' columns"
            )


def _collect_all_proteins(per_model_rankings: dict[str, pd.DataFrame]) -> set:
    """Collect unique proteins across all models."""
    all_proteins = set()
    for df in per_model_rankings.values():
        all_proteins.update(df["protein"].tolist())

    if not all_proteins:
        raise ValueError("No proteins found in any model ranking")

    return all_proteins


def _build_rank_lookups(
    per_model_rankings: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Build rank lookups for efficient access.

    Returns:
        Tuple of (max_ranks, rank_lookups) where:
        - max_ranks: Dict mapping model_name -> max rank (list length)
        - rank_lookups: Dict mapping model_name -> {protein: rank}
    """
    max_ranks: dict[str, float] = {}
    rank_lookups: dict[str, dict[str, float]] = {}

    for model_name, df in per_model_rankings.items():
        max_ranks[model_name] = float(df["final_rank"].max())
        rank_lookups[model_name] = dict(zip(df["protein"], df["final_rank"], strict=False))

    return max_ranks, rank_lookups


def _compute_protein_consensus(
    protein: str,
    model_names: list[str],
    rank_lookups: dict[str, dict[str, float]],
    max_ranks: dict[str, float],
    missing_rank: float,
    n_models: int,
    method: str,
) -> dict:
    """Compute consensus metrics for a single protein across models."""
    row = {"protein": protein}
    ranks = []
    n_present = 0

    # Collect ranks from all models
    for model_name in model_names:
        rank = rank_lookups[model_name].get(protein)
        if rank is None or pd.isna(rank):
            rank = missing_rank
        else:
            rank = float(rank)
            n_present += 1

        row[f"{model_name}_rank"] = rank
        ranks.append(rank)

    row["n_models_present"] = n_present

    # Compute uncertainty metrics
    row.update(_compute_uncertainty_metrics(ranks, n_present, n_models))

    # Compute consensus score
    row["consensus_score"] = _compute_consensus_score(method, ranks, model_names, max_ranks)

    return row


def _compute_uncertainty_metrics(
    ranks: list[float], n_present: int, n_models: int
) -> dict[str, float]:
    """Compute uncertainty metrics from ranks."""
    if len(ranks) > 1:
        rank_std = float(np.std(ranks, ddof=1))
        rank_mean = float(np.mean(ranks))
        rank_cv = rank_std / rank_mean if rank_mean > 0 else 0.0
    elif len(ranks) == 1:
        rank_std = 0.0
        rank_cv = 0.0
    else:
        rank_std = np.nan
        rank_cv = np.nan

    presence_fraction = n_present / n_models

    return {
        "rank_std": rank_std,
        "rank_cv": rank_cv,
        "agreement_strength": presence_fraction,
        "presence_fraction": presence_fraction,
    }


def _compute_consensus_score(
    method: str,
    ranks: list[float],
    model_names: list[str],
    max_ranks: dict[str, float],
) -> float:
    """Compute consensus score using specified method."""
    if method == "geometric_mean":
        return _compute_geometric_mean_score(ranks, model_names, max_ranks)
    if method == "borda":
        return _compute_borda_score(ranks, model_names, max_ranks)
    if method == "median":
        return _compute_median_score(ranks, max_ranks)

    raise ValueError(f"Unsupported method: {method}")


def _compute_geometric_mean_score(
    ranks: list[float], model_names: list[str], max_ranks: dict[str, float]
) -> float:
    """Compute geometric mean of normalized reciprocal ranks."""
    normalized_reciprocals = []
    for model_name, rank in zip(model_names, ranks, strict=False):
        n_list = max_ranks[model_name]
        normalized_reciprocals.append(n_list / rank)
    return float(gmean(normalized_reciprocals))


def _compute_borda_score(
    ranks: list[float], model_names: list[str], max_ranks: dict[str, float]
) -> float:
    """Compute Borda count normalized by number of models."""
    borda_scores = []
    for model_name, rank in zip(model_names, ranks, strict=False):
        max_r = max_ranks[model_name] + 1
        borda_scores.append(max_r - rank + 1)
    return sum(borda_scores) / len(model_names)


def _compute_median_score(ranks: list[float], max_ranks: dict[str, float]) -> float:
    """Compute median-based score (inverted for consistency)."""
    median_rank = float(np.median(ranks))
    max_possible = max(max_ranks.values()) + 1
    return max_possible - median_rank


def _warn_low_presence_in_top_results(result: pd.DataFrame) -> None:
    """Log warning for proteins with low presence in top results."""
    if result.empty:
        return

    low_presence_threshold = 0.5
    top_k = min(50, len(result))
    top_proteins = result.head(top_k)
    low_presence_proteins = top_proteins[top_proteins["presence_fraction"] < low_presence_threshold]

    if len(low_presence_proteins) > 0:
        logger.warning(
            f"Found {len(low_presence_proteins)} proteins in top {top_k} with "
            f"presence_fraction < {low_presence_threshold}. "
            "These proteins are less robust across models despite penalty-based ranking. "
            "Consider filtering by presence_fraction >= 0.5."
        )
