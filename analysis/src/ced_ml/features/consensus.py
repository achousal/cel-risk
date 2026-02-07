"""Cross-model consensus panel generation via Robust Rank Aggregation.

This module aggregates protein rankings from multiple models to create a single
consensus panel for clinical deployment. The workflow (per ADR-004):

1. Per-model ranking: OOF importance (primary) + Essentiality (secondary) +
   Stability (tie-break) combined into composite score
2. Cross-model RRA: Aggregate rankings via geometric mean of reciprocal ranks
3. Correlation clustering: Deduplicate highly correlated proteins (|r| > threshold)
4. Top-N selection: Extract final panel

Design rationale:
- Uses geometric mean RRA (penalizes proteins missing from some models)
- No external R dependencies (vs Stuart's p-value method)
- Reuses existing correlation pruning infrastructure
- Output compatible with --fixed-panel training
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gmean

from ced_ml.features.corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Results from cross-model consensus panel generation.

    Attributes:
        final_panel: List of proteins in final consensus panel (ordered by score).
        consensus_ranking: DataFrame with all proteins and consensus scores.
        per_model_rankings: DataFrame with per-model rankings for each protein.
        correlation_clusters: DataFrame with cluster assignments and pruning info.
        metadata: Dict with run parameters and statistics.
    """

    final_panel: list[str] = field(default_factory=list)
    consensus_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_model_rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_clusters: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_per_model_ranking(
    stability_df: pd.DataFrame,
    stability_col: str = "selection_fraction",
    oof_importance_df: pd.DataFrame | None = None,
    essentiality_df: pd.DataFrame | None = None,
    oof_weight: float = 0.6,
    essentiality_weight: float = 0.3,
    stability_weight: float = 0.1,
) -> pd.DataFrame:
    """Compute composite ranking for a single model.

    Uses OOF importance (primary) + Essentiality (secondary) + Stability (tie-break)
    per ADR-004 feature selection strategy.

    Args:
        stability_df: DataFrame with columns [protein, selection_fraction].
        stability_col: Column name for stability frequency.
        oof_importance_df: DataFrame with OOF grouped importance (columns: feature/protein,
            importance/mean_importance).
        essentiality_df: DataFrame with drop-column essentiality results (columns:
            cluster_id, representative/features, mean_delta_auroc).
        oof_weight: Weight for OOF importance (default 0.6).
        essentiality_weight: Weight for essentiality (default 0.3).
        stability_weight: Weight for stability (default 0.1).

    Returns:
        DataFrame with columns:
            - protein: Protein name
            - stability_freq: Selection frequency [0, 1]
            - stability_rank: Rank by stability (1 = best)
            - oof_importance: OOF importance score
            - oof_rank: Rank by OOF importance
            - essentiality: Drop-column delta AUROC
            - essentiality_rank: Rank by essentiality
            - composite_score: Weighted combination [0, 1]
            - final_rank: Rank by composite score (1 = best)
    """
    df = stability_df.copy()

    # Ensure protein column exists
    if "protein" not in df.columns:
        raise ValueError("stability_df must have 'protein' column")

    if stability_col not in df.columns:
        raise ValueError(f"stability_df must have '{stability_col}' column")

    # Rename for consistency
    df = df.rename(columns={stability_col: "stability_freq"})

    # Compute stability rank (1 = highest frequency)
    df = df.sort_values("stability_freq", ascending=False)
    df["stability_rank"] = range(1, len(df) + 1)

    # Normalize stability to [0, 1] using rank-based normalization
    n_proteins = len(df)
    df["norm_stability"] = (n_proteins - df["stability_rank"] + 1) / n_proteins

    # Initialize value columns (rank columns created during processing)
    df["oof_importance"] = np.nan
    df["essentiality"] = np.nan

    # Handle OOF importance (primary signal)
    if oof_importance_df is not None and len(oof_importance_df) > 0:
        # Detect column names (support both 'feature' and 'protein')
        oof_col = "feature" if "feature" in oof_importance_df.columns else "protein"
        imp_col = (
            "mean_importance" if "mean_importance" in oof_importance_df.columns else "importance"
        )

        oof_lookup = dict(
            zip(
                oof_importance_df[oof_col],
                oof_importance_df[imp_col],
                strict=False,
            )
        )
        df["oof_importance"] = df["protein"].map(oof_lookup)

        # Rank by OOF importance (higher = better)
        oof_valid = df[df["oof_importance"].notna()].copy()
        if len(oof_valid) > 0:
            oof_valid = oof_valid.sort_values("oof_importance", ascending=False)
            oof_rank_map = {p: r for r, p in enumerate(oof_valid["protein"], 1)}
            df["oof_rank"] = df["protein"].map(oof_rank_map)

            # Normalize OOF to [0, 1]
            n_oof = len(oof_valid)
            df["norm_oof"] = np.where(
                df["oof_rank"].notna(),
                (n_oof - df["oof_rank"] + 1) / n_oof,
                0.0,  # Missing = 0 contribution
            )
        else:
            df["oof_rank"] = np.nan
            df["norm_oof"] = 0.0
    else:
        df["oof_rank"] = np.nan
        df["norm_oof"] = 0.0

    # Handle essentiality (secondary signal)
    if essentiality_df is not None and len(essentiality_df) > 0:
        # Essentiality df has cluster_id, mean_delta_auroc
        # Map representative protein to essentiality score
        if "representative" in essentiality_df.columns:
            ess_col = "representative"
        elif "features" in essentiality_df.columns:
            # May be JSON list; take first feature
            ess_col = "features"
        else:
            ess_col = None

        if ess_col and "mean_delta_auroc" in essentiality_df.columns:
            if ess_col == "features":
                # Parse JSON and take first feature
                def parse_first_feature(x):
                    try:
                        features = json.loads(x) if isinstance(x, str) else x
                        return features[0] if features else None
                    except (json.JSONDecodeError, TypeError):
                        return x

                ess_lookup = dict(
                    zip(
                        essentiality_df[ess_col].apply(parse_first_feature),
                        essentiality_df["mean_delta_auroc"],
                        strict=False,
                    )
                )
            else:
                ess_lookup = dict(
                    zip(
                        essentiality_df[ess_col],
                        essentiality_df["mean_delta_auroc"],
                        strict=False,
                    )
                )
            df["essentiality"] = df["protein"].map(ess_lookup)

            # Rank by essentiality (higher delta = more essential)
            ess_valid = df[df["essentiality"].notna()].copy()
            if len(ess_valid) > 0:
                ess_valid = ess_valid.sort_values("essentiality", ascending=False)
                ess_rank_map = {p: r for r, p in enumerate(ess_valid["protein"], 1)}
                df["essentiality_rank"] = df["protein"].map(ess_rank_map)

                # Normalize essentiality to [0, 1]
                n_ess = len(ess_valid)
                df["norm_essentiality"] = np.where(
                    df["essentiality_rank"].notna(),
                    (n_ess - df["essentiality_rank"] + 1) / n_ess,
                    0.0,
                )
            else:
                df["essentiality_rank"] = np.nan
                df["norm_essentiality"] = 0.0
        else:
            df["essentiality_rank"] = np.nan
            df["norm_essentiality"] = 0.0
    else:
        df["essentiality_rank"] = np.nan
        df["norm_essentiality"] = 0.0

    # Compute composite score with configurable weights
    df["composite_score"] = (
        oof_weight * df["norm_oof"].fillna(0.0)
        + essentiality_weight * df["norm_essentiality"].fillna(0.0)
        + stability_weight * df["norm_stability"]
    )

    # Final rank by composite score
    df = df.sort_values("composite_score", ascending=False)
    df["final_rank"] = range(1, len(df) + 1)

    output_cols = [
        "protein",
        "stability_freq",
        "stability_rank",
        "oof_importance",
        "oof_rank",
        "essentiality",
        "essentiality_rank",
        "composite_score",
        "final_rank",
    ]

    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    return df[output_cols].copy()


def robust_rank_aggregate(
    per_model_rankings: dict[str, pd.DataFrame],
    method: str = "geometric_mean",
) -> pd.DataFrame:
    """Aggregate rankings across models using Robust Rank Aggregation.

    Args:
        per_model_rankings: Dict mapping model_name -> DataFrame with columns
            [protein, final_rank]. Each model may have different protein sets.
        method: Aggregation method:
            - "geometric_mean": Geometric mean of reciprocal ranks (default).
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
    valid_methods = {"geometric_mean", "borda", "median"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    if not per_model_rankings:
        raise ValueError("per_model_rankings cannot be empty")

    # Collect all proteins across models
    all_proteins = set()
    for df in per_model_rankings.values():
        all_proteins.update(df["protein"].tolist())

    if not all_proteins:
        raise ValueError("No proteins found in any model ranking")

    # Build rank matrix: rows = proteins, columns = models
    model_names = list(per_model_rankings.keys())
    n_models = len(model_names)

    # Get max ranks per model (for missing protein penalty)
    max_ranks = {}
    for model_name, df in per_model_rankings.items():
        max_ranks[model_name] = df["final_rank"].max()

    # Build result rows
    rows = []
    for protein in sorted(all_proteins):
        row = {"protein": protein}
        ranks = []
        ranks_present_only = []  # Ranks only from models where protein is present

        for model_name in model_names:
            df = per_model_rankings[model_name]
            protein_row = df[df["protein"] == protein]

            if len(protein_row) > 0:
                rank = float(protein_row["final_rank"].iloc[0])
                ranks_present_only.append(rank)
            else:
                # Missing protein gets global worst rank + 1 (penalty)
                # Use global max rank to ensure consistent penalty across models
                rank = max(max_ranks.values()) + 1

            row[f"{model_name}_rank"] = rank
            ranks.append(rank)

        # Count models where protein was actually present
        n_present = sum(
            1
            for model_name in model_names
            if protein in per_model_rankings[model_name]["protein"].values
        )
        row["n_models_present"] = n_present

        # Compute uncertainty metrics (using only ranks from models where protein is present)
        if len(ranks_present_only) > 1:
            rank_std = float(np.std(ranks_present_only, ddof=1))
            rank_mean = float(np.mean(ranks_present_only))
            rank_cv = rank_std / rank_mean if rank_mean > 0 else 0.0
        elif len(ranks_present_only) == 1:
            rank_std = 0.0
            rank_cv = 0.0
        else:
            rank_std = np.nan
            rank_cv = np.nan

        row["rank_std"] = rank_std
        row["rank_cv"] = rank_cv
        row["agreement_strength"] = n_present / n_models
        row["presence_fraction"] = n_present / n_models

        # Compute consensus score based on method
        if method == "geometric_mean":
            # Geometric mean of reciprocal ranks (higher = better)
            reciprocal_ranks = [1.0 / r for r in ranks]
            row["consensus_score"] = float(gmean(reciprocal_ranks))

        elif method == "borda":
            # Borda count: sum of (max_rank - rank + 1)
            # Normalized by number of models
            borda_scores = []
            for model_name, rank in zip(model_names, ranks, strict=False):
                max_r = max_ranks[model_name] + 1  # +1 for missing penalty
                borda_scores.append(max_r - rank + 1)
            row["consensus_score"] = sum(borda_scores) / n_models

        elif method == "median":
            # Median rank (lower = better, so we invert for consistency)
            median_rank = float(np.median(ranks))
            max_possible = max(max_ranks.values()) + 1
            row["consensus_score"] = max_possible - median_rank

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
    ]
    col_order += [f"{m}_rank" for m in model_names]
    result = result[col_order]

    # Log warning for proteins with low presence but appearing in top results
    if not result.empty:
        low_presence_threshold = 0.5
        top_k = min(50, len(result))
        top_proteins = result.head(top_k)
        low_presence_proteins = top_proteins[
            top_proteins["presence_fraction"] < low_presence_threshold
        ]
        if len(low_presence_proteins) > 0:
            logger.warning(
                f"Found {len(low_presence_proteins)} proteins in top {top_k} with presence_fraction < {low_presence_threshold}. "
                "These proteins may have falsely low rank_std/rank_cv due to missing from most models. "
                "Consider filtering by presence_fraction >= 0.5 for more robust consensus."
            )

    return result


def cluster_and_select_representatives(
    df_train: pd.DataFrame,
    proteins: list[str],
    consensus_scores: dict[str, float],
    corr_threshold: float = 0.85,
    corr_method: str = "spearman",
) -> tuple[pd.DataFrame, list[str]]:
    """Cluster correlated proteins and select representatives by consensus score.

    Unlike the standard corr_prune which uses selection_freq, this function
    uses consensus_score for representative selection.

    Args:
        df_train: Training data for correlation computation.
        proteins: List of proteins to cluster.
        consensus_scores: Dict mapping protein -> consensus score.
        corr_threshold: Correlation threshold for clustering.
        corr_method: Correlation method ("spearman" or "pearson").

    Returns:
        Tuple of:
            - DataFrame with cluster assignments and pruning info
            - List of kept representative proteins (sorted by consensus score)

    Important:
        Correlation matrix for clustering should be computed on training data only.
        Ensure df_train does not include validation or test samples to avoid
        information leakage into feature grouping.
    """
    logger.warning(
        "Correlation matrix for clustering should be computed on training data only. "
        "Ensure df_train does not include validation or test samples."
    )
    # Filter to valid proteins
    valid_proteins = [p for p in proteins if p in df_train.columns]
    if not valid_proteins:
        return pd.DataFrame(), []

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df_train, valid_proteins, method=corr_method)

    if corr_matrix.empty:
        # No valid correlations - return all as singletons
        rows = [
            {
                "protein": p,
                "cluster_id": i,
                "cluster_size": 1,
                "kept": True,
                "representative": p,
                "consensus_score": consensus_scores.get(p, 0.0),
            }
            for i, p in enumerate(valid_proteins, 1)
        ]
        df_clusters = pd.DataFrame(rows)
        return df_clusters, valid_proteins

    # Build correlation graph and find components
    adjacency = build_correlation_graph(corr_matrix, threshold=corr_threshold)
    components = find_connected_components(adjacency)

    # Select representative from each cluster by consensus score
    rows = []
    kept_proteins = []

    for cluster_id, component in enumerate(components, 1):
        # Find representative (highest consensus score)
        representative = max(component, key=lambda p: consensus_scores.get(p, 0.0))
        kept_proteins.append(representative)

        # Record all proteins in cluster
        for protein in component:
            rows.append(
                {
                    "protein": protein,
                    "cluster_id": cluster_id,
                    "cluster_size": len(component),
                    "kept": protein == representative,
                    "representative": representative,
                    "consensus_score": consensus_scores.get(protein, 0.0),
                    "removed_due_to_corr_with": (
                        "" if protein == representative else representative
                    ),
                }
            )

    # Create DataFrame
    df_clusters = pd.DataFrame(rows)
    df_clusters = df_clusters.sort_values(["kept", "consensus_score"], ascending=[False, False])

    # Sort kept proteins by consensus score
    kept_sorted = sorted(kept_proteins, key=lambda p: -consensus_scores.get(p, 0.0))

    return df_clusters, kept_sorted


def build_consensus_panel(
    model_stability: dict[str, pd.DataFrame],
    df_train: pd.DataFrame,
    stability_threshold: float = 0.90,
    corr_threshold: float = 0.85,
    target_size: int = 25,
    rra_method: str = "geometric_mean",
    corr_method: str = "spearman",
    model_oof_importance: dict[str, pd.DataFrame] | None = None,
    model_essentiality: dict[str, pd.DataFrame] | None = None,
    oof_weight: float = 0.6,
    essentiality_weight: float = 0.3,
    stability_weight: float = 0.1,
) -> ConsensusResult:
    """Build consensus panel from multiple models.

    Main entry point for cross-model consensus generation.

    Uses composite ranking per ADR-004: OOF importance (primary) +
    Essentiality (secondary) + Stability (tie-break).

    Args:
        model_stability: Dict mapping model_name -> DataFrame with columns
            [protein, selection_fraction]. Only proteins above stability_threshold
            are considered.
        df_train: Training data for correlation computation.
        stability_threshold: Minimum selection frequency to include protein.
        corr_threshold: Correlation threshold for clustering.
        target_size: Target panel size after pruning.
        rra_method: RRA aggregation method.
        corr_method: Correlation method for clustering.
        model_oof_importance: Dict mapping model_name -> OOF importance DataFrame
            (columns: feature/protein, importance/mean_importance).
        model_essentiality: Dict mapping model_name -> Essentiality DataFrame
            (columns: cluster_id, representative/features, mean_delta_auroc).
        oof_weight: Weight for OOF importance in composite score (default 0.6).
        essentiality_weight: Weight for essentiality in composite score (default 0.3).
        stability_weight: Weight for stability in composite score (default 0.1).

    Returns:
        ConsensusResult with final panel and intermediate data.
    """
    logger.info(f"Building consensus panel from {len(model_stability)} models")
    logger.info(
        f"Parameters: stability_threshold={stability_threshold}, "
        f"corr_threshold={corr_threshold}, target_size={target_size}, "
        f"composite_weights=(oof={oof_weight}, ess={essentiality_weight}, stab={stability_weight})"
    )

    # Step 1: Compute per-model rankings
    per_model_rankings = {}
    model_stats = {}

    for model_name, stability_df in model_stability.items():
        # Filter by stability threshold
        stable_df = stability_df[stability_df["selection_fraction"] >= stability_threshold].copy()

        if len(stable_df) == 0:
            logger.warning(
                f"Model {model_name}: No proteins meet stability threshold "
                f"{stability_threshold}"
            )
            continue

        # Get OOF importance and essentiality if available
        oof_importance_df = None
        essentiality_df = None
        has_oof = False
        has_essentiality = False

        if model_oof_importance:
            oof_importance_df = model_oof_importance.get(model_name)
            has_oof = oof_importance_df is not None and len(oof_importance_df) > 0
        if model_essentiality:
            essentiality_df = model_essentiality.get(model_name)
            has_essentiality = essentiality_df is not None and len(essentiality_df) > 0

        # Compute composite ranking
        ranking_df = compute_per_model_ranking(
            stability_df=stable_df,
            stability_col="selection_fraction",
            oof_importance_df=oof_importance_df,
            essentiality_df=essentiality_df,
            oof_weight=oof_weight,
            essentiality_weight=essentiality_weight,
            stability_weight=stability_weight,
        )

        per_model_rankings[model_name] = ranking_df

        model_stats[model_name] = {
            "n_stable_proteins": len(stable_df),
            "has_oof_importance": has_oof,
            "has_essentiality": has_essentiality,
        }

        logger.info(
            f"Model {model_name}: {len(stable_df)} stable proteins, "
            f"OOF={'yes' if has_oof else 'no'}, essentiality={'yes' if has_essentiality else 'no'}"
        )

    if len(per_model_rankings) < 2:
        raise ValueError(
            f"Need at least 2 models for consensus, got {len(per_model_rankings)}. "
            f"Check that models have aggregated stability results."
        )

    # Step 2: Cross-model RRA
    logger.info(f"Running RRA aggregation ({rra_method})...")
    consensus_df = robust_rank_aggregate(per_model_rankings, method=rra_method)

    logger.info(f"Consensus ranking: {len(consensus_df)} total proteins")

    # Degenerate consensus warning: detect identical rankings across models
    if "rank_std" in consensus_df.columns:
        zero_std_frac = (consensus_df["rank_std"] == 0.0).mean()
        if zero_std_frac > 0.9:
            logger.warning(
                "Consensus is degenerate: %.0f%% of proteins have identical rankings "
                "across all models (rank_std=0). Enable model_selector=true "
                "or use a diverse k_grid for meaningful cross-model consensus.",
                zero_std_frac * 100,
            )

    # Extract consensus scores for clustering
    consensus_scores = dict(
        zip(consensus_df["protein"], consensus_df["consensus_score"], strict=False)
    )

    # Step 3: Correlation clustering on top candidates
    # Take buffer of candidates (3x target size) for pruning
    buffer_size = min(target_size * 3, len(consensus_df))
    candidate_proteins = consensus_df.head(buffer_size)["protein"].tolist()

    logger.info(
        f"Correlation clustering on top {len(candidate_proteins)} candidates "
        f"(threshold={corr_threshold})..."
    )

    cluster_df, kept_proteins = cluster_and_select_representatives(
        df_train=df_train,
        proteins=candidate_proteins,
        consensus_scores=consensus_scores,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
    )

    n_clusters = cluster_df["cluster_id"].nunique() if not cluster_df.empty else 0
    logger.info(
        f"Clustering: {len(candidate_proteins)} -> {len(kept_proteins)} proteins "
        f"({n_clusters} clusters)"
    )

    # Step 4: Select top N
    final_panel = kept_proteins[:target_size]

    logger.info(f"Final panel: {len(final_panel)} proteins")

    # Build per-model rankings DataFrame for output
    per_model_rows = []
    for model_name, ranking_df in per_model_rankings.items():
        for _, row in ranking_df.iterrows():
            per_model_rows.append(
                {
                    "model": model_name,
                    "protein": row["protein"],
                    "stability_freq": row["stability_freq"],
                    "stability_rank": row["stability_rank"],
                    "oof_importance": row.get("oof_importance"),
                    "oof_rank": row.get("oof_rank"),
                    "essentiality": row.get("essentiality"),
                    "essentiality_rank": row.get("essentiality_rank"),
                    "composite_score": row["composite_score"],
                    "final_rank": row["final_rank"],
                }
            )

    per_model_df = pd.DataFrame(per_model_rows)

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(per_model_rankings),
        "models": list(per_model_rankings.keys()),
        "model_stats": model_stats,
        "parameters": {
            "stability_threshold": stability_threshold,
            "corr_threshold": corr_threshold,
            "target_size": target_size,
            "composite_ranking": {
                "oof_weight": oof_weight,
                "essentiality_weight": essentiality_weight,
                "stability_weight": stability_weight,
            },
            "rra_method": rra_method,
            "corr_method": corr_method,
        },
        "results": {
            "n_total_proteins": len(consensus_df),
            "n_candidate_proteins": len(candidate_proteins),
            "n_clusters": n_clusters,
            "n_kept_after_clustering": len(kept_proteins),
            "n_final_panel": len(final_panel),
        },
    }

    return ConsensusResult(
        final_panel=final_panel,
        consensus_ranking=consensus_df,
        per_model_rankings=per_model_df,
        correlation_clusters=cluster_df,
        metadata=metadata,
    )


def save_consensus_results(
    result: ConsensusResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Save consensus panel results to output directory.

    Args:
        result: ConsensusResult from build_consensus_panel.
        output_dir: Directory to save outputs.

    Returns:
        Dict mapping artifact name -> file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    # 1. Final panel (plain text, one protein per line)
    panel_txt_path = output_dir / "final_panel.txt"
    with open(panel_txt_path, "w") as f:
        for protein in result.final_panel:
            f.write(f"{protein}\n")
    paths["final_panel_txt"] = str(panel_txt_path)

    # 2. Final panel (CSV with details including uncertainty metrics)
    panel_csv_path = output_dir / "final_panel.csv"
    panel_df = pd.DataFrame({"protein": result.final_panel})
    panel_df["rank"] = range(1, len(panel_df) + 1)

    # Add all consensus metrics for final panel proteins
    for col in [
        "consensus_score",
        "n_models_present",
        "agreement_strength",
        "rank_std",
        "rank_cv",
    ]:
        if col in result.consensus_ranking.columns:
            col_map = dict(
                zip(
                    result.consensus_ranking["protein"],
                    result.consensus_ranking[col],
                    strict=False,
                )
            )
            panel_df[col] = panel_df["protein"].map(col_map)

    panel_df.to_csv(panel_csv_path, index=False)
    paths["final_panel_csv"] = str(panel_csv_path)

    # 3. Consensus ranking (all proteins with uncertainty metrics)
    ranking_path = output_dir / "consensus_ranking.csv"
    result.consensus_ranking.to_csv(ranking_path, index=False)
    paths["consensus_ranking"] = str(ranking_path)

    # 4. Per-model rankings
    per_model_path = output_dir / "per_model_rankings.csv"
    result.per_model_rankings.to_csv(per_model_path, index=False)
    paths["per_model_rankings"] = str(per_model_path)

    # 5. Correlation clusters
    if not result.correlation_clusters.empty:
        clusters_path = output_dir / "correlation_clusters.csv"
        result.correlation_clusters.to_csv(clusters_path, index=False)
        paths["correlation_clusters"] = str(clusters_path)

    # 6. Uncertainty summary (new file)
    uncertainty_summary_path = output_dir / "uncertainty_summary.csv"
    if not result.consensus_ranking.empty:
        # Extract final panel proteins with uncertainty metrics
        uncertainty_df = result.consensus_ranking[
            result.consensus_ranking["protein"].isin(result.final_panel)
        ].copy()

        # Sort by rank (same order as final_panel)
        uncertainty_df["panel_rank"] = uncertainty_df["protein"].map(
            {p: i for i, p in enumerate(result.final_panel, 1)}
        )
        uncertainty_df = uncertainty_df.sort_values("panel_rank")

        # Select columns to save, handling missing uncertainty metrics gracefully
        base_cols = ["protein", "panel_rank", "consensus_score"]
        optional_cols = [
            "n_models_present",
            "agreement_strength",
            "rank_std",
            "rank_cv",
        ]

        cols_to_save = base_cols.copy()
        # Add optional uncertainty columns if they exist
        for col in optional_cols:
            if col in uncertainty_df.columns:
                cols_to_save.append(col)

        # Add per-model ranks if available (exclude consensus_rank)
        model_rank_cols = [
            c
            for c in uncertainty_df.columns
            if c.endswith("_rank") and c not in ["consensus_rank", "panel_rank"]
        ]
        cols_to_save.extend(model_rank_cols)

        uncertainty_df[cols_to_save].to_csv(uncertainty_summary_path, index=False)
        paths["uncertainty_summary"] = str(uncertainty_summary_path)

    # 7. Metadata JSON (enhanced with uncertainty statistics)
    metadata_path = output_dir / "consensus_metadata.json"

    # Compute uncertainty statistics for final panel (if uncertainty metrics exist)
    if not result.consensus_ranking.empty:
        panel_uncertainty = result.consensus_ranking[
            result.consensus_ranking["protein"].isin(result.final_panel)
        ]

        # Check if uncertainty metrics are available
        has_uncertainty_metrics = all(
            col in panel_uncertainty.columns
            for col in ["agreement_strength", "rank_cv", "n_models_present"]
        )

        if has_uncertainty_metrics:
            uncertainty_stats = {
                "mean_agreement_strength": float(panel_uncertainty["agreement_strength"].mean()),
                "min_agreement_strength": float(panel_uncertainty["agreement_strength"].min()),
                "mean_rank_cv": float(panel_uncertainty["rank_cv"].mean()),
                "max_rank_cv": float(panel_uncertainty["rank_cv"].max()),
                "proteins_in_all_models": int(
                    (panel_uncertainty["n_models_present"] == result.metadata["n_models"]).sum()
                ),
                "proteins_in_majority_models": int(
                    (panel_uncertainty["n_models_present"] >= result.metadata["n_models"] / 2).sum()
                ),
            }

            # Add to metadata
            result.metadata["uncertainty"] = uncertainty_stats

    with open(metadata_path, "w") as f:
        json.dump(result.metadata, f, indent=2)
    paths["metadata"] = str(metadata_path)

    logger.info(f"Saved consensus results to {output_dir}")

    return paths
