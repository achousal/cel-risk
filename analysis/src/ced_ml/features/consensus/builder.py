"""Main consensus panel builder and result persistence.

Implements the three-stage workflow:
1. Per-model ranking (stability filter + OOF importance)
2. Cross-model rank aggregation (geometric mean of reciprocal ranks)
3. Correlation clustering and top-N selection
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregation import geometric_mean_rank_aggregate
from .clustering import cluster_and_select_representatives
from .ranking import compute_per_model_ranking

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


def build_consensus_panel(
    model_stability: dict[str, pd.DataFrame],
    df_train: pd.DataFrame,
    stability_threshold: float = 0.90,
    corr_threshold: float = 0.85,
    target_size: int = 25,
    rra_method: str = "geometric_mean",
    corr_method: str = "spearman",
    model_oof_importance: dict[str, pd.DataFrame] | None = None,
) -> ConsensusResult:
    """Build consensus panel from multiple models.

    Main entry point for cross-model consensus generation.

    Workflow:
        1. Filter proteins by stability threshold per model
        2. Rank survivors by OOF importance (or stability if OOF unavailable)
        3. Aggregate per-model ranks via geometric mean reciprocal ranks
        4. Correlation-cluster top candidates, select representatives
        5. Return top-N panel

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

    Returns:
        ConsensusResult with final panel and intermediate data.
    """
    logger.info(f"Building consensus panel from {len(model_stability)} models")
    logger.info(
        f"Parameters: stability_threshold={stability_threshold}, "
        f"corr_threshold={corr_threshold}, target_size={target_size}, "
        f"ranking=oof_importance_with_stability_filter"
    )

    # Step 1: Compute per-model rankings (stability filter + OOF importance)
    per_model_rankings, model_stats = _compute_all_model_rankings(
        model_stability=model_stability,
        stability_threshold=stability_threshold,
        model_oof_importance=model_oof_importance,
    )

    if len(per_model_rankings) < 2:
        raise ValueError(
            f"Need at least 2 models for consensus, got {len(per_model_rankings)}. "
            "Check that models have aggregated stability results."
        )

    # Step 2: Cross-model rank aggregation
    logger.info(f"Running rank aggregation method={rra_method}...")
    consensus_df = geometric_mean_rank_aggregate(per_model_rankings, method=rra_method)
    logger.info(f"Consensus ranking: {len(consensus_df)} total proteins")

    _check_degenerate_consensus(consensus_df)

    # Extract consensus scores for clustering
    consensus_scores = dict(
        zip(consensus_df["protein"], consensus_df["consensus_score"], strict=False)
    )

    # Step 3: Correlation clustering on top candidates
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
    per_model_df = _build_per_model_dataframe(per_model_rankings)

    # Build metadata
    metadata = _build_metadata(
        per_model_rankings=per_model_rankings,
        model_stats=model_stats,
        consensus_df=consensus_df,
        candidate_proteins=candidate_proteins,
        n_clusters=n_clusters,
        kept_proteins=kept_proteins,
        final_panel=final_panel,
        stability_threshold=stability_threshold,
        corr_threshold=corr_threshold,
        target_size=target_size,
        rra_method=rra_method,
        corr_method=corr_method,
    )

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

    # Save all artifacts
    paths["final_panel_txt"] = _save_panel_txt(result, output_dir)
    paths["final_panel_csv"] = _save_panel_csv(result, output_dir)
    paths["consensus_ranking"] = _save_consensus_ranking(result, output_dir)
    paths["per_model_rankings"] = _save_per_model_rankings(result, output_dir)

    if not result.correlation_clusters.empty:
        paths["correlation_clusters"] = _save_correlation_clusters(result, output_dir)

    paths["uncertainty_summary"] = _save_uncertainty_summary(result, output_dir)
    paths["metadata"] = _save_metadata(result, output_dir)

    logger.info(f"Saved consensus results to {output_dir}")

    return paths


# Private helper functions


def _compute_all_model_rankings(
    model_stability: dict[str, pd.DataFrame],
    stability_threshold: float,
    model_oof_importance: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Compute per-model rankings for all models.

    For each model: filter by stability threshold, then rank by OOF importance.
    """
    per_model_rankings = {}
    model_stats = {}

    for model_name, stability_df in model_stability.items():
        # Filter by stability threshold
        stable_df = stability_df[stability_df["selection_fraction"] >= stability_threshold].copy()

        if len(stable_df) == 0:
            logger.warning(
                f"Model {model_name}: No proteins meet stability threshold {stability_threshold}"
            )
            continue

        # Get OOF importance if available
        oof_importance_df = None
        has_oof = False

        if model_oof_importance:
            oof_importance_df = model_oof_importance.get(model_name)
            has_oof = oof_importance_df is not None and len(oof_importance_df) > 0

        # Rank by OOF importance (stability already filtered)
        ranking_df = compute_per_model_ranking(
            stability_df=stable_df,
            stability_col="selection_fraction",
            oof_importance_df=oof_importance_df,
        )

        per_model_rankings[model_name] = ranking_df
        model_stats[model_name] = {
            "n_stable_proteins": len(stable_df),
            "has_oof_importance": has_oof,
        }

        logger.info(
            f"Model {model_name}: {len(stable_df)} stable proteins, "
            f"OOF={'yes' if has_oof else 'no'}"
        )

    return per_model_rankings, model_stats


def _check_degenerate_consensus(consensus_df: pd.DataFrame) -> None:
    """Warn if consensus is degenerate (identical rankings across models)."""
    if "rank_std" not in consensus_df.columns:
        return

    zero_std_frac = (consensus_df["rank_std"] == 0.0).mean()
    if zero_std_frac > 0.9:
        logger.warning(
            "Consensus is degenerate: %.0f%% of proteins have identical rankings "
            "across all models (rank_std=0). Enable model_selector=true "
            "or use a diverse k_grid for meaningful cross-model consensus.",
            zero_std_frac * 100,
        )


def _build_per_model_dataframe(per_model_rankings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build per-model rankings DataFrame for output."""
    rows = []
    for model_name, ranking_df in per_model_rankings.items():
        for _, row in ranking_df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "protein": row["protein"],
                    "stability_freq": row.get("stability_freq"),
                    "oof_importance": row.get("oof_importance"),
                    "oof_rank": row.get("oof_rank"),
                    "final_rank": row["final_rank"],
                }
            )
    return pd.DataFrame(rows)


def _build_metadata(
    per_model_rankings: dict[str, pd.DataFrame],
    model_stats: dict,
    consensus_df: pd.DataFrame,
    candidate_proteins: list[str],
    n_clusters: int,
    kept_proteins: list[str],
    final_panel: list[str],
    stability_threshold: float,
    corr_threshold: float,
    target_size: int,
    rra_method: str,
    corr_method: str,
) -> dict:
    """Build metadata dictionary."""
    return {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(per_model_rankings),
        "models": list(per_model_rankings.keys()),
        "model_stats": model_stats,
        "parameters": {
            "stability_threshold": stability_threshold,
            "corr_threshold": corr_threshold,
            "target_size": target_size,
            "ranking_method": "oof_importance_with_stability_filter",
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


def _save_panel_txt(result: ConsensusResult, output_dir: Path) -> str:
    """Save final panel as plain text (one protein per line)."""
    path = output_dir / "final_panel.txt"
    with open(path, "w") as f:
        for protein in result.final_panel:
            f.write(f"{protein}\n")
    return str(path)


def _save_panel_csv(result: ConsensusResult, output_dir: Path) -> str:
    """Save final panel as CSV with details."""
    path = output_dir / "final_panel.csv"
    panel_df = pd.DataFrame({"protein": result.final_panel})
    panel_df["rank"] = range(1, len(panel_df) + 1)

    # Add consensus metrics
    for col in ["consensus_score", "n_models_present", "agreement_strength", "rank_std", "rank_cv"]:
        if col in result.consensus_ranking.columns:
            col_map = dict(
                zip(
                    result.consensus_ranking["protein"],
                    result.consensus_ranking[col],
                    strict=False,
                )
            )
            panel_df[col] = panel_df["protein"].map(col_map)

    panel_df.to_csv(path, index=False)
    return str(path)


def _save_consensus_ranking(result: ConsensusResult, output_dir: Path) -> str:
    """Save consensus ranking for all proteins."""
    path = output_dir / "consensus_ranking.csv"
    result.consensus_ranking.to_csv(path, index=False)
    return str(path)


def _save_per_model_rankings(result: ConsensusResult, output_dir: Path) -> str:
    """Save per-model rankings."""
    path = output_dir / "per_model_rankings.csv"
    result.per_model_rankings.to_csv(path, index=False)
    return str(path)


def _save_correlation_clusters(result: ConsensusResult, output_dir: Path) -> str:
    """Save correlation clusters."""
    path = output_dir / "correlation_clusters.csv"
    result.correlation_clusters.to_csv(path, index=False)
    return str(path)


def _save_uncertainty_summary(result: ConsensusResult, output_dir: Path) -> str:
    """Save uncertainty summary for final panel."""
    path = output_dir / "uncertainty_summary.csv"

    if result.consensus_ranking.empty:
        return str(path)

    # Extract final panel proteins with uncertainty metrics
    uncertainty_df = result.consensus_ranking[
        result.consensus_ranking["protein"].isin(result.final_panel)
    ].copy()

    # Sort by rank
    uncertainty_df["panel_rank"] = uncertainty_df["protein"].map(
        {p: i for i, p in enumerate(result.final_panel, 1)}
    )
    uncertainty_df = uncertainty_df.sort_values("panel_rank")

    # Select columns to save
    base_cols = ["protein", "panel_rank", "consensus_score"]
    optional_cols = ["n_models_present", "agreement_strength", "rank_std", "rank_cv"]

    cols_to_save = base_cols.copy()
    for col in optional_cols:
        if col in uncertainty_df.columns:
            cols_to_save.append(col)

    # Add per-model ranks
    model_rank_cols = [
        c
        for c in uncertainty_df.columns
        if c.endswith("_rank") and c not in ["consensus_rank", "panel_rank"]
    ]
    cols_to_save.extend(model_rank_cols)

    uncertainty_df[cols_to_save].to_csv(path, index=False)
    return str(path)


def _save_metadata(result: ConsensusResult, output_dir: Path) -> str:
    """Save metadata JSON with uncertainty statistics."""
    path = output_dir / "consensus_metadata.json"

    # Compute uncertainty statistics for final panel
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
            result.metadata["uncertainty"] = uncertainty_stats

    with open(path, "w") as f:
        json.dump(result.metadata, f, indent=2)
    return str(path)
