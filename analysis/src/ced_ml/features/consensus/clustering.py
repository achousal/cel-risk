"""Correlation-based clustering and representative selection for consensus panels.

Deduplicates highly correlated proteins using consensus score for selection.
"""

import logging

import pandas as pd

from ced_ml.features.corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)

logger = logging.getLogger(__name__)


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
    logger.debug(
        "Correlation matrix for clustering should be computed on training data only. "
        "Ensure df_train does not include validation or test samples."
    )

    valid_proteins = [p for p in proteins if p in df_train.columns]
    if not valid_proteins:
        return pd.DataFrame(), []

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df_train, valid_proteins, method=corr_method)

    if corr_matrix.empty:
        # No valid correlations - return all as singletons
        return _create_singleton_clusters(valid_proteins, consensus_scores)

    # Build correlation graph and find components
    adjacency = build_correlation_graph(corr_matrix, threshold=corr_threshold)
    components = find_connected_components(adjacency)

    # Select representative from each cluster by consensus score
    return _select_cluster_representatives(components, consensus_scores)


def _create_singleton_clusters(
    proteins: list[str], consensus_scores: dict[str, float]
) -> tuple[pd.DataFrame, list[str]]:
    """Create singleton clusters when no correlations are found."""
    rows = [
        {
            "protein": p,
            "cluster_id": i,
            "cluster_size": 1,
            "kept": True,
            "representative": p,
            "consensus_score": consensus_scores.get(p, 0.0),
        }
        for i, p in enumerate(proteins, 1)
    ]
    df_clusters = pd.DataFrame(rows)
    return df_clusters, proteins


def _select_cluster_representatives(
    components: list[list[str]], consensus_scores: dict[str, float]
) -> tuple[pd.DataFrame, list[str]]:
    """Select representative from each cluster by consensus score."""
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
                    "removed_due_to_corr_with": "" if protein == representative else representative,
                }
            )

    # Create DataFrame
    df_clusters = pd.DataFrame(rows)
    df_clusters = df_clusters.sort_values(["kept", "consensus_score"], ascending=[False, False])

    # Sort kept proteins by consensus score
    kept_sorted = sorted(kept_proteins, key=lambda p: -consensus_scores.get(p, 0.0))

    return df_clusters, kept_sorted
