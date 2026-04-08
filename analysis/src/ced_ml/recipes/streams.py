"""Stream discovery via correlation clustering and round-robin interleave.

Reuses existing correlation infrastructure from ced_ml.features.corr_prune.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from ced_ml.features.corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)

logger = logging.getLogger(__name__)


def discover_streams(
    data_df: pd.DataFrame,
    protein_list: list[str],
    *,
    threshold: float = 0.50,
    method: Literal["pearson", "spearman"] = "spearman",
) -> list[list[str]]:
    """Discover co-regulated protein streams via correlation clustering.

    Uses a lower threshold than dedup (0.50 vs 0.85) to find co-regulated
    groups rather than near-duplicate features.

    Parameters
    ----------
    data_df : pd.DataFrame
        Training data matrix with protein columns.
    protein_list : list[str]
        Proteins to cluster (must be columns in data_df).
    threshold : float
        |Spearman rho| threshold for defining stream membership.
    method : {"pearson", "spearman"}
        Correlation method.

    Returns
    -------
    list[list[str]]
        List of streams (connected components), each a sorted list of proteins.
        Streams are sorted by size descending, then alphabetically.
    """
    valid_proteins = [p for p in protein_list if p in data_df.columns]
    if not valid_proteins:
        logger.warning("No valid proteins found in data_df columns")
        return []

    corr_matrix = compute_correlation_matrix(data_df, valid_proteins, method=method)
    if corr_matrix.empty:
        return [[p] for p in valid_proteins]

    adjacency = build_correlation_graph(corr_matrix, threshold=threshold)
    components = find_connected_components(adjacency)

    # Sort streams: larger first, then alphabetically by first member
    components.sort(key=lambda c: (-len(c), c[0]))

    logger.info(
        "Discovered %d streams from %d proteins (threshold=%.2f)",
        len(components),
        len(valid_proteins),
        threshold,
    )
    return components


def interleave_round_robin(
    streams: list[list[str]],
    scores: dict[str, float],
) -> list[str]:
    """Round-robin interleave proteins across streams.

    Within each stream, proteins are ranked by score descending.
    Streams are sorted by their max score descending.
    Round-robin picks one protein per stream per round.

    Parameters
    ----------
    streams : list[list[str]]
        Protein streams from discover_streams().
    scores : dict[str, float]
        Score for each protein (e.g. consensus_score / observed_rra).
        Higher = more important.

    Returns
    -------
    list[str]
        Interleaved protein ordering.
    """
    if not streams:
        return []

    # Sort proteins within each stream by score descending
    ranked_streams = []
    for stream in streams:
        ranked = sorted(stream, key=lambda p: -scores.get(p, 0.0))
        ranked_streams.append(ranked)

    # Sort streams by max score descending
    ranked_streams.sort(key=lambda s: -scores.get(s[0], 0.0) if s else 0.0)

    # Round-robin interleave
    result: list[str] = []
    max_len = max(len(s) for s in ranked_streams)

    for round_idx in range(max_len):
        for stream in ranked_streams:
            if round_idx < len(stream):
                result.append(stream[round_idx])

    return result
