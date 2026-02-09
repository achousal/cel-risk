"""Cross-model consensus panel generation via geometric mean rank aggregation.

This package aggregates protein rankings from multiple models to create a single
consensus panel for clinical deployment. Three-step workflow:

1. Per-model ranking: Stability acts as a hard filter (proteins must meet
   the stability threshold). Survivors are ranked by OOF importance (descending).
   If OOF importance is unavailable, falls back to stability frequency ranking.
2. Cross-model aggregation: Aggregate per-model OOF importance ranks via
   geometric mean of normalized reciprocal ranks.
3. Correlation clustering: Deduplicate highly correlated proteins, select
   representatives by consensus score, and extract top-N panel.

Post-hoc interpretation (not part of ranking):
   Drop-column essentiality validation on the final consensus panel computes
   delta-AUROC (primary), delta-PR-AUC, and delta-Brier per cluster.

Design rationale:
- Uses geometric mean of reciprocal ranks (NOT formal Kolde RRA with beta-model
  p-values; see ADR-004 for rationale)
- Ranks are normalized by list length to ensure comparability across models with
  different numbers of proteins
- No external R dependencies
- Reuses existing correlation pruning infrastructure
- Output compatible with --fixed-panel training
"""

from .aggregation import geometric_mean_rank_aggregate
from .builder import ConsensusResult, build_consensus_panel, save_consensus_results
from .clustering import cluster_and_select_representatives
from .ranking import compute_per_model_ranking

__all__ = [
    "ConsensusResult",
    "build_consensus_panel",
    "cluster_and_select_representatives",
    "compute_per_model_ranking",
    "geometric_mean_rank_aggregate",
    "save_consensus_results",
]
