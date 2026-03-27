"""Feature selection and screening utilities."""

from .corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    compute_univariate_strength,
    find_connected_components,
    find_high_correlation_pairs,
    prune_and_refill_panel,
    prune_correlated_proteins,
    refill_panel_to_target_size,
    select_component_representative,
)
from .kbest import (
    ScreeningTransformer,
    compute_f_classif_scores,
    rank_features_by_score,
)
from .screening import (
    f_statistic_screen,
    mann_whitney_screen,
    screen_proteins,
    wald_screen,
)
from .screening_cache import ScreeningCache, get_screening_cache
from .stability import (
    bootstrap_stability_selection,
    compute_selection_frequencies,
    extract_stable_panel,
    rank_proteins_by_frequency,
)

__all__ = [
    # Correlation pruning
    "build_correlation_graph",
    "compute_correlation_matrix",
    "compute_univariate_strength",
    "find_connected_components",
    "find_high_correlation_pairs",
    "prune_and_refill_panel",
    "prune_correlated_proteins",
    "refill_panel_to_target_size",
    "select_component_representative",
    # K-best selection and screening
    "ScreeningTransformer",
    "compute_f_classif_scores",
    "rank_features_by_score",
    # Screening
    "f_statistic_screen",
    "mann_whitney_screen",
    "screen_proteins",
    "wald_screen",
    # Screening cache
    "ScreeningCache",
    "get_screening_cache",
    # Stability
    "bootstrap_stability_selection",
    "compute_selection_frequencies",
    "extract_stable_panel",
    "rank_proteins_by_frequency",
]
