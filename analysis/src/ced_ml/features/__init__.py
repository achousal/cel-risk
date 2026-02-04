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
from .panels import (
    build_multi_size_panels,
)
from .screening import (
    f_statistic_screen,
    mann_whitney_screen,
    screen_proteins,
)
from .screening_cache import ScreeningCache, get_screening_cache
from .stability import (
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
    # Panels
    "build_multi_size_panels",
    # Screening
    "f_statistic_screen",
    "mann_whitney_screen",
    "screen_proteins",
    # Screening cache
    "ScreeningCache",
    "get_screening_cache",
    # Stability
    "compute_selection_frequencies",
    "extract_stable_panel",
    "rank_proteins_by_frequency",
]
