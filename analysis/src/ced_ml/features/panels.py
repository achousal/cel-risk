"""Biomarker panel building with correlation pruning.

Constructs clinical biomarker panels from feature selection results by:
1. Building raw panels from selection frequencies (topN or frequency threshold)
2. Pruning highly correlated proteins using connected components
3. Refilling to target size from ranked candidate pool

Design:
- Pure functions operating on DataFrames and dictionaries
- No side effects: all outputs explicit
- Supports multiple panel sizes and correlation thresholds
- Graph-based correlation component detection
"""

from typing import Literal

import numpy as np
import pandas as pd

from .corr_prune import prune_and_refill_panel

__all__ = [
    "build_multi_size_panels",
]


def build_multi_size_panels(
    df: pd.DataFrame,
    y: np.ndarray | None,
    selection_freq: dict[str, float],
    panel_sizes: list[int],
    corr_threshold: float = 0.80,
    pool_limit: int = 3000,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    tiebreak_method: Literal["freq", "freq_then_univariate"] = "freq",
) -> dict[int, tuple[pd.DataFrame, list[str]]]:
    """Build multiple panels of different sizes with correlation pruning.

    Convenience wrapper around prune_and_refill_panel for building
    nested panels (e.g., 10, 25, 50, 100, 200 proteins).

    Args:
        df: DataFrame containing protein columns (typically TRAIN set)
        y: Binary outcome array (required if tiebreak_method="freq_then_univariate")
        selection_freq: Dict mapping protein -> selection frequency
        panel_sizes: List of target panel sizes (e.g., [10, 25, 50, 100])
        corr_threshold: Correlation threshold for pruning
        pool_limit: Maximum number of candidates to consider (default: 3000)
        corr_method: Correlation method ("pearson" or "spearman")
        tiebreak_method: How to select representative ("freq" or "freq_then_univariate")

    Returns:
        Dict mapping panel_size -> (component_map, final_panel)

    Example:
        >>> df = pd.DataFrame(...)  # TRAIN set
        >>> freqs = {'A': 0.9, 'B': 0.8, ...}  # 100 proteins
        >>> panels = build_multi_size_panels(
        ...     df, y, freqs, panel_sizes=[10, 25, 50],
        ...     corr_threshold=0.80
        ... )
        >>> panels[10][1]  # 10-protein panel
        ['A', 'C', 'D', ...]
        >>> panels[50][1]  # 50-protein panel
        ['A', 'C', 'D', ..., 'Z']
    """
    # Rank proteins by selection frequency
    ranked = sorted(selection_freq.keys(), key=lambda p: (-selection_freq[p], p))

    results = {}
    for size in sorted(panel_sizes):
        component_map, panel = prune_and_refill_panel(
            df=df,
            y=y,
            ranked_proteins=ranked,
            selection_freq=selection_freq,
            target_size=size,
            corr_threshold=corr_threshold,
            pool_limit=pool_limit,
            corr_method=corr_method,
            tiebreak_method=tiebreak_method,
        )
        results[size] = (component_map, panel)

    return results
