"""Feature stability tracking across CV folds.

Tracks which features appear consistently across repeated cross-validation splits
to identify robust biomarker panels. Supports frequency-based selection and
stable panel generation from CV results.

Design:
- Pure functions operating on selection logs (DataFrames with JSON columns)
- No side effects: all outputs explicit
- Selection frequency = proportion of CV splits where feature was selected
- Stability threshold = minimum frequency for inclusion in stable panel
"""

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_selection_frequencies(
    selection_log: pd.DataFrame,
    selection_col: str = "selected_proteins_split",
) -> dict[str, float]:
    """Compute selection frequency for each protein across CV splits.

    Selection frequency = (#splits where protein appears) / (#total splits)

    Args:
        selection_log: DataFrame with columns [repeat, fold, {selection_col}]
                      {selection_col} contains JSON lists of selected proteins
        selection_col: Name of column containing JSON protein lists

    Returns:
        Dict mapping protein -> selection frequency in [0.0, 1.0]

    Example:
        >>> log = pd.DataFrame({
        ...     'repeat': [0, 0, 1, 1],
        ...     'fold': [0, 1, 0, 1],
        ...     'selected_proteins_split': [
        ...         '["PROT_A", "PROT_B"]',
        ...         '["PROT_A", "PROT_C"]',
        ...         '["PROT_A", "PROT_B"]',
        ...         '["PROT_B", "PROT_C"]'
        ...     ]
        ... })
        >>> compute_selection_frequencies(log)
        {'PROT_A': 0.75, 'PROT_B': 0.75, 'PROT_C': 0.5}
    """
    if selection_log is None or selection_log.empty or selection_col not in selection_log.columns:
        return {}

    n_splits = len(selection_log)
    if n_splits == 0:
        return {}

    counts: dict[str, int] = {}
    for json_str in selection_log[selection_col]:
        try:
            proteins = json.loads(json_str) if isinstance(json_str, str) else []
        except (json.JSONDecodeError, TypeError):
            proteins = []

        for protein in set(proteins):  # Count once per split
            counts[protein] = counts.get(protein, 0) + 1

    return {protein: count / n_splits for protein, count in counts.items()}


def extract_stable_panel(
    selection_log: pd.DataFrame,
    n_repeats: int,
    stability_threshold: float = 0.75,
    selection_col: str = "selected_proteins_split",
    fallback_top_n: int = 20,
) -> tuple[pd.DataFrame, list[str], list[set]]:
    """Extract stable panel from repeated CV by requiring proteins appear in most repeats.

    Computes per-repeat union of selected proteins, then identifies proteins
    appearing in >= stability_threshold fraction of repeats. Differs from
    compute_selection_frequencies which operates on individual splits.

    Args:
        selection_log: DataFrame with columns [repeat, fold, {selection_col}]
        n_repeats: Number of CV repeats
        stability_threshold: Minimum fraction of repeats (0.0-1.0)
        selection_col: Name of column containing JSON protein lists
        fallback_top_n: If no proteins meet threshold, keep this many top proteins

    Returns:
        (panel_df, stable_proteins, repeat_unions)
        - panel_df: DataFrame with [protein, selection_freq, kept]
        - stable_proteins: List of proteins meeting stability threshold
        - repeat_unions: List of sets, one per repeat (union of proteins in that repeat)

    Example:
        >>> log = pd.DataFrame({
        ...     'repeat': [0, 0, 1, 1, 2, 2],
        ...     'fold': [0, 1, 0, 1, 0, 1],
        ...     'selected_proteins_split': [
        ...         '["A", "B"]', '["A", "C"]',  # repeat 0: union = {A, B, C}
        ...         '["A", "B"]', '["A", "D"]',  # repeat 1: union = {A, B, D}
        ...         '["A", "B"]', '["B", "C"]',  # repeat 2: union = {A, B, C}
        ...     ]
        ... })
        >>> panel, stable, unions = extract_stable_panel(log, n_repeats=3, stability_threshold=0.67)
        >>> stable  # A and B appear in 3/3 repeats
        ['A', 'B']
        >>> len(unions)
        3
    """
    logger.info(f"Stability panel extraction (threshold={stability_threshold:.2f})")

    if selection_log is None or selection_log.empty or selection_col not in selection_log.columns:
        logger.warning("Empty selection log - no stable panel to extract")
        empty_df = pd.DataFrame(columns=["protein", "selection_freq", "kept"])
        return empty_df, [], []

    # Compute union of selected proteins for each repeat
    repeat_unions: list[set] = []
    for repeat_id in range(n_repeats):
        repeat_data = selection_log[selection_log["repeat"] == repeat_id]
        repeat_union = set()

        for json_str in repeat_data[selection_col]:
            try:
                proteins = json.loads(json_str) if isinstance(json_str, str) else []
                repeat_union.update(proteins)
            except (json.JSONDecodeError, TypeError):
                continue

        repeat_unions.append(repeat_union)
        logger.debug(f"  Repeat {repeat_id}: {len(repeat_union)} unique proteins")

    # Compute frequency = fraction of repeats where protein appears
    all_proteins = sorted(set().union(*repeat_unions)) if repeat_unions else []

    rows = []
    for protein in all_proteins:
        freq = sum(protein in union for union in repeat_unions) / n_repeats
        rows.append(
            {
                "protein": protein,
                "selection_freq": freq,
                "kept": freq >= stability_threshold,
            }
        )

    panel_df = pd.DataFrame(rows)
    if panel_df.empty:
        logger.warning("No proteins found in selection log")
        return panel_df, [], repeat_unions

    # Sort: kept first, then by frequency (desc), then by name (asc)
    panel_df = panel_df.sort_values(
        ["kept", "selection_freq", "protein"], ascending=[False, False, True]
    ).reset_index(drop=True)

    stable_proteins = panel_df[panel_df["kept"]]["protein"].tolist()

    # Fallback if no proteins meet threshold
    if len(stable_proteins) == 0:
        n_fallback = min(fallback_top_n, len(panel_df))
        stable_proteins = panel_df.nlargest(n_fallback, "selection_freq")["protein"].tolist()
        panel_df["kept"] = panel_df["protein"].isin(stable_proteins)
        logger.warning(
            f"No proteins met threshold {stability_threshold:.2f} - using top {n_fallback} by frequency"
        )
    else:
        logger.info(
            f"  {len(stable_proteins)} proteins selected ≥{stability_threshold*100:.0f}% of repeats"
        )

        freq_min = panel_df[panel_df["kept"]]["selection_freq"].min()
        freq_max = panel_df[panel_df["kept"]]["selection_freq"].max()
        freq_median = panel_df[panel_df["kept"]]["selection_freq"].median()
        logger.info(
            f"  Selection frequency distribution: min={freq_min:.2f}, median={freq_median:.2f}, max={freq_max:.2f}"
        )

    logger.info(f"  Final stable panel: {len(stable_proteins)} proteins")
    logger.info(
        "  Rationale: Stability filtering ensures reproducibility across independent data splits"
    )

    return panel_df, stable_proteins, repeat_unions


def rank_proteins_by_frequency(selection_frequencies: dict[str, float]) -> list[str]:
    """Rank proteins by selection frequency (descending), breaking ties by name.

    Args:
        selection_frequencies: Dict mapping protein -> frequency

    Returns:
        List of proteins sorted by (frequency DESC, name ASC)

    Example:
        >>> freqs = {'PROT_C': 0.70, 'PROT_A': 0.90, 'PROT_B': 0.90}
        >>> rank_proteins_by_frequency(freqs)
        ['PROT_A', 'PROT_B', 'PROT_C']
    """
    if not selection_frequencies:
        return []

    # Sort by frequency (desc), then protein name (asc)
    return sorted(selection_frequencies.keys(), key=lambda p: (-selection_frequencies[p], p))
