"""Correlation-based feature pruning.

This module implements correlation-based redundancy removal for feature panels:
- Identify highly correlated protein pairs
- Group proteins into connected components via correlation threshold
- Select one representative protein per component (by selection frequency or univariate strength)
- Refill pruned panels to target size while maintaining low inter-feature correlation

All correlation analysis is performed on TRAIN data only to prevent test leakage.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    df: pd.DataFrame,
    proteins: list[str],
    method: Literal["pearson", "spearman"] = "spearman",
) -> pd.DataFrame:
    """Compute correlation matrix for protein features.

    Parameters
    ----------
    df : pd.DataFrame
        Training data containing protein columns
    proteins : List[str]
        List of protein column names to analyze
    method : {"pearson", "spearman"}
        Correlation method

    Returns
    -------
    pd.DataFrame
        Absolute correlation matrix (proteins x proteins)

    Notes
    -----
    - Missing values are median-imputed before correlation computation
    - Returns absolute correlations for redundancy detection
    - Only valid proteins (present in df) are included
    """
    valid_prots = [p for p in proteins if p in df.columns]
    if len(valid_prots) == 0:
        return pd.DataFrame()

    # Extract and clean data
    X = df[valid_prots].apply(pd.to_numeric, errors="coerce")

    # Median imputation for missing values
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0, skipna=True))

    # Compute correlation
    method_clean = (method or "pearson").strip().lower()
    if method_clean not in ("pearson", "spearman"):
        method_clean = "pearson"

    corr_matrix = X.corr(method=method_clean).abs()

    return corr_matrix


def find_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Identify protein pairs with correlation above threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Absolute correlation matrix (from compute_correlation_matrix)
    threshold : float
        Correlation threshold for identifying redundant pairs (0.0-1.0)

    Returns
    -------
    pd.DataFrame
        Columns: protein1, protein2, abs_corr
        Sorted by abs_corr descending
        Empty if no pairs exceed threshold

    Notes
    -----
    Only reports each pair once (protein1 < protein2 alphabetically).
    NaN correlations are excluded.
    """
    if corr_matrix.empty:
        return pd.DataFrame(columns=["protein1", "protein2", "abs_corr"])

    proteins = corr_matrix.index.tolist()
    rows = []

    for i in range(len(proteins)):
        for j in range(i + 1, len(proteins)):
            corr_val = float(corr_matrix.iloc[i, j])
            if np.isfinite(corr_val) and corr_val >= float(threshold):
                rows.append(
                    {
                        "protein1": proteins[i],
                        "protein2": proteins[j],
                        "abs_corr": corr_val,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["protein1", "protein2", "abs_corr"])

    result = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    return result


def build_correlation_graph(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.80,
) -> dict[str, set[str]]:
    """Build adjacency graph of highly correlated proteins.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Absolute correlation matrix
    threshold : float
        Correlation threshold for creating edges

    Returns
    -------
    Dict[str, Set[str]]
        Adjacency list: {protein: set of correlated proteins}

    Notes
    -----
    Graph is undirected: if A correlates with B, both adj[A] and adj[B] are updated.
    """
    if corr_matrix.empty:
        return {}

    proteins = corr_matrix.index.tolist()
    adj = {p: set() for p in proteins}

    for i, p1 in enumerate(proteins):
        for j in range(i + 1, len(proteins)):
            p2 = proteins[j]
            corr_val = float(corr_matrix.loc[p1, p2])
            if corr_val >= float(threshold):
                adj[p1].add(p2)
                adj[p2].add(p1)

    return adj


def find_connected_components(
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    """Find connected components in correlation graph via DFS.

    Parameters
    ----------
    adjacency : Dict[str, Set[str]]
        Adjacency list from build_correlation_graph

    Returns
    -------
    List[List[str]]
        List of components, each component is a sorted list of proteins

    Notes
    -----
    Isolated proteins (no high correlations) form singleton components.
    Components are sorted internally for reproducibility.
    """
    proteins = list(adjacency.keys())
    seen = set()
    components = []

    for protein in proteins:
        if protein in seen:
            continue

        # DFS to find component
        stack = [protein]
        component = []
        seen.add(protein)

        while stack:
            curr = stack.pop()
            component.append(curr)
            for neighbor in adjacency[curr]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        components.append(sorted(component))

    return components


def compute_univariate_strength(
    df: pd.DataFrame,
    y: np.ndarray,
    proteins: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute univariate strength for proteins using Mann-Whitney U test.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    y : np.ndarray
        Binary labels (0=control, 1=case)
    proteins : List[str]
        Proteins to analyze

    Returns
    -------
    Dict[str, Tuple[float, float]]
        {protein: (p_value, abs_mean_delta)}
        Lower p-value indicates stronger univariate signal

    Notes
    -----
    Proteins with insufficient data (< 30 samples, < 5 per class) are skipped.
    Used for tie-breaking when selection frequencies are equal.
    """
    y_clean = np.asarray(y).astype(int)
    result = {}

    for protein in proteins:
        if protein not in df.columns:
            continue

        # Extract and clean data
        x = pd.to_numeric(df[protein], errors="coerce")
        valid_mask = x.notna().to_numpy()

        # Require sufficient data
        if valid_mask.sum() < 30:
            continue
        if len(np.unique(y_clean[valid_mask])) < 2:
            continue

        x_valid = x[valid_mask].to_numpy(dtype=float)
        y_valid = y_clean[valid_mask]

        x_control = x_valid[y_valid == 0]
        x_case = x_valid[y_valid == 1]

        if len(x_control) < 5 or len(x_case) < 5:
            continue

        # Mann-Whitney U test
        try:
            try:
                _, p_val = stats.mannwhitneyu(
                    x_case, x_control, alternative="two-sided", method="asymptotic"
                )
            except TypeError:
                # Older scipy version
                _, p_val = stats.mannwhitneyu(x_case, x_control, alternative="two-sided")
            p_val = float(p_val)
        except Exception:
            logger.warning(
                f"Mann-Whitney U test failed for protein '{protein}'; "
                f"setting p_val=nan (n_case={len(x_case)}, n_control={len(x_control)})",
                exc_info=True,
            )
            p_val = np.nan

        # Mean difference
        mean_delta = float(np.nanmean(x_case) - np.nanmean(x_control))

        result[protein] = (p_val, abs(mean_delta))

    return result


def select_component_representative(
    component: list[str],
    selection_freq: dict[str, float] | None,
    tiebreak_method: Literal["freq", "freq_then_univariate"],
    univariate_strength: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Select one representative protein from a correlated component.

    Parameters
    ----------
    component : List[str]
        List of correlated proteins
    selection_freq : Optional[Dict[str, float]]
        Selection frequency from CV folds {protein: frequency}
    tiebreak_method : {"freq", "freq_then_univariate"}
        Method for selecting representative:
        - "freq": highest selection frequency (default)
        - "freq_then_univariate": use MW p-value for frequency ties
    univariate_strength : Optional[Dict[str, Tuple[float, float]]]
        Univariate statistics from compute_univariate_strength
        Required if tiebreak_method="freq_then_univariate"

    Returns
    -------
    str
        Selected representative protein

    Notes
    -----
    Selection priority:
    1. Highest selection frequency (primary)
    2. If freq tied: lowest MW p-value (stronger univariate signal)
    3. If p-value tied: largest absolute mean difference
    4. If all tied: alphabetical order (reproducible)
    """

    def sort_key(protein: str) -> tuple:
        # Primary: higher selection frequency
        freq = selection_freq.get(protein, np.nan) if selection_freq else np.nan
        freq_clean = freq if np.isfinite(freq) else 0.0
        k1 = -freq_clean  # Negative for descending sort

        # Secondary: univariate strength (if enabled)
        if tiebreak_method == "freq_then_univariate" and univariate_strength:
            p_val, effect_size = univariate_strength.get(protein, (np.nan, np.nan))
            p_val_clean = p_val if np.isfinite(p_val) else 1.0
            delta_clean = effect_size if np.isfinite(effect_size) else 0.0
            k2 = p_val_clean  # Ascending (lower p is better)
            k3 = -delta_clean  # Descending (larger effect is better)
            return (k1, k2, k3, protein)  # Alphabetical tie-break

        return (k1, protein)  # Alphabetical tie-break

    return sorted(component, key=sort_key)[0]


def prune_correlated_proteins(
    df: pd.DataFrame,
    y: np.ndarray | None,
    proteins: list[str],
    selection_freq: dict[str, float] | None,
    corr_threshold: float = 0.80,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    tiebreak_method: Literal["freq", "freq_then_univariate"] = "freq",
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse highly correlated proteins into components and select representatives.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    y : Optional[np.ndarray]
        Binary labels (required if tiebreak_method="freq_then_univariate")
    proteins : List[str]
        Proteins to prune
    selection_freq : Optional[Dict[str, float]]
        Selection frequency from CV {protein: freq}
    corr_threshold : float
        Absolute correlation threshold (0.0-1.0)
    corr_method : {"pearson", "spearman"}
        Correlation method
    tiebreak_method : {"freq", "freq_then_univariate"}
        Representative selection method

    Returns
    -------
    df_map : pd.DataFrame
        Mapping table with columns:
        - component_id: int component identifier
        - protein: str protein name
        - selection_freq: float selection frequency
        - kept: bool whether protein was kept as representative
        - rep_protein: str representative for this component
        - component_size: int number of proteins in component
    kept : List[str]
        List of kept representative proteins

    Notes
    -----
    All correlation analysis uses TRAIN data only.
    Empty input returns empty results.
    """
    valid_prots = [p for p in proteins if p in df.columns]
    if len(valid_prots) == 0:
        empty_df = pd.DataFrame(
            columns=[
                "component_id",
                "protein",
                "selection_freq",
                "kept",
                "rep_protein",
                "component_size",
            ]
        )
        return empty_df, []

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df, valid_prots, method=corr_method)

    # Build correlation graph and find components
    adjacency = build_correlation_graph(corr_matrix, threshold=corr_threshold)
    components = find_connected_components(adjacency)

    # Compute univariate strength if needed
    univariate_strength = None
    if tiebreak_method == "freq_then_univariate" and y is not None:
        univariate_strength = compute_univariate_strength(df, y, valid_prots)

    # Select representatives
    rows = []
    kept_proteins = []

    for comp_id, component in enumerate(components, start=1):
        representative = select_component_representative(
            component=component,
            selection_freq=selection_freq,
            tiebreak_method=tiebreak_method,
            univariate_strength=univariate_strength,
        )
        kept_proteins.append(representative)

        # Build mapping rows
        for protein in component:
            freq = selection_freq.get(protein, np.nan) if selection_freq else np.nan
            rows.append(
                {
                    "component_id": comp_id,
                    "protein": protein,
                    "selection_freq": freq,
                    "kept": (protein == representative),
                    "rep_protein": representative,
                    "component_size": len(component),
                }
            )

    # Create mapping DataFrame
    df_map = pd.DataFrame(rows).sort_values(
        ["kept", "selection_freq", "protein"],
        ascending=[False, False, True],
        na_position="last",
    )

    # Sort kept proteins by selection frequency
    kept_sorted = sorted(
        set(kept_proteins),
        key=lambda p: (-(selection_freq.get(p, 0.0) if selection_freq else 0.0), p),
    )

    return df_map, kept_sorted


def refill_panel_to_target_size(
    df: pd.DataFrame,
    kept_proteins: list[str],
    ranked_candidates: list[str],
    target_size: int,
    corr_threshold: float = 0.80,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    pool_limit: int = 3000,
) -> list[str]:
    """Refill pruned panel to target size while avoiding high correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    kept_proteins : List[str]
        Proteins already kept after initial pruning
    ranked_candidates : List[str]
        Ranked list of candidate proteins (by frequency/importance)
    target_size : int
        Desired final panel size
    corr_threshold : float
        Correlation threshold for exclusion
    corr_method : {"pearson", "spearman"}
        Correlation method
    pool_limit : int
        Maximum number of candidates to consider

    Returns
    -------
    List[str]
        Final panel (kept + refilled proteins)

    Notes
    -----
    Iterates through ranked candidates and adds proteins that:
    1. Are not already in kept_proteins
    2. Have correlation < threshold with all kept proteins

    Stops when target_size is reached or candidates exhausted.
    """
    if len(kept_proteins) >= target_size:
        return kept_proteins[:target_size]

    # Build candidate pool
    valid_candidates = [p for p in ranked_candidates if p in df.columns]
    pool = valid_candidates[: min(pool_limit, len(valid_candidates))]

    if not pool:
        return kept_proteins

    # Compute correlation matrix for pool
    corr_matrix = compute_correlation_matrix(df, pool, method=corr_method)
    if corr_matrix.empty:
        return kept_proteins

    # Fill to target size
    final_panel = list(kept_proteins)
    kept_set = set(final_panel)

    for candidate in pool:
        if len(final_panel) >= target_size:
            break

        if candidate in kept_set:
            continue

        # Check correlation with all kept proteins
        too_correlated = False
        for kept in final_panel:
            if candidate in corr_matrix.index and kept in corr_matrix.columns:
                corr_val = float(corr_matrix.loc[candidate, kept])
                if corr_val >= corr_threshold:
                    too_correlated = True
                    break

        if not too_correlated:
            final_panel.append(candidate)
            kept_set.add(candidate)

    return final_panel


def prune_and_refill_panel(
    df: pd.DataFrame,
    y: np.ndarray | None,
    ranked_proteins: list[str],
    selection_freq: dict[str, float],
    target_size: int,
    corr_threshold: float = 0.80,
    corr_method: Literal["pearson", "spearman"] = "spearman",
    tiebreak_method: Literal["freq", "freq_then_univariate"] = "freq",
    pool_limit: int = 3000,
) -> tuple[pd.DataFrame, list[str]]:
    """Build correlation-pruned panel of target size.

    This is the main entry point combining pruning and refilling:
    1. Take top-N proteins from ranked list
    2. Prune by correlation (keep representatives)
    3. Refill to target size from ranked list (avoiding correlations)

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    y : Optional[np.ndarray]
        Binary labels (for univariate tie-breaking)
    ranked_proteins : List[str]
        Proteins ranked by importance/frequency
    selection_freq : Dict[str, float]
        Selection frequency from CV
    target_size : int
        Desired final panel size
    corr_threshold : float
        Correlation threshold
    corr_method : {"pearson", "spearman"}
        Correlation method
    tiebreak_method : {"freq", "freq_then_univariate"}
        Representative selection method
    pool_limit : int
        Maximum candidates for refilling

    Returns
    -------
    df_map : pd.DataFrame
        Component mapping with additional refill metadata:
        - representative_flag: bool (same as kept)
        - removed_due_to_corr_with: str (empty if kept, else rep_protein)
        Refilled proteins have singleton components
    final_panel : List[str]
        Final panel proteins (size <= target_size)

    Notes
    -----
    If initial prune yields >= target_size proteins, no refilling occurs.
    Refilled proteins are added as singleton components to df_map.
    """
    # Step 1: Take top-N and prune
    valid_ranked = [p for p in ranked_proteins if p in df.columns]
    top_n = valid_ranked[: min(target_size, len(valid_ranked))]

    df_map, kept = prune_correlated_proteins(
        df=df,
        y=y,
        proteins=top_n,
        selection_freq=selection_freq,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        tiebreak_method=tiebreak_method,
    )

    # Add metadata columns
    df_map = df_map.copy()
    if not df_map.empty:
        df_map["representative_flag"] = df_map["kept"].astype(bool)
        df_map["removed_due_to_corr_with"] = np.where(df_map["kept"], "", df_map["rep_protein"])

    # Step 2: Check if refill needed
    if len(kept) >= target_size:
        return df_map, kept[:target_size]

    # Step 3: Refill to target size
    final_panel = refill_panel_to_target_size(
        df=df,
        kept_proteins=kept,
        ranked_candidates=valid_ranked,
        target_size=target_size,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        pool_limit=pool_limit,
    )

    # Step 4: Add refilled proteins to mapping
    if not df_map.empty and len(final_panel) > len(kept):
        max_comp_id = int(df_map["component_id"].max())
        refilled = final_panel[len(kept) :]

        refill_rows = []
        for i, protein in enumerate(refilled, start=1):
            refill_rows.append(
                {
                    "component_id": max_comp_id + i,
                    "protein": protein,
                    "selection_freq": float(selection_freq.get(protein, np.nan)),
                    "kept": True,
                    "rep_protein": protein,
                    "component_size": 1,
                    "representative_flag": True,
                    "removed_due_to_corr_with": "",
                }
            )

        df_map = pd.concat([df_map, pd.DataFrame(refill_rows)], ignore_index=True)

    return df_map, final_panel
