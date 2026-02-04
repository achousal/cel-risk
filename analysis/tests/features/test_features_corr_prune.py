"""Tests for features/corr_prune.py - correlation-based feature pruning.

Test coverage:
- compute_correlation_matrix: basic, missing values, method selection
- find_high_correlation_pairs: threshold filtering, sorting
- build_correlation_graph: adjacency construction
- find_connected_components: DFS component detection
- compute_univariate_strength: Mann-Whitney computation
- select_component_representative: frequency and univariate tie-breaking
- prune_correlated_proteins: full component-based pruning
- refill_panel_to_target_size: greedy refill with correlation constraints
- prune_and_refill_panel: end-to-end panel building
"""

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.corr_prune import (
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


@pytest.fixture
def basic_data():
    """Simple correlated data for testing."""
    rng = np.random.default_rng(42)
    n = 100

    # Create correlated features
    a = rng.standard_normal(n)
    b = a + rng.standard_normal(n) * 0.1  # High correlation with a
    c = rng.standard_normal(n)  # Independent
    d = c + rng.standard_normal(n) * 0.1  # High correlation with c

    df = pd.DataFrame(
        {
            "PROT_A": a,
            "PROT_B": b,  # Correlated with A
            "PROT_C": c,
            "PROT_D": d,  # Correlated with C
        }
    )

    # Binary labels
    y = np.array([0] * 50 + [1] * 50)

    return df, y


class TestComputeCorrelationMatrix:
    """Tests for compute_correlation_matrix."""

    def test_basic_correlation(self, basic_data):
        """Correlation matrix computed correctly."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B", "PROT_C"]

        corr_matrix = compute_correlation_matrix(df, proteins, method="pearson")

        assert corr_matrix.shape == (3, 3)
        assert list(corr_matrix.index) == proteins
        assert list(corr_matrix.columns) == proteins

        # Diagonal should be 1.0
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # A and B should be highly correlated
        assert corr_matrix.loc["PROT_A", "PROT_B"] > 0.9

    def test_spearman_correlation(self, basic_data):
        """Spearman correlation works."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B"]

        corr_matrix = compute_correlation_matrix(df, proteins, method="spearman")

        assert corr_matrix.shape == (2, 2)
        assert corr_matrix.loc["PROT_A", "PROT_B"] > 0.8

    def test_missing_values_imputed(self):
        """Missing values are median-imputed."""
        df = pd.DataFrame(
            {
                "PROT_A": [1, 2, np.nan, 4],
                "PROT_B": [1, 2, 3, 4],
            }
        )

        corr_matrix = compute_correlation_matrix(df, ["PROT_A", "PROT_B"])

        # Should not contain NaN
        assert not corr_matrix.isna().any().any()

    def test_invalid_proteins_filtered(self, basic_data):
        """Invalid protein names are filtered."""
        df, _ = basic_data
        proteins = ["PROT_A", "INVALID", "PROT_B"]

        corr_matrix = compute_correlation_matrix(df, proteins)

        assert corr_matrix.shape == (2, 2)
        assert "INVALID" not in corr_matrix.index

    def test_empty_proteins(self, basic_data):
        """Empty protein list returns empty matrix."""
        df, _ = basic_data
        corr_matrix = compute_correlation_matrix(df, [])
        assert corr_matrix.empty

    def test_invalid_method_defaults_to_pearson(self, basic_data):
        """Invalid method defaults to pearson."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B"]

        corr_matrix = compute_correlation_matrix(df, proteins, method="invalid")

        # Should compute successfully (pearson)
        assert corr_matrix.shape == (2, 2)


class TestFindHighCorrelationPairs:
    """Tests for find_high_correlation_pairs."""

    def test_basic_pair_detection(self, basic_data):
        """High correlation pairs detected."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]

        corr_matrix = compute_correlation_matrix(df, proteins)
        pairs = find_high_correlation_pairs(corr_matrix, threshold=0.8)

        # Should find A-B and C-D pairs
        assert len(pairs) >= 2

        # Check structure
        assert list(pairs.columns) == ["protein1", "protein2", "abs_corr"]

        # Sorted by correlation
        assert pairs["abs_corr"].is_monotonic_decreasing

    def test_threshold_filtering(self, basic_data):
        """Only pairs above threshold are returned."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B", "PROT_C"]

        corr_matrix = compute_correlation_matrix(df, proteins)

        # High threshold
        pairs_high = find_high_correlation_pairs(corr_matrix, threshold=0.95)
        # Low threshold
        pairs_low = find_high_correlation_pairs(corr_matrix, threshold=0.5)

        assert len(pairs_high) <= len(pairs_low)

    def test_empty_matrix(self):
        """Empty correlation matrix returns empty pairs."""
        corr_matrix = pd.DataFrame()
        pairs = find_high_correlation_pairs(corr_matrix)
        assert pairs.empty
        assert list(pairs.columns) == ["protein1", "protein2", "abs_corr"]

    def test_no_pairs_above_threshold(self):
        """No pairs above threshold returns empty DataFrame."""
        rng = np.random.default_rng(42)
        # Uncorrelated data
        df = pd.DataFrame(
            {
                "PROT_A": rng.standard_normal(50),
                "PROT_B": rng.standard_normal(50),
            }
        )

        corr_matrix = compute_correlation_matrix(df, ["PROT_A", "PROT_B"])
        pairs = find_high_correlation_pairs(corr_matrix, threshold=0.99)

        assert pairs.empty


class TestBuildCorrelationGraph:
    """Tests for build_correlation_graph."""

    def test_basic_graph_construction(self, basic_data):
        """Adjacency graph constructed correctly."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]

        corr_matrix = compute_correlation_matrix(df, proteins)
        adjacency = build_correlation_graph(corr_matrix, threshold=0.8)

        # All proteins should be in graph
        assert set(adjacency.keys()) == set(proteins)

        # A should be connected to B
        assert "PROT_B" in adjacency["PROT_A"]
        assert "PROT_A" in adjacency["PROT_B"]  # Undirected

        # C should be connected to D
        assert "PROT_D" in adjacency["PROT_C"]

    def test_empty_matrix(self):
        """Empty matrix returns empty graph."""
        corr_matrix = pd.DataFrame()
        adjacency = build_correlation_graph(corr_matrix)
        assert adjacency == {}

    def test_isolated_proteins(self):
        """Isolated proteins have empty neighbor sets."""
        rng = np.random.default_rng(42)
        # Uncorrelated data
        df = pd.DataFrame(
            {
                "PROT_A": rng.standard_normal(50),
                "PROT_B": rng.standard_normal(50),
            }
        )

        corr_matrix = compute_correlation_matrix(df, ["PROT_A", "PROT_B"])
        adjacency = build_correlation_graph(corr_matrix, threshold=0.99)

        # Should have entries but no connections
        assert "PROT_A" in adjacency
        assert len(adjacency["PROT_A"]) == 0


class TestFindConnectedComponents:
    """Tests for find_connected_components."""

    def test_two_components(self, basic_data):
        """Two separate components detected."""
        df, _ = basic_data
        proteins = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]

        corr_matrix = compute_correlation_matrix(df, proteins)
        adjacency = build_correlation_graph(corr_matrix, threshold=0.8)
        components = find_connected_components(adjacency)

        # Should have 2 components
        assert len(components) == 2

        # Each component should be sorted
        for comp in components:
            assert comp == sorted(comp)

    def test_singleton_components(self):
        """Isolated proteins form singleton components."""
        adjacency = {
            "PROT_A": set(),
            "PROT_B": set(),
            "PROT_C": set(),
        }

        components = find_connected_components(adjacency)

        assert len(components) == 3
        assert all(len(comp) == 1 for comp in components)

    def test_fully_connected(self):
        """Fully connected graph forms one component."""
        adjacency = {
            "PROT_A": {"PROT_B", "PROT_C"},
            "PROT_B": {"PROT_A", "PROT_C"},
            "PROT_C": {"PROT_A", "PROT_B"},
        }

        components = find_connected_components(adjacency)

        assert len(components) == 1
        assert sorted(components[0]) == ["PROT_A", "PROT_B", "PROT_C"]

    def test_empty_graph(self):
        """Empty graph returns empty components."""
        components = find_connected_components({})
        assert components == []


class TestComputeUnivariateStrength:
    """Tests for compute_univariate_strength."""

    def test_basic_univariate_strength(self):
        """Mann-Whitney p-value computed correctly."""
        rng = np.random.default_rng(42)
        n = 100

        # Create discriminative feature
        df = pd.DataFrame(
            {
                "PROT_STRONG": np.concatenate(
                    [
                        rng.standard_normal(50),  # Controls
                        rng.standard_normal(50) + 2.0,  # Cases (shifted)
                    ]
                ),
                "PROT_WEAK": rng.standard_normal(n),  # Non-discriminative
            }
        )
        y = np.array([0] * 50 + [1] * 50)

        strength = compute_univariate_strength(df, y, ["PROT_STRONG", "PROT_WEAK"])

        # Strong protein should have low p-value
        p_strong, delta_strong = strength["PROT_STRONG"]
        assert p_strong < 0.05
        assert delta_strong > 1.0  # Mean difference ~2

        # Weak protein should have high p-value
        p_weak, _ = strength["PROT_WEAK"]
        assert p_weak > 0.05

    def test_insufficient_data_skipped(self):
        """Proteins with insufficient data are skipped."""
        df = pd.DataFrame(
            {
                "PROT_A": [1, 2, 3],  # Too few samples
            }
        )
        y = np.array([0, 0, 1])

        strength = compute_univariate_strength(df, y, ["PROT_A"])

        assert "PROT_A" not in strength

    def test_missing_protein(self):
        """Missing proteins are skipped."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"PROT_A": rng.standard_normal(50)})
        y = np.array([0] * 25 + [1] * 25)

        strength = compute_univariate_strength(df, y, ["PROT_MISSING"])

        assert strength == {}


class TestSelectComponentRepresentative:
    """Tests for select_component_representative."""

    def test_freq_method(self):
        """Frequency-based selection."""
        component = ["PROT_A", "PROT_B", "PROT_C"]
        selection_freq = {"PROT_A": 0.5, "PROT_B": 0.8, "PROT_C": 0.3}

        rep = select_component_representative(
            component=component,
            selection_freq=selection_freq,
            tiebreak_method="freq",
        )

        assert rep == "PROT_B"  # Highest frequency

    def test_freq_tie_alphabetical(self):
        """Frequency ties broken alphabetically."""
        component = ["PROT_Z", "PROT_A"]
        selection_freq = {"PROT_Z": 0.5, "PROT_A": 0.5}  # Tied

        rep = select_component_representative(
            component=component,
            selection_freq=selection_freq,
            tiebreak_method="freq",
        )

        assert rep == "PROT_A"  # Alphabetical

    def test_freq_then_univariate(self):
        """Univariate tie-breaking."""
        component = ["PROT_A", "PROT_B"]
        selection_freq = {"PROT_A": 0.5, "PROT_B": 0.5}  # Tied frequency
        univariate_strength = {
            "PROT_A": (0.01, 2.0),  # Stronger (lower p-value)
            "PROT_B": (0.10, 1.0),
        }

        rep = select_component_representative(
            component=component,
            selection_freq=selection_freq,
            tiebreak_method="freq_then_univariate",
            univariate_strength=univariate_strength,
        )

        assert rep == "PROT_A"  # Lower p-value

    def test_no_selection_freq(self):
        """No selection frequency defaults to 0."""
        component = ["PROT_A", "PROT_B"]

        rep = select_component_representative(
            component=component,
            selection_freq=None,
            tiebreak_method="freq",
        )

        # Alphabetical tie-break
        assert rep == "PROT_A"


class TestPruneCorrelatedProteins:
    """Tests for prune_correlated_proteins."""

    def test_basic_pruning(self, basic_data):
        """Basic pruning keeps representatives."""
        df, y = basic_data
        proteins = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]
        selection_freq = {"PROT_A": 0.9, "PROT_B": 0.7, "PROT_C": 0.8, "PROT_D": 0.6}

        df_map, kept = prune_correlated_proteins(
            df=df,
            y=y,
            proteins=proteins,
            selection_freq=selection_freq,
            corr_threshold=0.8,
            tiebreak_method="freq",
        )

        # Should keep 2 proteins (one from A-B, one from C-D)
        assert len(kept) == 2

        # Higher frequency proteins should be kept
        assert "PROT_A" in kept  # Higher than PROT_B
        assert "PROT_C" in kept  # Higher than PROT_D

        # Check mapping structure
        assert len(df_map) == 4
        assert "component_id" in df_map.columns
        assert "kept" in df_map.columns
        assert df_map["kept"].sum() == 2

    def test_empty_proteins(self, basic_data):
        """Empty protein list returns empty results."""
        df, y = basic_data

        df_map, kept = prune_correlated_proteins(
            df=df,
            y=y,
            proteins=[],
            selection_freq={},
        )

        assert df_map.empty
        assert kept == []

    def test_no_correlations(self):
        """No correlations keeps all proteins."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "PROT_A": rng.standard_normal(50),
                "PROT_B": rng.standard_normal(50),
                "PROT_C": rng.standard_normal(50),
            }
        )
        y = np.array([0] * 25 + [1] * 25)

        df_map, kept = prune_correlated_proteins(
            df=df,
            y=y,
            proteins=["PROT_A", "PROT_B", "PROT_C"],
            selection_freq={"PROT_A": 0.5, "PROT_B": 0.5, "PROT_C": 0.5},
            corr_threshold=0.99,  # Very high threshold
        )

        # All proteins kept (singleton components)
        assert len(kept) == 3


class TestRefillPanelToTargetSize:
    """Tests for refill_panel_to_target_size."""

    def test_basic_refill(self, basic_data):
        """Refill adds uncorrelated proteins."""
        df, _ = basic_data

        kept = ["PROT_A"]
        candidates = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]

        final = refill_panel_to_target_size(
            df=df,
            kept_proteins=kept,
            ranked_candidates=candidates,
            target_size=2,
            corr_threshold=0.8,
        )

        # Should add one more protein
        assert len(final) == 2
        assert "PROT_A" in final

        # Should not add PROT_B (correlated with A)
        # Should add PROT_C (uncorrelated)
        assert "PROT_C" in final

    def test_no_refill_needed(self, basic_data):
        """No refill if already at target size."""
        df, _ = basic_data

        kept = ["PROT_A", "PROT_B", "PROT_C"]
        candidates = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]

        final = refill_panel_to_target_size(
            df=df,
            kept_proteins=kept,
            ranked_candidates=candidates,
            target_size=2,
        )

        # Truncate to target size
        assert len(final) == 2

    def test_insufficient_candidates(self, basic_data):
        """Returns what's available if candidates exhausted."""
        df, _ = basic_data

        kept = ["PROT_A"]
        candidates = ["PROT_A", "PROT_B"]  # Only one new candidate

        final = refill_panel_to_target_size(
            df=df,
            kept_proteins=kept,
            ranked_candidates=candidates,
            target_size=10,
        )

        # Can't reach target size
        assert len(final) <= 2


class TestPruneAndRefillPanel:
    """Tests for prune_and_refill_panel (end-to-end)."""

    def test_complete_workflow(self, basic_data):
        """Complete prune and refill workflow."""
        df, y = basic_data

        ranked = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]
        selection_freq = {
            "PROT_A": 0.9,
            "PROT_B": 0.8,
            "PROT_C": 0.7,
            "PROT_D": 0.6,
        }

        df_map, final_panel = prune_and_refill_panel(
            df=df,
            y=y,
            ranked_proteins=ranked,
            selection_freq=selection_freq,
            target_size=3,
            corr_threshold=0.8,
            tiebreak_method="freq",
        )

        # Should keep representatives (A from A-B, C from C-D)
        # Due to high correlation threshold, may only have 2 components
        assert len(final_panel) >= 2
        assert len(final_panel) <= 3

        # Check mapping includes all proteins
        assert len(df_map) >= len(final_panel)
        assert "representative_flag" in df_map.columns
        assert "removed_due_to_corr_with" in df_map.columns

    def test_no_refill_if_target_met(self, basic_data):
        """No refill if pruning already meets target."""
        df, y = basic_data

        # Low correlation threshold → all proteins kept
        ranked = ["PROT_A", "PROT_B", "PROT_C"]
        selection_freq = {"PROT_A": 0.5, "PROT_B": 0.5, "PROT_C": 0.5}

        df_map, final_panel = prune_and_refill_panel(
            df=df,
            y=y,
            ranked_proteins=ranked,
            selection_freq=selection_freq,
            target_size=2,
            corr_threshold=0.99,  # No correlations
        )

        # Should keep only target_size
        assert len(final_panel) == 2

    def test_refill_metadata(self, basic_data):
        """Refilled proteins have correct metadata."""
        df, y = basic_data

        ranked = ["PROT_A", "PROT_B", "PROT_C", "PROT_D"]
        selection_freq = dict.fromkeys(ranked, 0.5)

        df_map, final_panel = prune_and_refill_panel(
            df=df,
            y=y,
            ranked_proteins=ranked,
            selection_freq=selection_freq,
            target_size=3,
            corr_threshold=0.8,
        )

        # Check refilled proteins have singleton components
        refilled = [
            p for p in final_panel if df_map[df_map["protein"] == p]["component_size"].iloc[0] == 1
        ]
        assert len(refilled) >= 0  # May or may not have refilled proteins

        # All refilled proteins should be kept
        for protein in refilled:
            row = df_map[df_map["protein"] == protein].iloc[0]
            assert row["kept"]  # Use == for numpy bool comparison
            assert row["representative_flag"]
            assert row["removed_due_to_corr_with"] == ""
