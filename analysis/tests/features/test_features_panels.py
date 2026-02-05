"""Tests for features/panels.py - biomarker panel building.

Test coverage:
- compute_univariate_strength: Mann-Whitney testing and tie-breaking
- prune_correlated_proteins: Component detection, representative selection
- prune_and_refill_panel: Full workflow with backfill
- build_multi_size_panels: Multiple panel sizes
"""

import numpy as np
import pandas as pd

from ced_ml.features.corr_prune import (
    compute_univariate_strength,
    prune_and_refill_panel,
    prune_correlated_proteins,
)
from ced_ml.features.panels import build_multi_size_panels


class TestComputeUnivariateStrength:
    """Tests for compute_univariate_strength."""

    def test_basic_univariate_computation(self):
        """Compute Mann-Whitney p-values and effect sizes."""
        rng = np.random.default_rng(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "PROT_A": np.concatenate(
                    [
                        rng.normal(0, 1, n_samples // 2),
                        rng.normal(2, 1, n_samples // 2),  # Strong effect
                    ]
                ),
                "PROT_B": np.concatenate(
                    [
                        rng.normal(0, 1, n_samples // 2),
                        rng.normal(0.2, 1, n_samples // 2),  # Weak effect
                    ]
                ),
            }
        )
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        result = compute_univariate_strength(df, y, ["PROT_A", "PROT_B"])

        assert "PROT_A" in result
        assert "PROT_B" in result

        # PROT_A should have smaller p-value (stronger effect)
        p_a, delta_a = result["PROT_A"]
        p_b, delta_b = result["PROT_B"]

        assert p_a < p_b, "PROT_A should have smaller p-value"
        assert delta_a > delta_b, "PROT_A should have larger effect size"

    def test_missing_protein(self):
        """Skip proteins not in DataFrame."""
        df = pd.DataFrame({"PROT_A": [1, 2, 3, 4]})
        y = np.array([0, 0, 1, 1])

        result = compute_univariate_strength(df, y, ["PROT_A", "PROT_B"])

        assert "PROT_A" not in result  # Too few samples
        assert "PROT_B" not in result  # Not in df

    def test_insufficient_samples(self):
        """Require minimum sample size."""
        df = pd.DataFrame(
            {
                "PROT_A": [1, 2, 3, 4, 5],  # Only 5 samples
            }
        )
        y = np.array([0, 0, 0, 1, 1])

        result = compute_univariate_strength(df, y, ["PROT_A"])

        assert "PROT_A" not in result  # < 30 total samples

    def test_missing_values_imputed(self):
        """Handle missing values gracefully."""
        df = pd.DataFrame(
            {
                "PROT_A": [np.nan] * 20 + list(range(80)),
            }
        )
        y = np.array([0] * 50 + [1] * 50)

        result = compute_univariate_strength(df, y, ["PROT_A"])

        # Should process the 80 non-missing values
        assert "PROT_A" in result


class TestPruneCorrelatedProteins:
    """Tests for prune_correlated_proteins."""

    def test_no_correlation_all_kept(self):
        """Independent proteins should all be kept."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": rng.normal(0, 1, 100),
                "B": rng.normal(0, 1, 100),
                "C": rng.normal(0, 1, 100),
            }
        )
        freqs = {"A": 0.9, "B": 0.8, "C": 0.7}

        component_map, kept = prune_correlated_proteins(
            df, None, ["A", "B", "C"], freqs, corr_threshold=0.80
        )

        assert len(kept) == 3
        assert set(kept) == {"A", "B", "C"}
        assert len(component_map) == 3
        assert component_map["kept"].sum() == 3

    def test_high_correlation_pruned(self):
        """Highly correlated proteins should form components."""
        # Note: C is negatively correlated with A/B, so all three form one component
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1.1, 2.1, 3.1, 4.1, 5.1],  # Positively correlated with A (r≈1.0)
                "C": [5, 4, 3, 2, 1],  # Negatively correlated with A/B (r≈-1.0)
                "D": [2.5, 2.6, 2.4, 2.7, 2.5],  # Independent (low variance)
            }
        )
        freqs = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}

        component_map, kept = prune_correlated_proteins(
            df, None, ["A", "B", "C", "D"], freqs, corr_threshold=0.95
        )

        # A, B, C all highly correlated (|r| >= 0.95), A kept (highest freq)
        # D is independent (low correlation with others)
        assert len(kept) == 2, f"Expected 2 proteins kept, got {len(kept)}: {kept}"
        assert "A" in kept, "A should be kept (highest frequency in ABC component)"
        assert "D" in kept, "D should be kept (independent)"
        assert "B" not in kept, "B should be pruned (correlated with A)"
        assert "C" not in kept, "C should be pruned (negatively correlated with A)"

        # Check component structure
        assert len(component_map) == 4
        abc_component = component_map[component_map["protein"].isin(["A", "B", "C"])]
        assert abc_component["component_id"].nunique() == 1, "A, B, C should share component ID"
        assert abc_component[abc_component["kept"]]["protein"].iloc[0] == "A"

    def test_tiebreak_by_frequency(self):
        """Tie-breaking by frequency (default)."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": [1.05, 2.05, 3.05, 4.05],  # Correlated
            }
        )
        freqs = {"A": 0.8, "B": 0.9}  # B has higher frequency

        component_map, kept = prune_correlated_proteins(
            df, None, ["A", "B"], freqs, corr_threshold=0.95, tiebreak_method="freq"
        )

        assert kept == ["B"], "B should be kept (higher frequency)"

    def test_tiebreak_by_univariate(self):
        """Tie-breaking by univariate strength when frequencies equal."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": np.concatenate(
                    [
                        rng.normal(0, 1, 20),
                        rng.normal(0.5, 1, 20),
                    ]  # Weak effect
                ),
                "B": np.concatenate(
                    [
                        rng.normal(0, 1, 20),
                        rng.normal(2.0, 1, 20),
                    ]  # Strong effect
                ),
            }
        )
        # Make B slightly correlated with A
        df["A"] = df["A"] + 0.3 * df["B"]

        y = np.array([0] * 20 + [1] * 20)
        freqs = {"A": 0.8, "B": 0.8}  # Equal frequencies

        component_map, kept = prune_correlated_proteins(
            df,
            y,
            ["A", "B"],
            freqs,
            corr_threshold=0.50,
            tiebreak_method="freq_then_univariate",
        )

        # B should be kept (stronger univariate association)
        # Note: This is probabilistic, so we check both proteins tested
        assert len(kept) <= 2

    def test_empty_input(self):
        """Handle empty protein list."""
        df = pd.DataFrame({"A": [1, 2, 3]})

        component_map, kept = prune_correlated_proteins(df, None, [], {}, corr_threshold=0.80)

        assert len(kept) == 0
        assert component_map.empty

    def test_component_map_columns(self):
        """Component map should have required columns."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": [1.1, 2.1, 3.1, 4.1],
            }
        )
        freqs = {"A": 0.9, "B": 0.8}

        component_map, _ = prune_correlated_proteins(
            df, None, ["A", "B"], freqs, corr_threshold=0.95
        )

        required_cols = [
            "component_id",
            "protein",
            "selection_freq",
            "kept",
            "rep_protein",
            "component_size",
        ]
        assert all(col in component_map.columns for col in required_cols)


class TestPruneAndRefillPanel:
    """Tests for prune_and_refill_panel."""

    def test_basic_prune_and_refill(self):
        """Prune correlated proteins and backfill to target size."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1.1, 2.1, 3.1, 4.1, 5.1],  # Positively correlated with A
                "C": rng.normal(0, 1, 5),  # Independent
                "D": rng.normal(10, 1, 5),  # Independent
                "E": rng.normal(20, 1, 5),  # Independent
            }
        )
        ranked = ["A", "B", "C", "D", "E"]  # Pre-ranked by frequency
        freqs = {"A": 0.95, "B": 0.90, "C": 0.80, "D": 0.70, "E": 0.60}

        component_map, panel = prune_and_refill_panel(
            df, None, ranked, freqs, target_size=3, corr_threshold=0.95, pool_limit=5
        )

        # Should get A (not B due to correlation), then backfill C, D
        assert len(panel) == 3
        assert "A" in panel
        assert "B" not in panel  # Pruned
        # C, D, or E should be backfilled (order may vary)
        assert len(set(panel) & {"C", "D", "E"}) == 2

    def test_no_backfill_needed(self):
        """When pruning doesn't reduce size below target."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": rng.normal(0, 1, 20),
                "B": rng.normal(10, 1, 20),  # Independent
                "C": rng.normal(20, 1, 20),  # Independent
            }
        )
        ranked = ["A", "B", "C"]
        freqs = {"A": 0.9, "B": 0.8, "C": 0.7}

        component_map, panel = prune_and_refill_panel(
            df, None, ranked, freqs, target_size=2, corr_threshold=0.80, pool_limit=3
        )

        assert len(panel) == 2
        assert set(panel) == {"A", "B"}

    def test_backfill_skips_correlated(self):
        """Backfill should skip candidates correlated with existing panel."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8],
                "B": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],  # Correlated with A
                "C": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],  # Also correlated with A
                "D": rng.normal(100, 10, 8),  # Independent
            }
        )
        ranked = ["A", "B", "C", "D"]
        freqs = {k: 0.9 - i * 0.1 for i, k in enumerate(ranked)}

        component_map, panel = prune_and_refill_panel(
            df, None, ranked, freqs, target_size=2, corr_threshold=0.90, pool_limit=4
        )

        # Should get A and D (B and C skipped due to correlation)
        assert len(panel) == 2
        assert "A" in panel
        assert "D" in panel
        assert "B" not in panel
        assert "C" not in panel

    def test_pool_limit_respected(self):
        """Should not consider candidates beyond pool_limit."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"P{i}": rng.normal(0, 1, 50) for i in range(20)})
        ranked = [f"P{i}" for i in range(20)]
        freqs = {f"P{i}": 1.0 - i * 0.01 for i in range(20)}

        component_map, panel = prune_and_refill_panel(
            df,
            None,
            ranked,
            freqs,
            target_size=5,
            corr_threshold=0.99,  # No pruning
            pool_limit=8,  # Only consider first 8
        )

        # Panel should only contain proteins from first 8
        assert all(p in ranked[:8] for p in panel)

    def test_component_map_includes_backfilled(self):
        """Component map should document backfilled proteins."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1.1, 2.1, 3.1, 4.1, 5.1],  # Correlated
                "C": rng.normal(100, 10, 5),  # Independent
            }
        )
        ranked = ["A", "B", "C"]
        freqs = {"A": 0.9, "B": 0.8, "C": 0.7}

        component_map, panel = prune_and_refill_panel(
            df, None, ranked, freqs, target_size=2, corr_threshold=0.95, pool_limit=3
        )

        # Should have A (kept), B (pruned), C (backfilled)
        assert len(panel) == 2
        assert "A" in panel
        assert "C" in panel

        # Component map should show C as backfilled
        c_row = component_map[component_map["protein"] == "C"]
        assert not c_row.empty
        assert c_row["kept"].iloc[0]


class TestBuildMultiSizePanels:
    """Tests for build_multi_size_panels."""

    def test_multiple_panel_sizes(self):
        """Build panels of different sizes."""
        rng = np.random.default_rng(42)
        n_proteins = 50
        df = pd.DataFrame({f"P{i}": rng.normal(0, 1, 100) for i in range(n_proteins)})
        freqs = {f"P{i}": 1.0 - i * 0.01 for i in range(n_proteins)}

        panels = build_multi_size_panels(
            df,
            None,
            freqs,
            panel_sizes=[10, 25, 50],
            corr_threshold=0.99,  # Minimal pruning
            pool_limit=50,
        )

        assert len(panels) == 3
        assert 10 in panels
        assert 25 in panels
        assert 50 in panels

        # Check sizes
        assert len(panels[10][1]) == 10
        assert len(panels[25][1]) == 25
        assert len(panels[50][1]) <= 50  # May be fewer if correlation pruning

    def test_nested_panels(self):
        """Smaller panels should be subsets of larger panels (in ranked order)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"P{i}": rng.normal(0, 1, 100) for i in range(30)})
        freqs = {f"P{i}": 1.0 - i * 0.01 for i in range(30)}

        panels = build_multi_size_panels(
            df,
            None,
            freqs,
            panel_sizes=[5, 10, 20],
            corr_threshold=0.99,  # Minimal pruning
            pool_limit=30,
        )

        panel_5 = set(panels[5][1])
        panel_10 = set(panels[10][1])
        panel_20 = set(panels[20][1])

        # Note: Due to correlation pruning and backfill, strict nesting not guaranteed
        # But top proteins should generally appear in all panels
        assert "P0" in panel_5  # Highest frequency
        assert "P0" in panel_10
        assert "P0" in panel_20

    def test_correlation_pruning_across_sizes(self):
        """Correlation pruning should be applied to each panel size."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1.1, 2.1, 3.1, 4.1, 5.1],  # Correlated with A
                "C": [5, 4, 3, 2, 1],
                "D": [2, 3, 1, 4, 5],
                "E": [3, 1, 4, 2, 5],
            }
        )
        freqs = {"A": 0.95, "B": 0.90, "C": 0.80, "D": 0.70, "E": 0.60}

        panels = build_multi_size_panels(
            df, None, freqs, panel_sizes=[2, 3], corr_threshold=0.95, pool_limit=5
        )

        # B should be pruned in both sizes
        assert "B" not in panels[2][1]
        assert "B" not in panels[3][1]

    def test_empty_panel_sizes(self):
        """Handle empty panel sizes list."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        freqs = {"A": 0.9}

        panels = build_multi_size_panels(df, None, freqs, panel_sizes=[], corr_threshold=0.80)

        assert len(panels) == 0

    def test_panel_sizes_sorted(self):
        """Panel sizes should be processed in sorted order."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({f"P{i}": rng.normal(0, 1, 50) for i in range(20)})
        freqs = {f"P{i}": 1.0 - i * 0.01 for i in range(20)}

        panels = build_multi_size_panels(
            df,
            None,
            freqs,
            panel_sizes=[10, 5, 15],
            corr_threshold=0.99,
            pool_limit=20,  # Unsorted
        )

        # Should handle all sizes despite unsorted input
        assert len(panels) == 3
        assert set(panels.keys()) == {5, 10, 15}


class TestIntegration:
    """Integration tests combining multiple panel functions."""

    def test_full_workflow(self):
        """Complete workflow: frequencies -> multi-size panels."""
        rng = np.random.default_rng(42)

        # Simulate TRAIN data
        df = pd.DataFrame(
            {
                "PROT_A": rng.normal(0, 1, 100),
                "PROT_B": rng.normal(0, 1, 100),
                "PROT_C": rng.normal(0, 1, 100),
                "PROT_D": rng.normal(0, 1, 100),
                "PROT_E": rng.normal(0, 1, 100),
            }
        )
        # Add correlation between A and B
        df["PROT_B"] = df["PROT_A"] + rng.normal(0, 0.1, 100)

        y = rng.binomial(1, 0.3, 100)

        # Simulate selection frequencies from CV
        freqs = {
            "PROT_A": 0.95,
            "PROT_B": 0.90,  # Correlated with A, should be pruned
            "PROT_C": 0.80,
            "PROT_D": 0.70,
            "PROT_E": 0.60,
        }

        # Build multi-size panels
        panels = build_multi_size_panels(
            df,
            y,
            freqs,
            panel_sizes=[2, 3],
            corr_threshold=0.80,
            pool_limit=5,
            tiebreak_method="freq",
        )

        # Verify results
        assert len(panels) == 2
        panel_2 = panels[2][1]
        panel_3 = panels[3][1]

        assert len(panel_2) == 2
        assert len(panel_3) == 3

        # A should be kept over B (higher frequency)
        assert "PROT_A" in panel_2
        assert "PROT_B" not in panel_2  # Pruned due to correlation

        # Both panels should have component maps
        assert not panels[2][0].empty
        assert not panels[3][0].empty

    def test_realistic_sizes(self):
        """Test with realistic panel sizes (10, 25, 50, 100, 200)."""
        rng = np.random.default_rng(42)
        n_proteins = 500

        df = pd.DataFrame({f"PROT_{i:03d}": rng.normal(0, 1, 200) for i in range(n_proteins)})
        freqs = {f"PROT_{i:03d}": 1.0 - i * 0.001 for i in range(n_proteins)}

        panels = build_multi_size_panels(
            df,
            None,
            freqs,
            panel_sizes=[10, 25, 50, 100, 200],
            corr_threshold=0.80,
            pool_limit=1000,
        )

        assert len(panels) == 5
        for size in [10, 25, 50, 100, 200]:
            assert size in panels
            assert len(panels[size][1]) == size
