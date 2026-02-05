"""Tests for features/stability.py - feature stability tracking.

Test coverage:
- compute_selection_frequencies: basic, edge cases
- extract_stable_panel: per-repeat unions, threshold filtering, fallback
- rank_proteins_by_frequency: sorting and tie-breaking
"""

import json

import numpy as np
import pandas as pd

from ced_ml.features.stability import (
    compute_selection_frequencies,
    extract_stable_panel,
    rank_proteins_by_frequency,
)


class TestComputeSelectionFrequencies:
    """Tests for compute_selection_frequencies."""

    def test_basic_frequency_computation(self):
        """Selection frequency = count / total splits."""
        log = pd.DataFrame(
            {
                "repeat": [0, 0, 1, 1],
                "fold": [0, 1, 0, 1],
                "selected_proteins_split": [
                    '["PROT_A", "PROT_B"]',
                    '["PROT_A", "PROT_C"]',
                    '["PROT_A", "PROT_B"]',
                    '["PROT_B", "PROT_C"]',
                ],
            }
        )

        freqs = compute_selection_frequencies(log)

        assert freqs == {"PROT_A": 0.75, "PROT_B": 0.75, "PROT_C": 0.5}

    def test_custom_column_name(self):
        """Custom selection column name."""
        log = pd.DataFrame(
            {
                "custom_col": ['["A"]', '["A", "B"]', '["B"]'],
            }
        )

        freqs = compute_selection_frequencies(log, selection_col="custom_col")

        assert freqs == {"A": 2 / 3, "B": 2 / 3}

    def test_empty_log(self):
        """Empty log returns empty dict."""
        log = pd.DataFrame(columns=["repeat", "fold", "selected_proteins_split"])
        freqs = compute_selection_frequencies(log)
        assert freqs == {}

    def test_none_log(self):
        """None log returns empty dict."""
        freqs = compute_selection_frequencies(None)
        assert freqs == {}

    def test_missing_column(self):
        """Missing selection column returns empty dict."""
        log = pd.DataFrame({"repeat": [0], "fold": [0]})
        freqs = compute_selection_frequencies(log, selection_col="missing")
        assert freqs == {}

    def test_malformed_json(self):
        """Malformed JSON entries are skipped."""
        log = pd.DataFrame(
            {
                "selected_proteins_split": [
                    '["A", "B"]',
                    "NOT_JSON",
                    '["A", "C"]',
                    None,
                ],
            }
        )

        freqs = compute_selection_frequencies(log)

        # Only valid JSON entries counted (2 splits)
        assert freqs == {"A": 2 / 4, "B": 1 / 4, "C": 1 / 4}

    def test_duplicate_proteins_in_split(self):
        """Duplicates within a split counted once."""
        log = pd.DataFrame(
            {
                "selected_proteins_split": [
                    '["A", "A", "B"]',  # A appears twice in same split
                ],
            }
        )

        freqs = compute_selection_frequencies(log)

        assert freqs == {"A": 1.0, "B": 1.0}  # A counted once


class TestExtractStablePanel:
    """Tests for extract_stable_panel."""

    def test_stable_panel_basic(self):
        """Proteins appearing in >= threshold fraction of repeats are stable."""
        log = pd.DataFrame(
            {
                "repeat": [0, 0, 1, 1, 2, 2],
                "fold": [0, 1, 0, 1, 0, 1],
                "selected_proteins_split": [
                    '["A", "B"]',
                    '["A", "C"]',  # repeat 0: {A, B, C}
                    '["A", "B"]',
                    '["A", "D"]',  # repeat 1: {A, B, D}
                    '["A", "B"]',
                    '["B", "C"]',  # repeat 2: {A, B, C}
                ],
            }
        )

        panel, stable, unions = extract_stable_panel(
            log, n_repeats=3, stability_threshold=2 / 3  # 67%
        )

        # A: 3/3 repeats (100% >= 67%)
        # B: 3/3 repeats (100% >= 67%)
        # C: 2/3 repeats (67% >= 67%) - exactly at threshold, included
        # D: 1/3 repeats (33% < 67%)
        assert set(stable) == {"A", "B", "C"}
        assert panel[panel["kept"]]["protein"].tolist() == ["A", "B", "C"]

        # Verify repeat unions
        assert len(unions) == 3
        assert unions[0] == {"A", "B", "C"}
        assert unions[1] == {"A", "B", "D"}
        assert unions[2] == {"A", "B", "C"}

    def test_frequency_computation(self):
        """Selection frequency = repeats where protein appears / total repeats."""
        log = pd.DataFrame(
            {
                "repeat": [0, 1, 2],
                "fold": [0, 0, 0],
                "selected_proteins_split": [
                    '["A"]',  # repeat 0: {A}
                    '["A"]',  # repeat 1: {A}
                    '["B"]',  # repeat 2: {B}
                ],
            }
        )

        panel, _, _ = extract_stable_panel(log, n_repeats=3, stability_threshold=1.0)

        assert len(panel) == 2
        assert panel[panel["protein"] == "A"]["selection_freq"].values[0] == 2 / 3
        assert panel[panel["protein"] == "B"]["selection_freq"].values[0] == 1 / 3

    def test_fallback_when_no_stable(self):
        """If no proteins meet threshold, keep top N by frequency."""
        log = pd.DataFrame(
            {
                "repeat": [0, 1, 2],
                "fold": [0, 0, 0],
                "selected_proteins_split": [
                    '["A"]',  # repeat 0
                    '["B"]',  # repeat 1
                    '["C"]',  # repeat 2
                ],
            }
        )

        # No protein appears in 100% of repeats
        panel, stable, _ = extract_stable_panel(
            log, n_repeats=3, stability_threshold=1.0, fallback_top_n=2
        )

        # Fallback to top 2 proteins (all have freq=1/3, alphabetical order)
        assert len(stable) == 2
        assert stable == ["A", "B"]
        assert panel["kept"].sum() == 2

    def test_custom_selection_column(self):
        """Custom selection column name."""
        log = pd.DataFrame(
            {
                "repeat": [0, 1],
                "fold": [0, 0],
                "custom_col": ['["A"]', '["A"]'],
            }
        )

        panel, stable, _ = extract_stable_panel(
            log, n_repeats=2, stability_threshold=1.0, selection_col="custom_col"
        )

        assert stable == ["A"]

    def test_empty_log(self):
        """Empty log returns empty results."""
        log = pd.DataFrame(columns=["repeat", "fold", "selected_proteins_split"])

        panel, stable, unions = extract_stable_panel(log, n_repeats=3, stability_threshold=0.75)

        assert panel.empty
        assert stable == []
        assert unions == []

    def test_none_log(self):
        """None log returns empty results."""
        panel, stable, unions = extract_stable_panel(None, n_repeats=3, stability_threshold=0.75)

        assert panel.empty
        assert stable == []
        assert unions == []

    def test_malformed_json_skipped(self):
        """Malformed JSON entries are skipped when building unions."""
        log = pd.DataFrame(
            {
                "repeat": [0, 0, 1],
                "fold": [0, 1, 0],
                "selected_proteins_split": [
                    '["A"]',
                    "NOT_JSON",  # Skipped
                    '["A"]',
                ],
            }
        )

        panel, stable, unions = extract_stable_panel(log, n_repeats=2, stability_threshold=1.0)

        assert stable == ["A"]
        assert unions[0] == {"A"}  # repeat 0 (malformed skipped)
        assert unions[1] == {"A"}  # repeat 1

    def test_sorting_order(self):
        """Panel sorted by kept DESC, freq DESC, name ASC."""
        log = pd.DataFrame(
            {
                "repeat": [0, 1, 2],
                "fold": [0, 0, 0],
                "selected_proteins_split": [
                    '["A", "C"]',  # repeat 0
                    '["A", "B"]',  # repeat 1
                    '["A", "B"]',  # repeat 2
                ],
            }
        )

        panel, _, _ = extract_stable_panel(log, n_repeats=3, stability_threshold=2 / 3)

        # A and B are kept (appear in 2+ repeats)
        # Sorting: kept=True first, then freq DESC (A=3/3, B=2/3, C=1/3), then name ASC
        assert panel["protein"].tolist() == ["A", "B", "C"]
        assert panel["kept"].tolist() == [True, True, False]


class TestRankProteinsByFrequency:
    """Tests for rank_proteins_by_frequency."""

    def test_basic_ranking(self):
        """Proteins ranked by frequency DESC."""
        freqs = {"A": 0.90, "B": 0.80, "C": 0.70}

        ranked = rank_proteins_by_frequency(freqs)

        assert ranked == ["A", "B", "C"]

    def test_tie_breaking_by_name(self):
        """Ties broken by protein name ASC."""
        freqs = {"PROT_C": 0.90, "PROT_A": 0.90, "PROT_B": 0.80}

        ranked = rank_proteins_by_frequency(freqs)

        assert ranked == ["PROT_A", "PROT_C", "PROT_B"]

    def test_empty_frequencies(self):
        """Empty frequencies returns empty list."""
        ranked = rank_proteins_by_frequency({})
        assert ranked == []

    def test_single_protein(self):
        """Single protein returns single-element list."""
        ranked = rank_proteins_by_frequency({"A": 0.90})
        assert ranked == ["A"]


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_full_workflow(self):
        """Full workflow: compute frequencies -> build panel -> extract stable."""
        # Simulate 2 repeats × 3 folds = 6 splits
        log = pd.DataFrame(
            {
                "repeat": [0, 0, 0, 1, 1, 1],
                "fold": [0, 1, 2, 0, 1, 2],
                "selected_proteins_split": [
                    '["A", "B", "C"]',
                    '["A", "B", "D"]',
                    '["A", "C", "D"]',
                    '["A", "B", "C"]',
                    '["A", "B", "E"]',
                    '["A", "C", "E"]',
                ],
            }
        )

        # Step 1: Compute split-level frequencies
        split_freqs = compute_selection_frequencies(log)
        assert split_freqs["A"] == 6 / 6  # In all splits
        assert split_freqs["B"] == 4 / 6
        assert split_freqs["C"] == 4 / 6

        # Step 2: Extract stable panel (repeat-level)
        stable_panel, stable_prots, unions = extract_stable_panel(
            log, n_repeats=2, stability_threshold=1.0
        )
        # A, B, C appear in both repeat unions
        assert set(stable_prots) == {"A", "B", "C"}
        assert len(unions) == 2

    def test_realistic_cv_scenario(self):
        """Realistic scenario: 5-fold × 10 repeats."""
        rng = np.random.default_rng(42)

        # Simulate protein pool
        all_proteins = [f"PROT_{i:03d}" for i in range(50)]

        # Simulate selection: top 10 proteins selected consistently
        # remaining 40 selected randomly
        rows = []
        for repeat in range(10):
            for fold in range(5):
                # Top 10 always selected
                selected = all_proteins[:10].copy()
                # Add 5 random proteins
                selected.extend(rng.choice(all_proteins[10:], size=5, replace=False).tolist())
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "selected_proteins_split": json.dumps(selected),
                    }
                )

        log = pd.DataFrame(rows)

        # Compute frequencies
        freqs = compute_selection_frequencies(log)

        # Top 10 should have freq=1.0 (50/50 splits)
        for i in range(10):
            assert freqs[f"PROT_{i:03d}"] == 1.0

        # Extract stable panel with high threshold (1.0 = must appear in ALL repeats)
        panel, stable, _ = extract_stable_panel(log, n_repeats=10, stability_threshold=1.0)

        # Top 10 proteins should be stable (appear in 10/10 repeats = 100%)
        # Random proteins may appear in multiple repeats but not ALL repeats
        assert len(stable) >= 10  # At least the top 10
        assert all(f"PROT_{i:03d}" in stable for i in range(10))

        # With threshold=1.0, only proteins in ALL repeats are kept
        # Top 10 are the only proteins guaranteed to be in all repeats
        for protein in stable:
            freq = panel[panel["protein"] == protein]["selection_freq"].values[0]
            assert freq == 1.0  # All stable proteins must have 100% frequency
