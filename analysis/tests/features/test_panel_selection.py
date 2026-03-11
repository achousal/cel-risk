"""Tests for deterministic panel size selection rules.

Tests cover:
- Bootstrap non-inferiority test (Rule 1)
- Cross-seed stability gate (Rule 2)
- Passenger filtering (Rule 3)
- End-to-end panel selection on synthetic Pareto curves
"""

import json

import numpy as np
import pandas as pd
import pytest

from ced_ml.features.panel_selection import (
    NonInferiorityResult,
    PanelSelectionResult,
    StabilityResult,
    bootstrap_noninferiority_test,
    cross_seed_stability_check,
    decision_table_to_dict,
    filter_passengers,
    select_optimal_panel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_predictions():
    """Predictions where full and reduced models are identical."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.uniform(0, 1, size=500)
    # Make predictions correlated with labels
    y_pred = np.where(y_true == 1, y_pred * 0.3 + 0.6, y_pred * 0.3 + 0.1)
    return y_true, y_pred, y_pred.copy()


@pytest.fixture()
def degraded_predictions():
    """Predictions where reduced model is clearly worse."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_pred_full = np.where(
        y_true == 1,
        rng.uniform(0.6, 1.0, size=500),
        rng.uniform(0.0, 0.4, size=500),
    )
    # Add heavy noise to reduced predictions
    y_pred_reduced = y_pred_full + rng.normal(0, 0.3, size=500)
    y_pred_reduced = np.clip(y_pred_reduced, 0, 1)
    return y_true, y_pred_full, y_pred_reduced


@pytest.fixture()
def synthetic_pareto_curve():
    """Synthetic Pareto curve with known structure.

    Sizes: [100, 50, 25, 15, 10, 5]
    AUROC decreases as panel shrinks, with a knee around 15-25.
    """
    # AUROC values designed so that with delta=0.02 and full AUROC~0.886:
    #   k=100 (delta~0.000) and k=50 (delta~0.001) clearly pass
    #   k=25 (delta~0.008) passes
    #   k=15 (delta~0.016) passes (within 0.02)
    #   k=10 (delta~0.031) fails
    #   k=5  (delta~0.066) fails
    # Stds are low enough that CI_upper stays within margin for passing sizes.
    return [
        {
            "size": 100,
            "auroc_val": 0.886,
            "auroc_val_std": 0.008,
            "n_seeds": 5,
            "proteins": [f"P{i}" for i in range(100)],
        },
        {
            "size": 50,
            "auroc_val": 0.885,
            "auroc_val_std": 0.008,
            "n_seeds": 5,
            "proteins": [f"P{i}" for i in range(50)],
        },
        {
            "size": 25,
            "auroc_val": 0.878,
            "auroc_val_std": 0.009,
            "n_seeds": 5,
            "proteins": [f"P{i}" for i in range(25)],
        },
        {
            "size": 15,
            "auroc_val": 0.870,
            "auroc_val_std": 0.010,
            "n_seeds": 5,
            "proteins": [f"P{i}" for i in range(15)],
        },
        {
            "size": 10,
            "auroc_val": 0.855,
            "auroc_val_std": 0.020,
            "n_seeds": 5,
            "proteins": [f"P{i}" for i in range(10)],
        },
        {
            "size": 5,
            "auroc_val": 0.820,
            "auroc_val_std": 0.030,
            "n_seeds": 5,
            "proteins": [f"P{i}" for i in range(5)],
        },
    ]


# ---------------------------------------------------------------------------
# Rule 1: Non-inferiority test
# ---------------------------------------------------------------------------


class TestBootstrapNoninferiority:
    """Tests for bootstrap_noninferiority_test."""

    def test_identical_models_reject_h0(self, perfect_predictions):
        """Identical predictions should reject H0 (declare non-inferior)."""
        y_true, y_full, y_reduced = perfect_predictions
        result = bootstrap_noninferiority_test(
            y_true, y_full, y_reduced, delta=0.02, n_bootstrap=500, seed=42
        )
        assert isinstance(result, NonInferiorityResult)
        assert result.rejected is True
        assert result.delta_estimate == pytest.approx(0.0, abs=0.01)
        assert result.p_value < 0.05

    def test_degraded_model_fails_to_reject(self, degraded_predictions):
        """Clearly worse predictions should fail to reject H0."""
        y_true, y_full, y_reduced = degraded_predictions
        result = bootstrap_noninferiority_test(
            y_true, y_full, y_reduced, delta=0.02, n_bootstrap=500, seed=42
        )
        assert result.rejected is False
        assert result.delta_estimate > 0.02

    def test_strict_delta_harder_to_pass(self, perfect_predictions):
        """Stricter delta should be harder to pass."""
        y_true, y_full, y_reduced = perfect_predictions
        # Add small noise
        rng = np.random.default_rng(99)
        y_reduced_noisy = y_reduced + rng.normal(0, 0.05, size=len(y_reduced))
        y_reduced_noisy = np.clip(y_reduced_noisy, 0, 1)

        result_loose = bootstrap_noninferiority_test(
            y_true, y_full, y_reduced_noisy, delta=0.05, n_bootstrap=500
        )
        result_strict = bootstrap_noninferiority_test(
            y_true, y_full, y_reduced_noisy, delta=0.005, n_bootstrap=500
        )
        # Loose delta should be easier to pass
        assert result_loose.p_value <= result_strict.p_value

    def test_returns_correct_fields(self, perfect_predictions):
        """Result should have all expected fields."""
        y_true, y_full, y_reduced = perfect_predictions
        result = bootstrap_noninferiority_test(
            y_true, y_full, y_reduced, delta=0.02, n_bootstrap=100
        )
        assert hasattr(result, "rejected")
        assert hasattr(result, "p_value")
        assert hasattr(result, "delta_estimate")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "delta_margin")
        assert hasattr(result, "n_bootstrap")
        assert result.delta_margin == 0.02
        assert result.n_bootstrap > 0

    def test_empty_labels_returns_safe_default(self):
        """All-same labels should return a safe non-rejection."""
        y_true = np.ones(100)
        y_full = np.random.default_rng(0).uniform(size=100)
        y_reduced = y_full.copy()
        result = bootstrap_noninferiority_test(y_true, y_full, y_reduced)
        assert result.rejected is False
        assert result.n_bootstrap == 0


# ---------------------------------------------------------------------------
# Rule 2: Stability gate
# ---------------------------------------------------------------------------


class TestCrossSeedStability:
    """Tests for cross_seed_stability_check."""

    def test_stable_seeds(self):
        """Low variance across seeds should be stable."""
        aurocs = [0.89, 0.88, 0.90, 0.87, 0.89]
        result = cross_seed_stability_check(aurocs, cv_threshold=0.05)
        assert isinstance(result, StabilityResult)
        assert result.stable is True
        assert result.cv < 0.05

    def test_unstable_seeds(self):
        """High variance across seeds should be unstable."""
        aurocs = [0.95, 0.60, 0.85, 0.70, 0.50]
        result = cross_seed_stability_check(aurocs, cv_threshold=0.05)
        assert result.stable is False
        assert result.cv > 0.05

    def test_single_seed_always_stable(self):
        """Single seed has no variance, should be stable."""
        result = cross_seed_stability_check([0.89], cv_threshold=0.05)
        assert result.stable is True
        assert result.cv == 0.0
        assert result.n_seeds == 1

    def test_correct_statistics(self):
        """Mean and std should be computed correctly."""
        aurocs = [0.80, 0.90]
        result = cross_seed_stability_check(aurocs)
        assert result.mean_auroc == pytest.approx(0.85, abs=1e-10)
        # ddof=1 std
        expected_std = np.std([0.80, 0.90], ddof=1)
        assert result.std_auroc == pytest.approx(expected_std, abs=1e-10)


# ---------------------------------------------------------------------------
# Rule 3: Passenger filtering
# ---------------------------------------------------------------------------


class TestFilterPassengers:
    """Tests for filter_passengers."""

    def test_all_essential(self):
        """All proteins with significant delta should be kept."""
        df = pd.DataFrame(
            {
                "representative": ["P0", "P1", "P2"],
                "mean_delta_auroc": [0.05, 0.03, 0.04],
                "std_delta_auroc": [0.005, 0.004, 0.003],
                "n_folds": [5, 5, 5],
            }
        )
        essential, removed, n = filter_passengers(df, ["P0", "P1", "P2"])
        assert n == 0
        assert set(essential) == {"P0", "P1", "P2"}
        assert removed == []

    def test_passenger_removed(self):
        """Proteins with CI including zero should be removed."""
        df = pd.DataFrame(
            {
                "representative": ["P0", "P1", "P2"],
                "mean_delta_auroc": [0.05, 0.001, 0.04],
                "std_delta_auroc": [0.005, 0.010, 0.003],
                "n_folds": [5, 5, 5],
            }
        )
        essential, removed, n = filter_passengers(df, ["P0", "P1", "P2"])
        assert "P1" in removed
        assert "P1" not in essential
        assert n >= 1

    def test_none_essentiality_keeps_all(self):
        """None input should keep all proteins."""
        essential, removed, n = filter_passengers(None, ["P0", "P1"])
        assert essential == ["P0", "P1"]
        assert n == 0

    def test_empty_df_keeps_all(self):
        """Empty DataFrame should keep all proteins."""
        df = pd.DataFrame()
        essential, removed, n = filter_passengers(df, ["P0", "P1"])
        assert essential == ["P0", "P1"]
        assert n == 0


# ---------------------------------------------------------------------------
# End-to-end: select_optimal_panel
# ---------------------------------------------------------------------------


class TestSelectOptimalPanel:
    """Tests for select_optimal_panel end-to-end."""

    def test_selects_smallest_passing_panel(self, synthetic_pareto_curve):
        """Should select the smallest panel passing all rules."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
            delta_primary=0.02,
            delta_sensitivity=0.01,
        )
        assert isinstance(result, PanelSelectionResult)
        assert result.selected_size > 0
        assert result.selected_size <= 100
        assert len(result.selected_proteins) == result.selected_size
        assert result.delta_used == 0.02

    def test_sensitivity_panel_is_larger_or_equal(self, synthetic_pareto_curve):
        """Sensitivity (stricter delta) should require >= primary size."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
            delta_primary=0.02,
            delta_sensitivity=0.01,
        )
        if result.sensitivity_size > 0 and result.selected_size > 0:
            assert result.sensitivity_size >= result.selected_size

    def test_empty_curve_returns_empty(self):
        """Empty curve should return empty result."""
        result = select_optimal_panel(
            curve=[],
            full_auroc_by_seed=[0.89],
        )
        assert result.selected_size == 0
        assert result.selected_proteins == []

    def test_decision_table_populated(self, synthetic_pareto_curve):
        """Decision table should have one entry per curve point."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
        )
        assert len(result.decision_table) == len(synthetic_pareto_curve)

    def test_full_model_auroc_stored(self, synthetic_pareto_curve):
        """Full model AUROC should be stored in result."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
        )
        assert result.full_model_auroc == pytest.approx(np.mean(full_aurocs))

    def test_very_strict_delta_selects_full_model(self, synthetic_pareto_curve):
        """Very strict delta should force selection of largest panel."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
            delta_primary=0.001,
        )
        # With delta=0.001, only full model (or very close) should pass
        if result.selected_size > 0:
            assert result.selected_size >= 50

    def test_unstable_small_panels_skipped(self):
        """Panels with high cross-seed variance should be skipped."""
        curve = [
            {
                "size": 50,
                "auroc_val": 0.88,
                "auroc_val_std": 0.010,
                "n_seeds": 5,
                "proteins": [f"P{i}" for i in range(50)],
            },
            {
                "size": 10,
                "auroc_val": 0.87,
                "auroc_val_std": 0.080,  # Very unstable
                "n_seeds": 5,
                "proteins": [f"P{i}" for i in range(10)],
            },
        ]
        result = select_optimal_panel(
            curve=curve,
            full_auroc_by_seed=[0.886, 0.884, 0.890, 0.882, 0.888],
            delta_primary=0.03,
        )
        # k=10 should be skipped due to instability
        skipped_10 = [d for d in result.decision_table if d.size == 10]
        if skipped_10:
            assert not skipped_10[0].accepted or skipped_10[0].size != 10

    def test_constant_full_auroc_uses_correct_se(self):
        """Constant full_auroc_by_seed (known reference) should use SE = std_k / sqrt(n).

        Previously the code used SE = std_k (sample SD, not SE of mean),
        making CI_upper ~ delta + 1.645 * 0.033 ~ 0.056, always > 0.02.
        The correct SE = 0.033 / sqrt(30) ~ 0.006 gives CI ~ 0.012 < 0.02.
        """
        full_aurocs_constant = [0.870] * 30
        curve = [
            {
                "size": 10,
                "auroc_val": 0.868,
                "auroc_val_std": 0.033,
                "n_seeds": 30,
                "proteins": [f"P{i}" for i in range(10)],
            },
        ]
        result = select_optimal_panel(
            curve=curve,
            full_auroc_by_seed=full_aurocs_constant,
            delta_primary=0.02,
        )
        # SE = 0.033 / sqrt(30) ~ 0.006; CI_upper ~ 0.002 + 1.645*0.006 ~ 0.012 < 0.02
        assert result.selected_size == 10, (
            "Panel with delta=0.002 and SE=0.006 should pass non-inferiority "
            f"(CI_upper ~ 0.012 < 0.02), got selected_size={result.selected_size}"
        )
        ni = result.decision_table[0].noninferiority
        assert ni.ci_upper < 0.02, f"CI_upper={ni.ci_upper:.4f} should be < 0.02"

    def test_truly_inferior_panel_rejected(self):
        """Panel with large delta should still be rejected even with correct SE."""
        full_aurocs = [0.900] * 30
        curve = [
            {
                "size": 5,
                "auroc_val": 0.860,
                "auroc_val_std": 0.030,
                "n_seeds": 30,
                "proteins": [f"P{i}" for i in range(5)],
            },
        ]
        result = select_optimal_panel(
            curve=curve,
            full_auroc_by_seed=full_aurocs,
            delta_primary=0.02,
        )
        # delta = 0.04 >> 0.02, should be rejected
        assert result.selected_size == 0, "Panel with delta=0.04 should be rejected"

    def test_paired_noninferiority_with_per_seed_data(self):
        """Paired test using auroc_val_by_seed should produce tight CI."""
        import numpy as np

        rng = np.random.default_rng(99)
        n = 30
        # Full-model AUROCs with real cross-seed variance
        full_aurocs = (0.870 + rng.normal(0, 0.03, n)).tolist()
        # Panel AUROCs: slightly lower but highly correlated (same seed effect)
        panel_aurocs = [f - 0.002 + rng.normal(0, 0.003) for f in full_aurocs]

        curve = [
            {
                "size": 10,
                "auroc_val": float(np.mean(panel_aurocs)),
                "auroc_val_std": float(np.std(panel_aurocs, ddof=1)),
                "auroc_val_by_seed": panel_aurocs,
                "n_seeds": n,
                "proteins": [f"P{i}" for i in range(10)],
            },
        ]
        result = select_optimal_panel(
            curve=curve,
            full_auroc_by_seed=full_aurocs,
            delta_primary=0.02,
        )
        # Paired diffs have small variance (~0.003), so SE ~ 0.003/sqrt(30) ~ 0.0005
        # CI_upper ~ 0.002 + 1.645*0.0005 ~ 0.003 << 0.02
        assert (
            result.selected_size == 10
        ), "Paired test with correlated seeds should pass non-inferiority"
        ni = result.decision_table[0].noninferiority
        assert ni.ci_upper < 0.02, f"Paired CI_upper={ni.ci_upper:.4f} should be << 0.02"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for decision_table_to_dict."""

    def test_serializes_to_dict(self, synthetic_pareto_curve):
        """Should produce a JSON-serializable dict."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
        )
        d = decision_table_to_dict(result)
        assert isinstance(d, dict)
        assert "selected_size" in d
        assert "selected_proteins" in d
        assert "decisions" in d
        assert isinstance(d["decisions"], list)

        # All values should be JSON-serializable (no numpy types)
        json.dumps(d)  # Should not raise


# ---------------------------------------------------------------------------
# Integration: output file generation
# ---------------------------------------------------------------------------


class TestOutputFileGeneration:
    """Tests for the output wiring (optimal_panel.txt, optimal_panel_selection.json)."""

    def test_selection_json_roundtrip(self, synthetic_pareto_curve):
        """decision_table_to_dict should produce JSON that roundtrips cleanly."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
        )
        d = decision_table_to_dict(result)

        # Add metadata fields as the wiring does
        d["model_name"] = "LR_EN"
        d["run_id"] = "20260307_120000"
        d["n_seeds"] = 5
        d["full_auroc_by_seed"] = [float(v) for v in full_aurocs]

        serialized = json.dumps(d, indent=2)
        loaded = json.loads(serialized)

        assert loaded["model_name"] == "LR_EN"
        assert loaded["n_seeds"] == 5
        assert loaded["selected_size"] == d["selected_size"]
        assert len(loaded["decisions"]) == len(synthetic_pareto_curve)

    def test_optimal_panel_txt_content(self, synthetic_pareto_curve, tmp_path):
        """optimal_panel.txt should contain one protein per line."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]
        result = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
        )

        if result.selected_size > 0:
            panel_txt = tmp_path / "optimal_panel.txt"
            with open(panel_txt, "w") as f:
                for protein in result.selected_proteins:
                    f.write(protein + "\n")

            lines = panel_txt.read_text().strip().splitlines()
            assert len(lines) == result.selected_size
            assert all(line.startswith("P") for line in lines)

    def test_essentiality_integration(self, synthetic_pareto_curve):
        """Essentiality map should remove passengers from selected panel."""
        full_aurocs = [0.886, 0.884, 0.890, 0.882, 0.888]

        # First find what size gets selected without essentiality
        result_no_ess = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
        )

        if result_no_ess.selected_size == 0:
            pytest.skip("No panel selected without essentiality")

        selected_size = result_no_ess.selected_size

        # Create essentiality data with one passenger
        proteins = result_no_ess.selected_proteins
        ess_df = pd.DataFrame(
            {
                "representative": proteins,
                "mean_delta_auroc": [0.05] * (len(proteins) - 1) + [0.0001],
                "std_delta_auroc": [0.005] * (len(proteins) - 1) + [0.01],
                "n_folds": [5] * len(proteins),
            }
        )

        result_with_ess = select_optimal_panel(
            curve=synthetic_pareto_curve,
            full_auroc_by_seed=full_aurocs,
            essentiality={selected_size: ess_df},
        )

        # Should have removed the passenger
        assert result_with_ess.n_passengers_removed >= 1
        assert result_with_ess.selected_size < selected_size
