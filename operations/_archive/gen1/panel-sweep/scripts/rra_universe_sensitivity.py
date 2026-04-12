"""Sensitivity experiment: RRA significance with full-universe correction.

Compares the original RRA permutation test (N=per-model list length, BH over
126 tested proteins) against a corrected version (N=2920, BH over 2920) to
determine which of the 7-protein panel members survive when the full search
space is accounted for.

Motivation: Bourgon et al. 2010 (PNAS) and Zehetmayer & Posch 2012 (BMC
Bioinformatics) show that BH correction over a pre-filtered subset is only
valid when the filter is independent of the test statistic under the null.
In this pipeline, the same outcome labels are used for ML feature selection
and the RRA permutation test, violating the independence condition.

Run:
    python experiments/optimal-setup/panel-sweep/scripts/rra_universe_sensitivity.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add source to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ced_ml.features.consensus.significance import rra_permutation_test

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RUN_ID = "run_20260317_131842"
RUN_DIR = RESULTS_DIR / RUN_ID

MODELS = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]
STABILITY_THRESHOLD = 0.95
N_PERMS = 1_000_000
ALPHA = 0.05
UNIVERSE_SIZE = 2920  # total proteins on the SomaScan panel

OUTDIR = RESULTS_DIR / "experiments" / "rra_universe_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_per_model_rankings() -> dict[str, pd.DataFrame]:
    """Load per-model stability-filtered rankings from the original run."""
    from ced_ml.features.consensus.ranking import compute_per_model_ranking

    rankings = {}
    for model in MODELS:
        model_dir = RUN_DIR / model / "aggregated"

        # Load stability summary
        stability_path = model_dir / "panels" / "feature_stability_summary.csv"
        if not stability_path.exists():
            print(f"  SKIP {model}: no stability summary at {stability_path}")
            continue

        stability_df = pd.read_csv(stability_path)
        stable_df = stability_df[
            stability_df["selection_fraction"] >= STABILITY_THRESHOLD
        ].copy()
        print(f"  {model}: {len(stable_df)} stable proteins (>= {STABILITY_THRESHOLD})")

        # Load OOF importance (at importance/oof_importance__{model}.csv)
        oof_path = model_dir / "importance" / f"oof_importance__{model}.csv"
        oof_df = pd.read_csv(oof_path) if oof_path.exists() else None
        if oof_df is not None:
            # Rename columns to match expected format
            if "feature" in oof_df.columns and "protein" not in oof_df.columns:
                oof_df = oof_df.rename(columns={"feature": "protein"})
            print(f"    -> OOF importance loaded ({len(oof_df)} features)")
        else:
            print(f"    -> OOF importance NOT found at {oof_path}")

        rankings[model] = compute_per_model_ranking(
            stability_df=stable_df,
            stability_col="selection_fraction",
            oof_importance_df=oof_df,
        )

    return rankings


def main():
    print("=" * 70)
    print("RRA Universe Sensitivity Experiment")
    print(f"Run: {RUN_ID}")
    print(f"Universe size: {UNIVERSE_SIZE}")
    print(f"Permutations: {N_PERMS}")
    print(f"Alpha: {ALPHA}")
    print("=" * 70)

    # Load rankings
    print("\nLoading per-model rankings...")
    rankings = load_per_model_rankings()
    print(f"\n{len(rankings)} models loaded")

    # --- Original: no universe correction (legacy) ---
    print("\n" + "-" * 70)
    print("TEST 1: Legacy (no universe correction)")
    print("-" * 70)
    result_legacy = rra_permutation_test(
        per_model_rankings=rankings,
        n_perms=N_PERMS,
        alpha=ALPHA,
        seed=42,
        universe_size=None,
    )
    n_sig_legacy = result_legacy["significant"].sum()
    print(f"\nSignificant: {n_sig_legacy}/{len(result_legacy)}")
    print(result_legacy[["protein", "observed_rra", "perm_p", "bh_adjusted_p", "significant"]].head(10).to_string(index=False))

    # --- Corrected: universe_size = 2920 ---
    print("\n" + "-" * 70)
    print(f"TEST 2: Full-universe correction (N={UNIVERSE_SIZE})")
    print("-" * 70)
    result_corrected = rra_permutation_test(
        per_model_rankings=rankings,
        n_perms=N_PERMS,
        alpha=ALPHA,
        seed=42,
        universe_size=UNIVERSE_SIZE,
    )
    n_sig_corrected = result_corrected["significant"].sum()
    print(f"\nSignificant: {n_sig_corrected}/{len(result_corrected)}")
    print(result_corrected[["protein", "observed_rra", "perm_p", "bh_adjusted_p", "significant"]].head(10).to_string(index=False))

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON: Top 10 proteins")
    print("=" * 70)

    comparison = result_legacy[["protein", "observed_rra", "perm_p", "bh_adjusted_p", "significant"]].copy()
    comparison.columns = ["protein", "rra_score", "perm_p_legacy", "bh_p_legacy", "sig_legacy"]

    corrected_cols = result_corrected[["protein", "perm_p", "bh_adjusted_p", "significant"]].copy()
    corrected_cols.columns = ["protein", "perm_p_corrected", "bh_p_corrected", "sig_corrected"]

    merged = comparison.merge(corrected_cols, on="protein", how="left")
    merged = merged.sort_values("rra_score", ascending=False).head(10)

    print(merged.to_string(index=False))

    # Status of original 7 panel members
    panel_7 = ["tgm2_resid", "cpa2_resid", "itgb7_resid", "gip_resid",
                "cxcl9_resid", "cd160_resid", "muc2_resid"]

    print("\n" + "=" * 70)
    print("PANEL STATUS: Original 7 proteins")
    print("=" * 70)
    panel_status = merged[merged["protein"].isin(panel_7)].copy()
    if len(panel_status) < 7:
        # Some might be beyond top 10
        full_merged = comparison.merge(corrected_cols, on="protein", how="left")
        panel_status = full_merged[full_merged["protein"].isin(panel_7)].sort_values("rra_score", ascending=False)

    print(panel_status.to_string(index=False))

    survived = panel_status["sig_corrected"].sum()
    lost = 7 - survived
    print(f"\nSurvived full-universe correction: {survived}/7")
    if lost > 0:
        lost_proteins = panel_status[~panel_status["sig_corrected"]]["protein"].tolist()
        print(f"Lost: {', '.join(lost_proteins)}")

    # Save artifacts
    result_legacy.to_csv(OUTDIR / "rra_significance_legacy.csv", index=False)
    result_corrected.to_csv(OUTDIR / "rra_significance_corrected.csv", index=False)

    metadata = {
        "run_id": RUN_ID,
        "universe_size": UNIVERSE_SIZE,
        "n_perms": N_PERMS,
        "alpha": ALPHA,
        "stability_threshold": STABILITY_THRESHOLD,
        "n_proteins_tested": len(result_legacy),
        "n_significant_legacy": int(n_sig_legacy),
        "n_significant_corrected": int(n_sig_corrected),
        "panel_survived": int(survived),
        "panel_lost": lost_proteins if lost > 0 else [],
        "references": [
            "Bourgon et al. 2010 PNAS 107:9546 (independent filtering)",
            "Zehetmayer & Posch 2012 BMC Bioinformatics 13:81 (two-stage FDR)",
        ],
    }
    with open(OUTDIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to: {OUTDIR}")
    print("  - rra_significance_legacy.csv")
    print("  - rra_significance_corrected.csv")
    print("  - experiment_metadata.json")


if __name__ == "__main__":
    main()
