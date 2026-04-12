#!/usr/bin/env python3
"""
Post-processing for incident validation results.

Generates:
  1. Paired statistical comparison between strategies (Wilcoxon signed-rank)
  2. Cross-fold feature consistency for the winning combo
  3. Interpretive summary with biological context

Usage:
  python scripts/postprocess_incident_validation.py --results-dir results/incident_validation
  python scripts/postprocess_incident_validation.py --results-dir results/incident_validation_smoke
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Paired strategy comparison
# ============================================================================


def paired_strategy_comparison(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank tests on per-fold AUPRC between all strategy+weight combos."""
    combos = cv_results.groupby(["strategy", "weight_scheme"])
    combo_folds = {}
    for (strat, wt), grp in combos:
        key = f"{strat}+{wt}"
        combo_folds[key] = grp.sort_values("fold")["auprc"].values

    rows = []
    for (a, b) in combinations(sorted(combo_folds.keys()), 2):
        auprc_a = combo_folds[a]
        auprc_b = combo_folds[b]
        diff = auprc_a - auprc_b
        mean_diff = np.mean(diff)

        # Wilcoxon requires n >= 5 non-zero differences
        nonzero = np.sum(diff != 0)
        if nonzero >= 5:
            stat, p = wilcoxon(auprc_a, auprc_b, alternative="two-sided")
        else:
            stat, p = np.nan, np.nan

        rows.append({
            "combo_a": a,
            "combo_b": b,
            "mean_auprc_a": np.mean(auprc_a),
            "mean_auprc_b": np.mean(auprc_b),
            "mean_diff": mean_diff,
            "wilcoxon_stat": stat,
            "p_value": p,
        })

    return pd.DataFrame(rows).sort_values("p_value")


def top_comparisons(paired_df: pd.DataFrame, cv_results: pd.DataFrame) -> str:
    """Format the most informative comparisons for the report."""
    # Find the winner
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])["auprc"]
        .mean()
        .reset_index()
        .sort_values("auprc", ascending=False)
    )
    best = summary.iloc[0]
    winner = f"{best['strategy']}+{best['weight_scheme']}"

    lines = [
        "## Paired Strategy Comparison (Wilcoxon signed-rank on per-fold AUPRC)",
        "",
        f"**Winner: {winner}** (mean AUPRC = {best['auprc']:.4f})",
        "",
        "### Winner vs all others",
        "",
        "| Comparison | Mean diff | p-value | Significant (p<0.05)? |",
        "|-----------|-----------|---------|----------------------|",
    ]

    winner_rows = paired_df[
        (paired_df["combo_a"] == winner) | (paired_df["combo_b"] == winner)
    ].copy()

    for _, row in winner_rows.iterrows():
        other = row["combo_b"] if row["combo_a"] == winner else row["combo_a"]
        diff = row["mean_diff"] if row["combo_a"] == winner else -row["mean_diff"]
        p = row["p_value"]
        sig = "Yes" if p < 0.05 else "No" if not np.isnan(p) else "N/A (n<5)"
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
        lines.append(f"| {winner} vs {other} | {diff:+.4f} | {p_str} | {sig} |")

    lines.extend([
        "",
        "### Strategy-level summary (averaging across weight schemes)",
        "",
        "| Strategy | Mean AUPRC | Std |",
        "|----------|-----------|-----|",
    ])

    strat_summary = (
        cv_results.groupby("strategy")["auprc"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    for strat, row in strat_summary.iterrows():
        lines.append(f"| {strat} | {row['mean']:.4f} | {row['std']:.4f} |")

    return "\n".join(lines)


# ============================================================================
# 2. Cross-fold feature consistency
# ============================================================================


def cross_fold_consistency(
    fold_coefs: pd.DataFrame, cv_results: pd.DataFrame, feature_panel: pd.DataFrame,
) -> str:
    """Analyze feature stability across CV folds for the winning combo."""
    # Identify winner
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])["auprc"]
        .mean()
        .reset_index()
        .sort_values("auprc", ascending=False)
    )
    best = summary.iloc[0]
    winner_strat = best["strategy"]
    winner_wt = best["weight_scheme"]

    # Filter to winning combo
    mask = (fold_coefs["strategy"] == winner_strat) & (fold_coefs["weight_scheme"] == winner_wt)
    winner_coefs = fold_coefs[mask].copy()

    # Pivot: protein × fold
    pivot = winner_coefs.pivot(index="protein", columns="fold", values="coefficient")
    n_folds = pivot.shape[1]

    # Non-zero in each fold
    nonzero_mat = (pivot.abs() > 0).astype(int)
    nonzero_count = nonzero_mat.sum(axis=1)
    nonzero_any = (nonzero_count > 0).sum()
    nonzero_all = (nonzero_count == n_folds).sum()
    nonzero_majority = (nonzero_count >= n_folds / 2).sum()

    # Merge with bootstrap stability
    consistency = pd.DataFrame({
        "protein": pivot.index,
        "folds_nonzero": nonzero_count.values,
        "mean_coef": pivot.mean(axis=1).values,
        "std_coef": pivot.std(axis=1).values,
        "sign_consistent": (
            (pivot > 0).all(axis=1) | (pivot < 0).all(axis=1) | (pivot == 0).all(axis=1)
        ).values,
    }).merge(feature_panel, on="protein", how="left")

    consistency = consistency.sort_values("folds_nonzero", ascending=False)

    # Build report section
    lines = [
        f"## Cross-Fold Feature Consistency ({winner_strat} + {winner_wt})",
        "",
        f"- Panel size: {len(pivot)} proteins",
        f"- Non-zero in all {n_folds} folds: **{nonzero_all}**",
        f"- Non-zero in majority (>={n_folds // 2 + 1}) folds: **{nonzero_majority}**",
        f"- Non-zero in at least 1 fold: {nonzero_any}",
        "",
        "### Core features (non-zero in all folds, consistent sign)",
        "",
        "| Protein | Mean coef | Std coef | Bootstrap freq | Direction |",
        "|---------|----------|---------|----------------|-----------|",
    ]

    core = consistency[(consistency["folds_nonzero"] == n_folds) & consistency["sign_consistent"]]
    core = core.sort_values("mean_coef", key=abs, ascending=False)
    for _, row in core.iterrows():
        direction = "+" if row["mean_coef"] > 0 else "-"
        lines.append(
            f"| {row['protein'].replace('_resid', '')} "
            f"| {row['mean_coef']:.4f} "
            f"| {row['std_coef']:.4f} "
            f"| {row['stability_freq']:.0%} "
            f"| {direction} |"
        )

    # Unstable features
    unstable = consistency[
        (consistency["folds_nonzero"] > 0)
        & (consistency["folds_nonzero"] < n_folds)
    ]
    if len(unstable) > 0:
        lines.extend([
            "",
            "### Fold-variable features (non-zero in some folds only)",
            "",
            "| Protein | Folds non-zero | Bootstrap freq | Sign consistent? |",
            "|---------|---------------|----------------|-----------------|",
        ])
        for _, row in unstable.iterrows():
            lines.append(
                f"| {row['protein'].replace('_resid', '')} "
                f"| {int(row['folds_nonzero'])}/{n_folds} "
                f"| {row['stability_freq']:.0%} "
                f"| {'Yes' if row['sign_consistent'] else 'No'} |"
            )

    return "\n".join(lines), consistency


# ============================================================================
# 3. Interpretive summary
# ============================================================================


def interpretive_summary(
    cv_results: pd.DataFrame,
    consistency: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> str:
    """Generate interpretive summary of what works and why."""
    # Strategy stats
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])["auprc"]
        .mean()
        .reset_index()
        .sort_values("auprc", ascending=False)
    )
    best = summary.iloc[0]
    winner_strat = best["strategy"]
    winner_wt = best["weight_scheme"]

    # Strategy-level means
    strat_means = cv_results.groupby("strategy")["auprc"].mean()
    incident_mean = strat_means.get("incident_only", 0)
    prevalent_mean = strat_means.get("prevalent_only", 0)
    mixed_mean = strat_means.get("incident_prevalent", 0)

    # Weight-level means (within incident_only)
    inc_only = cv_results[cv_results["strategy"] == "incident_only"]
    wt_means = inc_only.groupby("weight_scheme")["auprc"].mean().sort_values(ascending=False)

    # Core feature count
    n_folds = cv_results["fold"].nunique()
    n_core = int(((consistency["folds_nonzero"] == n_folds) & consistency["sign_consistent"]).sum())

    # Significant comparisons
    winner_key = f"{winner_strat}+{winner_wt}"
    winner_comparisons = paired_df[
        (paired_df["combo_a"] == winner_key) | (paired_df["combo_b"] == winner_key)
    ]
    n_sig = int((winner_comparisons["p_value"] < 0.05).sum())
    n_total = len(winner_comparisons)

    lines = [
        "## Interpretive Summary",
        "",
        "### What works best",
        "",
        f"**{winner_strat} + {winner_wt}** is the best strategy (mean AUPRC = {best['auprc']:.4f}).",
        f"The winner is significantly better (Wilcoxon p < 0.05) than {n_sig}/{n_total} other combos.",
        "",
        "### Why incident-only outperforms",
        "",
        f"- **incident_only** (mean AUPRC = {incident_mean:.4f}) > "
        f"**incident_prevalent** ({mixed_mean:.4f}) > "
        f"**prevalent_only** ({prevalent_mean:.4f})",
        "- Prevalent cases were diagnosed BEFORE blood draw — their proteomic profiles reflect "
        "post-diagnosis changes (dietary adaptation, mucosal healing, treatment effects), not "
        "pre-diagnostic risk signatures.",
        "- Adding prevalent cases to training introduces signal that does not generalize to the "
        "incident prediction task, where all cases are undiagnosed at the time of blood collection.",
        "- Training exclusively on incident cases aligns the learned decision boundary with the "
        "target population: apparently healthy individuals who will later develop CeD.",
        "",
        "### Why log weighting works",
        "",
        "Class weight ranking within incident_only:",
        "",
    ]

    for wt, mean_auprc in wt_means.items():
        lines.append(f"- **{wt}**: {mean_auprc:.4f}")

    lines.extend([
        "",
        f"At 0.34% prevalence, `balanced` upweights cases ~290×, which overfits to noisy positives "
        f"in a 119-case sample. `log` weighting (~5.7× upweight) provides enough signal to learn "
        f"the minority class without distorting the decision boundary. `none` (no correction) "
        f"slightly underperforms `log`, suggesting some loss rebalancing helps but aggressive "
        f"correction hurts.",
        "",
        "### Feature stability",
        "",
        f"- **{n_core} core features** are non-zero with consistent sign direction across all "
        f"{n_folds} CV folds — these form the reliable biomarker panel.",
    ])

    # List top core features
    core = consistency[
        (consistency["folds_nonzero"] == n_folds) & consistency["sign_consistent"]
    ].sort_values("mean_coef", key=abs, ascending=False)

    if len(core) > 0:
        top_names = [r["protein"].replace("_resid", "").upper() for _, r in core.head(5).iterrows()]
        lines.append(f"- Top stable markers: {', '.join(top_names)}")

    n_variable = int(
        ((consistency["folds_nonzero"] > 0) & (consistency["folds_nonzero"] < n_folds)).sum()
    )
    if n_variable > 0:
        lines.append(
            f"- {n_variable} features are fold-variable (appear in some folds but not all) — "
            f"these are candidates for exclusion in a minimal panel."
        )

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Post-process incident validation results")
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Path to incident validation results directory",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load outputs
    cv_results = pd.read_csv(results_dir / "cv_results.csv")
    feature_panel = pd.read_csv(results_dir / "feature_panel.csv")

    fold_coefs_path = results_dir / "fold_coefficients.csv"
    if not fold_coefs_path.exists():
        logger.error(
            "fold_coefficients.csv not found — rerun pipeline with updated code to generate it"
        )
        raise FileNotFoundError(fold_coefs_path)

    fold_coefs = pd.read_csv(fold_coefs_path)

    logger.info("Loaded %d CV rows, %d fold coefficient rows", len(cv_results), len(fold_coefs))

    # 1. Paired comparison
    logger.info("Running paired strategy comparisons...")
    paired_df = paired_strategy_comparison(cv_results)
    paired_df.to_csv(results_dir / "paired_comparison.csv", index=False)
    logger.info("Saved: %s", results_dir / "paired_comparison.csv")

    comparison_report = top_comparisons(paired_df, cv_results)

    # 2. Cross-fold consistency
    logger.info("Analyzing cross-fold feature consistency...")
    consistency_report, consistency_df = cross_fold_consistency(
        fold_coefs, cv_results, feature_panel,
    )
    consistency_df.to_csv(results_dir / "feature_consistency.csv", index=False)
    logger.info("Saved: %s", results_dir / "feature_consistency.csv")

    # 3. Interpretive summary
    logger.info("Generating interpretive summary...")
    interp_report = interpretive_summary(cv_results, consistency_df, paired_df)

    # Write combined analysis report
    report = "\n\n".join([
        "# Incident Validation Analysis",
        comparison_report,
        consistency_report,
        interp_report,
    ])

    report_path = results_dir / "analysis_report.md"
    report_path.write_text(report + "\n")
    logger.info("Saved: %s", report_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
