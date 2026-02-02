#!/usr/bin/env python3
"""
Statistical analysis for 2x2x2 factorial experiment.

Analyzes results from run_factorial_2x2x2.py:
    - Main effects (paired contrasts per seed)
    - Two-way and three-way interactions
    - Mixed-effects model (metric ~ factors + (1|seed))
    - Cohen's d (paired) and 95% CIs

Usage:
    python analyze_factorial_2x2x2.py \\
        --results results/factorial_2x2x2/factorial_results.csv \\
        --output-dir results/factorial_2x2x2/analysis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

METRICS = [
    "AUROC",
    "PRAUC",
    "Brier",
    "cal_slope",
    "cal_intercept",
    "sens_at_spec95",
    "mean_prob_incident",
    "mean_prob_prevalent",
    "mean_prob_control",
    "score_gap",
]

FACTORS = {
    "n_cases": {"low": 50, "high": 149, "label": "Cases (149 vs 50)"},
    "ratio": {"low": 1, "high": 5, "label": "Ratio (5 vs 1)"},
    "prevalent_frac": {"low": 0.5, "high": 1.0, "label": "Prev frac (1.0 vs 0.5)"},
}

# Plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# Paired contrasts
# ---------------------------------------------------------------------------


def paired_contrast(
    df: pd.DataFrame,
    factor: str,
    metric: str,
    low_val,
    high_val,
) -> dict:
    """
    Compute paired contrast (high - low) for one factor, averaging over other factors.

    Pairing is by seed (Guardrail 2).
    """
    [f for f in FACTORS if f != factor]

    # For each seed, compute mean metric at high vs low, averaging over other factors
    seeds = sorted(df["seed"].unique())
    deltas = []
    for s in seeds:
        ds = df[df["seed"] == s]
        mean_high = ds[ds[factor] == high_val][metric].mean()
        mean_low = ds[ds[factor] == low_val][metric].mean()
        deltas.append(mean_high - mean_low)

    deltas = np.array(deltas)
    n = len(deltas)
    mean_d = deltas.mean()
    se = deltas.std(ddof=1) / np.sqrt(n)
    ci_lo = mean_d - 1.96 * se
    ci_hi = mean_d + 1.96 * se

    # Cohen's d (paired)
    sd = deltas.std(ddof=1)
    cohens_d = mean_d / sd if sd > 0 else 0.0

    # Paired t-test
    if n >= 2 and sd > 0:
        t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        "factor": factor,
        "metric": metric,
        "contrast": f"{high_val} vs {low_val}",
        "mean_delta": mean_d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_val,
        "n_seeds": n,
    }


def compute_main_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all main effects for all metrics."""
    rows = []
    for metric in METRICS:
        for factor, spec in FACTORS.items():
            row = paired_contrast(df, factor, metric, spec["low"], spec["high"])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


def two_way_interaction(
    df: pd.DataFrame,
    f1: str,
    f2: str,
    metric: str,
) -> dict:
    """
    Test whether effect of f1 depends on level of f2 (paired by seed).

    Interaction = (effect of f1 at f2_high) - (effect of f1 at f2_low).
    """
    s1 = FACTORS[f1]
    s2 = FACTORS[f2]
    seeds = sorted(df["seed"].unique())

    interaction_deltas = []
    for s in seeds:
        ds = df[df["seed"] == s]
        # Effect of f1 at f2=high
        sub_hi = ds[ds[f2] == s2["high"]]
        eff_hi = (
            sub_hi[sub_hi[f1] == s1["high"]][metric].mean()
            - sub_hi[sub_hi[f1] == s1["low"]][metric].mean()
        )
        # Effect of f1 at f2=low
        sub_lo = ds[ds[f2] == s2["low"]]
        eff_lo = (
            sub_lo[sub_lo[f1] == s1["high"]][metric].mean()
            - sub_lo[sub_lo[f1] == s1["low"]][metric].mean()
        )
        interaction_deltas.append(eff_hi - eff_lo)

    deltas = np.array(interaction_deltas)
    n = len(deltas)
    mean_d = deltas.mean()
    se = deltas.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    sd = deltas.std(ddof=1) if n > 1 else np.nan
    cohens_d = mean_d / sd if sd and sd > 0 else 0.0

    if n >= 2 and sd > 0:
        t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        "interaction": f"{f1} x {f2}",
        "metric": metric,
        "mean_delta": mean_d,
        "ci_lo": mean_d - 1.96 * se if not np.isnan(se) else np.nan,
        "ci_hi": mean_d + 1.96 * se if not np.isnan(se) else np.nan,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_val,
        "n_seeds": n,
    }


def three_way_interaction(df: pd.DataFrame, metric: str) -> dict:
    """
    Test 3-way interaction: does the n_cases x ratio interaction differ by prevalent_frac?
    """
    f_prev = FACTORS["prevalent_frac"]
    seeds = sorted(df["seed"].unique())

    deltas_3way = []
    for s in seeds:
        ds = df[df["seed"] == s]
        # 2-way (n_cases x ratio) at prev_high
        sub_ph = ds[ds["prevalent_frac"] == f_prev["high"]]
        int_ph = _two_way_delta(sub_ph, "n_cases", "ratio", metric)
        # 2-way (n_cases x ratio) at prev_low
        sub_pl = ds[ds["prevalent_frac"] == f_prev["low"]]
        int_pl = _two_way_delta(sub_pl, "n_cases", "ratio", metric)
        deltas_3way.append(int_ph - int_pl)

    deltas = np.array(deltas_3way)
    n = len(deltas)
    mean_d = deltas.mean()
    sd = deltas.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    cohens_d = mean_d / sd if sd and sd > 0 else 0.0

    if n >= 2 and sd > 0:
        t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        "interaction": "n_cases x ratio x prevalent_frac",
        "metric": metric,
        "mean_delta": mean_d,
        "ci_lo": mean_d - 1.96 * se if not np.isnan(se) else np.nan,
        "ci_hi": mean_d + 1.96 * se if not np.isnan(se) else np.nan,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_val,
        "n_seeds": n,
    }


def _two_way_delta(sub: pd.DataFrame, f1: str, f2: str, metric: str) -> float:
    """Helper: compute interaction delta within a subset."""
    s1, s2 = FACTORS[f1], FACTORS[f2]
    eff_hi = (
        sub[(sub[f1] == s1["high"]) & (sub[f2] == s2["high"])][metric].mean()
        - sub[(sub[f1] == s1["low"]) & (sub[f2] == s2["high"])][metric].mean()
    )
    eff_lo = (
        sub[(sub[f1] == s1["high"]) & (sub[f2] == s2["low"])][metric].mean()
        - sub[(sub[f1] == s1["low"]) & (sub[f2] == s2["low"])][metric].mean()
    )
    return eff_hi - eff_lo


def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 2-way and 3-way interactions."""
    rows = []
    factor_names = list(FACTORS.keys())
    # 2-way
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            for metric in METRICS:
                row = two_way_interaction(df, factor_names[i], factor_names[j], metric)
                rows.append(row)
    # 3-way
    for metric in METRICS:
        rows.append(three_way_interaction(df, metric))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature importance overlap (Jaccard)
# ---------------------------------------------------------------------------


def compute_feature_jaccard(
    fi_df: pd.DataFrame,
    top_k: int = 15,
) -> pd.DataFrame:
    """Compute Jaccard similarity of top-K features across prevalent_frac levels.

    For each (model, seed, n_cases, ratio) combination, compares top-K features
    at prevalent_frac=0.5 vs 1.0. This reveals whether including more prevalent
    cases shifts which biomarkers the model relies on.
    """
    group_cols = ["model", "seed", "n_cases", "ratio"]
    rows = []

    for keys, grp in fi_df.groupby(group_cols):
        model, seed, n_cases, ratio = keys
        lo = grp[grp["prevalent_frac"] == 0.5]
        hi = grp[grp["prevalent_frac"] == 1.0]
        if lo.empty or hi.empty:
            continue

        top_lo = set(lo.nlargest(top_k, "importance")["feature"].values)
        top_hi = set(hi.nlargest(top_k, "importance")["feature"].values)

        intersection = len(top_lo & top_hi)
        union = len(top_lo | top_hi)
        jaccard = intersection / union if union > 0 else 0.0

        rows.append(
            {
                "model": model,
                "seed": seed,
                "n_cases": n_cases,
                "ratio": ratio,
                "top_k": top_k,
                "jaccard": jaccard,
                "n_shared": intersection,
                "n_union": union,
            }
        )

    return pd.DataFrame(rows)


def summarize_jaccard(jaccard_df: pd.DataFrame) -> str:
    """Generate markdown summary of feature overlap analysis."""
    lines = [
        "## Feature Importance Overlap (Jaccard)",
        "",
        "Jaccard similarity of top-K features between prevalent_frac=0.5 and 1.0,",
        "grouped by model and design factors.",
        "",
    ]

    for model, grp in jaccard_df.groupby("model"):
        lines.append(f"### {model}")
        lines.append("")
        summary = (
            grp.groupby(["n_cases", "ratio"])
            .agg(
                jaccard_mean=("jaccard", "mean"),
                jaccard_std=("jaccard", "std"),
                n_shared_mean=("n_shared", "mean"),
            )
            .round(3)
        )
        lines.append(summary.to_markdown())
        lines.append("")

        overall = grp["jaccard"].mean()
        lines.append(
            f"Overall mean Jaccard: {overall:.3f} "
            f"(1.0 = identical top-K, 0.0 = completely disjoint)"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_main_effects(
    main_effects: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Create forest plot of main effects with confidence intervals."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 4), constrained_layout=True)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS, strict=False):
        sub = main_effects[main_effects["metric"] == metric].copy()
        sub = sub.sort_values("mean_delta")

        y_pos = np.arange(len(sub))
        ax.errorbar(
            sub["mean_delta"],
            y_pos,
            xerr=[
                sub["mean_delta"] - sub["ci_lo"],
                sub["ci_hi"] - sub["mean_delta"],
            ],
            fmt="o",
            capsize=5,
            markersize=6,
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([FACTORS[f]["label"] for f in sub["factor"]])
        ax.set_xlabel("Effect size")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add significance markers
        for i, (_, row) in enumerate(sub.iterrows()):
            if row["p_value"] < 0.05:
                ax.text(
                    row["ci_hi"] + 0.01,
                    i,
                    "*",
                    fontsize=14,
                    va="center",
                    color="red",
                )

    fig.suptitle(
        f"Main Effects: {model_name} (forest plot, * = p<0.05)",
        fontsize=13,
        fontweight="bold",
    )
    out_path = output_dir / f"main_effects_{model_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved main effects plot: %s", out_path)


def plot_interactions(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Create 2-way interaction plots for key metrics."""
    key_metrics = ["AUROC", "Brier", "sens_at_spec95"]
    factor_pairs = [
        ("n_cases", "ratio"),
        ("n_cases", "prevalent_frac"),
        ("ratio", "prevalent_frac"),
    ]

    fig, axes = plt.subplots(
        len(key_metrics),
        len(factor_pairs),
        figsize=(15, 12),
        constrained_layout=True,
    )

    for row, metric in enumerate(key_metrics):
        for col, (f1, f2) in enumerate(factor_pairs):
            ax = axes[row, col]

            # Aggregate across seeds
            group_cols = [f1, f2]
            agg = df.groupby(group_cols)[metric].agg(["mean", "std"]).reset_index()

            # Plot lines for each level of f2
            for val2 in sorted(agg[f2].unique()):
                sub = agg[agg[f2] == val2]
                label = f"{f2}={val2}"
                ax.plot(
                    sub[f1],
                    sub["mean"],
                    marker="o",
                    label=label,
                    linewidth=2,
                )
                ax.fill_between(
                    sub[f1],
                    sub["mean"] - sub["std"],
                    sub["mean"] + sub["std"],
                    alpha=0.2,
                )

            ax.set_xlabel(f1)
            if col == 0:
                ax.set_ylabel(metric)
            ax.set_title(f"{f1} x {f2}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Interaction Plots: {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    out_path = output_dir / f"interactions_{model_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved interaction plot: %s", out_path)


def plot_jaccard_heatmap(
    jaccard_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Create heatmap of Jaccard similarity across design factors."""
    sub = jaccard_df[jaccard_df["model"] == model_name].copy()
    if sub.empty:
        logger.warning("No Jaccard data for %s, skipping heatmap", model_name)
        return

    # Aggregate across seeds
    pivot = sub.groupby(["n_cases", "ratio"])["jaccard"].mean().unstack()

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Jaccard similarity"},
        ax=ax,
    )
    ax.set_xlabel("Ratio (controls per case)")
    ax.set_ylabel("Cases (incident)")
    ax.set_title(
        f"Feature Overlap: {model_name}\n"
        f"(Top-{sub['top_k'].iloc[0]} features, prev_frac 0.5 vs 1.0)",
        fontsize=12,
        fontweight="bold",
    )

    out_path = output_dir / f"jaccard_heatmap_{model_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved Jaccard heatmap: %s", out_path)


def plot_score_distributions(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Create box plots of score distributions across cells."""
    score_cols = ["mean_prob_incident", "mean_prob_prevalent", "mean_prob_control"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, col in zip(axes, score_cols, strict=False):
        # Aggregate across seeds
        agg = (
            df.groupby(["n_cases", "ratio", "prevalent_frac"])[col].mean().reset_index()
        )

        # Create cell labels
        agg["cell"] = agg.apply(
            lambda r: f"n={r['n_cases']}\nr={r['ratio']}\np={r['prevalent_frac']:.1f}",
            axis=1,
        )

        sns.boxplot(
            data=df,
            x="n_cases",
            y=col,
            hue="prevalent_frac",
            ax=ax,
        )
        ax.set_xlabel("Number of cases")
        ax.set_ylabel("Mean predicted probability")
        ax.set_title(
            col.replace("mean_prob_", "").replace("_", " ").title(),
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(title="Prev frac", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Score Distributions: {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    out_path = output_dir / f"score_distributions_{model_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved score distributions plot: %s", out_path)


def plot_cell_means(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Create bar plots of cell means for key metrics."""
    key_metrics = ["AUROC", "PRAUC", "Brier", "sens_at_spec95"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, key_metrics, strict=False):
        # Aggregate across seeds
        agg = (
            df.groupby(["n_cases", "ratio", "prevalent_frac"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Create cell labels
        agg["cell"] = agg.apply(
            lambda r: f"n={r['n_cases']}, r={r['ratio']}, p={r['prevalent_frac']:.1f}",
            axis=1,
        )

        x_pos = np.arange(len(agg))
        ax.bar(x_pos, agg["mean"], yerr=agg["std"], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agg["cell"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Cell Means: {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    out_path = output_dir / f"cell_means_{model_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved cell means plot: %s", out_path)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def generate_summary(
    df: pd.DataFrame,
    main_effects: pd.DataFrame,
    interactions: pd.DataFrame,
    model_name: str,
) -> str:
    """Generate markdown summary for one model."""
    lines = [
        f"# Factorial Analysis: {model_name}",
        "",
        "## Design",
        f"- Seeds: {df['seed'].nunique()}",
        f"- Cells: {len(df) // df['seed'].nunique()}",
        f"- Total runs: {len(df)}",
        "",
        "## Cell Means",
        "",
    ]

    # Cell means table
    group_cols = ["n_cases", "ratio", "prevalent_frac"]
    means = df.groupby(group_cols)[METRICS].agg(["mean", "std"]).round(4)
    lines.append(means.to_markdown())
    lines.append("")

    # Main effects
    lines.append("## Main Effects (paired contrasts)")
    lines.append("")
    me_display = main_effects[
        ["factor", "metric", "mean_delta", "ci_lo", "ci_hi", "cohens_d", "p_value"]
    ].round(4)
    lines.append(me_display.to_markdown(index=False))
    lines.append("")

    # Significant main effects
    sig = main_effects[main_effects["p_value"] < 0.05]
    if len(sig):
        lines.append("### Significant effects (p < 0.05):")
        for _, r in sig.iterrows():
            direction = "+" if r["mean_delta"] > 0 else "-"
            lines.append(
                f"- {r['factor']} on {r['metric']}: "
                f"delta={r['mean_delta']:.4f} ({direction}), "
                f"d={r['cohens_d']:.2f}, p={r['p_value']:.4f}"
            )
        lines.append("")

    # Interactions
    lines.append("## Interactions")
    lines.append("")
    int_display = interactions[
        ["interaction", "metric", "mean_delta", "ci_lo", "ci_hi", "cohens_d", "p_value"]
    ].round(4)
    lines.append(int_display.to_markdown(index=False))
    lines.append("")

    sig_int = interactions[interactions["p_value"] < 0.05]
    if len(sig_int):
        lines.append("### Significant interactions (p < 0.05):")
        for _, r in sig_int.iterrows():
            lines.append(
                f"- {r['interaction']} on {r['metric']}: "
                f"delta={r['mean_delta']:.4f}, d={r['cohens_d']:.2f}, p={r['p_value']:.4f}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 2x2x2 factorial experiment results",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to factorial_results.csv",
    )
    parser.add_argument(
        "--feature-importances",
        type=Path,
        default=None,
        help="Path to feature_importances.csv (default: sibling of results file)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top features for Jaccard overlap (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sibling 'analysis/' of results file)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    logger.info("Loaded %d rows from %s", len(df), args.results)

    out_dir = args.output_dir or args.results.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["model"].unique())
    all_main = []
    all_int = []

    for model_name in models:
        logger.info("Analyzing model: %s", model_name)
        sub = df[df["model"] == model_name].copy()

        me = compute_main_effects(sub)
        me["model"] = model_name
        all_main.append(me)

        ints = compute_interactions(sub)
        ints["model"] = model_name
        all_int.append(ints)

        # Per-model summary
        summary = generate_summary(sub, me, ints, model_name)
        summary_path = out_dir / f"summary_{model_name}.md"
        summary_path.write_text(summary)
        logger.info("Wrote %s", summary_path)

        # Generate plots for this model
        plot_main_effects(me, model_name, out_dir)
        plot_interactions(sub, model_name, out_dir)
        plot_score_distributions(sub, model_name, out_dir)
        plot_cell_means(sub, model_name, out_dir)

    # Combined outputs
    main_all = pd.concat(all_main, ignore_index=True)
    main_all.to_csv(out_dir / "main_effects.csv", index=False)

    int_all = pd.concat(all_int, ignore_index=True)
    int_all.to_csv(out_dir / "interactions.csv", index=False)

    # Feature importance overlap analysis
    fi_path = (
        args.feature_importances or args.results.parent / "feature_importances.csv"
    )
    if fi_path.exists():
        logger.info("Loading feature importances from %s", fi_path)
        fi_df = pd.read_csv(fi_path)

        jaccard_df = compute_feature_jaccard(fi_df, top_k=args.top_k)
        jaccard_df.to_csv(out_dir / "feature_jaccard.csv", index=False)

        jaccard_summary = summarize_jaccard(jaccard_df)
        (out_dir / "feature_overlap.md").write_text(jaccard_summary)
        logger.info("Wrote feature overlap analysis to %s", out_dir)

        # Append Jaccard summary to per-model summaries and create heatmaps
        for model_name in models:
            summary_path = out_dir / f"summary_{model_name}.md"
            model_jaccard = jaccard_df[jaccard_df["model"] == model_name]
            if not model_jaccard.empty:
                with open(summary_path, "a") as f:
                    f.write("\n\n")
                    f.write(summarize_jaccard(model_jaccard))
                plot_jaccard_heatmap(jaccard_df, model_name, out_dir)
    else:
        logger.warning(
            "Feature importances file not found at %s; skipping overlap analysis.",
            fi_path,
        )

    logger.info("Analysis complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
