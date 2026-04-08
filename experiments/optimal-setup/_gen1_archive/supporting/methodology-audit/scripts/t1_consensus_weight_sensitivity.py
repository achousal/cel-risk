#!/usr/bin/env python3
"""T1: Consensus Weight Sensitivity Analysis

Validates consensus weight configuration (0.6/0.3/0.1 for OOF/essentiality/stability)
by comparing top-k protein selection under multiple weight configurations.

Metrics computed:
- Jaccard overlap for top-k (k=10,20,50,100)
- Spearman/Kendall rank correlation for full rankings
- Selection frequency stability across configurations

Usage:
    # With real data from pipeline run
    python t1_consensus_weight_sensitivity.py --run-id 20260127_115115

    # With synthetic data (for testing/development)
    python t1_consensus_weight_sensitivity.py --synthetic

    # Custom weight configurations
    python t1_consensus_weight_sensitivity.py --run-id <RUN_ID> \
        --weight-configs "0.6,0.3,0.1" "0.5,0.4,0.1" "0.7,0.2,0.1"
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ced_ml.features.consensus import compute_per_model_ranking, geometric_mean_rank_aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """Weight configuration for consensus ranking."""

    oof_weight: float
    essentiality_weight: float
    stability_weight: float

    def __str__(self) -> str:
        return f"OOF={self.oof_weight:.1f}_Ess={self.essentiality_weight:.1f}_Stab={self.stability_weight:.1f}"

    @classmethod
    def from_tuple(cls, weights: tuple[float, float, float]) -> "WeightConfig":
        """Create from (oof, essentiality, stability) tuple."""
        return cls(oof_weight=weights[0], essentiality_weight=weights[1], stability_weight=weights[2])


def generate_synthetic_data(
    n_proteins: int = 500, n_models: int = 3, random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for testing weight sensitivity.

    Args:
        n_proteins: Number of proteins to simulate
        n_models: Number of models to simulate
        random_state: Random seed

    Returns:
        Dict with keys: 'stability', 'oof_importance', 'essentiality' for each model
    """
    logger.info(f"Generating synthetic data: {n_proteins} proteins, {n_models} models")
    rng = np.random.RandomState(random_state)

    data = {}
    model_names = [f"Model_{i+1}" for i in range(n_models)]
    protein_names = [f"Protein_{i:04d}" for i in range(n_proteins)]

    for model_name in model_names:
        # Stability: selection frequency [0, 1]
        stability = pd.DataFrame(
            {
                "protein": protein_names,
                "selection_fraction": rng.beta(2, 5, size=n_proteins),  # Skewed toward lower values
            }
        )

        # OOF importance: higher values = more important
        oof_importance = pd.DataFrame(
            {
                "feature": protein_names,
                "mean_importance": rng.gamma(2, 0.5, size=n_proteins),  # Right-skewed
            }
        )

        # Essentiality: mean delta AUROC (higher = more essential)
        essentiality = pd.DataFrame(
            {
                "representative": protein_names,
                "mean_delta_auroc": rng.exponential(0.02, size=n_proteins),  # Small positive values
            }
        )

        data[model_name] = {
            "stability": stability,
            "oof_importance": oof_importance,
            "essentiality": essentiality,
        }

    logger.info(f"✓ Generated synthetic data for {n_models} models")
    return data


def load_real_data(run_id: str, results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load real data from pipeline run.

    Args:
        run_id: Pipeline run ID
        results_dir: Path to results directory

    Returns:
        Dict with model data
    """
    run_dir = results_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.info(f"Loading data from run: {run_id}")

    # Find all model directories
    model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {run_dir}")

    data = {}
    for model_dir in model_dirs:
        model_name = model_dir.name

        # Load stability
        stability_file = model_dir / "aggregated" / "stability_summary.csv"
        if not stability_file.exists():
            logger.warning(f"Stability file not found for {model_name}, skipping")
            continue

        # Load OOF importance
        oof_file = model_dir / "aggregated" / "oof_grouped_importance_summary.csv"
        if not oof_file.exists():
            logger.warning(f"OOF importance file not found for {model_name}, skipping")
            continue

        # Load essentiality (optional)
        essentiality_file = model_dir / "aggregated" / "drop_column_importance_summary.csv"
        essentiality_df = None
        if essentiality_file.exists():
            essentiality_df = pd.read_csv(essentiality_file)

        data[model_name] = {
            "stability": pd.read_csv(stability_file),
            "oof_importance": pd.read_csv(oof_file),
            "essentiality": essentiality_df,
        }

        logger.info(f"✓ Loaded data for {model_name}")

    if not data:
        raise ValueError(f"No valid model data found in {run_dir}")

    logger.info(f"✓ Loaded data for {len(data)} models")
    return data


def compute_consensus_for_weights(
    model_data: dict[str, pd.DataFrame], weight_config: WeightConfig
) -> pd.DataFrame:
    """Compute consensus ranking for a given weight configuration.

    Args:
        model_data: Dict of model data
        weight_config: Weight configuration

    Returns:
        DataFrame with consensus ranking
    """
    per_model_rankings = {}

    for model_name, data in model_data.items():
        ranking = compute_per_model_ranking(
            stability_df=data["stability"],
            oof_importance_df=data["oof_importance"],
            essentiality_df=data["essentiality"],
            oof_weight=weight_config.oof_weight,
            essentiality_weight=weight_config.essentiality_weight,
            stability_weight=weight_config.stability_weight,
        )

        # Prepare for geometric_mean_rank_aggregate: needs columns [protein, final_rank]
        ranking_for_agg = ranking[["protein", "final_rank"]].copy()
        per_model_rankings[model_name] = ranking_for_agg

    # Apply geometric mean rank aggregation
    consensus = geometric_mean_rank_aggregate(per_model_rankings=per_model_rankings)

    return consensus


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_metrics(
    baseline_consensus: pd.DataFrame,
    comparison_consensus: pd.DataFrame,
    top_k_values: list[int] = [10, 20, 50, 100],
) -> dict[str, Any]:
    """Compute comparison metrics between two consensus rankings.

    Args:
        baseline_consensus: Baseline consensus ranking
        comparison_consensus: Comparison consensus ranking
        top_k_values: Values of k for top-k analysis

    Returns:
        Dict of metrics
    """
    metrics = {}

    # Get common proteins
    common_proteins = set(baseline_consensus["protein"]) & set(comparison_consensus["protein"])
    n_common = len(common_proteins)

    if n_common == 0:
        logger.warning("No common proteins between rankings")
        return {"n_common_proteins": 0}

    # Create rank dictionaries
    baseline_ranks = dict(zip(baseline_consensus["protein"], baseline_consensus["consensus_rank"]))
    comparison_ranks = dict(
        zip(comparison_consensus["protein"], comparison_consensus["consensus_rank"])
    )

    # Get ranks for common proteins
    common_baseline = np.array([baseline_ranks[p] for p in common_proteins])
    common_comparison = np.array([comparison_ranks[p] for p in common_proteins])

    # Spearman correlation
    spearman_r, spearman_p = spearmanr(common_baseline, common_comparison)
    metrics["spearman_r"] = float(spearman_r)
    metrics["spearman_p"] = float(spearman_p)

    # Kendall tau
    kendall_tau, kendall_p = kendalltau(common_baseline, common_comparison)
    metrics["kendall_tau"] = float(kendall_tau)
    metrics["kendall_p"] = float(kendall_p)

    # Jaccard overlap for top-k
    jaccard_scores = {}
    for k in top_k_values:
        if k > n_common:
            continue
        baseline_topk = set(baseline_consensus.nsmallest(k, "consensus_rank")["protein"])
        comparison_topk = set(comparison_consensus.nsmallest(k, "consensus_rank")["protein"])
        jaccard_scores[f"top_{k}"] = jaccard_similarity(baseline_topk, comparison_topk)

    metrics["jaccard_overlap"] = jaccard_scores
    metrics["n_common_proteins"] = n_common

    return metrics


def plot_results(
    results_df: pd.DataFrame, baseline_config: WeightConfig, output_dir: Path
) -> None:
    """Generate visualization plots for weight sensitivity analysis.

    Args:
        results_df: DataFrame with comparison results
        baseline_config: Baseline weight configuration
        output_dir: Output directory for plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # 1. Heatmap of Jaccard overlap for different k values
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract Jaccard data for each k
    k_values = [10, 20, 50, 100]
    for idx, k in enumerate(k_values):
        ax = axes[idx // 2, idx % 2]
        jaccard_data = []
        for _, row in results_df.iterrows():
            jaccard_data.append(row["jaccard_overlap"].get(f"top_{k}", np.nan))
        results_df[f"jaccard_top_{k}"] = jaccard_data

        # Create bar plot
        ax.barh(results_df["config_name"], results_df[f"jaccard_top_{k}"])
        ax.set_xlabel("Jaccard Similarity")
        ax.set_title(f"Top-{k} Protein Overlap")
        ax.set_xlim(0, 1)
        ax.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="0.8 threshold")
        ax.legend()

    plt.suptitle(f"Jaccard Overlap vs Baseline: {baseline_config}")
    plt.tight_layout()
    plt.savefig(output_dir / "jaccard_overlap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Rank correlation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.barh(results_df["config_name"], results_df["spearman_r"])
    ax1.set_xlabel("Spearman's ρ")
    ax1.set_title("Rank Correlation (Spearman)")
    ax1.set_xlim(-1, 1)
    ax1.axvline(0.9, color="red", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax1.legend()

    ax2.barh(results_df["config_name"], results_df["kendall_tau"])
    ax2.set_xlabel("Kendall's τ")
    ax2.set_title("Rank Correlation (Kendall)")
    ax2.set_xlim(-1, 1)
    ax2.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax2.legend()

    plt.suptitle(f"Rank Correlation vs Baseline: {baseline_config}")
    plt.tight_layout()
    plt.savefig(output_dir / "rank_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Summary heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create matrix for heatmap
    metrics_to_plot = ["spearman_r", "kendall_tau"]
    metrics_to_plot.extend([f"jaccard_top_{k}" for k in k_values if f"jaccard_top_{k}" in results_df.columns])

    heatmap_data = results_df[metrics_to_plot].T
    heatmap_data.columns = results_df["config_name"]

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.8,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Similarity Score"},
    )
    ax.set_title(f"Weight Sensitivity Summary vs Baseline: {baseline_config}")
    ax.set_ylabel("Metric")
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Plots saved to {output_dir}")


def run_analysis(
    model_data: dict[str, pd.DataFrame],
    weight_configs: list[WeightConfig],
    baseline_idx: int = 0,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run weight sensitivity analysis.

    Args:
        model_data: Dict of model data
        weight_configs: List of weight configurations to test
        baseline_idx: Index of baseline configuration
        output_dir: Output directory for results

    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Running analysis with {len(weight_configs)} weight configurations")

    # Compute consensus for all configurations
    consensus_rankings = {}
    for config in weight_configs:
        logger.info(f"Computing consensus for: {config}")
        consensus = compute_consensus_for_weights(model_data, config)
        consensus_rankings[str(config)] = consensus

    # Baseline configuration
    baseline_config = weight_configs[baseline_idx]
    baseline_consensus = consensus_rankings[str(baseline_config)]

    logger.info(f"Baseline configuration: {baseline_config}")

    # Compare each configuration to baseline
    results = []
    for i, config in enumerate(weight_configs):
        if i == baseline_idx:
            # Baseline vs itself
            results.append(
                {
                    "config_name": str(config),
                    "oof_weight": config.oof_weight,
                    "essentiality_weight": config.essentiality_weight,
                    "stability_weight": config.stability_weight,
                    "is_baseline": True,
                    "spearman_r": 1.0,
                    "spearman_p": 0.0,
                    "kendall_tau": 1.0,
                    "kendall_p": 0.0,
                    "jaccard_overlap": {f"top_{k}": 1.0 for k in [10, 20, 50, 100]},
                    "n_common_proteins": len(baseline_consensus),
                }
            )
            continue

        comparison_consensus = consensus_rankings[str(config)]
        metrics = compute_metrics(baseline_consensus, comparison_consensus)

        results.append(
            {
                "config_name": str(config),
                "oof_weight": config.oof_weight,
                "essentiality_weight": config.essentiality_weight,
                "stability_weight": config.stability_weight,
                "is_baseline": False,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / "weight_sensitivity_results.csv", index=False)
        logger.info(f"✓ Results saved to {output_dir / 'weight_sensitivity_results.csv'}")

        # Save detailed rankings for each configuration
        for config_name, consensus in consensus_rankings.items():
            safe_name = config_name.replace("=", "").replace(",", "_").replace(".", "")
            consensus.to_csv(output_dir / f"consensus_ranking_{safe_name}.csv", index=False)

        # Generate plots
        plot_results(results_df, baseline_config, output_dir)

        # Save summary report
        with open(output_dir / "summary_report.txt", "w") as f:
            f.write("T1: Consensus Weight Sensitivity Analysis\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis date: {datetime.now().isoformat()}\n")
            f.write(f"Baseline configuration: {baseline_config}\n")
            f.write(f"Number of configurations tested: {len(weight_configs)}\n")
            f.write(f"Number of models: {len(model_data)}\n\n")

            f.write("Summary Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Spearman ρ: {results_df[results_df['is_baseline']==False]['spearman_r'].mean():.3f}\n")
            f.write(f"Min Spearman ρ: {results_df[results_df['is_baseline']==False]['spearman_r'].min():.3f}\n")
            f.write(
                f"Mean Kendall τ: {results_df[results_df['is_baseline']==False]['kendall_tau'].mean():.3f}\n"
            )
            f.write(f"Min Kendall τ: {results_df[results_df['is_baseline']==False]['kendall_tau'].min():.3f}\n\n")

            # Top-k Jaccard statistics
            for k in [10, 20, 50, 100]:
                col = f"jaccard_top_{k}"
                if col in results_df.columns:
                    non_baseline = results_df[results_df["is_baseline"] == False]
                    f.write(f"Top-{k} Jaccard overlap:\n")
                    f.write(f"  Mean: {non_baseline[col].mean():.3f}\n")
                    f.write(f"  Min: {non_baseline[col].min():.3f}\n")
                    f.write(f"  Max: {non_baseline[col].max():.3f}\n\n")

            f.write("\nInterpretation:\n")
            f.write("-" * 80 + "\n")
            f.write("High correlation (ρ > 0.9, τ > 0.8) suggests weight choice is robust.\n")
            f.write("High Jaccard (> 0.8) for small k suggests top proteins are stable.\n")
            f.write("Lower overlap for larger k is expected (more variation in middle ranks).\n")

        logger.info(f"✓ Summary report saved to {output_dir / 'summary_report.txt'}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="T1: Consensus Weight Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-id", type=str, help="Pipeline run ID (e.g., 20260127_115115)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parents[3] / "results",
        help="Results directory (default: ../../../results)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real pipeline results",
    )
    parser.add_argument(
        "--weight-configs",
        nargs="+",
        help='Weight configurations as "oof,ess,stab" strings (default: predefined set)',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <results_dir>/empirical_validation/t1_<timestamp>)",
    )
    parser.add_argument(
        "--n-proteins", type=int, default=500, help="Number of proteins for synthetic data"
    )
    parser.add_argument(
        "--n-models", type=int, default=3, help="Number of models for synthetic data"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.synthetic and not args.run_id:
        parser.error("Either --run-id or --synthetic must be specified")

    # Load data
    if args.synthetic:
        logger.info("Using synthetic data")
        model_data = generate_synthetic_data(
            n_proteins=args.n_proteins, n_models=args.n_models
        )
    else:
        model_data = load_real_data(args.run_id, args.results_dir)

    # Define weight configurations
    if args.weight_configs:
        weight_configs = []
        for config_str in args.weight_configs:
            parts = [float(x) for x in config_str.split(",")]
            if len(parts) != 3:
                parser.error(f"Invalid weight config: {config_str} (expected 3 values)")
            weight_configs.append(WeightConfig.from_tuple(tuple(parts)))
    else:
        # Default configurations to test
        weight_configs = [
            WeightConfig(0.6, 0.3, 0.1),  # Baseline (current)
            WeightConfig(0.5, 0.4, 0.1),  # More balanced
            WeightConfig(0.7, 0.2, 0.1),  # OOF-dominant
            WeightConfig(0.4, 0.5, 0.1),  # Essentiality-dominant
            WeightConfig(0.5, 0.3, 0.2),  # Higher stability weight
            WeightConfig(0.33, 0.33, 0.34),  # Equal weights
            WeightConfig(0.8, 0.1, 0.1),  # Extreme OOF
        ]

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.synthetic:
            output_dir = args.results_dir / "empirical_validation" / f"t1_synthetic_{timestamp}"
        else:
            output_dir = args.results_dir / "empirical_validation" / f"t1_{args.run_id}_{timestamp}"

    # Run analysis
    results_df = run_analysis(
        model_data=model_data,
        weight_configs=weight_configs,
        baseline_idx=0,  # First config is baseline
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("T1: CONSENSUS WEIGHT SENSITIVITY ANALYSIS - SUMMARY")
    print("=" * 80)
    print(f"\nBaseline: {weight_configs[0]}")
    print(f"Configurations tested: {len(weight_configs)}")
    print(f"\nResults saved to: {output_dir}")
    print("\nKey Metrics (vs baseline):")
    print("-" * 80)

    non_baseline = results_df[results_df["is_baseline"] == False]
    print(f"Spearman ρ:  mean={non_baseline['spearman_r'].mean():.3f}, "
          f"min={non_baseline['spearman_r'].min():.3f}")
    print(f"Kendall τ:   mean={non_baseline['kendall_tau'].mean():.3f}, "
          f"min={non_baseline['kendall_tau'].min():.3f}")

    for k in [10, 20, 50]:
        col = f"jaccard_top_{k}"
        if col in non_baseline.columns:
            print(f"Jaccard@{k:3d}: mean={non_baseline[col].mean():.3f}, "
                  f"min={non_baseline[col].min():.3f}")

    print("\n" + "=" * 80)
    print("Review plots and detailed results in output directory")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
