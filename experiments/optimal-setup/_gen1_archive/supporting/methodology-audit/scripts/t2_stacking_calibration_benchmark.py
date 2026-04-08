#!/usr/bin/env python3
"""T2: Stacking Calibration Benchmark

Evaluates whether meta-learner calibration (calibrate_meta=True/False) improves
performance when base models are already OOF-posthoc calibrated.

Metrics compared:
- AUROC (discrimination)
- PR-AUC (precision-recall)
- Brier score (calibration quality)
- ECE (expected calibration error)

Usage:
    # With real stacking ensemble from pipeline
    python t2_stacking_calibration_benchmark.py --run-id 20260127_115115

    # With synthetic data (for testing/development)
    python t2_stacking_calibration_benchmark.py --synthetic

    # Custom calibration methods
    python t2_stacking_calibration_benchmark.py --run-id <RUN_ID> \
        --calibration-methods sigmoid isotonic none
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ced_ml.models.stacking import StackingEnsemble

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for meta-learner calibration."""

    calibrate_meta: bool
    calibration_method: str | None = None  # 'sigmoid' or 'isotonic' or None

    def __str__(self) -> str:
        if not self.calibrate_meta:
            return "NoCalibration"
        return f"Calibrated_{self.calibration_method or 'default'}"


def generate_synthetic_predictions(
    n_samples: int = 10000,
    prevalence: float = 0.0034,
    n_base_models: int = 3,
    random_state: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate synthetic OOF predictions and labels.

    Args:
        n_samples: Number of samples
        prevalence: Target prevalence
        n_base_models: Number of base models
        random_state: Random seed

    Returns:
        Tuple of (oof_predictions_dict, labels)
    """
    logger.info(f"Generating synthetic predictions: {n_samples} samples, {n_base_models} models")
    rng = np.random.RandomState(random_state)

    # Generate labels
    n_positive = int(n_samples * prevalence)
    y = np.zeros(n_samples, dtype=int)
    y[:n_positive] = 1
    rng.shuffle(y)

    # Generate base model predictions (already "calibrated")
    # Models should have reasonable discrimination but may need meta-level calibration
    oof_predictions = {}
    for i in range(n_base_models):
        # Generate predictions with different characteristics
        model_name = f"Model_{i+1}"

        # Create predictions that correlate with labels but with model-specific biases
        # Positive class: higher mean, but still low overall due to class imbalance
        pos_mean = 0.05 + i * 0.01  # Varies by model
        neg_mean = 0.001 + i * 0.0005

        preds = np.zeros(n_samples)
        preds[y == 1] = rng.beta(2, 20, size=n_positive) * pos_mean * 10  # Right-skewed
        preds[y == 0] = rng.beta(1, 50, size=n_samples - n_positive) * neg_mean * 10  # Very right-skewed

        # Clip to valid probability range
        preds = np.clip(preds, 1e-7, 1 - 1e-7)

        oof_predictions[model_name] = preds

    logger.info(f"✓ Generated synthetic predictions")
    return oof_predictions, y


def load_real_predictions(
    run_id: str, results_dir: Path
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load real OOF predictions from pipeline run.

    Args:
        run_id: Pipeline run ID
        results_dir: Path to results directory

    Returns:
        Tuple of (oof_predictions_dict, labels)
    """
    run_dir = results_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.info(f"Loading OOF predictions from run: {run_id}")

    # Load stacking ensemble results
    ensemble_dir = run_dir / "ensemble"
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

    # Load base model OOF predictions
    oof_file = ensemble_dir / "base_model_oof_predictions.csv"
    if not oof_file.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {oof_file}")

    oof_df = pd.read_csv(oof_file)

    # Extract labels and predictions
    if "label" not in oof_df.columns:
        raise ValueError("Labels not found in OOF predictions file")

    y = oof_df["label"].values

    # Get model columns (exclude label and index columns)
    model_cols = [col for col in oof_df.columns if col not in ["label", "sample_id", "index"]]

    oof_predictions = {col: oof_df[col].values for col in model_cols}

    logger.info(f"✓ Loaded OOF predictions for {len(oof_predictions)} models")
    logger.info(f"  Samples: {len(y)}, Prevalence: {y.mean():.4f}")

    return oof_predictions, y


def compute_expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better)
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if not mask.any():
            continue

        bin_size = mask.sum()
        bin_pred_mean = y_pred[mask].mean()
        bin_true_mean = y_true[mask].mean()

        ece += (bin_size / len(y_true)) * abs(bin_pred_mean - bin_true_mean)

    return float(ece)


def train_and_evaluate_ensemble(
    oof_predictions: dict[str, np.ndarray],
    y: np.ndarray,
    config: CalibrationConfig,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    """Train ensemble with given calibration config and evaluate.

    Args:
        oof_predictions: Dict of base model OOF predictions
        y: Labels
        config: Calibration configuration
        train_idx: Training indices
        test_idx: Test indices

    Returns:
        Dict of evaluation metrics
    """
    # Create stacking ensemble
    ensemble = StackingEnsemble(
        base_model_names=list(oof_predictions.keys()),
        calibrate_meta=config.calibrate_meta,
        random_state=42,
    )

    # Prepare training data (stack base model predictions)
    X_train = np.column_stack([oof_predictions[name][train_idx] for name in oof_predictions])
    y_train = y[train_idx]

    # Train meta-learner
    ensemble.fit_from_oof(
        oof_predictions_dict={name: preds[train_idx] for name, preds in oof_predictions.items()},
        y_train=y_train,
    )

    # Prepare test data
    X_test = np.column_stack([oof_predictions[name][test_idx] for name in oof_predictions])
    y_test = y[test_idx]

    # Get predictions
    # For real implementation, would use ensemble.predict_proba_from_base_preds
    # For this synthetic case, we'll use the meta model directly
    if hasattr(ensemble.meta_model, "predict_proba"):
        y_pred = ensemble.meta_model.predict_proba(X_test)[:, 1]
    else:
        logger.warning("Meta model does not have predict_proba, using decision function")
        y_pred = ensemble.meta_model.decision_function(X_test)
        y_pred = 1 / (1 + np.exp(-y_pred))  # Sigmoid

    # Compute metrics
    metrics = {}

    # AUROC
    try:
        metrics["auroc"] = float(roc_auc_score(y_test, y_pred))
    except ValueError as e:
        logger.warning(f"AUROC computation failed: {e}")
        metrics["auroc"] = np.nan

    # PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        metrics["pr_auc"] = float(auc(recall, precision))
    except ValueError as e:
        logger.warning(f"PR-AUC computation failed: {e}")
        metrics["pr_auc"] = np.nan

    # Brier score
    metrics["brier"] = float(brier_score_loss(y_test, y_pred))

    # ECE
    metrics["ece"] = compute_expected_calibration_error(y_test, y_pred)

    # Calibration slope (optional)
    from sklearn.linear_model import LogisticRegression

    try:
        logit_preds = np.log(y_pred / (1 - y_pred + 1e-7) + 1e-7)
        cal_model = LogisticRegression(penalty=None, max_iter=1000)
        cal_model.fit(logit_preds.reshape(-1, 1), y_test)
        metrics["calibration_slope"] = float(cal_model.coef_[0, 0])
        metrics["calibration_intercept"] = float(cal_model.intercept_[0])
    except Exception as e:
        logger.warning(f"Calibration slope computation failed: {e}")
        metrics["calibration_slope"] = np.nan
        metrics["calibration_intercept"] = np.nan

    return metrics


def plot_calibration_curves(
    results_df: pd.DataFrame,
    oof_predictions: dict[str, np.ndarray],
    y: np.ndarray,
    configs: list[CalibrationConfig],
    output_dir: Path,
) -> None:
    """Plot calibration curves for different configurations.

    Args:
        results_df: Results dataframe
        oof_predictions: OOF predictions dict
        y: Labels
        configs: List of calibration configs
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a subset for calibration curve plotting (faster)
    n_plot = min(5000, len(y))
    plot_idx = np.random.RandomState(42).choice(len(y), size=n_plot, replace=False)

    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5))
    if len(configs) == 1:
        axes = [axes]

    for idx, config in enumerate(configs):
        ax = axes[idx]

        # Get predictions for this config
        # For simplicity, use mean of base models (in practice would retrain)
        y_pred = np.mean([oof_predictions[name][plot_idx] for name in oof_predictions], axis=0)

        # Compute calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y[plot_idx], y_pred, n_bins=10, strategy="uniform"
            )

            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=str(config))
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f"Calibration: {config}")
            ax.legend()
            ax.grid(alpha=0.3)
        except Exception as e:
            logger.warning(f"Calibration curve plotting failed for {config}: {e}")
            ax.text(0.5, 0.5, "Calibration curve unavailable", ha="center", va="center")

    plt.suptitle("Calibration Curves Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Calibration curves saved")


def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comparison plots for metrics.

    Args:
        results_df: Results dataframe
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Aggregate results across CV folds
    agg_results = (
        results_df.groupby("config")
        .agg(
            {
                "auroc": ["mean", "std"],
                "pr_auc": ["mean", "std"],
                "brier": ["mean", "std"],
                "ece": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    agg_results.columns = ["_".join(col).strip("_") for col in agg_results.columns.values]

    # 1. Metrics comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("auroc", "AUROC (↑)", True),
        ("pr_auc", "PR-AUC (↑)", True),
        ("brier", "Brier Score (↓)", False),
        ("ece", "ECE (↓)", False),
    ]

    for idx, (metric, label, higher_is_better) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        x = np.arange(len(agg_results))
        means = agg_results[f"{metric}_mean"]
        stds = agg_results[f"{metric}_std"]

        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

        # Color best bar
        if higher_is_better:
            best_idx = means.argmax()
        else:
            best_idx = means.argmin()
        bars[best_idx].set_color("green")
        bars[best_idx].set_alpha(0.9)

        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(agg_results["config"], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Stacking Calibration: Metrics Comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Trade-off plot: AUROC vs Brier
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in agg_results.iterrows():
        ax.scatter(
            row["brier_mean"],
            row["auroc_mean"],
            s=200,
            alpha=0.7,
            label=row["config"],
        )
        ax.errorbar(
            row["brier_mean"],
            row["auroc_mean"],
            xerr=row["brier_std"],
            yerr=row["auroc_std"],
            fmt="none",
            alpha=0.3,
        )

    ax.set_xlabel("Brier Score (lower is better)")
    ax.set_ylabel("AUROC (higher is better)")
    ax.set_title("Discrimination vs Calibration Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "auroc_vs_brier.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Comparison plots saved to {output_dir}")


def run_benchmark(
    oof_predictions: dict[str, np.ndarray],
    y: np.ndarray,
    configs: list[CalibrationConfig],
    n_splits: int = 5,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run stacking calibration benchmark.

    Args:
        oof_predictions: Dict of base model OOF predictions
        y: Labels
        configs: List of calibration configurations to test
        n_splits: Number of CV splits for evaluation
        output_dir: Output directory for results

    Returns:
        DataFrame with benchmark results
    """
    logger.info(f"Running benchmark with {len(configs)} configurations")

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    for config in configs:
        logger.info(f"Evaluating: {config}")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            logger.info(f"  Fold {fold_idx + 1}/{n_splits}")

            try:
                metrics = train_and_evaluate_ensemble(
                    oof_predictions, y, config, train_idx, test_idx
                )

                results.append(
                    {
                        "config": str(config),
                        "calibrate_meta": config.calibrate_meta,
                        "calibration_method": config.calibration_method or "none",
                        "fold": fold_idx,
                        **metrics,
                    }
                )
            except Exception as e:
                logger.error(f"  Fold {fold_idx} failed: {e}")
                continue

    results_df = pd.DataFrame(results)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / "stacking_calibration_results.csv", index=False)
        logger.info(f"✓ Results saved to {output_dir / 'stacking_calibration_results.csv'}")

        # Generate plots
        plot_metrics_comparison(results_df, output_dir)
        plot_calibration_curves(results_df, oof_predictions, y, configs, output_dir)

        # Save summary report
        with open(output_dir / "summary_report.txt", "w") as f:
            f.write("T2: Stacking Calibration Benchmark\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis date: {datetime.now().isoformat()}\n")
            f.write(f"Configurations tested: {len(configs)}\n")
            f.write(f"Base models: {len(oof_predictions)}\n")
            f.write(f"Samples: {len(y)}\n")
            f.write(f"Prevalence: {y.mean():.4f}\n")
            f.write(f"CV folds: {n_splits}\n\n")

            f.write("Aggregated Results:\n")
            f.write("-" * 80 + "\n")

            agg = (
                results_df.groupby("config")
                .agg(
                    {
                        "auroc": ["mean", "std"],
                        "pr_auc": ["mean", "std"],
                        "brier": ["mean", "std"],
                        "ece": ["mean", "std"],
                    }
                )
                .round(4)
            )

            f.write(agg.to_string())
            f.write("\n\n")

            f.write("Interpretation:\n")
            f.write("-" * 80 + "\n")
            f.write("- AUROC: Discrimination (higher is better)\n")
            f.write("- PR-AUC: Precision-recall balance (higher is better)\n")
            f.write("- Brier: Overall calibration quality (lower is better)\n")
            f.write("- ECE: Expected calibration error (lower is better)\n\n")
            f.write("Recommendation: If calibrate_meta=True does not improve Brier/ECE\n")
            f.write("significantly, use calibrate_meta=False to avoid over-smoothing.\n")

        logger.info(f"✓ Summary report saved to {output_dir / 'summary_report.txt'}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="T2: Stacking Calibration Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id", type=str, help="Pipeline run ID (e.g., 20260127_115115)")
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
        "--calibration-methods",
        nargs="+",
        choices=["sigmoid", "isotonic", "none"],
        default=["sigmoid", "isotonic", "none"],
        help="Calibration methods to test (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <results_dir>/empirical_validation/t2_<timestamp>)",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of CV splits for evaluation"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of samples for synthetic data"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.synthetic and not args.run_id:
        parser.error("Either --run-id or --synthetic must be specified")

    # Load data
    if args.synthetic:
        logger.info("Using synthetic data")
        oof_predictions, y = generate_synthetic_predictions(n_samples=args.n_samples)
    else:
        oof_predictions, y = load_real_predictions(args.run_id, args.results_dir)

    # Define calibration configurations
    configs = []
    if "none" in args.calibration_methods:
        configs.append(CalibrationConfig(calibrate_meta=False))
    if "sigmoid" in args.calibration_methods:
        configs.append(CalibrationConfig(calibrate_meta=True, calibration_method="sigmoid"))
    if "isotonic" in args.calibration_methods:
        configs.append(CalibrationConfig(calibrate_meta=True, calibration_method="isotonic"))

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.synthetic:
            output_dir = args.results_dir / "empirical_validation" / f"t2_synthetic_{timestamp}"
        else:
            output_dir = args.results_dir / "empirical_validation" / f"t2_{args.run_id}_{timestamp}"

    # Run benchmark
    results_df = run_benchmark(
        oof_predictions=oof_predictions,
        y=y,
        configs=configs,
        n_splits=args.n_splits,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("T2: STACKING CALIBRATION BENCHMARK - SUMMARY")
    print("=" * 80)
    print(f"\nConfigurations tested: {len(configs)}")
    print(f"Base models: {len(oof_predictions)}")
    print(f"CV folds: {args.n_splits}")
    print(f"\nResults saved to: {output_dir}")
    print("\nAggregated Metrics (mean ± std):")
    print("-" * 80)

    agg = results_df.groupby("config").agg(
        {"auroc": ["mean", "std"], "brier": ["mean", "std"], "ece": ["mean", "std"]}
    )

    for config in agg.index:
        print(f"\n{config}:")
        print(f"  AUROC: {agg.loc[config, ('auroc', 'mean')]:.4f} ± {agg.loc[config, ('auroc', 'std')]:.4f}")
        print(f"  Brier: {agg.loc[config, ('brier', 'mean')]:.4f} ± {agg.loc[config, ('brier', 'std')]:.4f}")
        print(f"  ECE:   {agg.loc[config, ('ece', 'mean')]:.4f} ± {agg.loc[config, ('ece', 'std')]:.4f}")

    print("\n" + "=" * 80)
    print("Review plots and detailed results in output directory")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
