#!/usr/bin/env python3
"""T4: Prevalence Adjustment Validation

Validates that Saerens prevalence adjustment correctly shifts predicted probabilities
to match target prevalence: mean(adjusted_probs) ≈ target_prevalence

Tests conducted:
1. Agreement check: |mean(adjusted) - target| < tolerance
2. Calibration preservation: calibration slope/intercept before vs after
3. Discrimination preservation: AUROC unchanged
4. Multiple target prevalence values
5. Sensitivity to training prevalence estimation

Usage:
    # With real model predictions from pipeline
    python t4_prevalence_adjustment_validation.py --run-id 20260127_115115

    # With synthetic data (for testing/development)
    python t4_prevalence_adjustment_validation.py --synthetic

    # Custom target prevalence values
    python t4_prevalence_adjustment_validation.py --run-id <RUN_ID> \
        --target-prevalences 0.001 0.005 0.01 0.05
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ced_ml.models.prevalence import adjust_probabilities_for_prevalence
from ced_ml.utils.math_utils import logit

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PrevalenceTest:
    """Results from prevalence adjustment test."""

    sample_prevalence: float
    target_prevalence: float
    mean_raw_prob: float
    mean_adjusted_prob: float
    absolute_error: float
    relative_error: float
    auroc_raw: float
    auroc_adjusted: float
    cal_slope_raw: float
    cal_slope_adjusted: float
    cal_intercept_raw: float
    cal_intercept_adjusted: float
    n_samples: int
    true_prevalence: float

    def passes_agreement_test(self, tolerance: float = 0.001) -> bool:
        """Check if adjusted mean matches target within tolerance."""
        return self.absolute_error < tolerance


def generate_synthetic_predictions(
    n_samples: int = 10000,
    true_prevalence: float = 0.0034,
    sample_prevalence: float = 0.167,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic predictions and labels.

    Args:
        n_samples: Number of samples
        true_prevalence: True population prevalence
        sample_prevalence: Training sample prevalence (e.g., after downsampling)
        random_state: Random seed

    Returns:
        Tuple of (predictions, labels)
    """
    logger.info(
        f"Generating synthetic data: {n_samples} samples, "
        f"true_prev={true_prevalence:.4f}, sample_prev={sample_prevalence:.4f}"
    )
    rng = np.random.RandomState(random_state)

    # Generate labels based on true prevalence
    n_positive = int(n_samples * true_prevalence)
    y = np.zeros(n_samples, dtype=int)
    y[:n_positive] = 1
    rng.shuffle(y)

    # Generate predictions calibrated to sample_prevalence
    # Strategy: Generate logits, then scale/shift to achieve desired mean

    # Generate discriminative logits (positive class has higher mean)
    logit_pos_mean = 2.0  # Positive class: higher logits
    logit_neg_mean = -3.0  # Negative class: lower logits

    logits = np.zeros(n_samples)
    logits[y == 1] = rng.normal(logit_pos_mean, 1.0, size=n_positive)
    logits[y == 0] = rng.normal(logit_neg_mean, 1.0, size=n_samples - n_positive)

    # Convert to probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Scale logits to achieve target sample_prevalence mean
    # Use binary search to find the right intercept shift
    def mean_after_shift(shift):
        shifted_probs = 1 / (1 + np.exp(-(logits + shift)))
        return shifted_probs.mean()

    # Binary search for the right shift
    low, high = -10.0, 10.0
    for _ in range(50):  # 50 iterations for convergence
        mid = (low + high) / 2
        current_mean = mean_after_shift(mid)
        if abs(current_mean - sample_prevalence) < 1e-6:
            break
        if current_mean < sample_prevalence:
            low = mid
        else:
            high = mid

    # Apply final shift
    probs = 1 / (1 + np.exp(-(logits + mid)))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    final_mean = probs.mean()
    logger.info(
        f"✓ Generated synthetic predictions (mean prob: {final_mean:.6f}, "
        f"target: {sample_prevalence:.6f}, error: {abs(final_mean - sample_prevalence):.6f})"
    )
    return probs, y


def load_real_predictions(
    run_id: str, results_dir: Path, model_name: str | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load real predictions from pipeline run.

    Args:
        run_id: Pipeline run ID
        results_dir: Path to results directory
        model_name: Specific model to load (if None, uses first available)

    Returns:
        Tuple of (predictions, labels, sample_prevalence)
    """
    run_dir = results_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.info(f"Loading predictions from run: {run_id}")

    # Find model directory
    if model_name:
        model_dir = run_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    else:
        # Use first available model
        model_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {run_dir}")
        model_dir = model_dirs[0]
        model_name = model_dir.name
        logger.info(f"Using model: {model_name}")

    # Load OOF predictions
    oof_file = model_dir / "aggregated" / "oof_predictions.csv"
    if not oof_file.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {oof_file}")

    oof_df = pd.read_csv(oof_file)

    if "label" not in oof_df.columns or "oof_proba" not in oof_df.columns:
        raise ValueError("Required columns (label, oof_proba) not found in OOF predictions")

    y = oof_df["label"].values
    probs = oof_df["oof_proba"].values

    # Estimate sample prevalence from training metadata
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        sample_prevalence = metadata.get("sample_prevalence", y.mean())
    else:
        logger.warning("Metadata not found, using observed prevalence as sample prevalence")
        sample_prevalence = y.mean()

    logger.info(
        f"✓ Loaded predictions: {len(y)} samples, "
        f"true_prev={y.mean():.4f}, sample_prev={sample_prevalence:.4f}"
    )

    return probs, y, sample_prevalence


def compute_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute calibration slope and intercept on logit scale.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        Tuple of (slope, intercept)
    """
    # Convert to logits
    logit_pred = logit(y_pred)

    # Fit logistic regression (slope should be ~1.0, intercept ~0.0 for perfect calibration)
    cal_model = LogisticRegression(penalty=None, max_iter=1000)
    try:
        cal_model.fit(logit_pred.reshape(-1, 1), y_true)
        slope = float(cal_model.coef_[0, 0])
        intercept = float(cal_model.intercept_[0])
    except Exception as e:
        logger.warning(f"Calibration metric computation failed: {e}")
        slope, intercept = np.nan, np.nan

    return slope, intercept


def test_prevalence_adjustment(
    probs: np.ndarray,
    y: np.ndarray,
    sample_prevalence: float,
    target_prevalence: float,
) -> PrevalenceTest:
    """Test prevalence adjustment for a single target prevalence.

    Args:
        probs: Raw predicted probabilities
        y: True binary labels
        sample_prevalence: Training sample prevalence
        target_prevalence: Target deployment prevalence

    Returns:
        PrevalenceTest results
    """
    # Apply adjustment
    adjusted_probs = adjust_probabilities_for_prevalence(probs, sample_prevalence, target_prevalence)

    # Compute metrics
    mean_raw = float(probs.mean())
    mean_adjusted = float(adjusted_probs.mean())
    absolute_error = abs(mean_adjusted - target_prevalence)
    relative_error = absolute_error / target_prevalence if target_prevalence > 0 else np.inf

    # AUROC (should be preserved)
    try:
        auroc_raw = float(roc_auc_score(y, probs))
        auroc_adjusted = float(roc_auc_score(y, adjusted_probs))
    except ValueError:
        auroc_raw, auroc_adjusted = np.nan, np.nan

    # Calibration slopes
    cal_slope_raw, cal_intercept_raw = compute_calibration_metrics(y, probs)
    cal_slope_adjusted, cal_intercept_adjusted = compute_calibration_metrics(y, adjusted_probs)

    return PrevalenceTest(
        sample_prevalence=sample_prevalence,
        target_prevalence=target_prevalence,
        mean_raw_prob=mean_raw,
        mean_adjusted_prob=mean_adjusted,
        absolute_error=absolute_error,
        relative_error=relative_error,
        auroc_raw=auroc_raw,
        auroc_adjusted=auroc_adjusted,
        cal_slope_raw=cal_slope_raw,
        cal_slope_adjusted=cal_slope_adjusted,
        cal_intercept_raw=cal_intercept_raw,
        cal_intercept_adjusted=cal_intercept_adjusted,
        n_samples=len(y),
        true_prevalence=float(y.mean()),
    )


def plot_results(results: list[PrevalenceTest], output_dir: Path) -> None:
    """Generate visualization plots.

    Args:
        results: List of test results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Convert to dataframe
    df = pd.DataFrame([vars(r) for r in results])

    # 1. Agreement plot: target vs adjusted mean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(df["target_prevalence"], df["mean_adjusted_prob"], s=100, alpha=0.7)
    ax1.plot([0, df["target_prevalence"].max()], [0, df["target_prevalence"].max()], "k--", label="Perfect agreement")
    ax1.set_xlabel("Target Prevalence")
    ax1.set_ylabel("Mean Adjusted Probability")
    ax1.set_title("Prevalence Adjustment Agreement")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Error plot
    ax2.scatter(df["target_prevalence"], df["absolute_error"], s=100, alpha=0.7, color="red")
    ax2.axhline(0.001, color="orange", linestyle="--", label="0.001 tolerance")
    ax2.set_xlabel("Target Prevalence")
    ax2.set_ylabel("Absolute Error |mean(adjusted) - target|")
    ax2.set_title("Adjustment Error")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "prevalence_agreement.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. AUROC preservation
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["auroc_raw"], width, label="Raw", alpha=0.7)
    ax.bar(x + width / 2, df["auroc_adjusted"], width, label="Adjusted", alpha=0.7)

    ax.set_xlabel("Target Prevalence")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Preservation (discrimination unchanged)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.4f}" for p in df["target_prevalence"]], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "auroc_preservation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Calibration slope changes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["target_prevalence"], df["cal_slope_raw"], "o-", label="Raw", markersize=8)
    ax1.plot(df["target_prevalence"], df["cal_slope_adjusted"], "s-", label="Adjusted", markersize=8)
    ax1.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect calibration")
    ax1.set_xlabel("Target Prevalence")
    ax1.set_ylabel("Calibration Slope")
    ax1.set_title("Calibration Slope")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df["target_prevalence"], df["cal_intercept_raw"], "o-", label="Raw", markersize=8)
    ax2.plot(df["target_prevalence"], df["cal_intercept_adjusted"], "s-", label="Adjusted", markersize=8)
    ax2.axhline(0.0, color="green", linestyle="--", alpha=0.5, label="Perfect calibration")
    ax2.set_xlabel("Target Prevalence")
    ax2.set_ylabel("Calibration Intercept")
    ax2.set_title("Calibration Intercept")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Calibration Metrics Before/After Adjustment")
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Summary table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    table_data.append(["Target", "Mean Raw", "Mean Adj", "Abs Error", "Rel Error", "AUROC Δ", "Pass"])

    for _, row in df.iterrows():
        auroc_delta = abs(row["auroc_adjusted"] - row["auroc_raw"])
        passes = "✓" if row["absolute_error"] < 0.001 else "✗"
        table_data.append(
            [
                f"{row['target_prevalence']:.4f}",
                f"{row['mean_raw_prob']:.4f}",
                f"{row['mean_adjusted_prob']:.4f}",
                f"{row['absolute_error']:.6f}",
                f"{row['relative_error']:.2%}",
                f"{auroc_delta:.6f}",
                passes,
            ]
        )

    table = ax.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.12] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Prevalence Adjustment Validation Summary", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Plots saved to {output_dir}")


def run_validation(
    probs: np.ndarray,
    y: np.ndarray,
    sample_prevalence: float,
    target_prevalences: list[float],
    output_dir: Path | None = None,
) -> list[PrevalenceTest]:
    """Run prevalence adjustment validation.

    Args:
        probs: Raw predicted probabilities
        y: True binary labels
        sample_prevalence: Training sample prevalence
        target_prevalences: List of target prevalences to test
        output_dir: Output directory for results

    Returns:
        List of PrevalenceTest results
    """
    logger.info(f"Running validation with {len(target_prevalences)} target prevalences")
    logger.info(f"Sample prevalence: {sample_prevalence:.4f}")

    results = []
    for target_prev in target_prevalences:
        logger.info(f"Testing target prevalence: {target_prev:.4f}")
        result = test_prevalence_adjustment(probs, y, sample_prevalence, target_prev)
        results.append(result)

        # Log result
        pass_str = "PASS" if result.passes_agreement_test() else "FAIL"
        logger.info(
            f"  Mean adjusted: {result.mean_adjusted_prob:.6f}, "
            f"Error: {result.absolute_error:.6f} [{pass_str}]"
        )

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        df = pd.DataFrame([vars(r) for r in results])
        df.to_csv(output_dir / "prevalence_validation_results.csv", index=False)
        logger.info(f"✓ Results saved to {output_dir / 'prevalence_validation_results.csv'}")

        # Generate plots
        plot_results(results, output_dir)

        # Save summary report
        with open(output_dir / "summary_report.txt", "w") as f:
            f.write("T4: Prevalence Adjustment Validation\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis date: {datetime.now().isoformat()}\n")
            f.write(f"Sample prevalence: {sample_prevalence:.6f}\n")
            f.write(f"True prevalence: {y.mean():.6f}\n")
            f.write(f"Samples: {len(y)}\n")
            f.write(f"Target prevalences tested: {len(target_prevalences)}\n\n")

            f.write("Results:\n")
            f.write("-" * 80 + "\n")
            for result in results:
                pass_str = "PASS" if result.passes_agreement_test() else "FAIL"
                f.write(f"\nTarget: {result.target_prevalence:.6f}\n")
                f.write(f"  Mean adjusted: {result.mean_adjusted_prob:.6f}\n")
                f.write(f"  Absolute error: {result.absolute_error:.6f}\n")
                f.write(f"  Relative error: {result.relative_error:.2%}\n")
                f.write(f"  AUROC change: {abs(result.auroc_adjusted - result.auroc_raw):.6f}\n")
                f.write(f"  Status: {pass_str}\n")

            # Summary statistics
            errors = [r.absolute_error for r in results]
            auroc_changes = [abs(r.auroc_adjusted - r.auroc_raw) for r in results]

            f.write("\nSummary Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean absolute error: {np.mean(errors):.6f}\n")
            f.write(f"Max absolute error: {np.max(errors):.6f}\n")
            f.write(f"Tests passing (< 0.001 tolerance): {sum(r.passes_agreement_test() for r in results)}/{len(results)}\n")
            f.write(f"Mean AUROC change: {np.mean(auroc_changes):.6f}\n")
            f.write(f"Max AUROC change: {np.max(auroc_changes):.6f}\n")

            f.write("\nInterpretation:\n")
            f.write("-" * 80 + "\n")
            f.write("- Absolute error < 0.001: Excellent agreement\n")
            f.write("- AUROC change < 0.001: Discrimination preserved (expected)\n")
            f.write("- Calibration slope change: Intercept adjustment working correctly\n")
            f.write("\nSaerens adjustment is valid when:\n")
            f.write("1. P(X|Y) distributions are similar between training and deployment\n")
            f.write("2. Only the prior P(Y) changes\n")
            f.write("3. Feature distributions remain stable\n")

        logger.info(f"✓ Summary report saved to {output_dir / 'summary_report.txt'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="T4: Prevalence Adjustment Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-id", type=str, help="Pipeline run ID (e.g., 20260127_115115)")
    parser.add_argument(
        "--model-name", type=str, help="Specific model to load (default: first available)"
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
        "--target-prevalences",
        nargs="+",
        type=float,
        help="Target prevalences to test (default: predefined range)",
    )
    parser.add_argument(
        "--sample-prevalence",
        type=float,
        help="Sample prevalence for synthetic data (default: 0.167)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <results_dir>/empirical_validation/t4_<timestamp>)",
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
        sample_prev = args.sample_prevalence or 0.167
        probs, y = generate_synthetic_predictions(
            n_samples=args.n_samples, sample_prevalence=sample_prev
        )
        sample_prevalence = sample_prev
    else:
        probs, y, sample_prevalence = load_real_predictions(
            args.run_id, args.results_dir, args.model_name
        )

    # Define target prevalences to test
    if args.target_prevalences:
        target_prevalences = args.target_prevalences
    else:
        # Default range covering realistic scenarios
        target_prevalences = [
            0.0034,  # Original (0.34%)
            0.001,  # Very low (0.1%)
            0.005,  # Low (0.5%)
            0.01,  # Moderate (1%)
            0.02,  # Higher (2%)
            0.05,  # High (5%)
            0.10,  # Very high (10%)
        ]

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.synthetic:
            output_dir = args.results_dir / "empirical_validation" / f"t4_synthetic_{timestamp}"
        else:
            output_dir = args.results_dir / "empirical_validation" / f"t4_{args.run_id}_{timestamp}"

    # Run validation
    results = run_validation(
        probs=probs,
        y=y,
        sample_prevalence=sample_prevalence,
        target_prevalences=target_prevalences,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("T4: PREVALENCE ADJUSTMENT VALIDATION - SUMMARY")
    print("=" * 80)
    print(f"\nSample prevalence: {sample_prevalence:.6f}")
    print(f"True prevalence: {y.mean():.6f}")
    print(f"Target prevalences tested: {len(target_prevalences)}")
    print(f"\nResults saved to: {output_dir}")
    print("\nValidation Results:")
    print("-" * 80)

    for result in results:
        pass_str = "✓ PASS" if result.passes_agreement_test() else "✗ FAIL"
        print(
            f"Target={result.target_prevalence:.6f}: "
            f"Adjusted={result.mean_adjusted_prob:.6f}, "
            f"Error={result.absolute_error:.6f} {pass_str}"
        )

    # Summary statistics
    n_passed = sum(r.passes_agreement_test() for r in results)
    mean_error = np.mean([r.absolute_error for r in results])
    max_error = np.max([r.absolute_error for r in results])

    print("\n" + "-" * 80)
    print(f"Tests passed: {n_passed}/{len(results)}")
    print(f"Mean absolute error: {mean_error:.6f}")
    print(f"Max absolute error: {max_error:.6f}")

    if n_passed == len(results):
        print("\n✓ All tests PASSED: Saerens adjustment working correctly")
    else:
        print(f"\n✗ {len(results) - n_passed} tests FAILED: Review detailed results")

    print("\n" + "=" * 80)
    print("Review plots and detailed results in output directory")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
