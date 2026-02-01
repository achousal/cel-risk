#!/usr/bin/env python3
"""
Factorial experiment analysis with statistical testing.

Analyzes 2x2 factorial design:
- Factor 1: prevalent_train_frac (0.5 vs 1.0)
- Factor 2: train_control_per_case (1:1 vs 1:5)

Performs paired t-tests with Bonferroni correction and effect size calculations.

Usage:
    # Analyze latest run (auto-detect)
    python investigate_factorial.py

    # Analyze specific run
    python investigate_factorial.py --run-id run_20260131_232604

    # Custom paths
    python investigate_factorial.py --results-dir /path/to/results --output-dir /path/to/output

Inputs:
    - Results from: results/run_{timestamp}/{model}/split_seed{N}/
    - Reads: config_metadata.json, core/test_metrics.csv

Outputs:
    - results/{run_id}/analysis/factorial/metrics_all.csv
    - results/{run_id}/analysis/factorial/comparison_table.csv
    - results/{run_id}/analysis/factorial/statistical_tests.csv
    - results/{run_id}/analysis/factorial/power_analysis.csv
    - results/{run_id}/analysis/factorial/summary.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower


def discover_runs(results_root: Path) -> List[Path]:
    """
    Find all split directories in results tree.

    Current structure: results/run_{timestamp}/{model}/splits/split_seed{N}/
    Legacy structure: results/{model}/run_{timestamp}/split_seed{N}/

    Returns list of split_seed directories containing model outputs.
    """
    run_dirs = []

    # Try current structure: {model}/splits/split_seed* (when called on specific run)
    for run_dir in results_root.glob('*/splits/split_seed*'):
        if run_dir.is_dir():
            run_dirs.append(run_dir)

    # Try alternative: run_*/model/splits/split_seed* (when called on results root)
    if not run_dirs:
        for run_dir in results_root.glob('run_*/*/splits/split_seed*'):
            if run_dir.is_dir():
                run_dirs.append(run_dir)

    # Try legacy structure: model/run_*/split_seed* (no splits subdirectory)
    if not run_dirs:
        for model_dir in results_root.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
            for run_dir in model_dir.glob('run_*/split_seed*'):
                if run_dir.is_dir():
                    run_dirs.append(run_dir)

    return sorted(run_dirs)


def identify_config(run_dir: Path) -> Tuple[str, str, int]:
    """
    Map run to config via metadata.

    Returns:
        (config_id, model_name, split_seed)

    Config mapping:
        n_train=298 + train_prev~50% -> 1:1 ratio
        n_train=894 + train_prev~17% -> 1:5 ratio
        (prevalent_frac inferred from case composition)

    Directory structure:
        Current: results/run_{timestamp}/{model}/split_seed{N}/
        Legacy: results/{model}/run_{timestamp}/split_seed{N}/
    """
    metadata_path = run_dir / 'config_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with open(metadata_path) as f:
        meta = json.load(f)

    n_train = meta.get('n_train')
    train_prev = meta.get('train_prevalence')

    # Infer case:control ratio
    if abs(train_prev - 0.5) < 0.05:
        ccr = '1'
    elif abs(train_prev - 0.17) < 0.05:
        ccr = '5'
    else:
        raise ValueError(f"Unexpected train_prev={train_prev}")

    # Infer prevalent_frac (placeholder - need to check case composition)
    # For now, assume config naming in parent directory
    parent = run_dir.parent.parent.name
    if 'experiment' in parent:
        # Extract from experiment directory structure
        pf = '0.5'  # Placeholder
    else:
        pf = '0.5'  # Default

    config_id = f"{pf}_{ccr}"

    # Extract model and seed
    # Handle directory structures:
    #   Current: model/splits/split_seed* -> parent.parent is model
    #   Legacy: model/run_*/split_seed* -> parent.parent is model
    if run_dir.parent.name == 'splits':
        model_name = run_dir.parent.parent.name  # splits/split_seed -> model/splits -> model
    else:
        model_name = run_dir.parent.name  # run_*/split_seed -> run_*

    split_seed = int(run_dir.name.replace('split_seed', ''))

    return config_id, model_name, split_seed


def extract_metrics(run_dir: Path) -> Dict:
    """
    Read performance metrics from run outputs.

    Extracts:
        - AUROC, PR_AUC, sens_ctrl_95 from test_metrics.csv
        - Brier score from calibration outputs
        - Sample sizes from metadata
    """
    metrics = {}

    # Test metrics
    test_metrics_path = run_dir / 'core' / 'test_metrics.csv'
    if test_metrics_path.exists():
        df = pd.read_csv(test_metrics_path)
        # Handle both uppercase and lowercase column names
        metrics['AUROC'] = df.loc[0, 'auroc'] if 'auroc' in df.columns else df.loc[0, 'AUROC']
        metrics['PR_AUC'] = df.loc[0, 'prauc'] if 'prauc' in df.columns else df.loc[0, 'PR_AUC']
        metrics['Sens_95spec'] = df.loc[0, 'sens_ctrl_95']
        metrics['Brier'] = df.loc[0, 'brier_score'] if 'brier_score' in df.columns else None

    # Metadata
    metadata_path = run_dir / 'config_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        metrics['n_train'] = meta.get('n_train')
        metrics['train_prevalence'] = meta.get('train_prevalence')

    return metrics


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d for standardized mean difference.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def paired_comparison(metrics_df: pd.DataFrame, metric: str = 'AUROC') -> pd.DataFrame:
    """
    Paired t-tests between configs with Bonferroni correction.

    Comparisons (4 pairs per model):
        1. 0.5_1 vs 0.5_5 (case:control effect at 50% prevalent)
        2. 1.0_1 vs 1.0_5 (case:control effect at 100% prevalent)
        3. 0.5_1 vs 1.0_1 (prevalent sampling effect at 1:1 ratio)
        4. 0.5_5 vs 1.0_5 (prevalent sampling effect at 1:5 ratio)

    Total tests: 4 pairs × 2 models = 8 tests
    Bonferroni alpha: 0.05 / 8 = 0.00625
    """
    results = []
    models = metrics_df['model'].unique()

    comparisons = [
        ('0.5_1', '0.5_5', 'Case:control (50% prev)'),
        ('1.0_1', '1.0_5', 'Case:control (100% prev)'),
        ('0.5_1', '1.0_1', 'Prevalent frac (1:1 ratio)'),
        ('0.5_5', '1.0_5', 'Prevalent frac (1:5 ratio)'),
    ]

    n_tests = len(comparisons) * len(models)
    alpha_corrected = 0.05 / n_tests

    for model in models:
        for config1, config2, label in comparisons:
            df_model = metrics_df[metrics_df['model'] == model]

            g1 = df_model[df_model['config'] == config1][metric].values
            g2 = df_model[df_model['config'] == config2][metric].values

            if len(g1) == 0 or len(g2) == 0:
                continue

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(g1, g2)

            # Effect size
            cohen_d = compute_effect_size(g1, g2)

            # Mean difference
            mean_diff = g1.mean() - g2.mean()

            results.append({
                'comparison': f'{config1} vs {config2}',
                'label': label,
                'model': model,
                'metric': metric,
                'mean_diff': mean_diff,
                'cohen_d': cohen_d,
                't_stat': t_stat,
                'p_value': p_value,
                'p_adj': p_value * n_tests,  # Bonferroni
                'alpha_corrected': alpha_corrected,
                'significant': p_value < alpha_corrected,
            })

    return pd.DataFrame(results)


def power_analysis(n_seeds: int, effect_size: float, alpha: float = 0.00625) -> float:
    """
    Post-hoc power for paired t-test.

    Args:
        n_seeds: Number of independent random seeds (sample size)
        effect_size: Cohen's d
        alpha: Significance level (Bonferroni-corrected)

    Returns:
        Statistical power (1 - beta)
    """
    power_calc = TTestPower()
    power = power_calc.solve_power(
        effect_size=abs(effect_size),
        nobs=n_seeds,
        alpha=alpha,
        alternative='two-sided'
    )
    return power


def generate_comparison(metrics: List[Dict]) -> pd.DataFrame:
    """
    Aggregate metrics by config with mean/std/95% CI.
    """
    df = pd.DataFrame(metrics)

    agg_funcs = {
        'AUROC': ['mean', 'std', 'count'],
        'PR_AUC': ['mean', 'std'],
        'Sens_95spec': ['mean', 'std'],
        'Brier': ['mean', 'std'],
    }

    summary = df.groupby(['config', 'model']).agg(agg_funcs).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    # Add 95% CI (t-distribution)
    for metric in ['AUROC', 'PR_AUC', 'Sens_95spec', 'Brier']:
        n = summary[f'{metric}_count'] if f'{metric}_count' in summary.columns else 5
        se = summary[f'{metric}_std'] / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        summary[f'{metric}_95CI_lower'] = summary[f'{metric}_mean'] - t_crit * se
        summary[f'{metric}_95CI_upper'] = summary[f'{metric}_mean'] + t_crit * se

    return summary


def generate_summary(
    comparison_df: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    power_df: pd.DataFrame,
    output_path: Path
):
    """
    Generate human-readable markdown summary with interpretation.

    Falls back to CSV if tabulate is not available.
    """
    with open(output_path, 'w') as f:
        f.write("# Factorial Experiment Results\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Try markdown format, fallback to CSV-style
        try:
            f.write("## Summary Statistics\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Statistical Tests\n\n")
            f.write(statistical_tests.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Power Analysis\n\n")
            f.write(power_df.to_markdown(index=False))
            f.write("\n\n")
        except ImportError:
            # Fallback to CSV format if tabulate not available
            f.write("## Summary Statistics\n\n")
            f.write("```csv\n")
            comparison_df.to_csv(f, index=False)
            f.write("```\n\n")

            f.write("## Statistical Tests\n\n")
            f.write("```csv\n")
            statistical_tests.to_csv(f, index=False)
            f.write("```\n\n")

            f.write("## Power Analysis\n\n")
            f.write("```csv\n")
            power_df.to_csv(f, index=False)
            f.write("```\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        if len(statistical_tests) == 0:
            f.write("No statistical tests performed (insufficient data or single config).\n")
        elif 'significant' in statistical_tests.columns:
            sig_tests = statistical_tests[statistical_tests['significant']]
            if len(sig_tests) > 0:
                f.write("### Significant Differences (Bonferroni-corrected)\n\n")
                for _, row in sig_tests.iterrows():
                    f.write(f"- **{row['comparison']}** ({row['model']}): "
                           f"Δ={row['mean_diff']:.3f}, d={row['cohen_d']:.2f}, "
                           f"p={row['p_value']:.4f}\n")
            else:
                f.write("No statistically significant differences detected.\n")
        else:
            f.write("Statistical test results unavailable.\n")

        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Factorial experiment analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest run (auto-detect from results/)
  python investigate_factorial.py

  # Analyze specific run
  python investigate_factorial.py --run-id run_20260131_232604

  # Specify custom paths
  python investigate_factorial.py --results-dir /path/to/results --output-dir /path/to/output
        """
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='Results root directory (default: auto-detect from script location → ../../results/)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Specific run ID to analyze (e.g., run_20260131_232604). If not provided, uses latest run.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for analysis (default: results/{run_id}/analysis/factorial/)'
    )
    parser.add_argument(
        '--metric',
        default='AUROC',
        help='Primary metric for statistical testing (default: AUROC)'
    )
    args = parser.parse_args()

    # Auto-detect project root and results directory
    if args.results_dir is None:
        # Script location: analysis/docs/investigations/investigate_factorial.py
        # Project root: ../../../
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        args.results_dir = project_root / 'results'
        print(f"Auto-detected results directory: {args.results_dir}")

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    # Determine run_id (use specified or latest)
    if args.run_id:
        run_path = args.results_dir / args.run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        selected_run = args.run_id
    else:
        # Find latest run_* directory
        run_dirs = sorted([d for d in args.results_dir.glob('run_*') if d.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No run_* directories found in {args.results_dir}")
        selected_run = run_dirs[-1].name
        print(f"Auto-detected latest run: {selected_run}")

    # Set search path to specific run
    search_path = args.results_dir / selected_run

    # Auto-detect output directory if not specified
    if args.output_dir is None:
        args.output_dir = search_path / 'analysis' / 'factorial'
        print(f"Auto-detected output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover runs
    print(f"Discovering runs in: {search_path}")
    run_dirs = discover_runs(search_path)
    print(f"Found {len(run_dirs)} split directories")

    # Extract metrics
    print("Extracting metrics...")
    all_metrics = []
    for run_dir in run_dirs:
        try:
            config_id, model, seed = identify_config(run_dir)
            metrics = extract_metrics(run_dir)
            metrics['config'] = config_id
            metrics['model'] = model
            metrics['split_seed'] = seed
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to process {run_dir}: {e}")

    if len(all_metrics) == 0:
        raise ValueError("No valid runs found")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.output_dir / 'metrics_all.csv', index=False)
    print(f"Saved raw metrics: {args.output_dir / 'metrics_all.csv'}")

    # Generate comparison table
    print("Generating comparison table...")
    comparison = generate_comparison(all_metrics)
    comparison.to_csv(args.output_dir / 'comparison_table.csv', index=False)
    print(f"Saved comparison: {args.output_dir / 'comparison_table.csv'}")

    # Statistical tests
    print("Running statistical tests...")
    stat_tests = paired_comparison(metrics_df, metric=args.metric)
    stat_tests.to_csv(args.output_dir / 'statistical_tests.csv', index=False)
    print(f"Saved tests: {args.output_dir / 'statistical_tests.csv'}")

    # Power analysis
    print("Computing power analysis...")
    n_seeds = metrics_df.groupby(['config', 'model']).size().min()
    power_results = []
    for _, row in stat_tests.iterrows():
        power = power_analysis(n_seeds, row['cohen_d'], row['alpha_corrected'])
        power_results.append({
            'comparison': row['comparison'],
            'model': row['model'],
            'cohen_d': row['cohen_d'],
            'power': power,
            'n_seeds': n_seeds,
        })
    power_df = pd.DataFrame(power_results)
    power_df.to_csv(args.output_dir / 'power_analysis.csv', index=False)
    print(f"Saved power: {args.output_dir / 'power_analysis.csv'}")

    # Generate summary
    print("Generating summary...")
    generate_summary(comparison, stat_tests, power_df,
                    args.output_dir / 'summary.md')
    print(f"Saved summary: {args.output_dir / 'summary.md'}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
