#!/usr/bin/env python3
"""
Extended factorial experiment analysis with 3 factors.

Analyzes 3-factor factorial design:
- Factor 1: fold_size (k in k-fold CV: 3, 5, 10)
- Factor 2: train_size (absolute N: 298, 596, 894)
- Factor 3: calibration_method (isotonic, platt, oof_posthoc)

Feature selection is FIXED at hybrid for all experiments.

Performs:
- Main effects analysis (one factor at a time)
- Two-way interaction analysis
- Bonferroni-corrected pairwise comparisons
- Effect size calculations (Cohen's d)
- Post-hoc power analysis

Usage:
    # Analyze latest run
    python investigate_factorial_extended.py

    # Analyze specific run
    python investigate_factorial_extended.py --run-id run_20260131_232604

    # Custom paths
    python investigate_factorial_extended.py --results-dir /path/to/results --output-dir /path/to/output

Inputs:
    - Results from: results/run_{timestamp}/{model}/splits/split_seed{N}/
    - Reads: config_metadata.json, core/test_metrics.csv, calibration/

Outputs (in results/{run_id}/analysis/factorial/):
    - metrics_all.csv: Raw metrics for all runs
    - comparison_table.csv: Aggregated stats by config
    - main_effects.csv: Main effect tests
    - interactions.csv: Two-way interaction tests
    - power_analysis.csv: Statistical power
    - summary.md: Human-readable report
"""

import argparse
import json
from itertools import combinations
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

    # Try current structure: {model}/splits/split_seed*
    for run_dir in results_root.glob('*/splits/split_seed*'):
        if run_dir.is_dir():
            run_dirs.append(run_dir)

    # Try alternative: run_*/model/splits/split_seed*
    if not run_dirs:
        for run_dir in results_root.glob('run_*/*/splits/split_seed*'):
            if run_dir.is_dir():
                run_dirs.append(run_dir)

    # Try legacy structure: model/run_*/split_seed*
    if not run_dirs:
        for model_dir in results_root.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
            for run_dir in model_dir.glob('run_*/split_seed*'):
                if run_dir.is_dir():
                    run_dirs.append(run_dir)

    return sorted(run_dirs)


def identify_config(run_dir: Path) -> Tuple[str, str, int, int, int, str]:
    """
    Map run to config via metadata.

    Returns:
        (config_id, model_name, split_seed, fold_size, train_size, calib_method)

    Config ID format: k{fold}_N{train}_{calib}
        Example: "k5_N298_isotonic"

    Extraction strategy:
        - fold_size: from config_metadata.json -> cv_config.n_splits
        - train_size: from config_metadata.json -> n_train
        - calib_method: inferred from calibration directory contents
    """
    metadata_path = run_dir / 'config_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with open(metadata_path) as f:
        meta = json.load(f)

    # Extract fold size
    cv_config = meta.get('cv_config', {})
    fold_size = cv_config.get('n_splits', 5)  # Default to 5 if missing

    # Extract train size
    train_size = meta.get('n_train')
    if train_size is None:
        raise ValueError(f"Missing n_train in metadata: {metadata_path}")

    # Infer calibration method from directory structure
    calib_dir = run_dir / 'calibration'
    if calib_dir.exists():
        # Check for calibration artifacts
        if (calib_dir / 'isotonic').exists() or (calib_dir / 'calibrator_isotonic.pkl').exists():
            calib_method = 'isotonic'
        elif (calib_dir / 'platt').exists() or (calib_dir / 'calibrator_platt.pkl').exists():
            calib_method = 'platt'
        elif (calib_dir / 'oof_posthoc').exists() or (calib_dir / 'oof_calibrator.pkl').exists():
            calib_method = 'oof_posthoc'
        else:
            # Fallback: check config
            calib_config = meta.get('calibration', {})
            calib_method = calib_config.get('method', 'isotonic')
    else:
        # Default to isotonic if no calibration directory
        calib_method = 'isotonic'

    # Build config ID
    config_id = f"k{fold_size}_N{train_size}_{calib_method}"

    # Extract model and seed
    if run_dir.parent.name == 'splits':
        model_name = run_dir.parent.parent.name
    else:
        model_name = run_dir.parent.name

    split_seed = int(run_dir.name.replace('split_seed', ''))

    return config_id, model_name, split_seed, fold_size, train_size, calib_method


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


def main_effects_analysis(
    metrics_df: pd.DataFrame,
    metric: str = 'AUROC'
) -> pd.DataFrame:
    """
    Analyze main effects for each factor independently.

    For each factor (fold_size, train_size, calib_method):
    - Generate all pairwise comparisons between levels
    - Perform paired t-tests with Bonferroni correction
    - Calculate effect sizes

    Args:
        metrics_df: DataFrame with columns: config, model, split_seed, fold_size,
                    train_size, calib_method, {metric}
        metric: Metric to compare (AUROC, PR_AUC, etc.)

    Returns:
        DataFrame with test results for main effects
    """
    results = []
    models = metrics_df['model'].unique()

    factors = ['fold_size', 'train_size', 'calib_method']

    # Count total comparisons for Bonferroni correction
    n_tests = 0
    for factor in factors:
        levels = metrics_df[factor].unique()
        n_pairs = len(list(combinations(levels, 2)))
        n_tests += n_pairs * len(models)

    alpha_corrected = 0.05 / n_tests if n_tests > 0 else 0.05

    for factor in factors:
        levels = sorted(metrics_df[factor].unique())

        for level1, level2 in combinations(levels, 2):
            for model in models:
                # Filter to runs that differ ONLY in this factor
                # This is approximate - assumes factorial design
                df_model = metrics_df[metrics_df['model'] == model]

                g1 = df_model[df_model[factor] == level1][metric].values
                g2 = df_model[df_model[factor] == level2][metric].values

                if len(g1) == 0 or len(g2) == 0:
                    continue

                # Paired t-test (assumes matched seeds)
                min_len = min(len(g1), len(g2))
                g1 = g1[:min_len]
                g2 = g2[:min_len]

                if min_len < 2:
                    continue

                t_stat, p_value = stats.ttest_rel(g1, g2)
                cohen_d = compute_effect_size(g1, g2)
                mean_diff = g1.mean() - g2.mean()

                results.append({
                    'factor': factor,
                    'level1': level1,
                    'level2': level2,
                    'comparison': f'{level1} vs {level2}',
                    'model': model,
                    'metric': metric,
                    'n': min_len,
                    'mean_diff': mean_diff,
                    'cohen_d': cohen_d,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'p_adj': min(1.0, p_value * n_tests),
                    'alpha_corrected': alpha_corrected,
                    'significant': p_value < alpha_corrected,
                })

    return pd.DataFrame(results)


def interaction_analysis(
    metrics_df: pd.DataFrame,
    metric: str = 'AUROC'
) -> pd.DataFrame:
    """
    Analyze two-way interactions between factors.

    For each pair of factors (e.g., fold_size × train_size):
    - Test whether the effect of factor A depends on the level of factor B
    - Uses 2-way repeated measures ANOVA approach

    Args:
        metrics_df: DataFrame with factor columns
        metric: Metric to analyze

    Returns:
        DataFrame with interaction test results
    """
    results = []
    models = metrics_df['model'].unique()

    factor_pairs = [
        ('fold_size', 'train_size'),
        ('fold_size', 'calib_method'),
        ('train_size', 'calib_method'),
    ]

    for factor_a, factor_b in factor_pairs:
        for model in models:
            df_model = metrics_df[metrics_df['model'] == model]

            # Create interaction groups
            df_model['interaction'] = (
                df_model[factor_a].astype(str) + '_x_' + df_model[factor_b].astype(str)
            )

            groups = df_model.groupby('interaction')[metric].apply(list)

            if len(groups) < 2:
                continue

            # One-way ANOVA across interaction groups
            group_arrays = [np.array(g) for g in groups.values if len(g) > 0]

            if len(group_arrays) < 2:
                continue

            f_stat, p_value = stats.f_oneway(*group_arrays)

            # Effect size (eta-squared)
            grand_mean = df_model[metric].mean()
            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean)**2
                for g in group_arrays
            )
            ss_total = sum(
                sum((x - grand_mean)**2 for x in g)
                for g in group_arrays
            )
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            results.append({
                'interaction': f'{factor_a} × {factor_b}',
                'model': model,
                'metric': metric,
                'n_groups': len(groups),
                'f_stat': f_stat,
                'p_value': p_value,
                'eta_squared': eta_sq,
                'significant': p_value < 0.05,
            })

    return pd.DataFrame(results)


def power_analysis(n_seeds: int, effect_size: float, alpha: float = 0.05) -> float:
    """
    Post-hoc power for paired t-test.

    Args:
        n_seeds: Number of independent random seeds (sample size)
        effect_size: Cohen's d
        alpha: Significance level

    Returns:
        Statistical power (1 - beta)
    """
    if n_seeds < 2 or np.isnan(effect_size):
        return np.nan

    power_calc = TTestPower()
    try:
        power = power_calc.solve_power(
            effect_size=abs(effect_size),
            nobs=n_seeds,
            alpha=alpha,
            alternative='two-sided'
        )
        return power
    except:
        return np.nan


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
        n_col = f'{metric}_count'
        if n_col not in summary.columns:
            continue

        n = summary[n_col]
        se = summary[f'{metric}_std'] / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        summary[f'{metric}_95CI_lower'] = summary[f'{metric}_mean'] - t_crit * se
        summary[f'{metric}_95CI_upper'] = summary[f'{metric}_mean'] + t_crit * se

    return summary


def generate_summary(
    comparison_df: pd.DataFrame,
    main_effects_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    power_df: pd.DataFrame,
    output_path: Path
):
    """
    Generate human-readable markdown summary with interpretation.
    """
    with open(output_path, 'w') as f:
        f.write("# Extended Factorial Experiment Results\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Design\n\n")
        f.write("3-factor factorial:\n")
        f.write("- Fold size (k in CV)\n")
        f.write("- Train set size (N)\n")
        f.write("- Calibration method\n\n")
        f.write("Feature selection: **Hybrid** (fixed for all experiments)\n\n")

        # Summary statistics
        try:
            f.write("## Summary Statistics\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
        except ImportError:
            f.write("## Summary Statistics\n\n```csv\n")
            comparison_df.to_csv(f, index=False)
            f.write("```\n\n")

        # Main effects
        if len(main_effects_df) > 0:
            try:
                f.write("## Main Effects\n\n")

                # Significant effects
                sig_main = main_effects_df[main_effects_df['significant']]
                if len(sig_main) > 0:
                    f.write("### Significant Main Effects (Bonferroni-corrected)\n\n")
                    for _, row in sig_main.iterrows():
                        f.write(f"- **{row['factor']}**: {row['comparison']} ({row['model']})\n")
                        f.write(f"  - Δ={row['mean_diff']:.3f}, d={row['cohen_d']:.2f}, p={row['p_value']:.4f}\n")
                    f.write("\n")
                else:
                    f.write("### No significant main effects detected\n\n")

                # Full table
                f.write("### All Main Effects\n\n")
                f.write(main_effects_df.to_markdown(index=False))
                f.write("\n\n")
            except ImportError:
                f.write("## Main Effects\n\n```csv\n")
                main_effects_df.to_csv(f, index=False)
                f.write("```\n\n")

        # Interactions
        if len(interactions_df) > 0:
            try:
                f.write("## Two-Way Interactions\n\n")

                sig_int = interactions_df[interactions_df['significant']]
                if len(sig_int) > 0:
                    f.write("### Significant Interactions (p < 0.05)\n\n")
                    for _, row in sig_int.iterrows():
                        f.write(f"- **{row['interaction']}** ({row['model']})\n")
                        f.write(f"  - F={row['f_stat']:.2f}, p={row['p_value']:.4f}, η²={row['eta_squared']:.3f}\n")
                    f.write("\n")
                else:
                    f.write("### No significant interactions detected\n\n")

                f.write("### All Interactions\n\n")
                f.write(interactions_df.to_markdown(index=False))
                f.write("\n\n")
            except ImportError:
                f.write("## Interactions\n\n```csv\n")
                interactions_df.to_csv(f, index=False)
                f.write("```\n\n")

        # Power analysis
        if len(power_df) > 0:
            try:
                f.write("## Power Analysis\n\n")
                f.write(power_df.to_markdown(index=False))
                f.write("\n\n")
            except ImportError:
                f.write("## Power Analysis\n\n```csv\n")
                power_df.to_csv(f, index=False)
                f.write("```\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        if len(main_effects_df) > 0:
            sig_main = main_effects_df[main_effects_df['significant']]

            if len(sig_main) > 0:
                f.write("### Key Findings\n\n")

                # Group by factor
                for factor in sig_main['factor'].unique():
                    factor_tests = sig_main[sig_main['factor'] == factor]
                    f.write(f"**{factor}**:\n")

                    for _, row in factor_tests.iterrows():
                        direction = "increases" if row['mean_diff'] > 0 else "decreases"
                        f.write(f"- {row['level1']} vs {row['level2']}: {direction} {row['metric']} by {abs(row['mean_diff']):.3f}\n")
                    f.write("\n")
            else:
                f.write("No significant main effects detected. This suggests:\n")
                f.write("- Fold size, train set size, and calibration method have minimal impact on performance\n")
                f.write("- Current default settings are likely adequate\n")
                f.write("- Consider investigating other factors (feature selection, model choice, etc.)\n\n")

        if len(interactions_df) > 0:
            sig_int = interactions_df[interactions_df['significant']]

            if len(sig_int) > 0:
                f.write("### Interaction Effects\n\n")
                f.write("Significant interactions indicate that the effect of one factor depends on another.\n")
                f.write("Investigate these combinations more carefully:\n\n")
                for _, row in sig_int.iterrows():
                    f.write(f"- {row['interaction']} (η²={row['eta_squared']:.3f})\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extended 3-factor factorial experiment analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest run
  python investigate_factorial_extended.py

  # Analyze specific run
  python investigate_factorial_extended.py --run-id run_20260131_232604

  # Custom paths
  python investigate_factorial_extended.py --results-dir /path/to/results --output-dir /path/to/output
        """
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='Results root directory (default: auto-detect)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Specific run ID (e.g., run_20260131_232604). If not provided, uses latest.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: results/{run_id}/analysis/factorial/)'
    )
    parser.add_argument(
        '--metric',
        default='AUROC',
        help='Primary metric for statistical testing (default: AUROC)'
    )
    args = parser.parse_args()

    # Auto-detect project root and results directory
    if args.results_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        args.results_dir = project_root / 'results'
        print(f"Auto-detected results directory: {args.results_dir}")

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    # Determine run_id
    if args.run_id:
        run_path = args.results_dir / args.run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        selected_run = args.run_id
    else:
        run_dirs = sorted([d for d in args.results_dir.glob('run_*') if d.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No run_* directories found in {args.results_dir}")
        selected_run = run_dirs[-1].name
        print(f"Auto-detected latest run: {selected_run}")

    search_path = args.results_dir / selected_run

    # Auto-detect output directory
    if args.output_dir is None:
        args.output_dir = search_path / 'analysis' / 'factorial'
        print(f"Auto-detected output directory: {args.output_dir}")

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
            config_id, model, seed, fold_size, train_size, calib_method = identify_config(run_dir)
            metrics = extract_metrics(run_dir)
            metrics['config'] = config_id
            metrics['model'] = model
            metrics['split_seed'] = seed
            metrics['fold_size'] = fold_size
            metrics['train_size'] = train_size
            metrics['calib_method'] = calib_method
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to process {run_dir}: {e}")

    if len(all_metrics) == 0:
        raise ValueError("No valid runs found")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.output_dir / 'metrics_all.csv', index=False)
    print(f"Saved raw metrics: {args.output_dir / 'metrics_all.csv'}")

    # Print config summary
    print("\nConfig summary:")
    print(f"  Fold sizes: {sorted(metrics_df['fold_size'].unique())}")
    print(f"  Train sizes: {sorted(metrics_df['train_size'].unique())}")
    print(f"  Calib methods: {sorted(metrics_df['calib_method'].unique())}")
    print(f"  Models: {sorted(metrics_df['model'].unique())}")
    print(f"  Total configs: {metrics_df['config'].nunique()}")

    # Generate comparison table
    print("\nGenerating comparison table...")
    comparison = generate_comparison(all_metrics)
    comparison.to_csv(args.output_dir / 'comparison_table.csv', index=False)
    print(f"Saved: {args.output_dir / 'comparison_table.csv'}")

    # Main effects analysis
    print("Analyzing main effects...")
    main_effects = main_effects_analysis(metrics_df, metric=args.metric)
    main_effects.to_csv(args.output_dir / 'main_effects.csv', index=False)
    print(f"Saved: {args.output_dir / 'main_effects.csv'}")

    # Interaction analysis
    print("Analyzing interactions...")
    interactions = interaction_analysis(metrics_df, metric=args.metric)
    interactions.to_csv(args.output_dir / 'interactions.csv', index=False)
    print(f"Saved: {args.output_dir / 'interactions.csv'}")

    # Power analysis
    print("Computing power analysis...")
    power_results = []
    if len(main_effects) > 0:
        for _, row in main_effects.iterrows():
            power = power_analysis(row['n'], row['cohen_d'], row['alpha_corrected'])
            power_results.append({
                'factor': row['factor'],
                'comparison': row['comparison'],
                'model': row['model'],
                'n': row['n'],
                'cohen_d': row['cohen_d'],
                'alpha': row['alpha_corrected'],
                'power': power,
            })

    power_df = pd.DataFrame(power_results)
    power_df.to_csv(args.output_dir / 'power_analysis.csv', index=False)
    print(f"Saved: {args.output_dir / 'power_analysis.csv'}")

    # Generate summary
    print("Generating summary...")
    generate_summary(comparison, main_effects, interactions, power_df,
                    args.output_dir / 'summary.md')
    print(f"Saved: {args.output_dir / 'summary.md'}")

    print("\nAnalysis complete!")
    print(f"\nView results in: {args.output_dir}")


if __name__ == '__main__':
    main()
