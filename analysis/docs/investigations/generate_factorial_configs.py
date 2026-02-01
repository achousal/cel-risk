#!/usr/bin/env python3
"""
Generate config files for 3-factor factorial experiments.

Creates YAML configs for all combinations of:
- fold_size (k in CV)
- train_size (N)
- calibration_method

Feature selection is FIXED at hybrid for all configs.

Usage:
    # Generate full 3×3×3 = 27 configs
    python generate_factorial_configs.py --output-dir configs/factorial/

    # Generate subset (2×2×2 = 8 configs)
    python generate_factorial_configs.py \
        --fold-sizes 5 10 \
        --train-sizes 298 894 \
        --calib-methods isotonic platt \
        --output-dir configs/factorial_subset/

    # Quick test (minimal)
    python generate_factorial_configs.py --quick-test

Outputs:
    configs/factorial/k{fold}_N{train}_{calib}.yaml for each combination
"""

import argparse
from itertools import product
from pathlib import Path
from typing import List

import yaml


def calculate_control_ratio(n_train: int, n_cases: int = 149) -> int:
    """
    Calculate control:case ratio from total training size.

    Args:
        n_train: Total training samples
        n_cases: Number of case samples (incident + prevalent)

    Returns:
        Control per case ratio (rounded)

    Examples:
        n_train=298 → (298-149)/149 = 1
        n_train=596 → (596-149)/149 = 3
        n_train=894 → (894-149)/149 = 5
    """
    n_controls = n_train - n_cases
    ratio = round(n_controls / n_cases)
    return max(1, ratio)


def generate_config(
    fold_size: int,
    train_size: int,
    calib_method: str,
    split_seeds: List[int],
    models: List[str],
    base_config: dict = None
) -> dict:
    """
    Generate config dict for single factorial combination.

    Args:
        fold_size: k in k-fold CV (3, 5, 10)
        train_size: Total training samples (298, 596, 894)
        calib_method: isotonic, platt, or oof_posthoc
        split_seeds: List of random seeds
        models: List of model names
        base_config: Optional base config to extend

    Returns:
        Config dictionary
    """
    # Start with base or empty
    config = base_config.copy() if base_config else {}

    # CV configuration
    config['cv_config'] = {
        'n_splits': fold_size,
        'strategy': 'stratified',
        'shuffle': True
    }

    # Training configuration
    control_ratio = calculate_control_ratio(train_size)
    config['training'] = {
        'n_train': train_size,
        'train_control_per_case': control_ratio,
        'prevalent_train_frac': 0.5,  # Fixed (see ADR-002)
        'random_state': 42
    }

    # Feature selection (FIXED at hybrid)
    config['feature_selection'] = {
        'method': 'hybrid',
        'stability': {
            'threshold': 0.75,
            'min_votes': None  # Auto-calculate
        },
        'kbest': {
            'k': 100,
            'score_func': 'f_classif'
        },
        'screening': {
            'p_threshold': 0.1
        }
    }

    # Calibration configuration
    config['calibration'] = {
        'method': calib_method
    }

    # Models and seeds
    config['models'] = models
    config['split_seeds'] = split_seeds

    # Metadata
    config['experiment'] = {
        'type': 'factorial',
        'factors': {
            'fold_size': fold_size,
            'train_size': train_size,
            'calib_method': calib_method
        },
        'feature_selection': 'hybrid'
    }

    return config


def write_config(config: dict, output_path: Path):
    """Write config to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description='Generate factorial experiment configs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Factor levels
    parser.add_argument(
        '--fold-sizes',
        type=int,
        nargs='+',
        default=[3, 5, 10],
        help='Fold sizes (k in k-fold CV). Default: 3 5 10'
    )
    parser.add_argument(
        '--train-sizes',
        type=int,
        nargs='+',
        default=[298, 596, 894],
        help='Training set sizes (N). Default: 298 596 894'
    )
    parser.add_argument(
        '--calib-methods',
        type=str,
        nargs='+',
        default=['isotonic', 'platt', 'oof_posthoc'],
        help='Calibration methods. Default: isotonic platt oof_posthoc'
    )

    # Seeds and models
    parser.add_argument(
        '--split-seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Split seeds. Default: 0 1 2 3 4'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['LR_EN', 'XGBoost'],
        help='Models to train. Default: LR_EN XGBoost'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('configs/factorial'),
        help='Output directory for configs. Default: configs/factorial/'
    )

    # Convenience options
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Generate minimal 2×2×2 config for testing (overrides factor levels)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configs without writing files'
    )

    args = parser.parse_args()

    # Quick test override
    if args.quick_test:
        args.fold_sizes = [5, 10]
        args.train_sizes = [298, 894]
        args.calib_methods = ['isotonic', 'platt']
        args.split_seeds = [0, 1, 2]
        args.models = ['LR_EN']
        print("Quick test mode: 2×2×2 = 8 configs, 3 seeds, 1 model")

    # Generate all combinations
    combinations = list(product(
        args.fold_sizes,
        args.train_sizes,
        args.calib_methods
    ))

    print(f"\nGenerating {len(combinations)} configs:")
    print(f"  Fold sizes: {args.fold_sizes}")
    print(f"  Train sizes: {args.train_sizes}")
    print(f"  Calib methods: {args.calib_methods}")
    print(f"  Split seeds: {len(args.split_seeds)} seeds")
    print(f"  Models: {args.models}")
    print(f"  Total runs: {len(combinations)} × {len(args.split_seeds)} × {len(args.models)} = "
          f"{len(combinations) * len(args.split_seeds) * len(args.models)}")
    print()

    # Generate configs
    configs_created = []
    for fold_size, train_size, calib_method in combinations:
        # Config ID
        config_id = f"k{fold_size}_N{train_size}_{calib_method}"

        # Generate config
        config = generate_config(
            fold_size=fold_size,
            train_size=train_size,
            calib_method=calib_method,
            split_seeds=args.split_seeds,
            models=args.models
        )

        # Output path
        output_path = args.output_dir / f"{config_id}.yaml"

        if args.dry_run:
            print(f"Would create: {output_path}")
            print(yaml.dump(config, default_flow_style=False, sort_keys=False))
            print("=" * 80)
        else:
            write_config(config, output_path)
            configs_created.append(output_path)
            print(f"Created: {output_path}")

    if not args.dry_run:
        print(f"\nSuccessfully created {len(configs_created)} config files in {args.output_dir}")
        print("\nNext steps:")
        print("1. Review configs (optional)")
        print(f"2. Run experiments: for cfg in {args.output_dir}/*.yaml; do ced run-pipeline --config $cfg; done")
        print(f"3. Analyze results: python investigate_factorial_extended.py")


if __name__ == '__main__':
    main()
