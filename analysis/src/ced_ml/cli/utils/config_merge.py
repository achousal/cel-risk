"""
CLI config merging utilities.

Provides functions for merging configuration files with CLI arguments,
handling the priority resolution: CLI args > config file > defaults.
"""

from pathlib import Path
from typing import Any

import yaml


def merge_config_with_cli(
    config_path: Path | None,
    cli_kwargs: dict[str, Any],
    param_keys: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Merge configuration file with CLI arguments.

    Priority: CLI args > config file > defaults (None)

    Parameters
    ----------
    config_path : Path, optional
        Path to YAML configuration file. If None, only CLI args are used.
    cli_kwargs : dict
        Dictionary of CLI keyword arguments (from click command)
    param_keys : list of str
        List of parameter names to merge
    verbose : bool, default False
        If True, print messages about which config values are being used

    Returns
    -------
    dict
        Merged configuration with CLI args taking precedence

    Examples
    --------
    >>> config_path = Path('configs/optimize_panel.yaml')
    >>> cli_kwargs = {'cv_folds': 10, 'min_size': None}
    >>> param_keys = ['cv_folds', 'min_size', 'step_strategy']
    >>> merged = merge_config_with_cli(config_path, cli_kwargs, param_keys)
    >>> merged['cv_folds']
    10  # From CLI
    >>> merged['min_size']
    5   # From config (if present)
    """
    # Load config file if provided
    config_params: dict[str, Any] = {}
    if config_path and config_path.exists():
        try:
            with open(config_path) as f:
                config_params = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            raise ValueError(f"Failed to load config file {config_path}: {e}") from e

    # Merge config with CLI args (CLI takes precedence)
    merged = {}
    for key in param_keys:
        cli_value = cli_kwargs.get(key)
        config_value = config_params.get(key)

        if cli_value is not None:
            # CLI argument provided - always use it
            merged[key] = cli_value
        elif config_value is not None:
            # CLI not provided, but config has value - use config
            merged[key] = config_value
            if verbose:
                print(f"Using config value for {key}: {config_value}")
        else:
            # Neither CLI nor config provided - leave as None
            merged[key] = None

    return merged


def load_config_file(config_path: Path | None) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : Path, optional
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary (empty if path is None or file doesn't exist)

    Raises
    ------
    ValueError
        If config file exists but cannot be parsed

    Examples
    --------
    >>> config = load_config_file(Path('configs/training_config.yaml'))
    >>> config['models']
    ['LR_EN', 'RF', 'XGBoost']
    """
    if not config_path or not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config
    except (yaml.YAMLError, OSError) as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}") from e


def merge_nested_config(
    config_params: dict[str, Any],
    cli_kwargs: dict[str, Any],
    nested_key: str,
    param_keys: list[str],
) -> dict[str, Any]:
    """
    Merge nested configuration section with CLI arguments.

    Useful for configurations with nested sections like 'composite_ranking'
    or 'significance'.

    Parameters
    ----------
    config_params : dict
        Full configuration dictionary
    cli_kwargs : dict
        CLI keyword arguments
    nested_key : str
        Key for nested section in config (e.g., 'composite_ranking')
    param_keys : list of str
        List of parameter names to extract from nested section

    Returns
    -------
    dict
        Merged nested configuration

    Examples
    --------
    >>> config_params = {
    ...     'composite_ranking': {
    ...         'oof_weight': 0.6,
    ...         'essentiality_weight': 0.3
    ...     }
    ... }
    >>> cli_kwargs = {'oof_weight': 0.7}
    >>> merged = merge_nested_config(
    ...     config_params, cli_kwargs, 'composite_ranking',
    ...     ['oof_weight', 'essentiality_weight']
    ... )
    >>> merged['oof_weight']
    0.7  # From CLI
    >>> merged['essentiality_weight']
    0.3  # From config
    """
    nested_config = config_params.get(nested_key, {})
    merged = {}

    for key in param_keys:
        cli_value = cli_kwargs.get(key)
        nested_value = nested_config.get(key)

        if cli_value is not None:
            merged[key] = cli_value
        elif nested_value is not None:
            merged[key] = nested_value
        else:
            merged[key] = None

    return merged
