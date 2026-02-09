"""
CLI argument validation utilities.

Provides functions for validating CLI arguments and ensuring
consistency across commands.
"""

import re
from typing import Any


def validate_mutually_exclusive(
    arg1_name: str,
    arg1_value: Any,
    arg2_name: str,
    arg2_value: Any,
    error_message: str | None = None,
) -> None:
    """
    Validate that two arguments are not both provided.

    Parameters
    ----------
    arg1_name : str
        Name of first argument (e.g., '--results-dir')
    arg1_value : Any
        Value of first argument (None if not provided)
    arg2_name : str
        Name of second argument (e.g., '--run-id')
    arg2_value : Any
        Value of second argument (None if not provided)
    error_message : str, optional
        Custom error message. If not provided, generates a default message.

    Raises
    ------
    ValueError
        If both arguments are provided (both non-None)

    Examples
    --------
    >>> validate_mutually_exclusive('--results-dir', '/path', '--run-id', '12345')
    ValueError: --results-dir and --run-id are mutually exclusive. Use one or the other.

    >>> validate_mutually_exclusive('--results-dir', '/path', '--run-id', None)
    # No error raised
    """
    if arg1_value is not None and arg2_value is not None:
        if error_message is None:
            error_message = (
                f"{arg1_name} and {arg2_name} are mutually exclusive. " "Use one or the other."
            )
        raise ValueError(error_message)


def validate_run_id_format(run_id: str) -> bool:
    """
    Validate run_id follows expected format.

    Expected formats:
    - Timestamp: YYYYMMDD_HHMMSS (e.g., '20260127_115115')
    - Custom: alphanumeric with underscores (e.g., 'production_v1')

    Parameters
    ----------
    run_id : str
        Run ID to validate

    Returns
    -------
    bool
        True if valid format, False otherwise

    Examples
    --------
    >>> validate_run_id_format('20260127_115115')
    True

    >>> validate_run_id_format('production_v1')
    True

    >>> validate_run_id_format('invalid-run-id!')
    False
    """
    # Allow alphanumeric characters and underscores
    pattern = r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, run_id))


def validate_model_name(model_name: str) -> bool:
    """
    Validate model name follows expected format.

    Expected format: alphanumeric with underscores (e.g., 'LR_EN', 'XGBoost')

    Parameters
    ----------
    model_name : str
        Model name to validate

    Returns
    -------
    bool
        True if valid format, False otherwise

    Examples
    --------
    >>> validate_model_name('LR_EN')
    True

    >>> validate_model_name('XGBoost')
    True

    >>> validate_model_name('invalid-model!')
    False
    """
    # Allow alphanumeric characters and underscores
    pattern = r"^[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, model_name))
