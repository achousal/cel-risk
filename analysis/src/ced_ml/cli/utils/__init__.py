"""
CLI utility functions.

Provides shared utilities used across CLI commands including:
- Argument validation
- Seed list parsing
- Config merging
"""

from ced_ml.cli.utils.config_merge import merge_config_with_cli
from ced_ml.cli.utils.seed_parsing import parse_seed_list, parse_seed_range
from ced_ml.cli.utils.validation import (
    validate_model_name,
    validate_mutually_exclusive,
    validate_run_id_format,
)

__all__ = [
    "merge_config_with_cli",
    "parse_seed_list",
    "parse_seed_range",
    "validate_mutually_exclusive",
    "validate_run_id_format",
    "validate_model_name",
]
