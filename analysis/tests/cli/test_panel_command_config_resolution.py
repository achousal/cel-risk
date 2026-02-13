"""Tests for panel command CLI/config resolution helpers."""

from ced_ml.cli.commands.panel import _resolve_cli_or_config


def test_resolve_cli_or_config_preserves_false():
    """Explicit CLI False must not be overwritten by config True."""
    assert _resolve_cli_or_config(False, True, default=False) is False


def test_resolve_cli_or_config_preserves_zero():
    """Explicit CLI numeric zero must not be overwritten."""
    assert _resolve_cli_or_config(0, 3, default=1) == 0


def test_resolve_cli_or_config_uses_config_when_cli_missing():
    """Config value is used when CLI value is None."""
    assert _resolve_cli_or_config(None, 0.1, default=0.05) == 0.1


def test_resolve_cli_or_config_uses_default_when_both_missing():
    """Default is used when both CLI and config are None."""
    assert _resolve_cli_or_config(None, None, default=2) == 2
