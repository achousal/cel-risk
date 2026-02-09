"""Data-related CLI commands (save-splits, convert-to-parquet)."""

import click

from ced_ml.cli.options import (
    config_option,
    infile_option,
    n_splits_option,
    outdir_option,
)
from ced_ml.cli.save_splits import run_save_splits
from ced_ml.data.io import convert_csv_to_parquet


@click.command("save-splits")
@config_option
@infile_option
@outdir_option
@click.option(
    "--mode",
    type=click.Choice(["development", "holdout"]),
    default=None,
    help="Split mode: development (TRAIN/VAL/TEST) or holdout (TRAIN/VAL/TEST + HOLDOUT)",
)
@click.option(
    "--scenarios",
    multiple=True,
    default=None,
    help="Scenarios to generate (can be repeated)",
)
@n_splits_option
@click.option(
    "--val-size",
    type=float,
    default=None,
    help="Validation set proportion (0-1)",
)
@click.option(
    "--test-size",
    type=float,
    default=None,
    help="Test set proportion (0-1)",
)
@click.option(
    "--holdout-size",
    type=float,
    default=None,
    help="Holdout set proportion (only if mode=holdout)",
)
@click.option(
    "--seed-start",
    type=int,
    default=None,
    help="Starting random seed",
)
@click.option(
    "--prevalent-train-only",
    is_flag=True,
    default=None,
    help="Restrict prevalent cases to TRAIN set only (prevents reverse causality)",
)
@click.option(
    "--prevalent-train-frac",
    type=float,
    default=None,
    help="Fraction of prevalent cases to include in TRAIN (0-1)",
)
@click.option(
    "--train-control-per-case",
    type=float,
    default=None,
    help="Downsample TRAIN controls to N per case (e.g., 5 for 1:5 ratio)",
)
@click.option(
    "--eval-control-per-case",
    type=float,
    default=None,
    help="Downsample VAL/TEST controls to N per case",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing split files",
)
@click.option(
    "--temporal-split",
    is_flag=True,
    default=None,
    help="Enable temporal (chronological) validation splits",
)
@click.option(
    "--temporal-column",
    type=str,
    default=None,
    help="Column name for temporal ordering (e.g., 'CeD_date')",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (format: key=value or nested.key=value)",
)
@click.pass_context
def save_splits(ctx, config, **kwargs):
    """Generate train/val/test splits with stratification and optional downsampling."""

    # Collect CLI args
    cli_args = {k: v for k, v in kwargs.items() if k != "override"}
    overrides = list(kwargs.get("override", []))

    # Run split generation
    run_save_splits(
        config_file=config,
        cli_args=cli_args,
        overrides=overrides,
        log_level=ctx.obj.get("log_level"),
    )


@click.command("convert-to-parquet")
@click.argument(
    "csv_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output Parquet file path (default: same as input with .parquet extension)",
)
@click.option(
    "--compression",
    type=click.Choice(["snappy", "gzip", "brotli", "zstd", "none"]),
    default="snappy",
    help="Compression algorithm (default: snappy)",
)
def convert_to_parquet(csv_path, output, compression):
    """
    Convert proteomics CSV file to Parquet format.

    This command reads a CSV file and converts it to Parquet format with
    compression. Only columns needed for modeling are included (same as
    the training pipeline).

    CSV_PATH: Path to input CSV file

    Example:
        ced convert-to-parquet data/celiac_proteomics.csv
        ced convert-to-parquet data/celiac_proteomics.csv -o data/celiac.parquet --compression gzip
    """

    try:
        parquet_path = convert_csv_to_parquet(
            csv_path=csv_path,
            parquet_path=output,
            compression=compression,
        )
        click.echo(f"Successfully converted to: {parquet_path}")
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
