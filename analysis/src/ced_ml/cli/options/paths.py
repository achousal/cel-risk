"""Path-related CLI options."""

import click

infile_option = click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)


outdir_option = click.option(
    "--outdir",
    type=click.Path(),
    default="splits",
    help="Output directory for splits (default: splits/)",
)


results_dir_option = click.option(
    "--results-dir",
    type=click.Path(exists=True),
    help="Results directory containing trained models",
)


splits_dir_option = click.option(
    "--splits-dir",
    type=click.Path(exists=True),
    help="Directory containing split files",
)
