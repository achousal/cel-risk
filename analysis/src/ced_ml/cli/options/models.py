"""Model-related CLI options."""

import click

from ced_ml.data.schema import ModelName

model_option = click.option(
    "--model",
    type=click.Choice([m.value for m in ModelName], case_sensitive=False),
    multiple=True,
    default=None,
    help="Model(s) to train (can be repeated). Default: all models from config.",
)


split_seed_option = click.option(
    "--split-seed",
    type=int,
    multiple=True,
    default=None,
    help="Split seed(s) to use (can be repeated). Default: all seeds from config.",
)


n_splits_option = click.option(
    "--n-splits",
    type=int,
    default=None,
    help="Number of repeated splits with different seeds",
)
