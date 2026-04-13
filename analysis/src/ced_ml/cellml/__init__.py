"""ced_ml.cellml — experiment-level factorial orchestration.

The cellml package wraps the existing recipes/ machinery (panels, cells,
factorial configs) into an experiment-oriented CLI: `ced cellml ...`.

Every experiment is a single declarative spec.yaml (see
``ced_ml.cellml.schema.ExperimentSpec``) that names panels, axes, and
resources. The package orchestrates: resolve -> derive panels ->
generate cell configs -> LSF submit -> monitor -> compile -> validate.

This is additive. The older workflow (``ced derive-recipes`` +
``operations/cellml/submit_experiment.sh``) keeps working untouched.
"""
