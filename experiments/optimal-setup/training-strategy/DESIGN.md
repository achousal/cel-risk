# Training Strategy

## Goal

Determine whether incident-only training and associated class-weighting choices provide the best operating strategy.

## Scope

This experiment is separate from panel discovery. It answers:

- Which cohort definition should be used for training?
- How should controls be weighted relative to incident cases?
- Does the incident-only framing improve generalization for the intended prediction target?

## Execution assets

- `scripts/run_incident_validation.py`
- `scripts/postprocess_incident_validation.py`
- `scripts/submit_incident_validation.sh`
- `scripts/submit_incident_validation_parallel.sh`

## Artifact target

Primary outputs live under `results/incident_validation/`.
