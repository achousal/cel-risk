# dataset/

Write-once metadata and QC documentation for the celiac dataset.

## Files

- `fingerprint.yaml` — canonical dataset fingerprint per `rulebook/SCHEMA.md`.
  Write-once. If any field changes, a new project slug is required (e.g.
  `celiac-v2`).

## Pending documentation

As the V0 -> V1 transition progresses, fill in the following below this line:

### Table 1 (cohort demographics)
TODO: age, sex, BMI, genetic ancestry breakdown for the 148 incident cases
vs the 43,662 controls. Source: `analysis/` summary tables.

### QC summary
TODO: per-protein missingness profile, LOD handling policy, outlier removal
criteria. Cite the derivation script.

### Residualization policy
TODO: age/sex/BMI/batch residualization scheme (if applied) + justification.
Note whether residualization is applied to TRAIN only or to all splits, and
whether the residualization model is fit on TRAIN then applied downstream.

### Split generation provenance
30 seeds in `splits/` (100-129). Scenario-tagged filenames
(`split_meta_IncidentPlusPrevalent_seed*.json`). V0 evaluated multiple
training strategies; see `gates/v0-strategy/` for which scenarios exist on
disk vs which were evaluated.
