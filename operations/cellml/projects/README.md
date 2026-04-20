# projects/

Project-side of the CellML architecture. Each subdirectory is a dataset-scoped
project that runs gates against the shared rulebook at
`operations/cellml/rulebook/`.

## Layout

```
projects/
  registry.csv            project -> active rulebook version
  README.md               this file
  <project>/              one directory per cohort (e.g. celiac, ibd, t1d)
    README.md             orientation for that project
    dataset/              fingerprint + QC documentation
    gates/                v0-strategy, v1-recipe, ...
    tensions/             three subfolders: data-quality, gate-decisions, rule-vs-observation
    rulebook-snapshot/    hash.txt binding snapshot after rulebook tag
    archive/              superseded artifacts
```

## Gate lifecycle

Per `rulebook/SCHEMA.md`, each gate directory contains four files written in
order: `ledger.md` (PRE-run), `observation.md` (POST-run metrics, facts only),
`decision.md` (POST-run reasoning), `tensions.md` (delta log). Once
`observation.md` is written, `ledger.md` is immutable by convention.

## Registry

`registry.csv` maps each project to its currently active rulebook version.
Columns: `project`, `active_rulebook_version`, `created`, `status`.
Status values: `active`, `paused`, `retired`.

The current celiac binding (`rb-v0.0.0-unfinalized`) is a placeholder until
the first rulebook tag (`rb-v0.1.0`) is cut after ADR -> condensate migration
completes.
