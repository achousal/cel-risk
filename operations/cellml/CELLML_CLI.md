# `ced cellml` CLI

Experiment-level factorial orchestration. Wraps the existing
`ced derive-recipes` / `scripts/submit_experiment.sh` machinery behind a single
experiment-oriented CLI group. Additive — old workflow still works.

## Quick start

```bash
# 1. Scaffold a new experiment from a template
ced cellml init my_exp --template minimal

# 2. Check the plan (cell count, sample)
ced cellml plan experiments/my_exp/spec.yaml

# 3. Resolve semantic decisions (writes resolved_spec.yaml)
ced cellml resolve experiments/my_exp/spec.yaml

# 4. Generate per-cell configs
ced cellml generate experiments/my_exp/spec.yaml

# 5. Dry-run submit (writes runner scripts, prints bsub, no submission)
ced cellml submit my_exp --dry-run

# 6. Real submit
ced cellml submit my_exp

# 7. Watch the queue
ced cellml status my_exp

# 8. After jobs finish: compile + validate
ced cellml compile my_exp
ced cellml validate my_exp

# Or: one-shot full pipeline (dry-run mode stops after submit)
ced cellml run my_exp --dry-run
```

## Subcommands

| Command | What it does |
| --- | --- |
| `init NAME --template {minimal,svm,full}` | Scaffold `experiments/NAME/spec.yaml` |
| `plan SPEC [--override k=v ...]` | Load spec, apply overrides, print cell count |
| `resolve SPEC` | Run semantic resolution, dump `resolved_spec.yaml` |
| `generate SPEC` | Resolve + derive panels + write per-cell configs |
| `submit NAME [--dry-run] [--cells 1-N] [--wall] [--queue] [--project]` | Write runners, (optionally) `bsub` an LSF array |
| `status NAME` | `bjobs`-based state counts |
| `compile NAME` | Gather per-cell `test_metrics_summary.csv` into `compiled.csv` |
| `validate NAME` | Run `operations/cellml/validate_tree.R`, parse V1-V4 |
| `list` | Print the experiment registry |
| `show NAME` | Registry row + top 5 cells by PRAUC |
| `run NAME [--dry-run]` | `generate -> submit -> compile -> validate` |

## Spec format

See `experiments/templates/{minimal,svm,full}.yaml` for working
examples. Top-level keys:

- `name`, `description`
- `base_configs`: paths to base training / pipeline / splits YAMLs
- `trunks[]`: source data trunks (id + proteins_csv + optional sweep / feature CSV)
- `panels[]`: panel sources. Three modes:
  - `source: derived` — trunk + ordering + size_rule (delegates to `ced_ml.recipes.derive`)
  - `source: fixed_csv` — copy an existing CSV verbatim
  - `source: reference` — extract from a prior experiment (v1: raises
    `NotImplementedError` cleanly; v2 feature)
  Optional: `pinned_model` collapses the model axis for one panel.
- `axes`: factorial axes (`model`, `calibration`, `weighting`,
  `downsampling`, `scenario`, `feature_selection`, `control_ratio`)
- `seeds`: `{start, end}`
- `optuna`: trial budget + storage
- `resources`: LSF defaults (`wall`, `cores`, `mem_mb_per_core`, `queue`, `project`)

## Axis semantics

### Scenario axis

Scenario entries are `TrainingStrategy`-compatible dicts. When the
scenario axis is non-empty, each cell gets its own
`splits_config.yaml` overlay produced by
`TrainingStrategy.to_splits_overlay()` — the *same mechanism*
`generate_v0_configs` has always used for the v0 gate. The per-cell
`pipeline_hpc.yaml` has its `configs.splits` pointed at that file. When
the axis is empty (`scenario: []`), no per-cell splits config is
written and cell names keep the legacy `{model}_{cal}_{wt}_ds{ratio}`
form for backcompat.

### feature_selection axis

Wired through the config_gen API so v2 can add `multi_stage` without
another signature break. v1 only accepts `fixed_panel`; any other
value raises `NotImplementedError`.

### Panel sources

| Source | Description |
| --- | --- |
| `derived` | Delegates to `ced_ml.recipes.derive.derive_all_recipes` — a one-recipe in-memory `Manifest` is built from the `PanelSpec` fields. Unchanged derivation pipeline. |
| `fixed_csv` | CSV is copied to `experiments/<name>/panels/<panel_id>/panel.csv`. |
| `reference` | Pulls a panel from a prior registered experiment. v1 stub: raises `NotImplementedError`. v2 will walk the referenced `compiled.csv`, pick the winner by `extract={best_prauc,union,intersection}`, and copy that cell's `panel.csv`. |

## Migration from the old workflow

Both flows coexist:

```bash
# Old flow — still works, unchanged
ced derive-recipes --manifest operations/cellml/configs/manifest.yaml
bash operations/cellml/submit_experiment.sh --experiment factorial ...

# New flow — additive
ced cellml init my_exp --template svm
ced cellml generate experiments/my_exp/spec.yaml
ced cellml submit my_exp --dry-run
```

The new CLI does not rewrite `operations/cellml/submit_experiment.sh`
— it ports the same runner-script + `bsub` logic into Python for use
by `ced cellml submit`.

Backcompat is enforced by
`analysis/tests/cellml/test_config_gen_backcompat.py`, which snapshots
the cell count + column set for `operations/cellml/configs/manifest.yaml`
and fails if either drifts.

## Design decisions

- **Scenarios** use `TrainingStrategy.to_splits_overlay()` — same
  mechanism `generate_v0_configs` already uses. One code path, not two.
- **`feature_selection`** axis is wired in the API but only
  `fixed_panel` is supported in v1. Non-fixed values raise
  `NotImplementedError` with a clear message.
- **Panel sources**: `derived`, `fixed_csv`, `reference`. `reference`
  is a v1 stub because extraction rules need real downstream use cases
  to design them correctly.
- **LSF submit** is Python-via-`subprocess.run` calling `bsub`, since
  cel-risk standardizes on LSF. The legacy bash script in
  `operations/cellml/submit_experiment.sh` is preserved untouched.
- **Registry** is a CSV at `experiments/_registry.csv` with
  `fcntl.flock` around every write (stdlib only — no new deps).
- **Tests** mock all HPC interaction. No live `bsub` / `bjobs` /
  `Rscript` calls. Runner scripts and bsub command shape are asserted
  against mocked `subprocess.run` output.
- **Resolution stage** is separate from generation so humans can
  inspect `resolved_spec.yaml` before any panels are derived.
- **Templates** live in `experiments/templates/` and are copied with
  `name` rewritten at init time.
