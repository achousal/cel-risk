# Experiments

Consumer code that orchestrates `ced_ml` for specific research questions.
`ced_ml` (in `analysis/`) is the library; these directories are the callers.

## Active experiments

### `cellml/`
**CellML** — Factorial recipe sweep to determine optimal ML configuration for the
CeD proteomic risk model. Sweeps models × preprocessing recipes × feature orders,
then runs V0 gate + main training + holdout evaluation.

- `MASTER_PLAN.md` — single index for all phases and active run IDs
- `configs/` — manifest.yaml, recipe specs
- `analysis/` — post-hoc R/Python analysis scripts (operate on artifacts)
- `sweeps/` — sweep orchestration engine

### `incident-validation/`
**Incident Validation** — Validates the CeD risk model on incident (pre-diagnostic)
cases. Tests LR_EN and LinSVM_cal across training strategies and class-weight schemes.

- `scripts/` — `run_lr.py`, `run_svm.py`, `submit_lr_parallel.sh`, `submit_svm*.sh`
- `analysis/` — calibration, saturation, SHAP analysis + figures
- `RESULTS_LR_EN.md` — summary of LR_EN results

## Archived

### `_archive/gen1/`
First-generation analysis before the current ced_ml pipeline existed.

## Conventions

Each experiment:
1. Calls `ced_ml` via CLI (`ced train`, `ced run-pipeline`) or Python API.
2. Writes results under `results/<experiment>/` (namespaced from project root).
3. Logs land under `logs/<experiment>/` (mirrors results namespace).
4. Uses `--experiment <tag>` on `ced run-pipeline` to prefix run IDs.

Results and logs directories are gitignored; only scripts, configs, and analysis
code are tracked.
