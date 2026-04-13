# Operations

Orchestration code, configs, scripts, and post-hoc analysis that drives the
`ced_ml` library to produce concrete experiments.

**The library / operations / experiments split:**
- `analysis/src/ced_ml/` — the **library** (pure, experiment-agnostic).
- `operations/` — **how to run things**: configs, scripts, sweep specs, post-hoc analysis. Versioned.
- `results/` — **the experiments themselves**: each run is one experiment, registered in `results/experiment_registry.csv`. Gitignored.
- `logs/` — runtime logs mirroring the `results/` namespace. Gitignored.

## Active operations

### `cellml/`
**CellML** — Factorial recipe sweep to determine optimal ML configuration for the
CeD proteomic risk model. Sweeps models × preprocessing recipes × feature orders,
then runs V0 gate + main training + holdout evaluation.

- `MASTER_PLAN.md` — single index for all phases and active run IDs
- `DESIGN.md` — scientific design + recipe definitions
- `configs/manifest.yaml` — declarative recipe + factorial source of truth
- `analysis/` — post-hoc R/Python analysis (operates on artifacts)
- `sweeps/` — sweep orchestration engine (CellML-coupled today)
- `submit_experiment.sh`, `compile_factorial.py`, `monitor_factorial.py`, `validate_tree.R`

Outputs land in `results/cellml/{discovery,v0_gate,main,holdout,compiled,figures}/`.

### `incident-validation/`
**Incident Validation** — Validates the CeD risk model on incident (pre-diagnostic)
cases. Tests LR_EN and LinSVM_cal across training strategies and class-weight schemes.

- `README.md` — operation layout, usage, analysis script map
- `RESULTS_LR_EN.md` — summary of LR_EN findings
- `scripts/` — `run_lr.py`, `run_svm.py`, `submit_lr_parallel.sh`, `submit_svm*.sh`
- `analysis/` — calibration, DCA, SHAP, saturation (Python + R)

Outputs land in `results/incident-validation/{lr,linsvm_cal}/`.

## Archived

### `_archive/gen1/`
First-generation analysis from before the current `ced_ml` pipeline existed.
Frozen — do not run. Kept for provenance and so the manifest's trunk inputs
(which still reference these results) continue to resolve.

## Conventions

Each operation:
1. Calls `ced_ml` via CLI (`ced train`, `ced run-pipeline`) or Python API.
2. Writes results under `results/<operation>/` (namespaced from project root).
3. Logs land under `logs/<operation>/` (mirrors results namespace).
4. Tags runs with `--experiment <tag>` on `ced run-pipeline` / `ced train` so the auto-generated run_id is prefixed and the run is recorded in `results/experiment_registry.csv`.
