# CeliacRisks

**Production-grade ML pipeline for disease risk prediction from high-dimensional biomarker data**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-1422%20passing-success)
![Coverage](https://img.shields.io/badge/coverage-14%25-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

**Core features:**
- Nested cross-validation with Bayesian hyperparameter optimization
- Five feature selection strategies (hybrid stability, nested RFECV, post-hoc RFE, cross-model consensus, fixed panel)
- Permutation testing for statistical significance of predictive performance
- Cross-model consensus panel generation via Robust Rank Aggregation
- Multi-model ensemble with stacking (RF, XGBoost, SVM, Logistic Regression)
- HPC-ready (LSF/Slurm) with full provenance tracking

---

## Quick Start

```bash
git clone https://github.com/achousal/CeliacRisks.git
```

### Local
```bash
cd CeliacRisks/
pip install -e analysis/

conda activate ced_ml

ced run-pipeline
```

### HPC
```bash
cd CeliacRisks/
bash analysis/scripts/hpc_setup.sh

source analysis/venv/bin/activate  # Required if using venv

# Submit pipeline with LSF job dependency chains
ced run-pipeline --hpc

# Monitor jobs
bjobs -w | grep CeD_
```

**Pipeline flow:** Data → Split → Feature Selection → Model Training → Calibration → Ensemble → Evaluation

---

### 1. Run the Full Pipeline (Local)

```bash
cd CeliacRisks/
ced run-pipeline
```

**What happens:** Trains 4 models (LR_EN, RF, LinSVM_cal, XGBoost) across splits with nested CV, hyperparameter tuning, calibration, ensemble, and panel optimization. Results in `results/`.

### 2. Run the Full Pipeline (HPC)
Submit batch jobs with automated resource management and dependency chains:

```bash
cd CeliacRisks/
ced run-pipeline --hpc

# Monitor jobs
bjobs -w | grep CeD_
```

**Configure resources** in `analysis/configs/pipeline_hpc.yaml` (cores, memory, walltime, queue).

**What happens:**
1. Generates splits locally
2. Submits per-model training jobs to LSF
3. Chains aggregation, ensemble, and panel optimization as dependent jobs
4. Auto-detects everything from run-id

**No manual post-processing needed** - all steps are chained automatically.

### 3. CLI Commands

| Command | Description |
|---------|-------------|
| `ced run-pipeline` | End-to-end workflow orchestration (local or `--hpc`) |
| `ced save-splits` | Generate stratified train/val/test splits |
| `ced train` | Train a single model on one split |
| `ced train-ensemble` | Train stacking meta-learner on base model OOF predictions |
| `ced aggregate-splits` | Aggregate results across splits with bootstrap CIs |
| `ced optimize-panel` | Find minimal protein panel for a single model |
| `ced consensus-panel` | Cross-model consensus panel via Robust Rank Aggregation |
| `ced permutation-test` | Test statistical significance via label permutation |
| `ced eval-holdout` | Evaluate on held-out test set |
| `ced config` | Validate or diff config files |
| `ced convert-to-parquet` | Convert CSV to Parquet format |

Run `ced --help` or `ced <command> --help` for full usage.

---

### Configuration-Based Workflow

All commands use YAML configs in `analysis/configs/`:

- `pipeline_local.yaml` / `pipeline_hpc.yaml` - Models, paths, execution settings
- `training_config.yaml` - Feature selection, calibration, ensemble
- `splits_config.yaml` - Train/val/test ratios

---

### Evaluation
- **Discrimination**: AUROC, PR-AUC with bootstrap CIs
- **Calibration**: AUROC, Brier score, slope/intercept, calibration curves (OOF-posthoc)
- **Clinical Utility**: Decision curve analysis with auto-configured threshold ranges
- **Visualizations**: ROC/PR curves, calibration plots, risk distributions, DCA plots

### Production Features
- **Reproducibility**: Fixed seeds, YAML configs, Git commit tracking
- **HPC Support**: LSF/Slurm array jobs with automated post-processing (conda/venv compatible)
- **Provenance**: Complete metadata for every run (config, environment, timing)
- **Investigation Framework**: Factorial experiment design for methodological optimization

---

## Example Results: Celiac Disease

**Dataset:** 43,960 subjects (148 incident cases, 150 prevalent cases), 2,920 proteins, plus demographics (age, BMI, sex, genetic ancestry)

**Top Features:** TGM2, CXCL9, ITGB7, MUC2 (known CeD biomarkers)

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Project overview |
| [ARCHITECTURE.md](analysis/docs/ARCHITECTURE.md) | Technical architecture + code pointers |
| [ADRs](analysis/docs/adr/) | 15 architectural decisions (split strategy, calibration, ensembles, etc.) |
| [ARTIFACTS.md](analysis/docs/reference/ARTIFACTS.md) | Output structure |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

---

## Citation

```bibtex
@software{chousal2026celiacriskml,
  author = {Chousal, Andres and Chowell Lab},
  title = {CeliacRisks: ML Pipeline for Disease Risk Prediction},
  year = {2026},
  url = {https://github.com/achousal/CeliacRisks}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)
