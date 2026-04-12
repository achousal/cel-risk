# Incident Validation Experiment

Validates the CeD proteomic risk model on **incident (pre-diagnostic)** cases,
where serum was drawn before clinical diagnosis. This is the primary scientific
test of whether the model captures pre-symptomatic biology.

## Scientific question

Can the risk model trained on prevalent celiac cases generalize to incident cases,
and which training strategy (incident-only vs. incident+prevalent vs. prevalent-only)
and class-weight scheme (none / balanced / sqrt / log) performs best?

## Models tested

| Script | Models | Submit script |
|--------|--------|---------------|
| `scripts/run_lr.py` | LR_EN, SVM_L1, SVM_L2 | `scripts/submit_lr_parallel.sh` |
| `scripts/run_svm.py` | LinSVM_cal (L1, L2 penalty) | `scripts/submit_svm.sh`, `scripts/submit_svm_parallel.sh` |

## Directory layout

```
incident-validation/
├── README.md                       # this file
├── RESULTS_LR_EN.md                # LR_EN findings summary
├── scripts/
│   ├── run_lr.py                   # unified LR_EN + SVM_L1/L2 training (3 strategies × 4 weights)
│   ├── run_svm.py                  # LinSVM_cal (L1/L2 penalty, 3 strategies × 4 weights)
│   ├── submit_lr_parallel.sh       # HPC: one 14-job chain per model
│   ├── submit_svm.sh               # HPC: sequential SVM submission
│   └── submit_svm_parallel.sh      # HPC: parallel SVM job chains
└── analysis/
    ├── incident_validation_comparison.R    # figs 1-5 + auto-generated report
    ├── fig_forest_upset.R                  # figs 6b (forest) + 7b (UpSet)
    ├── compute_calibration_dca.py          # figs 6a (calibration) + 7a (DCA) + metrics
    ├── compute_shap_oof.py                 # SHAP values + OOF predictions
    ├── plot_shap.py                        # figs 9-11 (beeswarm, bar, dependence)
    ├── compute_saturation.py               # fig 8 (saturation curve)
    └── out/                                # generated figures, CSVs, report
```

## Results location

```
results/incident-validation/
├── lr/          # LR_EN + SVM_L1/L2 per-model subdirs
├── svm/         # LinSVM_cal per-penalty subdirs
├── compiled/    # cross-model summary tables
└── figures/     # publication figures
```

## Usage

```bash
# From cel-risk/ project root:

# Local smoke test
python operations/incident-validation/scripts/run_lr.py --model LR_EN --smoke

# HPC: submit all 3 models (14-job chains each)
bash operations/incident-validation/scripts/submit_lr_parallel.sh --model all

# HPC: SVM parallel
bash operations/incident-validation/scripts/submit_svm_parallel.sh
```

## Analysis scripts (artifact-first)

All analysis scripts read from `results/incident-validation/` artifacts produced
by the training scripts above. They do not recompute from raw data.

- `compute_calibration_dca.py` — reads model predictions → calibration + DCA curves
- `compute_saturation.py` — reads feature stability outputs → saturation curve
- `compute_shap_oof.py` + `plot_shap.py` — reads OOF predictions → SHAP figures
- `incident_validation_comparison.R` — reads compiled CSVs → figs 1-5 + report
