# panel_model_matrix

Protein-panel × classifier PR-AUC matrix for the cel-risk holdout_ds5 / IncidentOnly scenario.
Each row is a panel (top-20 proteins from one source model's OOF importance); each column is
a classifier trained on that panel. The intent is to separate panel-composition effects from
classifier-specific inductive biases, and to tie the empirical PR-AUC pattern back to the
closed-form properties of each model's loss.

## Design

- **Scenario**: IncidentOnly (148 incident CeD vs 43,662 controls pre-diagnosis)
- **Splits**: `splits_inc_holdout_ds5` — 1:5 downsampled, 70/30 holdout, 11 split seeds
- **Weighting**: `ds5` only (downsample is the imbalance remedy; `log` collapses calibrated
  probabilities and is excluded — see `docs/dev/HANDOFF.md`)
- **Panels** (`panels/`): top-20 proteins per model by OOF mean importance, from the completed
  `results/run_phase3_holdout_incident_only_ds5/<model>/aggregated/importance/oof_importance__<model>.csv`.
  Truncation to a common N=20 isolates composition from dimensionality.
- **Classifiers**: LinSVM_cal, LR_EN, RF, XGBoost (4 × 4 = 16 cells)
- **Hyperparameters**: each cell re-runs Optuna (`n_trials=50`) on the fixed panel — no panel
  carries tuned hyperparameters across classifiers.

## Panel identity overlap (Jaccard, top-20)

|            | LinSVM_cal | LR_EN | RF   | XGBoost |
|------------|-----------:|------:|-----:|--------:|
| LinSVM_cal |       1.00 |  0.67 | 0.48 |    0.38 |
| LR_EN      |       0.67 |  1.00 | 0.54 |    0.43 |
| RF         |       0.48 |  0.54 | 1.00 |    0.48 |
| XGBoost    |       0.38 |  0.43 | 0.48 |    1.00 |

LinSVM_cal and LR_EN share the most identity (linear-model family). RF and XGBoost drift
further apart despite both being tree ensembles — XGBoost's gain-based ranking penalizes
redundant high-correlation features more aggressively than RF's Gini importance.

## Model math — what each classifier *should* prefer

Each cell in the matrix carries a tag grounded in the loss-function term that governs its
response to panel composition.

| Model        | Loss / objective                                     | Feature-preference rule                                                        | Leverage equation term        | Tag                |
|--------------|------------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------|--------------------|
| LR_EN        | log-loss + α·(ρ‖β‖₁ + (1−ρ)‖β‖₂²)                    | Decorrelated features with stable logit contribution Σ βᵢ·xᵢ                   | ∂L/∂βᵢ = (p − y)·xᵢ − αβᵢ'    | logit-additive      |
| LinSVM_cal   | hinge: max(0, 1 − yᵢ·(w·xᵢ)) + C·‖w‖² → Platt        | Features that widen margin; saturates under heavy neg weighting → prior        | ∂L/∂w = −yᵢxᵢ·𝟙[margin<1]     | margin-limited      |
| RF           | CART Gini: ΔI = Σ (nₗ/n)·Iₗ                          | Tolerates correlated features; benefits from variance diversity                | variance-reduction per split  | variance-saturated  |
| XGBoost      | 2nd-order: Σ[gᵢfₜ + ½hᵢfₜ²] + γT + ½λ‖w‖²             | Sparse strong-signal features; γ/λ penalize redundant splits                   | gain = ½(Gₗ²/Hₗ + ...)        | gain-sparse         |

**Predictions before running the matrix:**
1. **LR_EN panel ≳ LinSVM_cal panel on both linear classifiers** — they share 67% identity and
   favor the same additive signal (logit-additive ≈ margin-limited at moderate C).
2. **XGBoost panel wins on XGBoost and likely RF** — gain-sparse ranking picks the strongest
   individual splits, which also reduce Gini impurity effectively.
3. **RF panel under-performs on XGBoost** — RF keeps correlated redundancies (Gini doesn't
   punish them) that XGBoost's λ/γ then waste splits on.
4. **LinSVM_cal panel under-performs on tree models** — proteins selected for margin support
   are not necessarily the highest-gain splitters.

These become falsifiable hypotheses: the finished matrix either matches the expected
off-diagonal pattern or the equation-tagged explanation fails and needs revision.

## Workflow

```
# 1. Panels already built. Rebuild if source importance files change:
python3 -c 'import csv, pathlib; ...'   # see commit history

# 2. Submit 16 cells on Minerva (LSF)
bash operations/panel_model_matrix/scripts/submit_minerva.sh
# DRY_RUN=1 bash ...  for a preview

# 3. Monitor:
ssh minerva 'bjobs -w | grep CeD_pmm_'

# 4. Aggregate into matrix once all cells complete:
python3 operations/panel_model_matrix/scripts/build_matrix.py

# 5. Plot heatmap (R):
Rscript operations/panel_model_matrix/scripts/plot_heatmap.R
```

Outputs land in `operations/panel_model_matrix/analysis/`:
- `panel_model_prauc.csv` (long form — prauc, auroc, brier_score, calibration_slope per cell)
- `panel_model_prauc_wide.csv` (4×4 matrix form)
- `panel_model_prauc_heatmap.png` (ggplot heatmap)

## Baseline (native-panel) diagonals — NOT matrix cells

For context, PRAUC from the pre-existing `run_phase3_holdout_incident_only_ds5` runs where each
model used its own full consensus panel (56–85 proteins, not truncated):

| Model       | n_splits | PRAUC (mean±sd) | AUROC |
|-------------|---------:|----------------:|------:|
| LinSVM_cal  |       11 |  0.572 ± 0.096  | 0.794 |
| LR_EN       |       11 |  0.626 ± 0.089  | 0.813 |
| RF          |       11 |  0.712 ± 0.068  | 0.856 |
| XGBoost     |       11 |  0.705 ± 0.076  | 0.862 |

These are **not** entered into the matrix (different dimensionality). Compare them to the
top-20 diagonals after the retrain to see how aggressive truncation affects each classifier.

## Files

```
operations/panel_model_matrix/
├── README.md                      ← this file
├── panels/                        ← 4 top-20 protein lists
│   ├── LinSVM_cal_top20.csv
│   ├── LR_EN_top20.csv
│   ├── RF_top20.csv
│   └── XGBoost_top20.csv
├── configs/                       ← 4 training + 4 pipeline YAMLs
│   ├── training_config_<model>_top20.yaml
│   └── pipeline_hpc_holdout_ds5_<model>_top20.yaml
├── scripts/
│   ├── submit_minerva.sh          ← LSF 16-cell submitter
│   ├── build_matrix.py            ← cell aggregator
│   └── plot_heatmap.R             ← ggplot heatmap
└── analysis/                      ← populated by build_matrix.py + plot_heatmap.R
```
