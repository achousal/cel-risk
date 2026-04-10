# Phase A: Model Gate

## Goal

Decide whether calibrated Linear SVM earns a dedicated optimization branch.

## Design

4 models x 3 strategies x 4 weights = 48 configurations, each evaluated via
5-fold CV with Optuna inner tuning. Shared recipe throughout.

### Models
- LR_EN (Elastic Net Logistic Regression)
- LinSVM_cal (LinearSVC + CalibratedClassifierCV)
- RF (Random Forest)
- XGBoost

### Strategies
- incident_only
- incident_prevalent
- prevalent_only

### Weight schemes
- none, balanced, sqrt, log

### Feature selection
Wald-based bootstrap stability (shared, model-agnostic, weight-agnostic).
Same panel for all 48 combos so comparison is fair.

### Tuning
Per model:
- LR_EN: C + l1_ratio
- LinSVM_cal: C only
- RF: n_estimators + max_depth + min_samples_leaf
- XGBoost: learning_rate + max_depth + n_estimators + subsample

All via Optuna, 50 trials, 3-fold inner CV, AUPRC objective.

## Provenance

```yaml
provenance:
  recipe_mode: shared
  recipe_source: experiments/optimal-setup/shared_recipe.yaml
  recipe_overrides: {}
  comparison_class: fair
```

## Decision rule

From shared_recipe.yaml:
- Best mean AUPRC wins.
- If 95% CIs overlap, prefer lower Brier.
- If still tied, prefer simpler model.
- Model earns dedicated branch if winner or CI overlaps winner.

## Outputs

```
results/model_gate/
├── config.json
├── cv_results.csv              # Per-fold, all 48 combos
├── model_comparison.csv        # Summary: model x strategy x weight
├── branch_decision.json        # {winner, branch_models, decision_rule}
├── pareto_analysis.csv         # AUROC vs Brier for all configs
├── shared_panel.csv            # Wald-based feature panel
├── summary_report.md
└── combos/                     # Per-fold coefficients
```

## Execution

```bash
# Smoke test:
cd cel-risk
python experiments/optimal-setup/model-gate/scripts/run_model_gate.py --smoke

# HPC:
bsub < experiments/optimal-setup/model-gate/scripts/submit_model_gate.sh
```
