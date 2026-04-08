# SVM Validation

## Goal

Select the best training recipe for calibrated LinearSVC and produce a
winner-conditional ranked protein list for downstream panel-size sweep.

## Design: Two phases in one script

### Phase 1A — Recipe selection (12 combos)

Replicate the incident validation with SVM instead of elastic-net logistic regression.

- **3 strategies:** incident_only, incident_prevalent, prevalent_only
- **4 weight schemes:** none, balanced, sqrt, log
- **Feature selection:** Wald-based bootstrap stability (shared, weight-agnostic)
  - Same panel for all 12 combos so strategy comparison is fair
- **Hyperparameter tuning:** Optuna tunes C only (log scale 1e-4 to 100, 50 trials)
- **CV:** 5-fold outer, 3-fold inner, AUPRC primary
- **Output:** winning (strategy, weight_scheme, median_C)

### Phase 1B — Winner-conditional feature ranking

Conditional on the Phase 1A winner:

1. Positive class matches the winning strategy:
   - incident_only: incident vs control
   - incident_prevalent: incident+prevalent vs control
   - prevalent_only: prevalent vs control
2. Class weighting matches the winning weight scheme
3. Model-based bootstrap stability:
   - Fit weighted LinearSVC (with winning C) per bootstrap resample
   - Keep top 200 features by |coef| per resample
   - Retain features selected in >= 70% of resamples
4. Correlation prune survivors (|r| > 0.85)
5. Rank by:
   - Primary: selection frequency (descending)
   - Secondary: mean standardized |coef| (descending)
   - Tertiary: RRA rank (ascending, tiebreak only)
6. **Output:** `winner_order.csv` — one ordered list

### Final evaluation

Refit winning config on full dev set, evaluate on locked 20% test set with
2000-resample bootstrap CIs.

## Why two feature selections?

Phase 1A uses Wald (univariate, weight-agnostic) so all 12 combos share the same
features — the strategy comparison is fair. Phase 1B uses model-based selection
(LinearSVC with the winning weights) so the feature ranking reflects the actual
decision surface. The downstream sweep uses only the Phase 1B ranking.

## Model

```python
LinearSVC(C=C, class_weight=cw, max_iter=5000, dual='auto')
# wrapped in:
CalibratedClassifierCV(method='sigmoid', cv=5)
```

## Downstream handoff

Step 2 (svm-sweep) takes from this experiment:
- **Fixed:** strategy, weight_scheme (from Phase 1A winner)
- **Fixed:** feature ordering (from Phase 1B winner_order.csv)
- **Sweep:** panel size p = prefixes of winner_order.csv
- **Tune:** C only at each p

Sensitivity analysis: also sweep RRA, importance, pathway orders under the winning
recipe to check robustness to ordering priors.

## Execution

```bash
# Smoke test (local):
cd cel-risk
source analysis/.venv/bin/activate
python experiments/optimal-setup/svm-validation/scripts/run_svm_validation.py --smoke

# HPC:
cd cel-risk
bsub < experiments/optimal-setup/svm-validation/scripts/submit_svm_validation.sh
```

## Outputs

```
results/svm_validation/
├── config.json
├── cv_results.csv                  # Per-fold, all 12 combos
├── strategy_comparison.csv         # Summary ranked by AUPRC
├── feature_coefficients.csv        # Final model (shared panel)
├── test_predictions.csv
├── shared_panel.csv                # Phase 1A Wald panel
├── winner_order.csv                # Phase 1B ranked list (top-level copy)
├── paired_comparison.csv           # SVM vs LR delta
├── summary_report.md
├── manifest/
│   ├── winner_order.csv            # Full ranking with metadata
│   ├── manifest_meta.json          # Strategy, weight, C, thresholds
│   ├── model_bootstrap_log.csv
│   └── model_corr_prune_map.csv
└── combos/
    └── coefs_*.csv                 # Per-fold coefficients
```
