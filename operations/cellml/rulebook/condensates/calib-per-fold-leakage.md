---
type: condensate
depends_on: ["[[equations/brier-decomp]]", "[[equations/platt-scaling]]", "[[equations/beta-calib]]", "[[equations/isotonic-calib]]"]
applies_to:
  - "pipelines with hyperparameter tuning inside nested CV"
  - "small-to-medium sample sizes (n cases < 500)"
  - "any calibrator fit inside the same fold used for model selection"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v4-calibration
    delta: "per_fold Brier reliability is optimistically biased relative to oof_posthoc because the calibrator indirectly sees hyperparameter-selected predictions"
    date: "2026-01-22"
    source: "ADR-008, operations/cellml/DESIGN.md V4"
falsifier: |
  Direction claim: \Delta(REL_perfold - REL_oofposthoc) < 0 with 95% bootstrap
  CI excluding 0 (per-fold is OPTIMISTIC, so REL appears lower than it really
  is). If per_fold and oof_posthoc produce Equivalence (|Delta REL| < 0.01
  AND 95% CI \subset [-0.02, 0.02]) across at least 3 datasets with n_cases
  >= 500 and model-selection via nested CV, this condensate is weakened. At
  least 5 such datasets -> retire.
---

# Per-fold calibration introduces optimism bias because the calibrator fits on hyperparameter-selected predictions

## Claim

When a calibrator (Platt, beta, or isotonic — see [[equations/platt-scaling]], [[equations/beta-calib]], [[equations/isotonic-calib]]) is fit **inside the same CV fold** used to select hyperparameters (`per_fold` strategy, the scikit-learn default via `CalibratedClassifierCV`), the calibrator indirectly sees validation data that already informed the hyperparameter choice. The reliability component of the Brier score ([[equations/brier-decomp]]) is then optimistically biased: REL on the VAL split looks lower than it will be on TEST.

The `oof_posthoc` strategy — fit a **single** calibrator on aggregated out-of-fold predictions AFTER nested CV has selected hyperparameters — eliminates this leakage because the OOF predictions themselves are never used for hyperparameter selection.

## Mechanism

Per-fold calibration workflow: within each outer fold, fit base model, pick hyperparameters on the inner validation slice, then fit a calibrator on that same validation slice. The calibrator fits on predictions that have already been selected to look good on that slice (the hyperparameters were chosen to maximize a score on the slice), so the calibrator learns a mapping that is tuned to the slice's idiosyncrasies.

OOF posthoc workflow: nested CV completes. The OOF predictions $\hat{p}_i^\text{OOF}$ are predictions on rows that the base model never saw during training AND were not used to select hyperparameters for that row (hyperparameters are selected on the INNER validation slice, which is disjoint from the outer held-out fold that produces the OOF prediction). A single calibrator is then fit on $\{(\hat{p}_i^\text{OOF}, y_i)\}_{i=1}^{n}$. The calibration-fit data is genuinely held out from both base-model training AND hyperparameter selection.

The optimism manifests as under-estimated REL: the calibrator fits the noise pattern of the inner validation slice, which reduces REL in-sample but does not generalize. Van Calster et al. (2019) document this specifically for small-$n$ clinical prediction models, where the inner validation slice has high variance and the calibrator's noise-fitting is proportionally worse.

## Actionable rule

- The V4 (calibration) gate MUST use `oof_posthoc` as the reference strategy. `per_fold` is allowed only as a comparator, and the gate logs the delta.
- `CalibrationConfig.strategy` (see `analysis/src/ced_ml/config/schema.py:301-337`) enumerates `{none, per_fold, oof_posthoc}`. `oof_posthoc` is the factorial default.
- Reporting: any public-facing Brier score or reliability diagram MUST cite which calibration strategy was used and whether the calibrator was fit on held-out data.
- Alternative path: an explicit 4-way split (TRAIN / VAL / CALIB / TEST) also removes the leakage but costs sample size. Rejected in ADR-008 for this cohort because $n_\text{case} = 148$ cannot afford a separate CALIB slice.

## Boundary conditions

- Applies when nested CV tunes hyperparameters AND a calibrator is fit inside the inner validation slice. Does NOT apply when there is no hyperparameter tuning (the calibrator is then fit on data that was only used for calibration, not selection).
- Applies most strongly when $n_\text{case}$ is small (the inner validation slice has high variance, so calibrator noise-fitting is severe). On very large cohorts ($n_\text{case} > 10{,}000$) the bias may fall below the 0.01 Equivalence threshold of the fixed falsifier rubric, in which case `per_fold` and `oof_posthoc` are effectively exchangeable.
- Does NOT apply to `none` (no calibration) — no calibrator exists, so no leakage path exists. `none` may still be the right choice when the base model is already well-calibrated (common for L2-penalized logistic regression).

## Evidence

| Dataset | n_cases | Phenomenon | Source gate |
|---|---|---|---|
| Celiac (UKBB) | 148 | ADR-008 rejects `per_fold` as default on leakage grounds; observational comparison was deferred to the factorial V4 gate | V4 calibration (post-V3 lock) |

## Related

- [[equations/brier-decomp]] — REL is the affected component
- [[equations/platt-scaling]], [[equations/beta-calib]], [[equations/isotonic-calib]] — three calibrator families affected
- [[condensates/calib-parsimony-order]] — complements this condensate; addresses which calibrator family to prefer under Equivalence
- ADR-008 (cel-risk) — decision record this condensate formalizes
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v4-calibration]]
