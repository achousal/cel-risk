---
type: condensate
depends_on: ["[[equations/oof-stacking]]"]
applies_to:
  - "stacking ensembles of 2+ base models"
  - "any meta-learner trained on stacked base-model predictions"
  - "settings where base models vary in overfitting capacity (e.g., tree ensembles + linear models)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v6-ensemble-comparison
    delta: "meta-learner fed in-fold predictions assigns weight concentration to the base model with the highest in-sample AUROC, which need not be the best out-of-sample model"
    date: "2026-01-22"
    source: "ADR-007, operations/cellml/DESIGN.md V6"
falsifier: |
  Direction claim: \Delta(AUROC_OOF_stack - AUROC_infold_stack) > 0 with 95%
  bootstrap CI excluding 0 on the TEST split. If OOF stacking and in-fold
  stacking produce Equivalence (|Delta| < 0.01 AND 95% CI \subset [-0.02, 0.02])
  across at least 3 datasets with n >= 500 and M >= 3 base models, this
  condensate is weakened. At least 5 such datasets -> retire.
---

# Stacking meta-learners trained on in-fold predictions concentrate weight on the hardest-overfitting base model

## Claim

A stacking ensemble's meta-learner must be trained on **out-of-fold (OOF)** base-model predictions, never on in-fold (in-sample) predictions. Feeding in-fold predictions to the meta-learner makes the stack training matrix $Z$ artificially inflated for whichever base model overfits hardest (e.g., XGBoost on a small-$n$ cohort), because that model's in-sample predictions are near-perfectly correlated with $y$. The meta-learner then assigns near-unit weight to the overfit model, which does NOT generalize.

## Mechanism

See [[equations/oof-stacking]]. The meta-learner assumes that the distribution of its input $Z$ is the same at train and deploy time. In-fold predictions violate this: $Z_\text{train}^\text{in-fold}$ is nearly noiseless relative to $y$ (because the base model fit those rows), whereas $Z_\text{deploy}$ will be noisy (the base model has not seen deploy rows). The meta-learner's variance-minimizing solution on the in-fold stack picks the base model that most closely reproduces $y$ in-sample — which is the most flexible (and typically most overfitting) model, not necessarily the best-generalizing one.

OOF predictions fix this: each $Z_{i,m}^\text{OOF}$ comes from a model that did NOT see row $i$, so the distribution of $Z_\text{train}^\text{OOF}$ matches $Z_\text{deploy}$ in the sense required by standard statistical learning guarantees (Wolpert 1992; Van der Laan et al. 2007).

## Actionable rule

- The V6 (ensemble comparison) gate MUST train the meta-learner on OOF predictions only. In-fold predictions are a search-space violation, not a tunable axis.
- `StackingEnsemble.fit_from_oof()` (see `analysis/src/ced_ml/models/stacking.py`) is the canonical API. Alternative code paths that feed in-fold predictions are disallowed.
- All base models in a stack MUST share identical fold assignments (same seed, same splitter) so $Z$ rows align. The factorial's 30 shared splits (seeds 100–129) satisfy this by construction.
- Meta-learner is Logistic Regression with L2 penalty. The $\lambda$ hyperparameter is tuned via an additional inner CV loop on $Z$.

## Boundary conditions

- Applies whenever base models differ in overfitting capacity. For stacks of models with identical capacity (e.g., two linear models with the same regularization), in-fold vs OOF stacking can produce similar meta-weights.
- Does NOT apply when there is only one base model — the "stack" reduces to a single calibrator on that model's predictions, which is the standard calibration workflow (see [[condensates/calib-per-fold-leakage]]).
- Requires fold assignments to be identical across base models. If base models are trained in separate pipelines with different random seeds, the OOF stack rows become misaligned and the meta-learner trains on inconsistent data.

## Evidence

| Dataset | n | M | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 148 cases / 43,662 controls | 4 (LR_EN, LinSVM_cal, RF, XGBoost) | OOF stacking is the adopted approach; in-fold stacking was rejected on leakage grounds in ADR-007 without a head-to-head empirical test, so the evidence supporting this condensate is mechanistic rather than observational on this cohort | V6 ensemble comparison (post-factorial informational step) |

## Related

- [[equations/oof-stacking]] — the stacking math, including the OOF requirement
- ADR-007 (cel-risk) — decision record this condensate formalizes
- ADR-005 — provides the OOF predictions consumed by the stack
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v6-ensemble-comparison]]
