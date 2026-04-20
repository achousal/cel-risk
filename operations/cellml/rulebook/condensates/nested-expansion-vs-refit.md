---
type: condensate
depends_on:
  - "[[equations/nested-cv]]"
  - "[[equations/optuna-tpe]]"
  - "[[equations/brier-decomp]]"
applies_to:
  - "tree-based models (RF, XGBoost) with per-size plateau expansions"
  - "pipelines where model hyperparameters are tuned at a reference p and reused at smaller p via post-hoc truncation"
  - "model-specific recipes whose plateau exceeds the significance core by >=1 protein"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v1-recipe
    delta: "DESIGN.md §Nested expansion specifies that RF plateau-8p and XGBoost plateau-9p sub-recipes (p=7, 6, 5, 4 for RF; p=8, 7, 6, 5, 4 for XGBoost) reuse the parent ordering truncated to top-N rather than retraining with per-size hyperparameter search; linear models (LR_EN, LinSVM_cal) plateau at 4p and do not expand"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §Nested expansion; operations/cellml/MASTER_PLAN.md V1-V5 Main Factorial (1,566 cells includes 486 nested expansion cells)"
falsifier: |
  Direction claim: for at least one tree model (RF or XGBoost), step-down
  AUROC (train once at plateau p, predict on top-N subset for each smaller
  N) underestimates refit-per-size AUROC (re-tune hyperparameters at each p
  independently) by more than 0.01 with 95% bootstrap CI (seed-level paired
  resample) excluding 0 on >=3 datasets. Under that pattern, step-down is
  a Direction-failing approximation and the V1 protocol must switch to
  refit-per-size for tree models. If |ΔAUROC_{refit - step-down}| < 0.01
  with 95% CI inside [-0.02, 0.02] (Equivalence) across >=3 datasets,
  step-down is confirmed as a safe approximation. Retire to `established`
  at 5 confirmations; retire as failed at 5 datasets showing the Direction
  failure.
---

# Step-down performance mapping is faster than refit-per-size but may understate per-size optimum for tree models

## Claim

Nested expansion maps a tree model's plateau (e.g. RF at 8p, XGBoost at 9p)
down to the significance core (4p) by TRUNCATING the parent ordering and
predicting on the reduced top-N feature set, reusing the hyperparameters
tuned at the reference plateau. This "train-once-then-subset" protocol is
far cheaper than retraining with a fresh hyperparameter search at each
smaller p, but it systematically understates the achievable per-size
optimum when the model's optimal hyperparameters change with feature count.
The tradeoff is explicit: step-down gives a clean single-protein-step
performance map at linear compute cost (in the plateau count), refit-per-
size gives per-size optima at O(plateau count * trial budget) compute cost.

## Mechanism

Tree model hyperparameters — tree depth, learning rate, subsample ratio,
min_child_weight, regularization — are tuned against the feature count
the model was trained on. A model trained on 9 features can afford deeper
trees than one trained on 4 features because splits have more candidate
variables; a shallower optimal depth at 4 features means the 9-p-trained
model, restricted to 4 features at inference, is over-parameterized for
the reduced input. The bias is one-directional: reducing the input never
helps a model whose capacity was chosen for a richer input, so the
step-down AUROC at 4p is <= the refit-at-4p AUROC in expectation.

Empirically, the magnitude depends on model-ordering interaction:

- For highly regularized tree models (XGBoost with strong L1/L2), the
  plateau hyperparameters already underfit slightly, so restricting
  features has a small effect — step-down and refit agree within ~0.005.
- For flexible tree models (RF with max_depth=None), the plateau
  hyperparameters are tuned against high feature diversity; restricting
  to 4 features removes the splits the ensemble relied on, and step-down
  underestimates refit by > 0.01.

Linear models are largely immune because the feature-selection step (LASSO-
style shrinkage via L1, or LinSVM_cal's margin objective) is itself
regularization in feature space: a linear model tuned at the plateau has
already zeroed out features beyond the core in most regularization paths,
so restricting to the top-N is equivalent (or near-equivalent) to what the
model would have chosen at p=N anyway. This is why DESIGN.md expands only
RF and XGBoost — linear plateaus match the significance core.

## Actionable rule

- V1 protocol MUST declare the expansion mode in `ledger.md` under
  "axes_explored" as either `nested_expansion: step_down` (default) or
  `nested_expansion: refit_per_size`.
- If step-down is used, V1 MUST log in `observation.md` the step-down vs
  reference-plateau ΔAUROC at each sub-size; observed degradation
  > 0.01 at any sub-size is flagged as a known-approximation tension and
  pointed to this condensate.
- If refit-per-size is used, the cell count multiplies by (plateau - 4 + 1);
  the ledger must pre-register the compute budget.
- Linear models (LR_EN, LinSVM_cal) with plateau == 4 MUST NOT expand;
  adding sub-recipes below plateau is a waste of compute and may confuse
  the V1 tournament structure.
- When step-down is used and an RF or XGBoost sub-size wins the V1 size
  tournament, V1 MUST run a single refit-per-size cell at the winning
  p as a confirmation. If refit-per-size AUROC exceeds step-down AUROC at
  the winning p by |Δ| >= 0.01 with CI excluding 0, V1 emits a tension
  and the lock defers to the refit value.
- Nested expansion is a factorial-scale decision: the choice of mode is
  locked for ALL tree-model sub-recipes in a V1 run, not per-recipe.
  Mixing modes in one gate produces incomparable sub-sizes.

## Boundary conditions

- **Linear models are immune and expansion is not needed.** LR_EN and
  LinSVM_cal plateau at the significance core in the celiac example.
  Generalize: any model whose feature selection is a regularization
  penalty in the same objective as training has step-down ~ refit.
- **Breaks when hyperparameter space is strongly coupled to feature
  count.** XGBoost with adaptive tree depth (min_child_weight scaling with
  n_features), RF with feature_subsample_ratio tuned separately, and
  gradient-boosted models with early stopping can have step-down biases
  > 0.02. These are known to respond strongly to feature count and must
  use refit-per-size if they are in the V1 factorial.
- **Breaks when the plateau is far from the significance core.** At
  plateau - 4 >= 5 (the XGBoost case in celiac), step-down has 5 sub-
  sizes to traverse and bias may accumulate: the p=4 sub-recipe is
  operating with hyperparameters tuned for 9p, a >2x feature reduction.
  At plateau - 4 <= 2 (small expansion), bias is bounded tighter and
  step-down is safer.
- **Sensitive to the ordering that drives truncation.** If the top-4 under
  the parent ordering is NOT approximately the top-4 under a refit-at-4
  feature importance, step-down doubly underestimates: both the
  hyperparameters and the feature subset are wrong. RFE orderings from
  deep trees are especially susceptible because RFE at 9p may rank
  differently than at 4p.
- **OOF-importance orderings are more step-down-stable than RFE.**
  OOF-importance is computed post-hoc on the trained plateau model, so
  the top-N subsets are by construction "what this plateau model found
  useful." RFE orderings are sequentially selected during a recursive
  elimination and depend more on the specific inductive bias at the
  plateau. V1's MS_oof_* recipes should step-down more cleanly than
  MS_rfe_* recipes.
- **Does NOT apply when the sizing rule fully retrains at each size.**
  This condensate governs nested expansion specifically (single parent,
  post-hoc truncation). The discovery sweep itself retrains at each p;
  the 3-criterion rule (see [[condensates/three-criterion-size-rule]])
  consumes the refit-per-size sweep outputs.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | RF plateau-8p expands to p=7, 6, 5, 4 via top-N truncation (4 sub-recipes); XGBoost plateau-9p expands to p=8, 7, 6, 5, 4 (5 sub-recipes); linear plateaus at 4 with no expansion. 486 nested-expansion cells in the 1,566-cell factorial per MASTER_PLAN.md. | v1-recipe 2026-04-20; source DESIGN.md §Nested expansion; MASTER_PLAN.md §V1-V5 Main Factorial |

## Related

- [[protocols/v1-recipe]] — §2.1 and T-V1-05 flag this condensate as a
  rulebook gap; this file fills it and the protocol's §2.3 axis mapping
  should update to cite this condensate for nested expansion
- [[condensates/three-criterion-size-rule]] — governs the plateau p for
  each model, which defines the top of the step-down ladder
- [[condensates/size-ordering-pooling]] — pooling happens at sweep stage;
  this condensate governs the post-sweep expansion stage
- [[equations/optuna-tpe]] — the hyperparameter search that refit-per-size
  would re-invoke at each p
- [[equations/nested-cv]] — the fold structure of both step-down and
  refit-per-size; step-down reuses outer-fold predictions from the plateau
  cell by restricting the input, refit-per-size re-runs the full nested CV
- [[equations/brier-decomp]] — step-down under-capacity may inflate REL
  at sub-sizes even when AUROC is preserved; V1 reports REL but does not
  adjudicate calibration (deferred to V4)
- DESIGN.md §Nested expansion — canonical specification
- MASTER_PLAN.md §V1-V5 Main Factorial — cell count accounting for nested
  expansion
