---
type: equation
symbol: "F_total"
depends_on: []
computational_cost: "O(K_out * R * K_in * T)"
assumptions:
  - "outer and inner splits are stratified on the binary label"
  - "each outer train partition is large enough that a K_in-way split leaves at least 2 positives per inner validation fold"
  - "hyperparameter tuning is confined to the inner loop per outer-fold; the same hyperparameters never feed back into outer-fold construction"
failure_modes:
  - "tuning on outer folds or leaking tuned hyperparameters across outer folds reintroduces optimism bias"
  - "reducing inner folds below 5 without also widening trial budget destabilizes tuning at rare-event prevalence"
  - "at very low prevalence, fixed K_in may produce inner folds with zero positives — dynamic K_in reduction is required (see ADR-005 safeguard)"
---

# Nested CV fold decomposition and total fit count

## Statement

Given:

- $K_\text{out}$ outer folds
- $R$ outer-fold repeats (re-shuffled stratified partitions)
- $K_\text{in}$ inner folds for hyperparameter tuning
- $T$ inner hyperparameter trials per outer fold

Total fits:

$$F_\text{total} = K_\text{out} \cdot R \cdot K_\text{in} \cdot T$$

Out-of-fold prediction count per sample is exactly $R$ (each sample is held out once per repeat).

The unbiased performance estimator is:

$$\widehat{\text{AUROC}}_{\text{OOF}} = \text{AUROC}\left(\{y_i, \hat{p}_i\}_{i=1}^{n}\right)$$

where $\hat{p}_i$ is the mean predicted probability for sample $i$ across the $R$ repeats in which $i$ fell in the outer held-out partition, and each outer model was trained using hyperparameters selected by the inner CV on that outer's training partition only.

For celiac pipeline defaults ($K_\text{out}=5$, $R=10$, $K_\text{in}=5$, $T=200$): $F_\text{total} = 50{,}000$ fits per model.

## Derivation

Varma & Simon (2006) show that single-loop CV used both to tune and to evaluate produces an optimistic estimate of generalization performance, with bias that scales with tuning freedom. The bias arises because the held-out fold influences hyperparameter choice, then those hyperparameters are evaluated on the same held-out data.

Nested CV removes this bias by isolating tuning to an inner loop: each outer held-out fold's predictions come from a model whose hyperparameters were chosen using only that outer fold's training partition. The outer held-out fold is never seen by any component during tuning, so OOF predictions are an unbiased estimate of deployment performance on data drawn from the same distribution.

The $K_\text{out} \cdot R$ decomposition (as opposed to a single large $K_\text{out}$) is a variance-reduction choice: 10 repeats of 5-fold gives 50 total outer folds with preserved positive balance per fold, whereas a single 50-fold split at rare-event prevalence risks inner folds with too few positives.

## Boundary conditions

- **Isolation required.** Hyperparameter choice in outer fold $i$ must depend only on the training rows of outer fold $i$. Sharing tuned hyperparameters across outer folds reintroduces optimism.
- **Inner fold positive floor.** At prevalence $\pi$ with outer train size $n_\text{train}$, each inner validation fold holds approximately $\pi \cdot n_\text{train} / K_\text{in}$ positives. Require $\ge 2$ per fold. ADR-005 safeguards reduce $K_\text{in}$ dynamically when this condition is violated (e.g. in per-fold calibration CV).
- **Tuned search budget.** Stable tuning at prevalence $< 0.01$ typically requires $T \ge 100$ with random or Bayesian search; $T < 50$ observed to produce noisy winners.
- **No peeking on $R$ repeats.** Each repeat is an independent stratified partition. Repeats are averaged on the probability scale (not the AUROC scale) to preserve the exchangeability needed for downstream permutation testing. See [[equations/perm-test-pvalue]].

## Worked reference

Celiac pipeline parameters: $K_\text{out}=5$, $R=10$, $K_\text{in}=5$, $T=200$.

$$F_\text{total} = 5 \cdot 10 \cdot 5 \cdot 200 = 50{,}000 \text{ fits per model}$$

At $n = 43{,}810$ samples and $\pi = 0.00337$:

- Outer train partition: $\approx 35{,}048$ samples, $\approx 118$ positives
- Inner validation fold (of 5): $\approx 7{,}010$ samples, $\approx 24$ positives
- $\gg 2$ positives per inner fold — safeguard not triggered for tuning; separate reduction applies to per-fold calibration.

Each of the $n = 43{,}810$ samples contributes $R = 10$ OOF probability predictions (one per repeat). These are averaged, then the single pooled $\{y_i, \hat{p}_i\}$ vector is used for AUROC.

## Sources

- Varma & Simon (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7(1):91.
- ADR-005 (nested CV $5 \times 10 \times 5$).
- `analysis/src/ced_ml/models/training.py::oof_predictions_with_nested_cv` — reference implementation.
- `analysis/src/ced_ml/models/nested_cv.py` — StratifiedKFold wiring.

## Used by

- [[condensates/nested-cv-prevents-tuning-optimism]]
- [[equations/perm-test-pvalue]] — the inner pipeline that re-runs under each permutation has this exact structure
- [[condensates/perm-validity-full-pipeline]] — requires the full nested-CV inner pipeline per permutation
<!-- TODO: verify slug exists after batch merge — protocols/v0-strategy.md and v1-recipe.md should both cite this equation -->
