---
type: equation
symbol: "r, \\pi_\\text{train}"
depends_on: ["[[equations/stratified-split-proportions]]"]
computational_cost: "O(n_\\text{control})"
assumptions:
  - "controls are exchangeable within-split (random downsampling is valid)"
  - "downsampling is applied per-split AFTER stratified allocation, not before"
  - "downstream probability output is adjusted back to the population prevalence before any calibration or decision threshold is applied"
failure_modes:
  - "probability outputs are interpreted directly as population-scale risk without prevalence adjustment"
  - "downsampling applied before splitting leaks held-out controls into TRAIN"
  - "variance inflation if r is too small relative to the class-conditional feature variability (insufficient negative signal)"
---

# Control downsampling at ratio r shifts training prevalence deterministically and reduces per-split control count by factor 1/r_factor

## Statement

Given $n_\text{case}$ cases and $n_\text{control}$ controls in a split, downsampling controls to a case:control ratio of $1:r$ retains:

$$n_\text{control}^\text{down} = \min(r \cdot n_\text{case}, \; n_\text{control})$$

The empirical training prevalence becomes:

$$\pi_\text{train} = \frac{n_\text{case}}{n_\text{case} + n_\text{control}^\text{down}} = \frac{1}{1 + r}$$

(exact when $r \cdot n_\text{case} < n_\text{control}$, which holds whenever the cohort is imbalanced).

Compute reduction factor (fits per training pass) compared to no downsampling:

$$\text{speedup} = \frac{n_\text{control}}{n_\text{control}^\text{down}} = \frac{n_\text{control}}{r \cdot n_\text{case}}$$

## Derivation

With $n_\text{case}$ positives and $n_\text{control}$ negatives in a split, random downsampling selects $r \cdot n_\text{case}$ controls uniformly without replacement. The retained dataset has exactly $n_\text{case}$ cases and $r \cdot n_\text{case}$ controls, so the rate of positives is

$$\pi_\text{train} = \frac{n_\text{case}}{n_\text{case} + r \cdot n_\text{case}} = \frac{1}{1 + r}$$

which depends only on $r$, not on the original prevalence. Any model trained on the downsampled data estimates $P(Y=1 | X, \pi_\text{train})$, not $P(Y=1 | X, \pi)$. To recover the population-scale risk, apply the Bayes prevalence correction:

$$P(Y=1 | X, \pi) = \frac{\pi \cdot \frac{P(Y=1 | X, \pi_\text{train})}{\pi_\text{train}}}{\pi \cdot \frac{P(Y=1 | X, \pi_\text{train})}{\pi_\text{train}} + (1 - \pi) \cdot \frac{1 - P(Y=1 | X, \pi_\text{train})}{1 - \pi_\text{train}}}$$

This is the odds-ratio shift derived from Bayes' rule under the label-shift assumption (features $X$ are class-conditionally stationary; only the class marginal changes). See Elkan (2001), *The foundations of cost-sensitive learning*.

## Boundary conditions

- Applies only when downsampling is applied **per-split after stratified allocation**. Downsampling before the split leaks controls across splits.
- Discrimination metrics (AUROC, ranking-based) are invariant to prevalence shift — they depend only on score ordering. Calibration metrics (Brier, ECE, reliability) are NOT invariant and require prevalence adjustment before evaluation.
- Variance of the estimated decision boundary increases as $r \to 1$ (too few negatives). The 5:1 ratio chosen in ADR-003 is a compute/signal trade; ratios below 2:1 risk under-sampling the negative class manifold.
- Does NOT apply when the label-shift assumption fails (i.e., when $P(X | Y)$ changes between training and evaluation). Label-shift corrections are unreliable under covariate shift.

## Worked reference

Celiac (UKBB) training split after 50/25/25 allocation: $n_\text{case} = 74$, $n_\text{control} = 21{,}831$, population prevalence $\pi = 0.00338$.

With $r = 5$:

$$n_\text{control}^\text{down} = 5 \cdot 74 = 370$$
$$\pi_\text{train} = \frac{74}{74 + 370} = \frac{1}{6} = 0.1667$$
$$\text{speedup} = \frac{21{,}831}{370} \approx 59$$

Confirming ADR-003's "$\sim 60\times$ computational savings". The training prevalence of 16.7% diverges from population prevalence 0.338% by a factor of $\sim 49$, so any calibrated probability output MUST be adjusted before being reported as risk.

## Sources

- Elkan (2001). *The foundations of cost-sensitive learning*. IJCAI.
- Saerens, Latinne, Decaestecker (2002). *Adjusting the outputs of a classifier to new a priori probabilities*. Neural Computation.
- ADR-003 (cel-risk), 2026-01-20.

## Used by

- [[condensates/downsample-preserves-discrimination-cuts-compute]]
- [[condensates/downsample-requires-prevalence-adjustment]]
<!-- TODO: verify slug exists after batch 2/3 merge -->
- [[protocols/v0-strategy]] — referenced when locking `train_control_per_case`
