---
type: equation
symbol: "p_beta"
depends_on: ["[[equations/platt-scaling]]"]
computational_cost: "O(n) per iteration of logistic regression on (log s, log(1-s)); typically O(n)"
assumptions:
  - "base-model scores $s_i \\in (0, 1)$ are strictly inside the open unit interval (apply $\\epsilon$-clipping if not)"
  - "the reliability curve is monotonic but may have asymmetric tail behavior that Platt's symmetric sigmoid cannot capture"
  - "calibration data is held out from base-model training"
failure_modes:
  - "scores at exactly 0 or 1 cause $\\log s$ or $\\log(1-s)$ to diverge; require clipping"
  - "three free parameters overfit on small calibration sets ($n_\\text{cal} < 50$)"
  - "when true miscalibration is monotonic and symmetric, beta is overkill and logistic_intercept or Platt suffice"
---

# Beta calibration generalizes Platt by fitting asymmetric sigmoid tails

## Statement

Given a base-model probability score $s_i \in (0, 1)$, Kull et al. (2017) fit a three-parameter calibration

$$\hat{p}_i = \frac{1}{1 + \exp\bigl(-a \log s_i + b \log(1 - s_i) - c\bigr)}$$

with parameters $(a, b, c)$ estimated by maximum likelihood. This is equivalent to logistic regression with two derived features $\log s_i$ and $\log(1 - s_i)$ plus an intercept.

**Parameter count**: 3 (vs 2 for Platt, 1 for logistic_intercept).

**Why beta**: the mapping arises as the posterior under the assumption that $s \mid y = 0$ and $s \mid y = 1$ follow Beta distributions (rather than Platt's equal-variance Gaussians). Beta distributions admit asymmetric tail shapes, so the resulting calibration curve can bend differently near 0 and near 1.

## Derivation

If $s \mid y = 1 \sim \mathrm{Beta}(\alpha_1, \beta_1)$ and $s \mid y = 0 \sim \mathrm{Beta}(\alpha_0, \beta_0)$, Bayes' rule gives

$$\frac{p(y=1 \mid s)}{p(y=0 \mid s)} = \frac{p(y=1)}{p(y=0)} \cdot \frac{s^{\alpha_1 - 1} (1-s)^{\beta_1 - 1}}{s^{\alpha_0 - 1} (1-s)^{\beta_0 - 1}}.$$

Taking logs and grouping terms yields a linear function of $\log s$ and $\log(1 - s)$ with an intercept absorbing the prior ratio. Re-parameterizing as $(a, b, c)$ and exponentiating recovers the logistic form above. The MLE problem is standard logistic regression with two derived features.

When $a + b = 0$ the beta calibration reduces to Platt scaling. When $a = b = 0$ it reduces to logistic_intercept. This strict nesting is what justifies the parsimony ordering logistic_intercept $\prec$ beta $\prec$ isotonic (see [[condensates/calib-parsimony-order]]) — each successive calibrator is a superset of the previous one, so added complexity can only help if the simpler form is genuinely under-parameterized.

## Boundary conditions

- Requires $s_i \in (0, 1)$ open. Implementations clip $s_i$ to $[\epsilon, 1 - \epsilon]$ with $\epsilon \approx 10^{-3}$ to avoid divergence of the log terms.
- Designed for monotonic reliability curves with potentially asymmetric tail bending (common for tree ensembles whose probability estimates are squashed toward 0/1 differently).
- Does NOT handle non-monotonic miscalibration. For that, use isotonic regression ([[equations/isotonic-calib]]).
- Three parameters. On calibration sets with $n_\text{cal} \lesssim 50$ positive cases, prefer Platt (2 params) or logistic_intercept (1 param) to reduce variance of the calibrator itself.
- Kull et al. (2017) reported beta outperforming Platt on a majority of the OpenML-CC18 benchmark when reliability curves were asymmetric, and tying otherwise.

## Worked reference

Base model: RF on celiac cohort. OOF probability scores concentrate near 0 (low prevalence) with a long right tail.

Platt fit gives $(\hat{A}, \hat{B}) = (-3.2, -6.1)$ with REL (from [[equations/brier-decomp]]) of $8.7 \times 10^{-6}$ on the VAL split.

Beta fit gives $(\hat{a}, \hat{b}, \hat{c}) = (1.8, 0.9, -5.4)$ — $\hat{a} \neq \hat{b}$ signals asymmetric tails (low end bent more than high end). REL drops to $4.2 \times 10^{-6}$. $\Delta\mathrm{REL} = -4.5 \times 10^{-6}$, which is well inside the Equivalence band of the fixed falsifier rubric ($|\Delta| < 0.01$).

**Parsimony verdict**: Equivalence with logistic_intercept -> prefer logistic_intercept per [[condensates/calib-parsimony-order]]. Beta's extra degrees of freedom do not earn their keep on this cohort.

## Sources

- Kull, Silva Filho, Flach (2017). "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." *AISTATS*.
- ADR-008 (cel-risk), 2026-01-22.

## Used by

- [[condensates/calib-parsimony-order]]
- [[condensates/calib-per-fold-leakage]]
- [[equations/platt-scaling]] — beta nests Platt (Platt $\equiv$ beta with $a = -b$)
- [[equations/isotonic-calib]] — isotonic nests beta (monotone regression dominates any parametric monotone form in-sample)
