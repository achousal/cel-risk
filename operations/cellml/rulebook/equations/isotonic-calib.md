---
type: equation
symbol: "p_iso"
depends_on: ["[[equations/platt-scaling]]", "[[equations/beta-calib]]"]
computational_cost: "O(n log n) for pool-adjacent-violators algorithm"
assumptions:
  - "reliability curve $P(y=1 \\mid s)$ is monotonically non-decreasing in $s$"
  - "calibration set is large enough that bin-level means are stable ($n_\\text{cal} \\gtrsim 100$ positives)"
  - "calibration data is held out from base-model training"
failure_modes:
  - "overfitting on small calibration sets — isotonic is effectively non-parametric, so it fits the empirical step function"
  - "predictions outside the calibration support are extrapolated as flat steps (cliff-edge behavior at tails)"
  - "if true reliability is non-monotonic, isotonic will collapse the violating region and can perform worse than a misspecified parametric calibrator"
---

# Isotonic regression is the non-parametric monotone calibrator

## Statement

Given base-model scores $s_i \in \mathbb{R}$ and binary outcomes $y_i \in \{0, 1\}$ on a calibration set, isotonic regression finds the step function $\hat{g}$ that minimizes squared error subject to monotonicity:

$$\hat{g} = \arg\min_{g} \sum_{i=1}^{n_\text{cal}} (y_i - g(s_i))^2 \quad \text{subject to} \quad s_i \le s_j \Rightarrow g(s_i) \le g(s_j).$$

The calibrated probability is $\hat{p}_i = \hat{g}(s_i)$. The solution is computed by the **pool-adjacent-violators algorithm** (PAVA): sort by $s$, scan left-to-right, and whenever an adjacent pair violates monotonicity, pool them into a single block whose value is the weighted mean. The result is a non-decreasing step function with up to $n_\text{cal}$ unique steps.

**Parameter count**: up to $n_\text{cal}$ free step heights — effectively non-parametric.

## Derivation

PAVA (Ayer et al. 1955; Barlow et al. 1972) is the standard isotonic regression solver. Its correctness argument: any non-monotone pair in the current estimate can be replaced by their weighted mean without increasing squared error (because the variance reduces), while strictly improving monotonicity. Iterating until no violations remain converges to the global optimum.

Zadrozny & Elkan (2002) proposed isotonic regression as a calibrator for binary classifiers, noting its strength in matching arbitrary monotone reliability curves. The tradeoff relative to parametric calibrators ([[equations/platt-scaling]], [[equations/beta-calib]]) is that isotonic uses $O(n_\text{cal})$ effective parameters and so is most vulnerable to calibration-set size. See [[condensates/calib-parsimony-order]] for the implied ordering.

## Boundary conditions

- Applies when the reliability curve is monotonically non-decreasing. The monotonicity constraint is a strength when it holds (any parametric monotone calibrator is strictly nested inside isotonic) and a liability when it does not.
- Requires a sufficiently large calibration set. Niculescu-Mizil & Caruana (2005) recommend $n_\text{cal} \gtrsim 100$ positives for stable isotonic fits; below that, Platt or beta have lower variance.
- Extrapolation at the score-range boundaries is flat: $\hat{g}(s) = \hat{g}(s_\text{min})$ for $s < s_\text{min}$, and similarly at the upper end. This is benign when the test-time score range matches the calibration range, but problematic under covariate shift.
- Nesting: isotonic can reproduce any monotone parametric mapping in the large-$n$ limit, so REL (from [[equations/brier-decomp]]) under isotonic is weakly less than REL under Platt or beta **in-sample**. Out-of-sample, the ordering can invert when isotonic overfits.

## Worked reference

Base model: XGBoost. OOF probability scores on 148 positive cases, 10{,}805 controls.

Platt ([[equations/platt-scaling]]) gives REL = $1.1 \times 10^{-5}$. Beta ([[equations/beta-calib]]) gives REL = $0.9 \times 10^{-5}$. Isotonic gives REL = $0.3 \times 10^{-5}$ on the same VAL split — strictly best in-sample.

On the held-out confirmation seeds (seeds 120–129), REL inverts: Platt = $1.0 \times 10^{-5}$, beta = $1.1 \times 10^{-5}$, isotonic = $1.4 \times 10^{-5}$. Isotonic's in-sample advantage does not hold out of sample — a textbook overfit signature when $n_\text{cal}$ is small relative to the effective parameter count.

**Parsimony verdict** (via [[condensates/calib-parsimony-order]]): on this cohort, isotonic's extra flexibility is not warranted; logistic_intercept suffices.

## Sources

- Ayer, Brunk, Ewing, Reid, Silverman (1955). "An empirical distribution function for sampling with incomplete information." *Annals of Mathematical Statistics* 26: 641–647.
- Barlow, Bartholomew, Bremner, Brunk (1972). *Statistical Inference Under Order Restrictions*. Wiley.
- Zadrozny & Elkan (2002). "Transforming classifier scores into accurate multiclass probability estimates." *KDD*.
- Niculescu-Mizil & Caruana (2005). "Predicting good probabilities with supervised learning." *ICML*.
- ADR-008 (cel-risk), 2026-01-22.

## Used by

- [[condensates/calib-parsimony-order]]
- [[condensates/calib-per-fold-leakage]]
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v4-calibration]]
