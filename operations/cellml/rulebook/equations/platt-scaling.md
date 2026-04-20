---
type: equation
symbol: "p_platt"
depends_on: []
computational_cost: "O(n) per iteration of logistic regression; typically O(n) total"
assumptions:
  - "base model emits a continuous score $s_i$ (decision function, margin, or uncalibrated probability)"
  - "the score-to-probability mapping is monotonic and approximately sigmoidal"
  - "calibration data is held out from base-model training"
failure_modes:
  - "severe miscalibration shapes (e.g., bimodal, sigmoid-reversed) break the two-parameter form"
  - "calibrator fit on same data used for base-model selection -> optimistic Brier score (see [[condensates/calib-per-fold-leakage]])"
  - "small calibration sets inflate variance of $(A, B)$ estimates"
---

# Platt scaling maps scores to probabilities via a two-parameter logistic

## Statement

Given a base-model score $s_i \in \mathbb{R}$ (signed distance to hyperplane, log-odds, or uncalibrated probability), Platt (1999) fits a logistic function

$$\hat{p}_i = \frac{1}{1 + \exp(A s_i + B)}$$

with parameters $(A, B)$ estimated by maximum likelihood on a calibration set $\{(s_i, y_i)\}$:

$$(\hat{A}, \hat{B}) = \arg\min_{A, B} \; -\sum_{i=1}^{n_\text{cal}} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right].$$

The calibrator has **two parameters** — $A$ (slope / sharpness) and $B$ (intercept / bias). This is the simplest calibration mapping that can correct both slope and intercept miscalibration.

The **logistic_intercept** variant fixes $A = 1$ and fits only $B$ — a **one-parameter** calibration that corrects intercept (base-rate shift) but not slope.

## Derivation

Platt (1999) motivated the two-parameter logistic as the posterior $p(y = 1 \mid s)$ under the Gaussian generative assumption that $s \mid y = 0$ and $s \mid y = 1$ are normally distributed with equal variance. Under those assumptions the posterior is logistic in $s$ with slope and intercept determined by the class-conditional means and variances.

Niculescu-Mizil & Caruana (2005) showed empirically that Platt scaling is most effective for classifiers with sigmoidal reliability curves (SVM, boosted trees) and least effective for classifiers already well-calibrated (logistic regression, naive Bayes on large samples) or with bimodal/multi-modal miscalibration (e.g., bagged decision trees).

Maximum likelihood for $(A, B)$ reduces to standard logistic regression with $s_i$ as the single predictor. Platt's original paper added a mild prior (Laplace smoothing on the target) to prevent degenerate fits at extreme scores; most modern implementations use unsmoothed MLE.

## Boundary conditions

- Two-parameter Platt applies when the reliability curve is approximately sigmoid. For multi-modal miscalibration, isotonic regression is preferred (see [[equations/isotonic-calib]]).
- One-parameter **logistic_intercept** applies when the base model has correct slope but shifted intercept (common for well-ranked models with base-rate drift between train and deployment cohorts). It has one degree of freedom, so it is the **most parsimonious** of the cel-risk calibration strategies (see [[condensates/calib-parsimony-order]]).
- Calibration set MUST be disjoint from the set used to select base-model hyperparameters. Fitting $(A, B)$ on training-fold OOF predictions that already informed model selection leaks model-selection optimism into the calibrator.
- Does NOT apply when scores are non-monotonic in $p(y = 1)$ — e.g., a multi-class one-vs-rest decomposition where the score ordering does not match the binary task.

## Worked reference

Base model: LinSVM_cal. Decision-function scores on 148 calibration samples range over $s \in [-2.1, 3.4]$.

Two-parameter Platt MLE yields $(\hat{A}, \hat{B}) = (-0.62, -5.4)$ (negative $A$ because higher score should yield higher $\hat{p}$; the sign is a function of how $s$ is defined). Example prediction: for $s = 1.0$, $\hat{p} = 1/(1 + \exp(-0.62 \cdot 1.0 - 5.4)) = 1/(1 + \exp(-6.02)) \approx 0.998$. For $s = -1.0$, $\hat{p} \approx 1/(1 + \exp(-4.78)) \approx 0.992$. Calibrated positive-class probabilities are now in $[0, 1]$ and reliability-curve matches on the calibration set.

One-parameter **logistic_intercept** on the same scores (fix $A = 1$) yields $\hat{B} = -5.7$. This corrects the base-rate offset only; slope sharpness is inherited from the base model.

## Sources

- Platt (1999). "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." *Advances in Large Margin Classifiers*.
- Niculescu-Mizil & Caruana (2005). "Predicting good probabilities with supervised learning." *ICML*.
- ADR-008 (cel-risk), 2026-01-22.

## Used by

- [[condensates/calib-parsimony-order]]
- [[condensates/calib-per-fold-leakage]]
- [[equations/beta-calib]] — beta adds one more parameter on top of Platt
