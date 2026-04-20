---
type: equation
symbol: "tau_spec"
depends_on: []
computational_cost: "O(n log n) to sort VAL-set predictions; O(n) to scan for quantile"
assumptions:
  - "calibrated probabilities $\\hat{p}_i$ on the VAL split are a representative sample of the deployment distribution"
  - "the target specificity $\\text{spec}^{*}$ is achievable given the score distribution of the controls"
  - "specificity is computed on VAL-set controls (label = 0)"
failure_modes:
  - "extreme targets (e.g., spec = 0.99) unreachable when VAL has too few controls — threshold falls back to the maximum predicted probability"
  - "VAL-set specificity optimism leaks into TEST if threshold is reselected after peeking (see [[condensates/threshold-on-val-not-test]])"
  - "ties in predicted probability at the target quantile -> deterministic tie-break rule required for reproducibility"
---

# Fixed-specificity threshold selects the 1-spec* quantile of VAL-set control scores

## Statement

Given calibrated probabilities $\hat{p}_i$ and labels $y_i$ on the VAL split, and a target specificity $\text{spec}^{*} \in (0, 1)$ (e.g., $\text{spec}^{*} = 0.95$), the fixed-specificity threshold is

$$\tau_{\text{spec}} \;=\; Q_{1 - \text{spec}^{*}}\!\left( \{ \hat{p}_i : y_i = 0 \} \right),$$

the $(1 - \text{spec}^{*})$-th empirical quantile of predicted probabilities among **controls** (label 0) in the VAL set.

At this threshold:

$$\text{Specificity}(\tau_{\text{spec}}) \;=\; \Pr[\hat{p}_i < \tau_{\text{spec}} \mid y_i = 0] \;\approx\; \text{spec}^{*}.$$

Sensitivity at $\tau_{\text{spec}}$ is then measured as

$$\text{Sensitivity}(\tau_{\text{spec}}) \;=\; \Pr[\hat{p}_i \ge \tau_{\text{spec}} \mid y_i = 1]$$

reported on the same VAL split (for threshold selection) and re-measured on TEST for the unbiased operating-point estimate.

## Derivation

Specificity equals $1 - \text{FPR}$, so fixing specificity at $\text{spec}^{*}$ is equivalent to fixing FPR at $1 - \text{spec}^{*}$. On a finite sample of $n_0$ controls, the FPR as a function of threshold $\tau$ is the empirical survival function $\hat{F}_0(\tau) = n_0^{-1} \sum_{i : y_i = 0} \mathbb{1}[\hat{p}_i \ge \tau]$, which is a step function that decreases from 1 (at $\tau = 0$) to 0 (at $\tau = \max_i \hat{p}_i$). The threshold achieving target FPR $1 - \text{spec}^{*}$ is the $(1 - \text{spec}^{*})$-th quantile of the control-score distribution.

This mirrors the standard ROC-curve construction: fixed specificity corresponds to fixing the x-coordinate of the ROC curve and reading off the y-coordinate (sensitivity). The operating-point selection is agnostic to the shape of the positive-class distribution — only the **control** distribution enters the threshold rule.

## Boundary conditions

- **Target reachability**: if there are fewer than $n_0 \cdot (1 - \text{spec}^{*})$ distinct control scores above the max positive score, the target cannot be cleanly achieved. Round up the quantile (more conservative specificity) rather than accept unreported target drift. Example: $n_0 = 100$, $\text{spec}^{*} = 0.99$ requires 1 control score above the threshold — any tie breaks the guarantee.
- **Tie-breaking**: when multiple control scores equal the target quantile, the threshold is ambiguous. Deterministic convention used in cel-risk: take the **upper** tied value, producing specificity $\ge \text{spec}^{*}$ (more conservative).
- Applies at threshold-selection time on the VAL split. The sensitivity reported at $\tau_{\text{spec}}$ on VAL is an upper bound on TEST-set sensitivity because VAL was used to select $\tau_{\text{spec}}$ — TEST re-evaluation is required for an unbiased sensitivity estimate. See [[condensates/threshold-on-val-not-test]].
- Does NOT apply when the clinical target is Youden's J ($\text{spec} + \text{sens} - 1$) or max $F_1$ — those are alternative threshold objectives and shift which score summary enters the selection. Youden and max $F_1$ are also supported as comparison objectives in cel-risk (ADR-010), but fixed-spec is the default for screening applications.

## Worked reference

VAL-set: $n_\text{VAL} = 10{,}953$, $n_\text{case,VAL} = 37$, $n_\text{control,VAL} = 10{,}916$. Target $\text{spec}^{*} = 0.95$.

Target quantile: $1 - 0.95 = 0.05$. Threshold is the 5th percentile of control scores **from the top** — i.e., the $0.95 \cdot 10{,}916 = 10{,}370$th-ranked control score. Sorted ascending, this is the score at position 10{,}370, which for a calibrated XGBoost model on the celiac cohort is $\tau_{\text{spec}} = 0.0082$.

At $\tau_{\text{spec}} = 0.0082$:
- VAL controls with $\hat{p} \ge 0.0082$: 546 out of 10{,}916 -> specificity = $1 - 546/10{,}916 = 0.950$ (target hit exactly due to large $n_0$).
- VAL cases with $\hat{p} \ge 0.0082$: 22 out of 37 -> sensitivity = $22/37 = 0.595$.

**Locked operating point**: $\tau = 0.0082$. TEST evaluation (never peeked at) will yield an unbiased sensitivity estimate at this fixed specificity target.

## Sources

- Steyerberg (2019). *Clinical Prediction Models*, 2nd ed., Springer — Chapter 11 (thresholds and decision rules).
- Pepe (2003). *The Statistical Evaluation of Medical Tests for Classification and Prediction*. Oxford.
- ADR-009 (cel-risk), 2026-01-20; ADR-010 (cel-risk), 2026-01-20.

## Used by

- [[condensates/threshold-on-val-not-test]]
- [[condensates/fixed-spec-for-screening]]
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v5-confirmation]]
