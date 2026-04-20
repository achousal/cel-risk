---
type: equation
symbol: "BS"
depends_on: []
computational_cost: "O(n) given binned predictions; O(n log n) for binning"
assumptions:
  - "predictions $\\hat{p}_i \\in [0,1]$ can be partitioned into $K$ bins by predicted value"
  - "within each bin, $\\hat{p}_i$ is approximately constant (i.e., bin widths are narrow relative to calibration curvature)"
  - "binary outcome $y_i \\in \\{0, 1\\}$"
failure_modes:
  - "coarse binning inflates REL and deflates RES (binning artifact, not calibration error)"
  - "very fine binning produces empty bins for rare-outcome cohorts -> unstable estimates"
  - "decomposition ignores sharpness beyond resolution; a constant predictor has RES=0 and arbitrarily small BS if prevalence is extreme"
---

# Brier score decomposes into reliability, resolution, and uncertainty

## Statement

For binary outcome $y_i \in \{0, 1\}$ and predicted probability $\hat{p}_i \in [0, 1]$ over $n$ samples, the Brier score is

$$\mathrm{BS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2.$$

Murphy (1973) showed that binning predictions into $K$ bins with $n_k$ samples per bin, bin-mean prediction $\bar{p}_k$, and bin-mean outcome $\bar{y}_k$, yields the decomposition

$$\mathrm{BS} \;=\; \underbrace{\frac{1}{n} \sum_{k=1}^{K} n_k (\bar{p}_k - \bar{y}_k)^2}_{\text{REL (reliability)}} \;-\; \underbrace{\frac{1}{n} \sum_{k=1}^{K} n_k (\bar{y}_k - \bar{y})^2}_{\text{RES (resolution)}} \;+\; \underbrace{\bar{y}(1 - \bar{y})}_{\text{UNC (uncertainty)}}$$

where $\bar{y} = n^{-1} \sum_i y_i$ is the overall base rate.

- **REL**: squared deviation between predicted and observed bin frequencies. Zero iff the model is perfectly calibrated (bin-mean prediction = bin-mean outcome in every bin). **Lower is better.**
- **RES**: spread of bin-mean outcomes around the base rate. Captures how much the model separates classes. **Higher is better.**
- **UNC**: irreducible variance of $y$ given the base rate alone. Depends only on prevalence; a property of the cohort, not the model.

## Derivation

Start from $\mathrm{BS} = n^{-1} \sum_i (\hat{p}_i - y_i)^2$. Partition samples into bins $k$ where all $\hat{p}_i$ in bin $k$ share prediction $\bar{p}_k$. Within bin $k$, expand $(\bar{p}_k - y_i)^2 = (\bar{p}_k - \bar{y}_k)^2 + 2(\bar{p}_k - \bar{y}_k)(\bar{y}_k - y_i) + (\bar{y}_k - y_i)^2$. The middle cross-term sums to zero within the bin. Summing over bins gives $\mathrm{BS} = \mathrm{REL} + (n^{-1} \sum_k n_k \bar{y}_k(1 - \bar{y}_k))$. The second term is rewritten using the identity $\bar{y}(1-\bar{y}) - n^{-1}\sum_k n_k (\bar{y}_k - \bar{y})^2 = n^{-1}\sum_k n_k \bar{y}_k(1 - \bar{y}_k)$ (variance decomposition for a Bernoulli), yielding $\mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}$.

## Boundary conditions

- Decomposition is exact only when predictions are constant within each bin. Typical implementations bin by quantiles of $\hat{p}$ and use the bin mean as $\bar{p}_k$; the result is a within-bin averaging that introduces a small binning artifact.
- **Low-prevalence cohorts**: UNC is small (e.g., for $\bar{y} = 0.003$, $\mathrm{UNC} \approx 0.003$). A trivial predictor $\hat{p}_i = \bar{y}$ for all $i$ has $\mathrm{REL} = 0$ and $\mathrm{RES} = 0$, so $\mathrm{BS} \approx \mathrm{UNC}$. Improvement over baseline is measured by the Brier Skill Score $\mathrm{BSS} = 1 - \mathrm{BS}/\mathrm{UNC}$.
- Sensitive to calibration only through the REL term. Comparing models by BS alone can hide a calibration regression that is compensated by a resolution gain (or vice versa). For calibration comparisons (e.g., V4 in DESIGN.md), report REL separately.
- Binning scheme affects REL and RES estimates but not UNC. Best practice: bootstrap with fixed $K=10$ quantile bins (common in the medical-ML literature).

## Worked reference

Cohort: $n = 10{,}953$, $\bar{y} = 0.00338$, so $\mathrm{UNC} = 0.00338 \times 0.99662 = 0.00337$.

Predictions from a calibrated model (fake numbers for illustration), binned into $K = 10$ deciles of $\hat{p}$. Bin 10 (top decile): $n_{10} = 1{,}095$, $\bar{p}_{10} = 0.022$, $\bar{y}_{10} = 0.020$. Bins 1–9 all have $\bar{p}_k \approx \bar{y}_k$ near 0.002.

$$\mathrm{REL} \approx \frac{1{,}095 (0.022 - 0.020)^2}{10{,}953} \approx 4 \times 10^{-7}, \qquad \mathrm{RES} \approx \frac{1{,}095 (0.020 - 0.00338)^2}{10{,}953} \approx 2.5 \times 10^{-5}.$$

$$\mathrm{BS} = 4 \times 10^{-7} - 2.5 \times 10^{-5} + 0.00337 \approx 0.00335.$$

The model is essentially perfectly calibrated (REL $\approx 0$) and has modest resolution (RES $\approx 7.5\%$ of UNC). The Brier Skill Score is $1 - 0.00335/0.00337 \approx 0.006$ — skill is real but small in absolute terms, which is typical of very-low-prevalence cohorts.

## Sources

- Brier (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review* 78: 1–3.
- Murphy (1973). "A new vector partition of the probability score." *Journal of Applied Meteorology* 12: 595–600.
- Steyerberg (2019). *Clinical Prediction Models*, 2nd ed., Springer — Chapter 15.
- ADR-008 (cel-risk), 2026-01-22.

## Used by

- [[condensates/calib-per-fold-leakage]]
- [[condensates/calib-parsimony-order]]
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v4-calibration]]
