---
type: condensate
depends_on:
  - "[[equations/isotonic-calib]]"
  - "[[equations/platt-scaling]]"
  - "[[equations/beta-calib]]"
  - "[[equations/brier-decomp]]"
  - "[[condensates/calib-parsimony-order]]"
applies_to:
  - "post-hoc calibrator selection at V4 when VAL or TEST contains rows with scores outside the calibrator's OOF fit range"
  - "low-prevalence cohorts where OOF score range can differ meaningfully from VAL/TEST"
  - "pipelines where base-model score distribution is narrower on OOF than on downstream evaluation slices"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v4-calibration
    delta: "isotonic's flat extrapolation kicks in when VAL-slice max score exceeds OOF-slice max by > 0.05; the REL bootstrap over seeds does not capture this boundary bias because it resamples within the observed range, so a support-mismatch cohort can look REL-Equivalent yet have isotonic underperform parametric calibrators on tail-region calibration"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §V4; v4-calibration protocol T-V4-04 §3.5 out-of-support behavior; equations/isotonic-calib.md §Boundary conditions"
falsifier: |
  If on >=3 cohorts, TEST-boundary-region Brier (scores in the top/bottom
  5% of the probability distribution) shows |delta-Brier(isotonic - beta)|
  < 0.005 with CI subset of [-0.01, 0.01], isotonic's boundary risk is
  weakened (Equivalence claim on the cross-metric boundary-region Brier).
  If on >=5 cohorts, retire in favor of the in-sample REL-only rule.
---

# Isotonic regression underperforms parametric calibrators on TEST rows outside the calibrator's OOF fit range

## Claim

Isotonic regression produces **flat-tail** predictions outside the VAL/OOF
support range on which it was fit. When TEST contains rows with scores
beyond the calibrator's OOF fit range (either above the max or below the
min), isotonic's flat extrapolation introduces a systematic calibration
bias at the tails that parametric calibrators (Platt,
`logistic_intercept`, beta) do not exhibit. Consequently, on cohorts with
a meaningful VAL/OOF vs TEST score-support mismatch, isotonic
underperforms parametric calibrators on tail-region calibration even when
the global REL bootstrap (over seeds, within the observed range) reports
Equivalence.

## Mechanism

Isotonic regression via PAVA produces a piecewise-constant monotone
function on the convex hull of the OOF score set. Outside that convex
hull — i.e., for any row with `score > max(OOF_scores)` or
`score < min(OOF_scores)` — the isotonic predictor extrapolates by
**reverting to the nearest observed bin**: the last (or first) step value
of the fitted function. This is a deliberate consequence of the
step-function nature of PAVA and is documented in
[[equations/isotonic-calib]] §Boundary conditions.

Parametric calibrators extrapolate via their functional form:
[[equations/platt-scaling]] and `logistic_intercept` extrapolate as
sigmoids (saturating to 0 or 1 but continuing to respond monotonically to
score), and [[equations/beta-calib]] extrapolates as an asymmetric
sigmoid. Parametric extrapolation may be biased (the functional form can
be wrong at the tail), but it is **continuous and monotone** in score —
so a row one unit beyond the training range receives a smoothly
different prediction than a row two units beyond. Isotonic's flat-tail
behavior collapses that gradient to zero, which produces systematic
mis-scoring of the most extreme TEST rows.

The key diagnostic is the seed-level bootstrap blind spot:
[[protocols/v4-calibration]] §3.3 specifies bootstrapping over
seeds, which resamples rows **within** the observed OOF score range. Rows
that are outside the range on a given seed's TEST slice but inside the
range on another seed's slice will resample in and out of the boundary
region across bootstrap replicates, so the bootstrap CI captures within-
range variance but underestimates boundary-region bias. The Brier
contribution of a single boundary-region row is small in absolute terms
but can be large in relative terms at that row's probability level — and
precisely those rows are often the decision-relevant extremes (top-k
sensitivity cases, bottom-k ruled-out screening candidates). On low-
prevalence cohorts (celiac `pi ~ 0.00338`), boundary cases are mostly
control-side — the TEST score max can exceed the OOF max when a TEST
control happens to score higher than any control in the OOF set, which
by combinatorics is a regular occurrence as TEST size grows.

## Actionable rule

- V4's [[protocols/v4-calibration]] §3.5 MUST log OOF-slice score range
  and VAL-slice score range at gate entry. If max(VAL_score) exceeds
  max(OOF_score) by > 0.05, OR if min(VAL_score) falls below
  min(OOF_score) by > 0.05, flag as a boundary-risk tension in
  `tensions.md` and **prefer parametric calibrators** (`logistic_intercept`
  or `beta`) unless isotonic shows Direction on the in-range REL. In a
  boundary-risk regime, Equivalence is not a sufficient basis to lock
  isotonic.
- V4's `observation.md` MUST additionally log, for each cohort where
  TEST is also available (e.g., at V5 dry-run), the fraction of TEST rows
  outside the OOF fit range and the boundary-region Brier contribution
  per calibrator on those rows. This makes the boundary-bias empirically
  measurable and lets the cross-cohort evidence accumulate toward the
  falsifier.
- If > 5% of TEST rows are outside the OOF fit range, isotonic MUST NOT
  be locked even if in-range REL favors it by Direction, unless the
  tail-region Brier under isotonic is also at least as good as under the
  best parametric alternative. The rationale is that the bulk-region REL
  advantage does not offset systematic tail-region miscalibration at
  decision-relevant extremes.
- Parsimony per [[condensates/calib-parsimony-order]] already biases V4
  toward `logistic_intercept`; this condensate reinforces that bias
  specifically in boundary-risk regimes.
- V4's `ledger.md` MUST cite this condensate in the §2.5 axis-condensate
  mapping under "Out-of-support behavior."

## Boundary conditions

- Applies when VAL/OOF is NOT a representative sample of TEST with
  respect to the base-model score range. Representativeness can be
  checked at gate entry via the range-overlap diagnostic above.
- Does NOT apply when VAL is a genuine representative sample of TEST
  (e.g., under random split with large n, the score ranges are expected
  to overlap fully in probability). On sufficiently large cohorts with
  random splitting, the boundary-mismatch event becomes increasingly
  rare and the flat-tail bias is bounded.
- Does NOT apply to score distributions with natural bounded support
  that matches VAL and TEST by construction (e.g., sigmoid-output models
  with scores already pre-compressed to `[eps, 1-eps]` by the base model,
  where VAL and TEST are both in the same bounded range).
- Does NOT apply when base-model miscalibration is genuinely non-
  monotonic per [[protocols/v4-calibration]] §4.3: in that regime
  isotonic is the only family that can fit the shape, and locking
  isotonic is correct even with boundary-bias cost. In that case, the
  tension is "upstream" (the V2 model lock selected a model with
  pathological probability outputs) and is handled by the V2 protocol,
  not here.
- Boundary-bias magnitude scales with base-model score variance and
  inversely with OOF size. On large OOF sets (`n_cal_positives >> 30`
  per [[condensates/calib-parsimony-order]] boundary condition), the
  probability of TEST falling outside the OOF range decreases, and the
  condensate's effect weakens proportionally.

## Evidence

| Dataset | n_OOF_positives | score-range VAL vs OOF overlap | Calibrator flagged | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | ~148 (TRAIN-slice OOF) | VAL max can exceed OOF max by > 0.05 on some seeds (flagged in protocol T-V4-04) | isotonic flagged as boundary-risk in those seeds | v4-calibration protocol T-V4-04; DESIGN.md §V4; equations/isotonic-calib.md §Boundary conditions |

The precise per-seed support overlap fractions are recorded in V4's
`observation.md` and accumulate toward cross-cohort evidence as additional
datasets land. The celiac cohort alone provides one confirmation; the
falsifier requires at least 3 non-overlapping cohorts before this
condensate can be weakened or promoted to `established`.

## Related

- [[protocols/v4-calibration]] — protocol that cited this gap (Known
  tension T-V4-04 "Isotonic extrapolation at support mismatch"; §3.5
  out-of-support behavior; §4.3 non-monotonic reliability fallback)
- [[equations/isotonic-calib]] — PAVA step-function mechanics that
  mechanistically ground the flat-tail behavior
- [[equations/platt-scaling]] and [[equations/beta-calib]] — parametric
  families whose functional-form extrapolation is the contrast case
- [[condensates/calib-parsimony-order]] — reinforces the parsimony bias
  toward `logistic_intercept` in boundary-risk regimes
- [[equations/brier-decomp]] — Brier decomposition that scopes REL to
  in-range behavior, motivating the supplementary tail-region Brier
  diagnostic this condensate mandates
- ADR-008 (cel-risk) — canonical source for the `oof_posthoc` mandate;
  this condensate extends ADR-008's out-of-support concern into a
  falsifiable rule
