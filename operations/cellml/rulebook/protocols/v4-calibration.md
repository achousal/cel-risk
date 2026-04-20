---
type: protocol
gate: v4-calibration
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v3-imbalance"
  - "oof_predictions: per-cell OOF posterior from the V3-locked (weighting, downsampling) configuration"
outputs:
  - "locks: [calibration_strategy]"
axes_explored:
  - "calibration ∈ {logistic_intercept, beta, isotonic}"
axes_deferred:
  - "threshold: deferred to v5-confirmation (fixed-specificity rule applied on VAL, evaluated on TEST)"
  - "operating-point objective (fixed-spec vs Youden vs max F1): ADR-010, not a V4 axis"
  - "seed-split confirmation: deferred to v5-confirmation"
depends_on:
  - "[[condensates/calib-parsimony-order]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[condensates/calib-per-fold-leakage]]"
  - "[[condensates/nested-cv-prevents-tuning-optimism]]"
  - "[[condensates/three-way-split-prevents-threshold-leakage]]"
  - "[[equations/platt-scaling]]"
  - "[[equations/beta-calib]]"
  - "[[equations/isotonic-calib]]"
  - "[[equations/brier-decomp]]"
  - "[[equations/nested-cv]]"
---

# V4 Calibration Protocol

Post-hoc calibration gate. V4 selects one of three candidate calibrators
({`logistic_intercept`, `beta`, `isotonic`}) for the model locked at V2 and
the imbalance configuration locked at V3. The decision is made against the
reliability component (REL) of the Brier score ([[equations/brier-decomp]])
with a strict parsimony tiebreaker under Equivalence
([[condensates/calib-parsimony-order]]). Calibration is applied **after**
the training-time decisions so that calibrator selection does not conflate
with model-architecture or imbalance-handling choices
([[condensates/calib-per-fold-leakage]]).

---

## 1. Pre-conditions

Before V4 ledger can be written:

1. **V3 must be locked.** [[protocols/v3-imbalance]] must have emitted a
   `decision.md` naming the locked (`weighting`, `downsampling`) pair and
   the claim type (Direction or Equivalence-parsimony) that justified it.
   V4 inherits those locks as fixed cell-level configuration. V4 does NOT
   re-open the V3 axes. If V3 returned Inconclusive on the winning
   recipe × model, V4 cannot run on that configuration until the V3 tension
   is resolved.
2. **V2 model lock is inherited.** V4 runs on the single model locked at
   [[protocols/v2-model]]; it does not test calibration conditional on
   alternative models. Cross-model calibration sensitivity is a post-tree
   concern, not a V4 axis.
3. **V1 recipe and panel are inherited.** The feature set is frozen; V4
   does not revisit panel composition.
4. **V0 data partitioning is inherited.** Training strategy, prevalent
   fraction, and control ratio are fixed. V4 operates inside the 50/25/25
   stratified split defined at V0 per
   [[condensates/three-way-split-prevents-threshold-leakage]]; calibrators
   are fit on OOF predictions derived from the TRAIN slice's nested CV and
   evaluated on VAL (threshold selection at V5 uses VAL; TEST is
   quarantined).
5. **OOF predictions from nested CV are available per cell.** Per
   [[equations/nested-cv]] and [[condensates/nested-cv-prevents-tuning-optimism]],
   each cell produces a per-sample OOF posterior under the nested 5×10×5×200
   structure. V4's calibrator is fit on the aggregated OOF predictions for
   the V3-locked cell, NOT inside each inner fold — this is the
   `oof_posthoc` strategy mandated by
   [[condensates/calib-per-fold-leakage]]. The `per_fold` strategy is
   **forbidden** at V4: it fits the calibrator on the same slice used to
   select hyperparameters, so REL on VAL is optimistically biased relative
   to TEST.
6. **Calibration-set size floor.** $n_\text{cal\_positives} \ge 30$ in the
   OOF set, per [[condensates/calib-parsimony-order]] boundary condition.
   Below this, even `logistic_intercept`'s single parameter has high
   variance and V4 returns Inconclusive per §4 rather than locking a
   noise-fit calibrator. On celiac: $n_\text{cases} = 148$ with OOF
   available on TRAIN (50%) comfortably satisfies the floor.
7. **Rulebook snapshot bound at gate entry.** Per SCHEMA.md §Versioning,
   V4 ledger frontmatter must cite `rulebook_snapshot: "rb-v{x.y.z}"` at
   entry and remain immutable through decision.

---

## 2. Search space

V4 crosses exactly one axis: `calibration ∈ {logistic_intercept, beta,
isotonic}`. Each level is a strictly nested parametric family
([[equations/platt-scaling]] → [[equations/beta-calib]] →
[[equations/isotonic-calib]]), which is what justifies the parsimony
ordering and the pairwise comparison structure below.

### 2.1 Fitting strategy (NOT an axis — a non-negotiable)

All three calibrators MUST be fit using the `oof_posthoc` strategy:

- Run the V3-locked cell's nested CV to completion.
- Aggregate per-sample OOF posteriors $\{\hat{p}_i^\text{OOF}\}$ across
  outer folds — these are predictions on rows that the base model never
  saw during training AND that were not used to select hyperparameters
  for that row.
- Fit ONE calibrator on $\{(\hat{p}_i^\text{OOF}, y_i)\}$ per
  candidate family.

`per_fold` fitting (the scikit-learn `CalibratedClassifierCV` default) is
forbidden. [[condensates/calib-per-fold-leakage]] is explicit: fitting a
calibrator inside the same fold used for hyperparameter selection lets it
learn the noise pattern of the inner validation slice, driving REL
artificially low on VAL. The rule's actionable line — "The V4 (calibration)
gate MUST use `oof_posthoc` as the reference strategy. `per_fold` is
allowed only as a comparator, and the gate logs the delta" — is enforced
here as: V4 reports REL under `oof_posthoc`; any `per_fold` number recorded
in `observation.md` is diagnostic only and cannot enter the lock decision.

### 2.2 Level: `logistic_intercept`

One-parameter variant of Platt scaling ([[equations/platt-scaling]]):
slope $A = 1$ fixed, intercept $B$ fit by MLE on the OOF set. This is the
simplest calibrator in the ordering — it corrects base-rate drift but
cannot change sharpness. Per [[condensates/calib-parsimony-order]], this
is the parsimony default: when the equivalence criterion holds across all
pairwise comparisons, `logistic_intercept` is locked.

### 2.3 Level: `beta`

Three-parameter sigmoid ([[equations/beta-calib]]) fitting
$(a, b, c)$ via logistic regression on $(\log s, \log(1-s))$. Captures
asymmetric tail behavior that a symmetric sigmoid cannot. Beta strictly
nests Platt (which strictly nests `logistic_intercept`): when $a = b = 0$,
beta reduces to `logistic_intercept`. The extra freedom can only reduce
in-sample REL; whether it pays out-of-sample is the V4 question.

### 2.4 Level: `isotonic`

Non-parametric monotone regression via PAVA
([[equations/isotonic-calib]]) with up to $n_\text{cal}$ effective
parameters. Strictly dominates any parametric monotone calibrator
in-sample. Most vulnerable to calibration-set size: celiac's OOF set has
only ~148 positives, so isotonic's bias-variance tradeoff is
unfavorable unless the base model's miscalibration shape is
genuinely non-sigmoidal. The isotonic worked reference in
[[equations/isotonic-calib]] documents exactly this inversion on this
cohort (in-sample REL lowest, out-of-sample REL highest).

### 2.5 Axis-condensate mapping

| Decision | Grounding | Load-bearing claim |
|---|---|---|
| Use `oof_posthoc`, forbid `per_fold` | [[condensates/calib-per-fold-leakage]] | Calibrators fit inside the hyperparameter-selection fold are optimistically biased; the factorial default is `oof_posthoc` |
| Parsimony ordering logistic_intercept ≺ beta ≺ isotonic | [[condensates/calib-parsimony-order]] | Simpler calibrators have lower out-of-sample variance when in-sample REL is statistically indistinguishable |
| REL (not BS) is the primary metric | [[equations/brier-decomp]] | BS = REL − RES + UNC; discrimination (RES) was decided at V1/V2; UNC is a dataset property; V4 adjudicates only REL |
| Cross-fold aggregation | [[equations/nested-cv]], [[condensates/nested-cv-prevents-tuning-optimism]] | OOF predictions are genuinely held out from both base-model training and hyperparameter selection |
| Calibration set is TRAIN-slice OOF, evaluation is VAL | [[condensates/three-way-split-prevents-threshold-leakage]] | Any post-training parameter fit (including calibrator parameters) must absorb into the VAL slice, not TEST |

---

## 3. Success criteria

All V4 decisions use the fixed falsifier rubric in SCHEMA.md. The primary
metric is $\Delta\mathrm{REL}$ on the VAL split, with 95% bootstrap CI over
1000 outer-fold resamples (paired across calibrators, seed-level unit).

### 3.1 Claim types

| Claim type | Criterion | V4 decision |
|---|---|---|
| **Direction** (A > B on improvement, i.e., $\mathrm{REL}_A < \mathrm{REL}_B$) | $\|\Delta\mathrm{REL}\| \ge 0.005$ AND 95% bootstrap CI excludes 0 | The more-flexible calibrator wins if and only if it improves REL by Direction |
| **Equivalence** (A ≈ B) | $\|\Delta\mathrm{REL}\| < 0.005$ AND 95% bootstrap CI $\subset [-0.01, 0.01]$ | Parsimony tiebreaker applies — lock the simpler calibrator |
| **Inconclusive** | Neither Direction nor Equivalence met | See §4.1 — lock `logistic_intercept` by parsimony |

Note: V4 uses a tighter REL band ($\|\Delta\mathrm{REL}\| \ge 0.005$ for
Direction, $<0.005$ for Equivalence, CI $\subset [-0.01, 0.01]$) than the
AUROC-default rubric (0.02 / 0.01 / $[-0.02, 0.02]$). Rationale: REL on
rare-event cohorts is small in absolute terms (celiac worked reference in
[[equations/brier-decomp]] gives $\mathrm{REL} \sim 10^{-5}$ and the UNC
floor is 0.003), so the rubric's 0.02 band would swamp any meaningful
calibration difference. The tightened band was introduced per ADR-008 and
is load-bearing for V4 discriminability on low-prevalence cohorts; it is
NOT a silent override of the SCHEMA.md rubric — the ledger must declare
the band explicitly at gate entry.

### 3.2 Decision tree (cite [[condensates/calib-parsimony-order]] §Actionable rule)

Evaluated strictly top-down:

1. **isotonic vs beta Direction.** If
   $\mathrm{REL}_\text{isotonic} < \mathrm{REL}_\text{beta}$ by the
   Direction criterion ($\|\Delta\mathrm{REL}\| \ge 0.005$ AND CI excludes
   0) → lock `isotonic`.
2. **Else: beta vs logistic_intercept Direction.** If
   $\mathrm{REL}_\text{beta} < \mathrm{REL}_\text{logistic\_intercept}$ by
   Direction → lock `beta`.
3. **Else (Equivalence or Inconclusive for all pairwise comparisons)** →
   lock `logistic_intercept` (parsimony tiebreaker per
   [[condensates/calib-parsimony-order]]).

Parsimony-tie resolution follows `[[condensates/parsimony-tiebreaker-when-equivalence]]`
with this axis's order: **`logistic_intercept` ≺ `beta` ≺ `isotonic`**. The
meta-condensate is the parent rule that governs Equivalence-triggered
tiebreakers across the factorial; [[condensates/calib-parsimony-order]] is
the axis-specific condensate that instantiates the calibrator ordering
here, and V4 decision ledgers under Equivalence MUST cite both.

The top-down ordering is non-negotiable: starting from the most-complex
end and demanding Direction to justify each step up is the exact
mechanism by which parsimony is enforced. A bottom-up search ("does beta
beat logistic_intercept?") would lock beta on a spurious Direction from
noise-fitting. The top-down structure matches the nested structure of
the calibrator families ([[equations/beta-calib]] §Derivation: "When
$a + b = 0$ the beta calibration reduces to Platt scaling. When
$a = b = 0$ it reduces to logistic_intercept").

### 3.3 Bootstrap unit (seed, not fold, not sample)

Paired bootstrap of $\Delta\mathrm{REL}$:

- Unit: seed (outer fold / split seed). 20 selection seeds (100–119), per
  [[protocols/v0-strategy]] §1 — the 10-seed confirmation set (120–129)
  is quarantined for V5.
- Statistic: $\Delta\mathrm{REL}_s = \mathrm{REL}_A^{(s)} - \mathrm{REL}_B^{(s)}$
  computed on the VAL slice for each seed, paired across calibrators.
- $B = 1000$ resamples with replacement over seeds.
- 95% CI from the empirical 2.5%/97.5% quantiles of $\{\Delta\mathrm{REL}_s^{*}\}$.

Per-sample bootstrap is forbidden (it conflates measurement variance with
within-seed dependence). Per-fold bootstrap (inner CV folds) is also
forbidden (inner folds are tuning objects, not evaluation objects, per
[[condensates/nested-cv-prevents-tuning-optimism]]).

### 3.4 V4–V2 cross-metric check (advisory, not a lock gate)

V4 optimizes REL. V2 optimized AUROC (and Brier-based Pareto). Because
$\mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}$
([[equations/brier-decomp]]), a calibrator that reduces REL cannot
mechanically reduce RES (RES is a function of $\{\bar{y}_k\}$ and
$\bar{y}$, both determined by the data, not the calibrator) — so AUROC
is invariant to calibration at rank level. **However, sensitivity at a
fixed specificity IS affected**: recalibration changes the threshold
$\tau_\text{spec}$ (via its effect on the control score distribution)
and can therefore change sensitivity. V4 logs sensitivity-at-spec (0.95,
the ADR-010 default) under each calibrator on VAL as a diagnostic
alongside REL. If $|\Delta\text{sens}_{0.95}| \ge 0.05$ between the
locked and an alternative calibrator despite REL Equivalence, flag a
cross-metric tension in `tensions.md` — this is a V5 concern (the V5
threshold will be selected on the V4-locked calibrator's VAL
predictions). The cross-metric flag does NOT veto the V4 lock — parsimony
under Equivalence is the adjudicator per
[[condensates/calib-parsimony-order]], but the flag is forwarded to V5
so the confirmation gate is aware.

### 3.5 Out-of-support behavior

Isotonic regression extrapolates flat at score-range boundaries
([[equations/isotonic-calib]] §Boundary conditions); Platt/beta
extrapolate via their parametric form. At gate entry, V4 must record the
VAL-slice score range and the OOF-slice score range: if the VAL-slice
max score exceeds the OOF-slice max by more than 0.05, isotonic's
extrapolation flat behavior kicks in on some VAL rows and the REL
comparison is not apples-to-apples. The ledger documents the support
overlap; a tension flag is written if the overlap is incomplete.

---

## 4. Fallbacks

When the decision tree in §3.2 returns Inconclusive (neither Direction
nor Equivalence holds on at least one pairwise comparison):

### 4.1 Parsimony lock under Inconclusive

V4 locks `logistic_intercept` by parsimony per
[[condensates/calib-parsimony-order]] §Actionable rule step 3
("Else (Equivalence or Inconclusive for all pairwise comparisons) → pick
`logistic_intercept`"). Rationale: Inconclusive means the data cannot
distinguish the calibrators at the chosen sample size; defaulting to the
simpler family minimizes out-of-sample variance. This is the same
mechanism used in V0 and V1 for Equivalence fallthrough, extended to the
Inconclusive case because the nested structure of the calibrator
families makes the simpler form the safe default even in the absence of
Equivalence evidence. The `decision.md` explicitly names the claim type
as `Inconclusive` and cites this fallback.

### 4.2 Calibration-set size below floor

If $n_\text{cal\_positives} < 30$ (e.g., a sub-cohort analysis where the
TRAIN OOF set is trimmed), V4 does NOT run comparison bootstrap — the
variance on every calibrator's REL estimate is too high for discrimination
to be meaningful. Lock `logistic_intercept` and write a tension flagging
that calibration choice is not empirically supported on this cohort;
deployment calibration should be re-fit on a larger cohort per
[[condensates/calib-parsimony-order]] boundary condition.

### 4.3 Non-monotonic reliability

If the base model's reliability curve on OOF is non-monotonic (checked via
pool-adjacent-violators: more than 2 flat regions or a locally decreasing
segment covering $\ge 10\%$ of the score range), `logistic_intercept` and
`beta` cannot fit the shape ([[equations/isotonic-calib]] §Boundary
conditions). V4 in this case is NOT a parsimony decision — `isotonic` is
the only family that can fit, and V4 locks it regardless of
$\Delta\mathrm{REL}$ vs the other two (they are effectively
misspecified). A separate tension flag is written because this signals
a deeper problem in the base model (the V2 lock may have selected a
model with pathological probability outputs). Per
[[condensates/calib-parsimony-order]]: "Does NOT apply when the base
model is non-monotonically miscalibrated."

### 4.4 `per_fold` comparator deviates beyond Equivalence

If the diagnostic `per_fold` fit (§2.1) differs from the `oof_posthoc`
fit on the same calibrator family by $|\Delta\mathrm{REL}| \ge 0.005$
with CI excluding 0, write a tension. The `per_fold` vs `oof_posthoc`
delta is the empirical validation of
[[condensates/calib-per-fold-leakage]]; a large delta confirms the
condensate and should be logged as an evidence row. A small delta does
NOT relax the `oof_posthoc` mandate (the mandate is mechanistic, not
empirical).

### 4.5 No relaxation of `oof_posthoc`

If all three `oof_posthoc` calibrators return Inconclusive and the
cohort is too small to discriminate at the 0.005 REL band, the correct
response is **NOT** to relax to `per_fold` (which would appear to
improve REL by leakage). Lock `logistic_intercept` per §4.1 and accept
that the V4 calibration decision is under-powered on this cohort.

---

## 5. Post-conditions

On V4 `decision.md` write:

1. **Locks (on Direction).** If the §3.2 tree locks a non-default
   calibrator, V4 emits:
   - `calibration_strategy` — one of {logistic_intercept, beta, isotonic}
   - claim type — Direction (which pair)
   - $\Delta\mathrm{REL}$ point estimate and 95% CI relative to
     `logistic_intercept`
2. **Locks (on Equivalence / Inconclusive via parsimony).**
   `calibration_strategy = logistic_intercept` with the claim type
   declared as `Equivalence` or `Inconclusive` and the parsimony rule
   from [[condensates/calib-parsimony-order]] cited explicitly.
3. **Diagnostics table.** Per-calibrator REL (point + 95% CI) on VAL,
   sensitivity-at-spec (0.95) on VAL, and the `per_fold` vs `oof_posthoc`
   delta per calibrator. The diagnostic sensitivity-at-spec is the bridge
   to V5 — it is NOT the V5 sensitivity estimate (V5 selects the
   threshold on VAL and evaluates on TEST; V4 diagnostics are VAL-only).
4. **Calibrated OOF predictions forwarded.** The locked calibrator is
   applied to the V3-locked cell's OOF predictions, producing calibrated
   per-sample posteriors $\{\hat{p}_i^\text{cal,OOF}\}$. These become V5's
   inputs for threshold selection on VAL and held-out evaluation on TEST.
5. **Cross-metric tensions.** Any flag raised in §3.4, §3.5, §4.3, or
   §4.4 is written to `tensions.md` and routed to
   `projects/<name>/tensions/rule-vs-observation/` per SCHEMA.md
   §Rulebook updates.
6. **Claim type MUST be declared explicitly.** Per SCHEMA.md §Fixed
   falsifier rubric, `decision.md` under `## Actual claim type (per rubric)`
   states exactly one of {Direction, Equivalence, Inconclusive} for the
   V4 axis, and identifies which pairwise comparison in §3.2 provided
   the lock.

What advances to V5:

- The calibrated pipeline: V0 locks ∘ V1 recipe ∘ V2 model ∘ V3 imbalance
  ∘ V4 calibrator — the full configuration that V5 will subject to
  held-out seed validation.
- The calibrated per-sample OOF posteriors on selection seeds (100–119),
  needed for V5's paired bootstrap of generalization deltas.
- Any advisory cross-metric flag (§3.4) so V5 can elect to report
  sensitivity-at-spec with an alternative calibrator as a sensitivity
  analysis (the V4 lock stands; the V5 report includes the sensitivity
  check).

What remains open:

- Threshold rule (fixed-spec at ADR-010 default target 0.95, Youden, or
  max $F_1$) — declared in the V5 ledger per
  [[condensates/threshold-on-val-not-test]].
- Holdout generalization verdict (V5).

---

## Known tensions

### T-V4-01: REL-Direction band is tighter than the SCHEMA default

SCHEMA.md's fixed falsifier rubric uses 0.02 for Direction and 0.01 for
Equivalence on AUROC / PR-AUC / Brier. V4 REL uses 0.005 / 0.005 / CI
$\subset [-0.01, 0.01]$. Justification is in §3.1 (REL on rare-event
cohorts is $\sim 10^{-5}$; the SCHEMA band would swamp all signal). The
band was introduced per ADR-008 and is consistent with
[[condensates/calib-parsimony-order]]'s falsifier wording ("Direction:
$|\Delta\mathrm{REL}| \ge 0.02$" is the condensate's own retirement
threshold, which applies to a hypothetical 3+ dataset aggregate delta —
not the per-gate discrimination band). The tension is: the SCHEMA
rubric should probably add an explicit metric-specific rule for REL, or
the rubric should allow protocols to tighten (never loosen) bands with
ledger justification. Flag as candidate SCHEMA.md PATCH.

### T-V4-02: V4 optimizes REL; AUROC is invariant but sensitivity-at-spec is not

At rank level, recalibration is monotone and AUROC is unchanged. But
sensitivity-at-fixed-spec depends on the $(1 - \mathrm{spec}^{*})$-th
quantile of **control** scores on VAL
([[equations/fixed-spec-threshold]]), which changes after calibration.
A calibrator that compresses the control score range near 0 can lower
$\tau_{\mathrm{spec}}$ and raise sensitivity (or vice versa). The §3.4
advisory surfaces this, but V4 does not adjudicate it — V5 inherits the
tradeoff. The worry: parsimony-locking `logistic_intercept` under REL
Equivalence could leave substantial sensitivity-at-spec gains from
`isotonic` on the table. This is an explicit design choice, not an
oversight: V4 is the calibration-quality gate, not the operating-point
gate.

### T-V4-03: `per_fold` diagnostic is forbidden from the lock

§2.1 and §4.5 forbid `per_fold` from entering the lock decision. If a
future dataset shows `oof_posthoc` and `per_fold` in true Equivalence on
REL across all three calibrators (the [[condensates/calib-per-fold-leakage]]
falsifier's Equivalence condition at $n_\text{cases} \ge 500$), the
mandate could relax. V4 does not test this path; the condensate's
falsifier does.

### T-V4-04: Isotonic extrapolation at support mismatch

§3.5 flags an incomplete VAL/OOF support overlap as a tension rather
than a lock veto. If the overlap gap is large (max VAL score > OOF max
by more than 0.05), isotonic's flat extrapolation introduces a bias
that the REL bootstrap does not capture (the bootstrap resamples seeds
within the observed range). No current condensate addresses this;
candidate rulebook TODO: `condensates/isotonic-extrapolation-at-boundary.md`.

### T-V4-05: V4 is the last axis-explored gate before V5

After V4, the full pipeline is locked. Any retuning at V5 is forbidden
per [[protocols/v5-confirmation]] and would constitute a quarantine
breach. V4's parsimony defaults (`logistic_intercept` on
Equivalence/Inconclusive) are therefore load-bearing for V5
generalization: a V4 over-fit that locks `isotonic` on a spurious
Direction from seed noise would propagate into V5 as an optimistic
selection-set REL that V5 would then flag as generalization failure.
The V4 → V5 coupling is acknowledged and is why §3.2's top-down
ordering is strict.

---

## Sources

- `operations/cellml/DESIGN.md` §V4, §Factorial Factors §Parsimony Ordering
- `operations/cellml/MASTER_PLAN.md` §V4
- `operations/cellml/DECISION_TREE_AUDIT.md` §1.1 (V4 parsimony), §1.2
  (V4 CI overlap test), §Resolution log (2026-04-08: V3→V4 reordering)
- ADR-008 (OOF post-hoc calibration) — canonical source for the
  `oof_posthoc` mandate
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
