---
type: protocol
gate: v2-model
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v1-recipe"
  - "locked: [training_strategy, prevalent_train_frac, train_control_per_case] from v0-strategy"
  - "locked: [recipe_id, panel_composition, panel_size, ordering_strategy] from v1-recipe"
  - "passlist: stage-1 permutation-passed models from v0-strategy"
outputs:
  - "locks: [model]"
  - "forward_if_pareto: [non_dominated_model_set]"
axes_explored:
  - "model ∈ {LR_EN, LinSVM_cal, RF, XGBoost}"
axes_deferred:
  - "calibration: deferred to v4-calibration"
  - "weighting: deferred to v3-imbalance"
  - "downsampling: deferred to v3-imbalance (joint with weighting)"
  - "threshold: deferred to post-factorial operating-point selection"
  - "ensemble: deferred to v6-ensemble (informational, non-locking)"
depends_on:
  - "[[condensates/feature-selection-needs-model-gate]]"
  - "[[condensates/nested-cv-prevents-tuning-optimism]]"
  - "[[condensates/optuna-warm-start-couples-gates]]"
  - "[[condensates/calib-per-fold-leakage]]"
  - "[[condensates/calib-parsimony-order]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[condensates/perm-validity-full-pipeline]]"
  - "[[equations/brier-decomp]]"
  - "[[equations/nested-cv]]"
  - "[[equations/optuna-tpe]]"
  - "[[equations/perm-test-pvalue]]"
---

# V2 Model Protocol

Factorial gate that selects the classifier family for the recipe locked at
[[protocols/v1-recipe]]. V2 opens after V1 has fixed `panel_composition`,
`panel_size`, and `ordering_strategy`, and before [[protocols/v3-imbalance]]
resolves weighting and downsampling.

V2 is structured as a **Pareto tournament on two orthogonal axes**:
discrimination (AUROC) and **calibration quality decoupled from
discrimination** (Brier reliability term, REL, per [[equations/brier-decomp]]).
All dominance and equivalence claims at V2 MUST be expressed on bootstrap
CIs over outer folds — point-estimate dominance is explicitly forbidden and
is the anti-pattern this protocol names and rejects (see §3.1 and Known
tensions §T-V2-01).

---

## 1. Pre-conditions

Before V2 `ledger.md` can be written:

1. **V1 must be locked.** [[protocols/v1-recipe]] must have emitted a
   `decision.md` with `recipe_id`, `panel_composition`, `panel_size`, and
   `ordering_strategy` locked under the fixed falsifier rubric (or, on
   Inconclusive, the non-dominated set of candidate recipes explicitly
   forwarded). V2 runs per surviving recipe; if V1 forwarded a Pareto set,
   V2 runs independently per recipe and its locks are scoped to each.
2. **V0 locks inherited.** `training_strategy`, `prevalent_train_frac`, and
   `train_control_per_case` are fixed cell-level configuration inherited
   from [[protocols/v0-strategy]]. V2 does not revisit V0 axes.
3. **Stage-1 model gate passlist available.** Per
   [[condensates/feature-selection-needs-model-gate]], V0's Stage-1
   permutation gate (`p < 0.05` per [[equations/perm-test-pvalue]], full-
   pipeline permutation per [[condensates/perm-validity-full-pipeline]])
   enumerates the models eligible for locking. V2 runs the full axis
   {LR_EN, LinSVM_cal, RF, XGBoost} for diagnostic breadth but a model that
   failed the Stage-1 gate CANNOT be locked — it can only appear on the
   Pareto front as a flagged candidate requiring a V1-level re-gate.
4. **Nested-CV budget fixed.** Per
   [[condensates/nested-cv-prevents-tuning-optimism]] and
   [[equations/nested-cv]]: $K_\text{out} = 5$, $R = 10$ repeats (or the
   20 selection seeds 100–119 as the outer replication structure), $K_\text{in} = 5$,
   $T = 200$ Optuna trials per cell. Dropping any factor below the
   condensate's boundary conditions requires a logged exception.
5. **Warm-start declared or refused.** Per
   [[condensates/optuna-warm-start-couples-gates]], if V2 warm-starts from
   V0 scout params, the V2 ledger MUST record the source gate fingerprint,
   `warm_start_top_k`, and the cohort / search-space delta. Default is
   warm-start OFF; any use is a declared coupling (see Known tensions
   §T-V2-04).
6. **Calibration family pinned (non-locking) for REL measurement.** V2
   measures REL on `oof_posthoc` predictions under a pinned reference
   calibrator per [[condensates/calib-per-fold-leakage]] — the V2 REL axis
   is NOT a calibration choice (that is V4). The pinned reference is
   `logistic_intercept` applied to OOF predictions, chosen because it is
   the parsimony floor of [[condensates/calib-parsimony-order]] and thus
   does not privilege any model's miscalibration pattern. The V2 REL
   ranking is conditional on this pin; a switched pin is re-evaluated at
   V4.
7. **Rulebook snapshot bound at gate entry.** Per SCHEMA.md §Versioning, V2
   ledger frontmatter must cite `rulebook_snapshot: "rb-v{x.y.z}"` at
   entry and remain immutable through decision.

---

## 2. Search space

V2 has a single explored axis (`model`) under all V0 and V1 locks. The
"search space" is the pairwise comparison structure and the Pareto
criterion, not a grid of configuration values — configuration was locked
upstream.

### 2.1 `model` ∈ {LR_EN, LinSVM_cal, RF, XGBoost}

All four families are always evaluated; the Stage-1 gate passlist (§1.3)
governs which may be **locked**, not which may be **run**. Running all
four preserves the diagnostic Pareto front for V6 ensemble comparison.

- **LR_EN** — L1/L2 penalized logistic regression. 2 hyperparameters
  (`C`, `l1_ratio`). Parsimony rank 1.
- **LinSVM_cal** — linear SVM with probability calibration via the pinned
  reference calibrator. 2 hyperparameters (`C`, `class_weight` slack).
  Parsimony rank 2.
- **RF** — Random Forest. 5–7 hyperparameters (`n_estimators`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `max_features`). Parsimony rank 3.
- **XGBoost** — gradient-boosted trees. 10+ hyperparameters
  (`learning_rate`, `n_estimators`, `max_depth`, `subsample`,
  `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`,
  `reg_lambda`). Parsimony rank 4.

The axis is closed: **do not invent additional models**. Ensemble
combinations are an explicit V6 follow-up per DESIGN.md §Design
Constraint: Single-Model Primary and are not in the V2 search space.

### 2.2 Axis-condensate mapping

| Axis element | Grounding condensate / equation | Load-bearing claim |
|---|---|---|
| Model eligibility (which families may lock) | [[condensates/feature-selection-needs-model-gate]] | Only Stage-1 permutation-passed models may lock; failed models may appear on the Pareto front but are flagged non-lockable |
| Inner-loop tuning inside each cell | [[equations/nested-cv]], [[condensates/nested-cv-prevents-tuning-optimism]] | Strict outer/inner isolation; 5×10×5×200 budget |
| Hyperparameter search inside inner loop | [[equations/optuna-tpe]] | TPE + MedianPruner default at $T \ge 50$; sampler seed pinned |
| Warm-start handling | [[condensates/optuna-warm-start-couples-gates]] | Declared or refused in ledger; $k/T \le 0.05$ when enabled; source gate fingerprint recorded |
| Discrimination axis of the Pareto front | AUROC on OOF predictions | Bootstrap CI over outer folds (20 selection seeds), 1000 resamples per SCHEMA §Fixed falsifier rubric |
| Calibration-quality axis of the Pareto front | [[equations/brier-decomp]] REL term only | Orthogonal to AUROC (resolution/refinement tracks AUROC; REL does not); DECISION_TREE_AUDIT §1.5 motivates the switch from raw Brier |
| REL measurement pin (non-locking) | [[condensates/calib-per-fold-leakage]], [[condensates/calib-parsimony-order]] | `oof_posthoc` + parsimony-floor calibrator to avoid per-fold optimism and to avoid privileging any model's miscalibration |

### 2.3 Pareto front construction

For each surviving V1 recipe, compute per-model:

- $\text{AUROC}_m$ = mean OOF AUROC across 20 selection seeds (100–119),
  aggregated per seed then averaged.
- $\text{REL}_m$ = mean OOF reliability component of the Brier
  decomposition ([[equations/brier-decomp]]) under the pinned reference
  calibrator, same seed aggregation.
- 95% bootstrap CIs on both metrics and on pairwise deltas, 1000
  resamples of outer folds (seeds as the paired resample unit), per
  SCHEMA.md §Fixed falsifier rubric.

A model $m$ is **Pareto non-dominated** if there is no other model $m'$
such that:

- $\text{AUROC}_{m'} > \text{AUROC}_m$ AND $\text{REL}_{m'} < \text{REL}_m$,
  with **both** deltas meeting the Direction criterion (|Δ| ≥ 0.02 on
  AUROC; |Δ| ≥ 0.005 on REL per §3.2 below) AND **both** 95% bootstrap CIs
  excluding 0.

Models on the Pareto front are candidates for locking (§3). Models
strictly dominated by a front member are eliminated from the lock and
from V6 ensemble input.

---

## 3. Success criteria

All V2 decisions use the fixed falsifier rubric from SCHEMA.md §Fixed
falsifier rubric. V2's lock is a **Dominance** claim per the rubric:
Direction holds independently on AUROC AND REL, both under bootstrap CI,
for one model against every other Pareto-front member. If Dominance does
not hold, V2 either locks by Equivalence + parsimony tiebreaker (§3.3) or
forwards the non-dominated set to V3/V4 (§4).

### 3.1 CI-based claims are mandatory; point-estimate dominance is forbidden

**Anti-pattern the protocol rejects:** "RF AUROC 0.724 > LinSVM_cal AUROC
0.723, therefore RF wins." This comparison reports a 0.001 difference
without uncertainty quantification and treats it as a Direction claim.
Under the fixed falsifier rubric (Direction requires |Δ| ≥ 0.02 AND 95%
bootstrap CI excluding 0), this is at most an **Inconclusive** claim and
likely an **Equivalence** claim — never Direction.

V2 `ledger.md` and `decision.md` MUST:

1. Report every pairwise comparison as a claim type under the SCHEMA
   rubric ({Direction, Equivalence, Dominance, Inconclusive}).
2. Report the point estimate AND the 95% bootstrap CI on every delta.
3. Declare the actual claim type explicitly under `## Actual claim type
   (per rubric)`.
4. NEVER use the phrase "X beats Y" or "X dominates Y" or "X wins over Y"
   when only point estimates have been compared. Language must name the
   claim type.

`DECISION_TREE_AUDIT.md` §1.2 documents the bug this rule corrects: the
pre-audit V2 implementation ranked models by point-estimate means without
CI, which would make an 0.001 AUROC gap locking-worthy. Any V2
`observation.md` or `decision.md` that emits a lock claim on a pairwise
delta without an accompanying 95% bootstrap CI is a rulebook violation
and MUST be rejected by the tension detector.

Implementation-level corollary: `validate_tree.R`'s V2 step
MUST compute CIs and MUST NOT lock on point-estimate means. The 2026-04-
08 resolution log entry (DECISION_TREE_AUDIT §5, "P1: V2/V3/V5
uncertainty — Implemented. Bootstrap CIs at all levels: V2 (Pareto, 1000
iters)") binds this requirement; this protocol inherits it and forbids
regression.

### 3.2 Claim types at V2

| Claim type | Criterion | V2 decision |
|---|---|---|
| **Direction** on one axis (A > B) | `|Δ| >= δ_axis` AND 95% bootstrap CI (1000 outer-fold resamples) excludes 0 | One axis separates A and B; check the other axis before calling Dominance |
| **Dominance** (A ≻ B, multi-axis) | Direction criterion holds on BOTH AUROC AND REL simultaneously | A dominates B on the Pareto front; B eliminated |
| **Equivalence** on one axis (A ≈ B) | `|Δ| < ε_axis` AND 95% bootstrap CI ⊂ [−δ_axis, δ_axis] | Tie on that axis; check the other axis for Direction |
| **Inconclusive** | Neither Direction nor Equivalence met on at least one axis | Both models forward as non-dominated (Pareto front alive) |

Axis-specific thresholds (non-negotiable per SCHEMA metric-specific rules
and the REL scale in [[equations/brier-decomp]] for low-prevalence
cohorts):

- **AUROC**: $\delta = 0.02$, $\varepsilon = 0.01$ (SCHEMA default).
- **REL**: $\delta = 0.005$, $\varepsilon = 0.002$. The tighter scale
  reflects REL's natural range on rare-event cohorts (UNC $\approx 0.003$
  on celiac per [[equations/brier-decomp]] §Worked reference), where an
  AUROC-scale 0.02 threshold would be a non-informative wide band.
  Documented as a V2-specific metric rule below the SCHEMA default.

**Counts** (panel size $p$ already locked at V1; not a V2 axis) are not
compared at V2.

### 3.3 Parsimony tiebreaker under Equivalence

When the Pareto front reduces to a set of models that are mutually
Equivalent on BOTH axes (or when Direction holds on one axis and
Equivalence on the other in ways that don't resolve to Dominance), apply
the **model complexity ordering**:

$$\text{LR\_EN} \;(1) \;\prec\; \text{LinSVM\_cal} \;(2) \;\prec\; \text{RF} \;(3) \;\prec\; \text{XGBoost} \;(4)$$

Ordering rationale (from DESIGN.md §Parsimony Ordering): LR_EN is a
single linear model with L1/L2 penalty (fewest effective parameters).
LinSVM_cal adds a kernel-like margin objective. RF is an ensemble of
trees. XGBoost is a sequential ensemble with the most hyperparameters.
Under Equivalence, lock the lowest-complexity model on the Pareto front.

Parsimony-tie resolution follows `[[condensates/parsimony-tiebreaker-when-equivalence]]`
with this axis's order: **LR_EN ≺ LinSVM_cal ≺ RF ≺ XGBoost**. The
meta-condensate is the parent rule that governs Equivalence-triggered
tiebreakers across the factorial; the V2-specific complexity order above
is the local instantiation invoked here.

**Status of the tiebreaker:** DESIGN.md codifies this ordering and
`DECISION_TREE_AUDIT.md` §1.1 / §5 (P2: V2/V3 parsimony —
"Implemented. MODEL_COMPLEXITY: LR_EN(1) < LinSVM_cal(2) < RF(3) <
XGBoost(4)") adopts it. **There is no condensate that grounds the
ordering as a falsifiable rule.** V2 uses it as a design-level tiebreaker
pending promotion to a condensate. See Known tensions §T-V2-03 and the
TODO for `condensates/parsimony-tiebreaker.md`.

### 3.4 Dominance composition (the V2 lock gate)

V2 locks exactly one model iff a Dominance claim holds: one model on the
Pareto front Dominates every other front member (Direction on both AUROC
and REL, both CIs excluding 0). This is the **happy path**.

When Dominance does not hold, the Pareto front has more than one
non-dominated member. V2 then branches by claim structure:

1. **Equivalence across all front members on both axes.** Apply §3.3
   parsimony tiebreaker. Lock the lowest-complexity front member.
2. **Mixed Direction and Equivalence without Dominance.** Example: three
   front members where A > B on AUROC (Direction) but A ≈ B on REL
   (Equivalence), with C elsewhere on the front. The rubric does not
   emit a Dominance claim; lock deferred.
3. **Any Inconclusive.** At least one pairwise comparison is
   Inconclusive. Lock deferred.

Cases 2 and 3 forward to V3/V4 per §4.1.

### 3.5 Stage-1 lockability check

Even when Dominance holds, the Dominant model must be on the V0 Stage-1
permutation passlist (§1.3). If a Dominant front member failed Stage-1,
V2 does NOT lock it — it flags the contradiction (a recipe-restricted
model that passed V1 but failed the whole-feature-universe permutation
gate) and forwards to V4/V5 for cohort-level investigation. This is a
defensive guard against permutation-gate failure modes documented in
[[condensates/feature-selection-needs-model-gate]].

---

## 4. Fallbacks

### 4.1 Forward the full Pareto front to V3/V4 when Dominance does not hold

When V2 cannot lock a single model via Dominance or Equivalence +
parsimony, **all non-dominated models** advance to V3 (imbalance) as
candidates. V3 runs its 3×3 weighting × downsampling grid **independently
per forwarded model**. The rationale: model × imbalance-handling
interactions may reorder the Pareto front. V4 (calibration) similarly
runs per surviving model. V5 (seed-split confirmation) may still
eliminate models whose V3/V4 winners fail the confirmation check.

This preserves the Pareto front rather than collapsing it prematurely and
is the `MASTER_PLAN.md` §V2 intent under the decision tree.

### 4.2 Widen trials before declaring Inconclusive (single-dissenter rule)

If Inconclusive is driven by a **single** ambiguous pairwise comparison
(one pair at the boundary between Direction and Equivalence on one axis,
all other pairs cleanly resolved), re-run the two affected models at
$T = 400$ trials (twice the default) before declaring Inconclusive. Per
[[condensates/nested-cv-prevents-tuning-optimism]] boundary condition
("more complex spaces may require $T = 400$"), a budget widening for
XGBoost or RF is within-spec. This is a **power check, not a
hyperparameter-peeking violation**, because the axis under test is model
family, not hyperparameters within a family.

Widening for LR_EN or LinSVM_cal beyond their default budget is NOT
permitted — their search spaces saturate well below $T = 200$ and extra
trials add noise.

### 4.3 Stage-1 failure on the Dominant candidate

Per §3.5, if Dominance points to a Stage-1-failed model, V2 does not lock.
Record the contradiction and forward the full front (including the
Stage-1-failed front leader, flagged) to V4. Resolution is a cohort-level
question: either the permutation gate was miscalibrated on this recipe or
the Dominance claim itself reflects a recipe-specific signal that does
not survive whole-feature-universe permutation. V4 is the wrong gate to
resolve this — flag to human review.

### 4.4 Pinned reference calibrator is misspecified

If the V2 REL axis shows suspicious REL inversion relative to expected
model families (e.g., XGBoost returns REL < LR_EN by a large margin,
suggesting the `logistic_intercept` pin is under-fitting XGBoost's
non-monotonic miscalibration), re-run with `none` calibration as a
sensitivity check. If REL ordering changes, flag the recipe for V4
attention and forward the full front. Do NOT substitute a different pin
mid-V2 — the pin is a declared prior, not a tunable axis.

### 4.5 Missing or corrupt CI artefacts

If the bootstrap CI artefact for any V2 pairwise delta is missing or
malformed, V2 does NOT fall back to point-estimate ranking. The correct
response is to re-run the bootstrap (seed fixed, 1000 resamples). A V2
run that cannot produce CIs is a failed V2 run and must be re-executed,
not downgraded.

---

## 5. Post-conditions

On V2 `decision.md` write:

1. **Lock (on Dominance).** If §3.4 case A holds:
   - `model = <locked value>` — the single Pareto-dominant model.
   - `claim_type = Dominance` — AUROC Direction AND REL Direction, both
     CIs excluding 0.
   - `pareto_front_at_lock = [<locked model>]` — singleton.
2. **Lock (on Equivalence via parsimony).** If §3.4 case 1 holds:
   - `model = <parsimony winner>` — the lowest-complexity front member.
   - `claim_type = Equivalence` — all pairs within front met Equivalence
     on both axes.
   - `parsimony_ordering_invoked = "LR_EN < LinSVM_cal < RF < XGBoost"`
     — explicit citation.
   - `pareto_front_at_lock = [<all front members>]` — recorded for V6
     ensemble comparison input.
3. **Forward (on mixed / Inconclusive).** If §3.4 cases 2 or 3 hold:
   - `model = null` — no lock.
   - `non_dominated_model_set = [<front members>]` — forwarded to V3.
   - `claim_type = Inconclusive` OR `claim_type = Mixed (Direction + Equivalence)`
     — named explicitly.
4. **Pairwise comparison table.** Always recorded, regardless of
   outcome. For each of the $\binom{4}{2} = 6$ model pairs: AUROC delta,
   REL delta, AUROC 95% CI, REL 95% CI, claim type per axis.
5. **Claim type MUST be declared explicitly.** Per SCHEMA.md §Fixed
   falsifier rubric, `decision.md` under `## Actual claim type (per
   rubric)` states exactly one of {Dominance, Direction, Equivalence,
   Inconclusive, Mixed} for V2 as a whole AND for each pairwise axis.
   The ledger's `## Predictions with criteria` section must be matched
   one-to-one against these claim types.
6. **Stage-1 lockability audit.** For the Dominant (or parsimony-winning)
   model, confirm Stage-1 passlist membership and log the p-value. If
   §3.5 triggered (Dominance on a Stage-1-failed model), log the
   contradiction and mark lock deferred.
7. **Pareto-front diagnostics for V6.** Record the full 4-model
   AUROC × REL scatter with CI ellipses — this is V6's input regardless
   of whether V2 locked.
8. **Tensions auto-populate.** Any pairwise comparison where the V2
   ledger predicted Direction but observation returned Equivalence or
   Inconclusive (or vice versa) writes a tension to
   `projects/<name>/tensions/rule-vs-observation/`, keyed on the
   relevant condensate. The tension detector does this; V2 does not
   write tensions manually.

What advances to V3:

- The locked `model` (on Dominance / parsimony) — V3's 9-cell imbalance
  grid runs for the single locked model. OR
- The `non_dominated_model_set` (on forward) — V3 runs its 9-cell grid
  independently per model in the set.
- All V0 and V1 locks inherited unchanged.
- The pinned reference calibrator fingerprint (V4 re-evaluates).

What remains open at V2:

- Calibration strategy (V4 — this gate's REL pin is not a lock).
- Weighting × downsampling imbalance handling (V3).
- Seed-split confirmation (V5).
- Ensemble comparison (V6 — informational, Pareto-front-consuming).

---

## Known tensions

First-class tensions for this protocol, documented here so the tension
detector can link observations back to them.

### T-V2-01: Point-estimate dominance (resolved by mandate)

`DECISION_TREE_AUDIT.md` §1.2 identifies the pre-audit V2 implementation
as operating on point-estimate Pareto dominance without CIs: "Two models
0.001 apart in Brier will show one dominating the other, even if that
difference is within seed-to-seed variance." §3.1 of this protocol
mandates CI-based claims for every pairwise delta and explicitly names
the anti-pattern ("RF AUROC 0.724 > LinSVM_cal 0.723 therefore RF
wins"). The 2026-04-08 audit resolution log
(DECISION_TREE_AUDIT.md §5, P1 uncertainty item) adopted bootstrap CIs
at V2 with 1000 iterations. This protocol adopts that fix and forbids
regression. Any V2 observation.md lacking CIs on pairwise deltas is a
rulebook violation.

### T-V2-02: Pareto axis collinearity (resolved by decomposition)

`DECISION_TREE_AUDIT.md` §1.5: "AUROC and Brier are correlated: Brier =
reliability + refinement, and refinement tracks discrimination (AUROC).
The Pareto front is nearly one-dimensional." §2.2 and §2.3 of this
protocol resolve this by replacing the y-axis from raw Brier to the REL
component of the Brier decomposition per [[equations/brier-decomp]] §1.
REL is orthogonal to AUROC by construction (REL measures the squared
deviation between predicted and observed bin frequencies; resolution /
refinement captures discrimination). The 2026-04-08 resolution log
("P1: V2 Pareto collinear axes — Implemented. Switched from AUROC vs
Brier to AUROC vs reliability (orthogonal).") adopted this switch;
this protocol inherits it. Residual risk: REL is sensitive to the
calibrator pin (§2.2 row 7); V2 pins `logistic_intercept` for
neutrality, but a misspecified pin can still collapse the axis (Fallback
§4.4).

### T-V2-03: Parsimony tiebreaker has no condensate (TODO)

The model complexity ordering (LR_EN < LinSVM_cal < RF < XGBoost) is
codified in DESIGN.md §Parsimony Ordering and adopted by the 2026-04-08
audit resolution log (P2 parsimony item), but **no condensate in the
rulebook grounds it as a falsifiable rule**. V2 uses it as a design-
level tiebreaker pending promotion.

**TODO:** Draft `condensates/parsimony-tiebreaker.md` with:
- Claim: "under Equivalence, selecting the lower-complexity model family
  yields equal-or-better TEST AUROC with 95% CI crossing 0."
- Mechanism: bias-variance tradeoff at small n (simpler families have
  lower variance out-of-sample).
- Actionable rule: V2 / V3 / V4 tiebreaker hierarchy and the complexity
  orderings for each axis.
- Falsifier: direction claim — if complex models beat simpler ones on
  TEST (Direction on AUROC) despite Equivalence on OOF, the ordering is
  weakened across 3+ datasets → retire.
- Sources: DESIGN.md §Parsimony Ordering, DECISION_TREE_AUDIT.md §1.1
  / §5 P2 resolution, Van Calster et al. (2019) on bias-variance in
  clinical prediction.

Gap logged for promotion when V2's first observation on a second dataset
provides evidence.

### T-V2-04: Warm-start coupling to V0 (declare or refuse)

Per [[condensates/optuna-warm-start-couples-gates]], V2 must declare
whether it warm-starts from V0 scout params. The celiac pipeline exposes
`warm_start_params_file` + `warm_start_top_k` as explicit switches and
`extract_scout_params.py` as dedicated tooling, so V2 inherits the
coupling decision. §1.5 of this protocol requires the V2 ledger to
record source gate fingerprint, $k/T$ ratio, and cohort / search-space
delta when warm-start is ON. Default is OFF for V2 because V2's recipe
is V1-locked, which narrows the feature space relative to V0's
representative recipes — search-space mismatch per the condensate's
mechanism §2. When V2 enables warm-start, the V2 tension ledger flags
it for review even if the run completed.

### T-V2-05: REL pin is a declared prior, not a lock

§1.6 pins `logistic_intercept` on OOF predictions as the V2 REL
measurement reference. This is not a calibration lock (V4's job) but a
measurement contract. A misspecified pin can collapse the V2 Pareto axis
(see T-V2-02 residual risk and Fallback §4.4). If V4 later locks a
different calibrator and the V2 Pareto ordering under the V4-locked
calibrator disagrees with the V2 lock, V5 (seed-split confirmation) is
the adjudicator. V2 does NOT re-open under V4's lock; the
forward-recorded `pareto_front_at_lock` field enables V4 / V5 to audit
the decision trail.

### T-V2-06: Stage-1 failure on Dominant candidate (escalate, don't downgrade)

§3.5 / §4.3: if the Dominant front member failed the V0 Stage-1
permutation gate, V2 does not lock it and does not substitute the
second-best front member. Instead, the contradiction is flagged for
human review. Rationale: the Dominance claim is genuine on this recipe's
feature subset, but the whole-feature-universe permutation gate rejected
the model — this is a cohort-level signal (e.g., the recipe sliced the
feature universe in a way that isolated a weak-model signal) and is a
candidate rulebook update to
[[condensates/feature-selection-needs-model-gate]]. Logged as a first-
class tension for cohort investigation, not a V2 lock decision.

---

## Sources

- `operations/cellml/DESIGN.md` — V2 positioning in §Validation Decision
  Tree, model complexity ordering in §Parsimony Ordering, Pareto dominance
  axes in §V2
- `operations/cellml/MASTER_PLAN.md` — V2 definition and decision
  architecture
- `operations/cellml/DECISION_TREE_AUDIT.md` — V2 point-estimate bug
  (§1.2), Pareto collinearity (§1.5), parsimony undefined (§1.1),
  resolution log (§5: P1 uncertainty, P1 Pareto, P2 parsimony)
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
- `validate_tree.R` — reference V2 implementation with bootstrap CIs
  (2026-04-08 audit resolution)
