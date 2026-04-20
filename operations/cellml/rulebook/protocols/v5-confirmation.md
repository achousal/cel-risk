---
type: protocol
gate: v5-confirmation
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v4-calibration"
  - "calibrated_oof_predictions: from V4-locked calibrator on selection seeds 100-119"
  - "seed_quarantine_manifest: validation seeds 120-129 must have never been inspected at V0-V4"
outputs:
  - "locks: [holdout_generalization_verdict]"
axes_explored: []  # no axes — V5 is a validation gate, not a selection gate
axes_deferred: []
depends_on:
  - "[[condensates/three-way-split-prevents-threshold-leakage]]"
  - "[[condensates/threshold-on-val-not-test]]"
  - "[[condensates/nested-cv-prevents-tuning-optimism]]"
  - "[[condensates/fixed-spec-for-screening]]"
  - "[[equations/fixed-spec-threshold]]"
  - "[[equations/nested-cv]]"
  - "[[equations/brier-decomp]]"
  - "[[equations/stratified-split-proportions]]"
  - "[[equations/perm-test-pvalue]]"
---

# V5 Confirmation Protocol

Held-out seed validation. V5 is the final gate in the cel-risk tree. It
does **not** explore any axes — the pipeline is fully locked by V4 — and
its sole purpose is to verify that the (V0 ∘ V1 ∘ V2 ∘ V3 ∘ V4) locked
configuration generalizes from the 20 selection seeds (100–119) to 10
held-out validation seeds (120–129). Per
[[condensates/three-way-split-prevents-threshold-leakage]], the threshold
is selected on VAL and applied to TEST without re-optimization.
`MASTER_PLAN.md` §V5 and `DESIGN.md` §V5 are the load-bearing spec
references; this protocol writes the gate-level falsifier rubric around
them.

---

## 1. Pre-conditions

V5 runs only when all of the following are in place. Any failure here
does not "fall back" to a relaxed V5 — it bounces back to the appropriate
prior gate.

1. **V4 must be locked.** [[protocols/v4-calibration]] must have emitted
   `decision.md` naming `calibration_strategy ∈ {logistic_intercept, beta,
   isotonic}` and its claim type. If V4 returned Inconclusive and locked
   `logistic_intercept` by parsimony, that IS a valid V4 lock for V5
   purposes — V5 does not re-open it.
2. **Full pipeline locked in order V0 → V4.** Each of {training_strategy,
   control_ratio, recipe, panel_composition, panel_size, ordering_strategy,
   model, weighting, downsampling, calibration_strategy} has an explicit
   locked value from its respective gate. V5 inherits all ten; it does
   not re-select any.
3. **Selection/validation seed split enforced.** Per
   [[protocols/v0-strategy]] §1 and `MASTER_PLAN.md` §V5, seeds 100–119
   (20 seeds) are the selection set used at V0–V4 for all Direction /
   Equivalence claims; seeds 120–129 (10 seeds) are the validation
   (held-out) set, which V5 is running for the first time. This partition
   is structural — V5 cannot be entered if any prior gate reported
   results computed on seeds 120–129.
4. **Seed quarantine attestation.** The V5 ledger MUST include a
   `quarantine_attestation` field certifying that the validation seeds
   were never inspected — specifically:
   - no model fits on seeds 120–129 appeared in V0–V4 decision inputs,
   - no tuning (Optuna trials, calibrator parameters, threshold scans)
     consumed seeds 120–129,
   - compilation outputs used as V1–V4 inputs were filtered to seeds
     100–119 at the compile step, with a checksum recorded in the gate
     ledger.

   A failed attestation means the TEST (validation-seed) quarantine was
   broken and V5 cannot run — the affected prior-gate decisions must be
   re-emitted on the 20-seed set only before V5 is entered.
5. **TEST split within each seed never peeked at.** In addition to the
   seed-level quarantine, the 25% TEST slice per split (distinct from
   the 25% VAL slice per [[condensates/three-way-split-prevents-threshold-leakage]]
   and [[equations/stratified-split-proportions]]) must not have
   contributed to any V0–V4 fit, tuning, or threshold decision. The
   three-way split is the structural invariant that makes V5's TEST
   metric an unbiased generalization estimate.
6. **Calibrated OOF predictions on selection seeds are available.** V4
   emits calibrated per-sample posteriors for seeds 100–119. V5 regenerates
   the analogous predictions on seeds 120–129 by running the V0–V4 locked
   pipeline against those splits — a "replay" of the pipeline, NOT a
   re-tune. The sampler seed, Optuna storage, and V4 calibrator parameters
   are all fixed from the selection-set run.
7. **Rulebook snapshot bound at gate entry.** Per SCHEMA.md §Versioning.

---

## 2. Search space

**V5 explores no axes.** `axes_explored: []` in frontmatter is the
correct value.

This is the structural distinction between V5 and every prior gate. V0
explored training strategy. V1 explored recipe. V2 explored model. V3
explored imbalance. V4 explored calibration. V5 does NOT explore — it
only verifies that the axis values locked at V0–V4 continue to hold on
seeds the selection process never saw.

This non-exploration property is load-bearing. If V5 had an axis, V5's
outputs would be another tuning surface, and the 10 validation seeds
would themselves become a selection set — requiring yet another
held-out confirmation below V5, which does not exist. The 30-seed
design (20 select + 10 validate) is a single-level held-out test; it is
not a recursive structure.

### 2.1 What V5 measures (all on the V4-locked pipeline replayed on seeds 120–129)

| Measurement | Split | Purpose |
|---|---|---|
| Selection-set AUROC | seeds 100–119, TEST slice | Point estimate from the V0–V4 lock, for comparison |
| Validation-set AUROC | seeds 120–129, TEST slice | Held-out generalization estimate |
| Selection-set REL (post-V4 calibrator) | seeds 100–119, VAL slice | Calibration quality at V4 lock |
| Validation-set REL | seeds 120–129, VAL slice | Held-out calibration stability |
| Threshold $\tau_\text{spec}$ per seed | seeds 120–129, VAL slice | Selected on VAL per [[equations/fixed-spec-threshold]] |
| Sensitivity at $\tau_\text{spec}$ | seeds 120–129, TEST slice | Unbiased operating-point sensitivity per [[condensates/threshold-on-val-not-test]] |

Each measurement is computed per-seed and aggregated across seeds
(mean + 95% bootstrap CI over the 10 validation seeds, or over the 20
selection seeds where the selection-set number is needed).

### 2.2 Threshold rule (inherited from ADR-010; not a V5 axis)

The threshold objective — fixed-specificity at $\mathrm{spec}^{*} = 0.95$,
per [[condensates/fixed-spec-for-screening]] and
[[equations/fixed-spec-threshold]] — is pre-registered and declared in
the V5 ledger. It is NOT selected by V5. Alternative objectives
(Youden's $J$, max $F_1$) are ADR-010 documented options, but swapping
objectives after peeking at TEST is a search-space violation
([[condensates/threshold-on-val-not-test]] §Actionable rule:
"Pre-registration: the threshold rule ... is declared in the V4 ledger;
swapping rules after peeking at TEST is a search-space violation").

For each validation seed $s \in \{120, \ldots, 129\}$:

$$\tau_\text{spec}^{(s)} = Q_{1 - \mathrm{spec}^{*}}\!\left( \{ \hat{p}_i^{\mathrm{cal}, (s)} : y_i = 0, i \in \mathrm{VAL}_s \} \right)$$

and sensitivity is read on TEST$_s$ at that $\tau_\text{spec}^{(s)}$
([[condensates/threshold-on-val-not-test]]).

### 2.3 Axis-condensate mapping

| Decision | Grounding | Load-bearing claim |
|---|---|---|
| Seed split 20 selection + 10 validation | `MASTER_PLAN.md` §V5, `DECISION_TREE_AUDIT.md` §3.7 resolution | Tree cross-validation requires held-out seeds; 20/10 was adopted 2026-04-08 |
| Three-way TRAIN/VAL/TEST at 50/25/25 | [[condensates/three-way-split-prevents-threshold-leakage]], [[equations/stratified-split-proportions]] | Structural invariant from V0; TEST absorbs no post-training fit |
| Threshold on VAL, evaluate on TEST | [[condensates/threshold-on-val-not-test]], [[equations/fixed-spec-threshold]] | Selecting $\tau$ on TEST produces optimistically biased sensitivity |
| `spec*` = 0.95 default | [[condensates/fixed-spec-for-screening]], ADR-010 | Screening operating-point convention |
| No retuning at V5 | [[condensates/nested-cv-prevents-tuning-optimism]] applied at the decision-tree level | Tuning on the evaluation set produces upward-biased metrics; V5 is the evaluation set for V0–V4 decisions |
| Calibration-aware REL report | [[equations/brier-decomp]] | V4 locked the calibrator on selection seeds; V5 confirms REL stability on validation seeds |

---

## 3. Success criteria

All V5 decisions use the fixed falsifier rubric in SCHEMA.md. The primary
metric is $\Delta\mathrm{AUROC}_\text{set}$ between the selection-set
AUROC and the validation-set AUROC, computed on the TEST slice of each
seed and aggregated across seeds.

### 3.1 Claim types

| Claim type | Criterion | V5 verdict |
|---|---|---|
| **Equivalence** (selection ≈ validation) | $\|\Delta\mathrm{AUROC}_\text{set}\| < 0.01$ AND 95% bootstrap CI $\subset [-0.02, 0.02]$ | **PASS** — generalization confirmed |
| **Direction (selection >> validation)** | $\mathrm{AUROC}_\text{sel} - \mathrm{AUROC}_\text{val} \ge 0.02$ AND 95% CI excludes 0 | **FAIL** — selection-set optimism detected |
| **Direction (validation >> selection)** | $\mathrm{AUROC}_\text{val} - \mathrm{AUROC}_\text{sel} \ge 0.02$ AND 95% CI excludes 0 | **WARN** — validation-set outperformance; diagnose before trust |
| **Inconclusive** | Neither Equivalence nor Direction met | **WARN** — power-limited; report as-is, do not treat as PASS |

The expected-case V5 verdict is PASS via Equivalence: the locked pipeline
should produce substantively identical AUROC on held-out seeds. A Direction
(sel >> val) outcome is the diagnostic signal that the V0–V4 selection
process over-fit the 20 selection seeds — the gates' Direction /
Equivalence claims were driven by seed-specific noise, not by
reproducible pipeline behavior.

### 3.2 Bootstrap unit (seed, paired across sel/val sets)

- For selection-set AUROC: mean AUROC across seeds 100–119 on TEST slice,
  with 95% CI from bootstrap over seeds (1000 resamples).
- For validation-set AUROC: mean AUROC across seeds 120–129 on TEST slice,
  with 95% CI from bootstrap over seeds.
- For $\Delta\mathrm{AUROC}_\text{set}$: the two bootstrap distributions
  are independent (different seed pools), so the CI is computed on
  $\bar{A}_\text{sel}^{*} - \bar{A}_\text{val}^{*}$ drawn as paired
  bootstrap samples within each pool, then differenced. Equivalently,
  the 95% CI is constructed via the delta distribution
  $\{\bar{A}_\text{sel}^{*(b)} - \bar{A}_\text{val}^{*(b)}\}_{b=1}^{1000}$.

### 3.3 REL stability as a secondary check (advisory)

Compute $\Delta\mathrm{REL}_\text{set} = \mathrm{REL}_\text{sel} -
\mathrm{REL}_\text{val}$ on VAL slices using the V4-locked calibrator.
Per the tightened V4 band (§3.1 of [[protocols/v4-calibration]]):
$\|\Delta\mathrm{REL}_\text{set}\| \ge 0.005$ with CI excluding 0 is a
REL-Direction signal and is reported alongside the AUROC verdict. A REL
Direction in the presence of an AUROC Equivalence means the model
discriminates the same on validation seeds but calibrates differently —
a softer failure than full generalization loss, but still a reason to
flag the V4 calibrator lock in `tensions.md`.

### 3.4 Sensitivity-at-spec confirmation (primary operational metric)

Per [[condensates/threshold-on-val-not-test]], for each validation seed
$s \in \{120, \ldots, 129\}$:

1. Select $\tau_\text{spec}^{(s)}$ on VAL$_s$ per
   [[equations/fixed-spec-threshold]] at $\mathrm{spec}^{*} = 0.95$.
2. Evaluate sensitivity on TEST$_s$ at that $\tau_\text{spec}^{(s)}$.
3. Report mean and 95% CI of $\mathrm{sens}_{0.95}^{(s)}$ across the 10
   validation seeds.

This is the final deploy-time sensitivity estimate. The per-seed
structure (select τ on VAL$_s$, evaluate on TEST$_s$) preserves
between-seed variance as a genuine generalization signal. A single
τ_global selected on pooled VAL across seeds would under-represent
between-seed variance and is explicitly rejected.

### 3.5 Pass/fail/warn routing

| Condition | Verdict | Next step |
|---|---|---|
| §3.1 Equivalence AND §3.3 $\|\Delta\mathrm{REL}\| < 0.005$ | PASS | Lock the full pipeline; proceed to holdout deployment |
| §3.1 Equivalence AND §3.3 $\|\Delta\mathrm{REL}\| \ge 0.005$ Direction | PASS-with-flag | Pipeline locks; REL instability is a V4 tension routed to `tensions.md` |
| §3.1 Direction (sel >> val) | FAIL | §4 routes to selection-set optimism tension; do NOT retune |
| §3.1 Direction (val >> sel) | WARN | §4 routes to cohort / split-leakage diagnostic |
| §3.1 Inconclusive | WARN | Report verdict with explicit power caveat; do NOT treat as PASS |

---

## 4. Fallbacks

V5 has no fallback that involves "relaxing" the generalization bar or
re-running the tree. Fallbacks route failure to **diagnostic** work, not
to a new factorial pass.

### 4.1 FAIL: selection-set optimism

When §3.1 returns Direction (sel >> val) with $\Delta \ge 0.02$ and CI
excluding 0:

1. **Do NOT retune on V5.** Retuning on validation seeds would break the
   quarantine and make V5 a second selection pass rather than a
   confirmation. Any retuning that consumes seeds 120–129 invalidates all
   subsequent held-out claims on this cohort.
2. **Do NOT relock any prior gate on validation seeds.** The V0–V4 locks
   were made under the fixed falsifier rubric on the 20-seed selection
   set; the fact that they over-fit that set is the finding, not a
   relaxation opportunity.
3. **Write a project-level tension** naming "selection-set optimism
   detected at V5 on celiac" and routing to
   `projects/celiac/tensions/rule-vs-observation/`. The tension is
   keyed on the gate whose locked decision shows the largest per-seed
   variance between selection and validation — diagnostic, not
   prescriptive.
4. **Escalate to rulebook update.** A selection-set optimism finding on
   the celiac cohort is evidence that the 20-seed selection budget may
   be insufficient for stable V1–V4 decisions (see Known tensions §T-V5-01).
   Accumulating this across ≥3 cohorts is a condensate-promotion signal
   (candidate: "20 selection seeds is an insufficient sample for stable
   multi-gate factorial decisions at rare-event prevalence").
5. **Deploy-decision consequence.** A FAIL verdict means the locked
   pipeline is NOT cleared for clinical deployment from this factorial.
   The correct path forward is a new cohort or a re-scoped factorial
   with more selection seeds, not a V5 rerun.

The "do not retune at V5" rule is non-negotiable — it is the failure mode
this protocol is specifically designed to name and forbid. Per
[[condensates/nested-cv-prevents-tuning-optimism]] (applied at the
decision-tree level rather than at the within-cell CV level), tuning on
the evaluation set produces an upward-biased estimate. If V5 becomes a
tuning surface, V0–V5 collectively have no held-out estimate at all.

### 4.2 WARN: validation >> selection

If §3.1 returns Direction with validation AUROC meaningfully HIGHER than
selection AUROC, something structural is suspicious — possibly:

- split-leakage on seeds 120–129 (features or labels contaminated across
  TRAIN/VAL/TEST for those seeds),
- non-stationarity between seed ranges (e.g., if seeds index cohort
  subsets rather than random shuffles — not the case for celiac, but
  possible on other cohorts),
- a bug in the pipeline replay that accidentally improves the model on
  validation seeds (e.g., a different Optuna best-trial pick due to
  sampler reseeding).

V5 writes a WARN verdict and a diagnostic tension. Deployment decision
is held until the diagnostic resolves.

### 4.3 WARN: Inconclusive

If §3.1 returns Inconclusive (|Δ| between 0.01 and 0.02 or CI not
contained in $[-0.02, 0.02]$), V5 does NOT elevate Inconclusive to PASS
by relaxing the band. The result is reported honestly as
"generalization inconclusive at 10 validation seeds" and the decision
to deploy the pipeline is an explicit human call with the
under-powered verdict in hand. The condensate path forward is to
increase the validation seed set on a future cohort; V5 does not do
this retroactively.

### 4.4 No quarantine-breach recovery

If the §1 quarantine attestation fails — i.e., it turns out the
validation seeds were peeked at during V0–V4 — V5 cannot run honestly.
The only valid recovery is a fresh cohort or a fresh split-seed pool
with a clean quarantine. V5 does not have a "partial peek" branch.

### 4.5 Permutation-test replay (optional diagnostic)

On a FAIL or Inconclusive verdict, a full-pipeline permutation test on
seeds 120–129 ([[equations/perm-test-pvalue]] with $B \ge 1000$, full
nested-CV inner pipeline per
[[condensates/nested-cv-prevents-tuning-optimism]]) provides an
independent check on whether the validation-set performance is above
chance. This is diagnostic, not a gate — a passing permutation on
validation seeds does NOT rescue a FAIL verdict.

---

## 5. Post-conditions

On V5 `decision.md` write:

1. **`holdout_generalization_verdict`** — one of {PASS, PASS-with-flag,
   FAIL, WARN}, with the claim type (§3.1) declared explicitly.
2. **Full locked pipeline recorded** — the complete V0 → V4 chain, with
   each lock's gate version and claim type, assembled as the single
   "locked pipeline" identifier that V5 validated:
   - training_strategy
   - prevalent_train_frac (if applicable)
   - train_control_per_case
   - recipe_id
   - panel_composition (protein list)
   - panel_size
   - ordering_strategy
   - model
   - weighting
   - downsampling
   - calibration_strategy
3. **Primary metrics table** — selection-set AUROC (mean + 95% CI across
   20 seeds), validation-set AUROC (mean + 95% CI across 10 seeds),
   $\Delta\mathrm{AUROC}_\text{set}$ (point + CI), plus REL and
   sensitivity-at-spec (0.95) analogs.
4. **Per-seed validation table** — 10 rows (seeds 120–129), each with
   TEST AUROC, VAL threshold $\tau_\text{spec}^{(s)}$, TEST sensitivity
   at $\tau_\text{spec}^{(s)}$, and VAL/TEST REL.
5. **Claim type MUST be declared explicitly.** Per SCHEMA.md, state
   Equivalence / Direction(sel>>val) / Direction(val>>sel) / Inconclusive
   under `## Actual claim type (per rubric)`.
6. **Tensions and diagnostics.** Any WARN or FAIL verdict routes a
   tension note to `projects/<name>/tensions/rule-vs-observation/`.
   PASS-with-flag routes a V4 REL-stability tension. Clean PASS routes
   no tensions.
7. **Deploy-readiness stamp (PASS only).** On PASS, the V5 `decision.md`
   emits a `pipeline_deploy_ready: true` field. On any non-PASS verdict
   this field is `false` regardless of how close the metrics were.

What advances beyond V5:

- On PASS: the full locked pipeline is cleared for the holdout
  confirmation step (`MASTER_PLAN.md` §Phase 5) — an independent
  cohort fit. V5 is NOT the holdout; it is the decision-tree
  cross-validation that guards against selection-set overfitting
  before committing HPC wallclock to an independent cohort.
- On FAIL / WARN / Inconclusive: no downstream automation. The
  diagnostic tension is human-adjudicated before any further
  factorial runs on this cohort.

What V5 does NOT produce:

- No new axis locks. V5 is not a selection gate.
- No deployment threshold below the pre-registered $\mathrm{spec}^{*}$
  — the threshold is the per-seed $\tau_\text{spec}^{(s)}$ selected on
  VAL$_s$; deployment requires a new fit on the full cohort (outside the
  30-seed factorial) with its own VAL threshold selection.
- No V6 input. V5's PASS verdict permits V6 ensemble comparison to
  proceed informationally (DESIGN.md §V6), but V6 is a separate
  post-tree step that does not feed back into V5.

---

## Known tensions

Acknowledged per `DECISION_TREE_AUDIT.md`; surfaced here because V5
inherits and cannot resolve them internally.

### T-V5-01: 20 selection seeds may be insufficient

`DECISION_TREE_AUDIT.md` §3.7 action (2026-04-08) adopted the 20/10
split per recommendation, but `MASTER_PLAN.md` §Open Questions #4 is
explicit: "20/10 seed split means 33% fewer seeds for selection. Is 20
seeds sufficient for stable V1–V4 decisions?" is an **unresolved
question**, not a settled one. V5's FAIL mode is the direct empirical
answer: if V5 shows selection >> validation, the answer was "no" on
this cohort. A systematic FAIL across cohorts would promote a
condensate mandating a larger selection seed pool. V5 does NOT rerun
with more selection seeds on a FAIL — that would require restructuring
the cohort seed pool, not a V5 protocol change.

### T-V5-02: Quarantine-breach risk (the retune-on-failure temptation)

On a FAIL verdict, the natural (and wrong) instinct is to retune one or
more prior gates on validation seeds to "fix" the generalization. §4.1
forbids this explicitly — it is the failure mode V5 is designed to name
and forbid. The audit log (`DECISION_TREE_AUDIT.md` §Resolution log)
records the 20/10 split as the mechanism that makes the quarantine
enforceable; this protocol operationalizes the enforcement at the
decision level. The risk is real because HPC wallclock investment in
V0–V4 is large and the pressure to recover the run is correspondingly
large. The protocol's answer: the FAIL verdict IS the deliverable; a
retune-to-PASS is a fabrication.

### T-V5-03: Cross-metric verdict composition

§3.5 composes AUROC verdict and REL verdict into a single routing
decision (PASS, PASS-with-flag, FAIL, WARN). A plausible edge case is
AUROC-PASS + REL-FAIL (discrimination generalizes but calibration does
not). This protocol classifies it as PASS-with-flag, treating REL as
advisory at V5. An alternative would be to treat REL-FAIL as a FAIL
verdict, forcing a V4 recalibration rerun. The current design privileges
deployment-critical discrimination over calibration stability, which is
consistent with `DESIGN.md` §V4 positioning calibration as post-hoc. If
a future cohort shows REL-FAIL with deployment-relevant consequences
(e.g., sensitivity-at-spec collapses on validation), the classification
may be revisited.

### T-V5-04: Sensitivity-at-spec variance on 10 validation seeds

Per [[equations/fixed-spec-threshold]] §Boundary conditions, $n_0 \ge
1000$ VAL controls is the rule of thumb for stable quantile estimation.
Celiac has $n_\text{control,VAL} \approx 10{,}916$ per seed, so
within-seed $\tau_\text{spec}$ is stable. But across 10 validation seeds,
the sensitivity mean has variance proportional to 10 — the reported CI
on $\mathrm{sens}_{0.95}$ is correspondingly wide. A FAIL on sensitivity
at fixed-spec is only meaningful if it exceeds the wide CI; V5's
verdict is keyed on AUROC, not on sensitivity, for this reason.
Sensitivity is reported as the operational deploy-relevant metric but
does not enter the gate's claim-type determination.

### T-V5-05: V5 is not the holdout

V5 is decision-tree cross-validation (seed-level held-out), not
data-level holdout. The `MASTER_PLAN.md` §Phase 5 holdout run is a
separate, fresh, independent-cohort fit that consumes the V5-PASSED
pipeline as its input configuration. If the factorial's entire 30-seed
split pool shares a common cohort structure (it does, for celiac —
same UKBB cohort, just different random seeds), V5 does NOT rule out
cohort-level systematic failure modes. Those are the holdout's job.
This tension is design-level: no protocol change makes V5 a holdout.

---

## Sources

- `operations/cellml/DESIGN.md` §V5, §Validation Decision Tree
- `operations/cellml/MASTER_PLAN.md` §V5, §Open Questions #4
- `operations/cellml/DECISION_TREE_AUDIT.md` §3.7 (no tree
  cross-validation), §Resolution log (2026-04-08: 20/10 seed split
  implemented)
- ADR-001 (50/25/25 split) — structural invariant V5 depends on
- ADR-009 (threshold on validation) — threshold rule V5 inherits
- ADR-010 (fixed specificity) — operating-point objective V5 pre-registers
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
