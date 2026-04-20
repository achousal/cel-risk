---
type: protocol
gate: v6-ensemble-comparison
informational: true   # new field — signals this gate does NOT lock; see SCHEMA extensions §T-V6-02
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v5-confirmation"
  - "v5_verdict: must be PASS or PASS-with-flag (FAIL/WARN/Inconclusive skip V6)"
  - "v2_non_dominated_set: the Pareto-front survivors forwarded from v2-model"
  - "calibrated_oof_predictions: per-base-model, per-sample OOF posteriors from the V0→V4 locked pipeline replayed across all candidate models on selection seeds 100–119"
outputs:
  - "report: ensemble_verdict ∈ {informationally_preferred, baseline_retained, inconclusive} (informational)"
axes_explored:
  - "ensemble_strategy ∈ {none, mean_proba, oof_stacking}"
axes_deferred: []
depends_on:
  - "[[condensates/stacking-requires-oof-not-infold]]"
  - "[[condensates/nested-cv-prevents-tuning-optimism]]"
  - "[[equations/oof-stacking]]"
  - "[[equations/brier-decomp]]"
  - "[[equations/nested-cv]]"
---

# V6 Ensemble Comparison Protocol

Post-tree informational ensemble gate. V6 runs **after** V5 has confirmed
generalization of the V0→V4 locked single-model pipeline and asks a single
question: would an ensemble of the V2 non-dominated candidate models
outperform the V5-locked pipeline on discrimination (AUROC) **without**
regressing calibration (REL on Brier)? V6 is the **only informational
gate** in the cel-risk validation tree — it does NOT lock anything. It
produces a recommendation; the V5-locked pipeline remains the deployed
pipeline regardless of V6's verdict. `DESIGN.md` §"Design Constraint:
Single-Model Primary, Ensemble Comparison" and
`DECISION_TREE_AUDIT.md` §3.4 + §Resolution log (2026-04-08: "Implemented
as V6. Post-tree informational comparison. ... Human decides.") are the
load-bearing spec references. ADR-007 (OOF stacking) provides the
ensemble mechanics; this protocol formalizes the non-locking gate
structure that the 2026 restructure imposed on ADR-007's original
"Accepted" posture.

---

## 1. Pre-conditions

V6 runs only when the full validation tree has PASSED at V5. Any failure
here means V6 is **skipped**, not relaxed — ensembling cannot rescue a
selection-set-optimism failure at V5.

1. **V5 must have emitted PASS or PASS-with-flag.**
   [[protocols/v5-confirmation]] §3.5 defines the routing: a PASS verdict
   (Equivalence between selection-set and validation-set AUROC) or a
   PASS-with-flag verdict (Equivalence on AUROC with a REL-stability flag)
   are the two V5 outcomes that permit V6 entry. If V5 returned FAIL
   (selection >> validation Direction), WARN (validation >> selection or
   diagnostic), or Inconclusive, V6 is skipped — the pipeline is not
   generalization-confirmed, and an ensemble comparison against a
   non-generalizing baseline is meaningless. The V6 ledger records
   `skipped_reason: "v5_verdict = {FAIL|WARN|Inconclusive}"` in that
   case.
2. **Full V0→V4 pipeline locked and V5-validated.** V6 inherits every
   V0→V4 lock without modification: `training_strategy`, `control_ratio`,
   `recipe_id`, `panel_composition`, `panel_size`, `ordering_strategy`,
   `model` (the V2-locked single-model winner), `weighting`,
   `downsampling`, `calibration_strategy`. V6 does NOT re-select any of
   these. The V2-locked single model is the "baseline" (`ensemble_strategy
   = none`) against which the two ensemble strategies are compared.
3. **V2 non-dominated set must have ≥ 2 members.** Per
   [[protocols/v2-model]] §2.3, V2 forwards the Pareto-front survivors
   (non-dominated on AUROC vs REL). If V2 locked a single model by
   Dominance (i.e., the Pareto front collapsed to one), V6 cannot run —
   there is no ensemble to build. The V6 ledger records
   `skipped_reason: "v2_non_dominated_set has < 2 members"` in that case,
   routing to §4 (inconclusive report) rather than to an exploration V6
   was not designed to perform.
4. **Calibrated OOF predictions on the V2 non-dominated set are
   available.** Each base model in the candidate set must have
   per-sample calibrated OOF posteriors $\hat{p}_{i,m}^{\text{cal,OOF}}$
   on selection seeds 100–119, produced by replaying the V0→V4 locked
   pipeline against that model. The V4 calibrator family used for the
   V2-winning model is re-applied per base model here — i.e., each base
   model gets its own fit of the V4-locked calibrator family on its own
   OOF predictions. This is NOT a re-run of V4 (the calibrator **family**
   is locked; only the per-model calibrator parameters are re-fit).
5. **Fold alignment across base models.** Per
   [[condensates/stacking-requires-oof-not-infold]] §Actionable rule,
   all base models in the stack MUST share identical fold assignments.
   The factorial's 30 shared splits (seeds 100–129, with 100–119 used
   for V6 selection and 120–129 quarantined as per V5) satisfy this by
   construction. If any base model's OOF predictions were produced on a
   different seed schedule, V6 cannot run honestly — lock
   `skipped_reason: "fold alignment violated"` and route to §4.
6. **Validation-set (120–129) quarantine preserved.** V6 uses the 20
   selection seeds only for building and evaluating the ensemble, mirroring
   V0–V4's data budget. The 10 validation seeds remain held out because
   V6 is not a generalization gate (V5 already did that on the
   single-model pipeline). Optional: if the human operator elects to
   **report** an ensemble validation-set number alongside the informational
   comparison, that number is reported as a secondary informational
   measurement only — the V5 verdict on the single-model pipeline is the
   deploy-readiness signal, not any V6 number.
7. **Rulebook snapshot bound at gate entry.** Per SCHEMA.md §Versioning.

---

## 2. Search space

V6 crosses exactly one axis: `ensemble_strategy ∈ {none, mean_proba,
oof_stacking}`. **No other ensemble strategies are in scope.** Blending
with learned weights outside the OOF-stacking formulation, weighted
arithmetic mean, geometric mean, rank-averaging, and other ensemble
variants are explicitly out of scope for V6 — adding them post-hoc would
make V6 a tuning surface against the V5 PASS baseline and re-introduce
selection-set optimism, the exact failure mode V5 exists to rule out.
Proposed additions must go through a rulebook MINOR bump, not an ad-hoc
V6 extension.

### 2.1 Level: `none` (reference baseline)

The V5-locked single-model pipeline. This is the comparison reference for
V6 — every informational judgment is **against** `none`, never between
the two ensemble strategies in isolation. The calibrated OOF posteriors
for the V2-winning single model form the baseline's per-sample predictions
$\hat{p}_i^{\text{base}} = \hat{p}_{i, m^{*}}^{\text{cal,OOF}}$ where
$m^{*}$ is the V2 lock.

### 2.2 Level: `mean_proba`

Equal-weight arithmetic mean of the calibrated OOF posteriors across all
$M$ members of the V2 non-dominated set:

$$\hat{p}_i^{\text{mean}} = \frac{1}{M} \sum_{m=1}^{M} \hat{p}_{i,m}^{\text{cal,OOF}}.$$

No meta-learner, no weight estimation, no extra parameters. This is the
simplest ensemble in the ordering: if a 1-parameter-per-model weighted
average were preferred, the `oof_stacking` level would win by definition
(it strictly nests `mean_proba` via $\beta_0 = 0, \beta_m = 1/M$). The
`mean_proba` level exists because it is the diagnostic null hypothesis
for stacking: if the meta-learner cannot beat equal weights on OOF, the
stacked weights are noise.

### 2.3 Level: `oof_stacking`

L2-penalized logistic regression meta-learner fit on the stacked
calibrated OOF posteriors per [[equations/oof-stacking]]:

$$Z_{i,m} = \hat{p}_{i,m}^{\text{cal,OOF}}, \qquad \hat{p}_i^{\text{stack}} = \sigma\!\left(\beta_0 + \sum_{m=1}^{M} \beta_m Z_{i,m}\right),$$

with $\hat{\beta}$ estimated by penalized MLE under an L2 penalty
$\lambda \|\beta\|_2^2$, where $\lambda$ is tuned via an additional inner
CV loop on $Z$. Per [[condensates/stacking-requires-oof-not-infold]]
§Actionable rule: in-fold predictions are **forbidden** as meta-learner
inputs — feeding in-fold predictions makes the meta-learner concentrate
weight on the hardest-overfitting base model, which does not generalize.
`StackingEnsemble.fit_from_oof()` in
`analysis/src/ced_ml/models/stacking.py` is the canonical API. ADR-007
documents the original adoption; [[equations/oof-stacking]] documents the
math.

Post-stack calibration is optional; when applied, it must be fit on
held-out predictions relative to the stacking fit (a second CV loop) per
[[equations/oof-stacking]] §Boundary conditions. For V6, post-stack
calibration is declared in the V6 ledger as either "enabled" (isotonic on
a second inner fold) or "disabled" (raw sigmoid output); the default is
**disabled** to keep `oof_stacking` and `mean_proba` on comparable
parameter-count footings for the comparison. If enabled, it counts as
part of the ensemble strategy and is reported alongside the verdict.

### 2.4 Axis-condensate mapping

| Decision | Grounding | Load-bearing claim |
|---|---|---|
| Meta-learner fed OOF, not in-fold | [[condensates/stacking-requires-oof-not-infold]], [[equations/oof-stacking]] | In-fold stacking concentrates weight on the hardest-overfitting base model and does not generalize |
| Fold alignment across base models | [[equations/oof-stacking]] §Boundary conditions | $Z$ rows must correspond to the same samples for all base models; different seeds invalidate the stack |
| Selection seeds only (100–119) for V6 | [[condensates/nested-cv-prevents-tuning-optimism]] applied at the decision-tree level | Tuning/comparing on validation seeds (120–129) would break V5's quarantine and make V6 another selection surface |
| Comparison reference is `none`, not head-to-head between ensembles | DESIGN.md §"Design Constraint: Single-Model Primary, Ensemble Comparison" | V6 asks whether an ensemble would improve the V5-locked pipeline, not whether `mean_proba` beats `oof_stacking` in isolation |
| Higher bar than V1–V4 for an ensemble to be preferred | DESIGN.md §V6 ("higher bar: gain > δ=0.02 AND full CI above 0 ... to account for the interpretability cost"); DECISION_TREE_AUDIT.md §Resolution log 2026-04-08 | The single-model winner has an interpretability value the ensemble does not; the informational preference threshold reflects that cost |
| V4 calibrator family re-fit per base model | [[equations/brier-decomp]] | REL is a function of the calibrator's fit on that model's scores; re-fitting per base model preserves the "calibrated probabilities" premise of the ensemble inputs |

---

## 3. Success criteria (informational — read carefully)

V6 does NOT use the rubric to **lock** anything. V6 uses the rubric to
**report** whether an ensemble is informationally preferred against the
V5-locked baseline. The distinction is load-bearing for this protocol;
see §"Why V6 is informational, not locking" below.

### 3.1 Informational preference criterion

An ensemble strategy $E \in \{\text{mean\_proba}, \text{oof\_stacking}\}$
is **informationally preferred** over `none` if and only if **both**:

1. **Direction on AUROC against the V5-locked baseline.**
   $\mathrm{AUROC}_E - \mathrm{AUROC}_{\text{none}} \ge 0.02$ AND 95%
   bootstrap CI over selection-set seeds (100–119, 1000 resamples)
   excludes 0. This is the SCHEMA.md default Direction criterion, NOT a
   relaxed band. The "higher bar" language in DESIGN.md §V6 is
   operationalized here as the full CI above 0 (not merely not
   containing 0).
2. **Equivalence on Brier REL against the V5-locked baseline.** Per
   [[equations/brier-decomp]]: $|\Delta\mathrm{REL}| < 0.005$ AND 95% CI
   $\subset [-0.01, 0.01]$ on the same bootstrap unit. This is the V4
   REL band (not the SCHEMA default), matching
   [[protocols/v4-calibration]] §3.1 — V6 inherits V4's tighter REL rule
   because the question is whether the ensemble preserves calibration
   quality, not whether it meets the discrimination band. An ensemble
   that achieves Direction on AUROC but regresses REL (Direction in the
   wrong direction) is **not** informationally preferred — V6 reports
   this case as `baseline_retained` with a cross-metric note.

Both conditions MUST hold. If AUROC Direction holds and REL is
Inconclusive or REL-Direction (adverse), the verdict is
`baseline_retained`. If REL Equivalence holds and AUROC is
Equivalence/Inconclusive, the verdict is `baseline_retained` (no
discrimination improvement to trade against interpretability). The joint
requirement is the "without regressing calibration" half of V6's
question.

### 3.2 Verdict table

| Condition | V6 verdict | What `decision.md` records |
|---|---|---|
| §3.1 (AUROC Direction AND REL Equivalence) holds for exactly one of {`mean_proba`, `oof_stacking`} | `informationally_preferred` | The preferred ensemble strategy, its $\Delta\mathrm{AUROC}$ + CI, its $\Delta\mathrm{REL}$ + CI, and the note that the V5-locked pipeline remains deployed |
| §3.1 holds for BOTH `mean_proba` and `oof_stacking` | `informationally_preferred` | Both strategies reported; `oof_stacking` preferred for the report because it is the more-specified ensemble, with `mean_proba` as the diagnostic null-hypothesis companion showing the stacking gain is not attributable to equal weighting alone |
| §3.1 holds for NEITHER | `baseline_retained` | $\Delta$ metrics + CIs for both ensembles, explicit note that neither meets the joint criterion |
| V2 forwarded < 2 non-dominated models | `inconclusive` | Comparison cannot be made; see §4.1 |
| V5 verdict was not PASS/PASS-with-flag | (V6 not run) | `skipped_reason` recorded per §1.1 |

### 3.3 Bootstrap unit (seed, paired across ensemble strategies)

- Unit: split seed (selection seeds 100–119 only, $n = 20$).
- Statistic: $\Delta\mathrm{AUROC}_s^{(E)} = \mathrm{AUROC}_E^{(s)} -
  \mathrm{AUROC}_{\text{none}}^{(s)}$ and $\Delta\mathrm{REL}_s^{(E)}
  = \mathrm{REL}_E^{(s)} - \mathrm{REL}_{\text{none}}^{(s)}$, computed
  per seed on the TEST slice (AUROC) and VAL slice (REL) per the
  three-way split convention inherited from V0–V4.
- $B = 1000$ bootstrap resamples with replacement over seeds, paired
  across `none`, `mean_proba`, and `oof_stacking` (same resample indices
  reused across strategies so the CI captures the strategy effect, not
  the seed effect).
- 95% CIs from empirical 2.5% / 97.5% quantiles of
  $\{\Delta\mathrm{AUROC}_s^{*(b,E)}\}$ and
  $\{\Delta\mathrm{REL}_s^{*(b,E)}\}$.

Per-sample bootstrap is forbidden (conflates within-seed dependence).
Per-fold (inner CV) bootstrap is forbidden (inner folds are tuning
objects per [[condensates/nested-cv-prevents-tuning-optimism]]).

### 3.4 Explicit anti-pattern: V6 does NOT lock

A V6 verdict of `informationally_preferred` does **not** relock the
pipeline, does **not** change what gets deployed, and does **not**
invalidate the V5 PASS verdict. The V5-locked pipeline is the deployed
pipeline regardless of V6's verdict. A protocol file that locks at an
informational gate is a **rubric violation** — it would make V6 a
selection surface against the V5 PASS baseline, which would require a
V7 confirmation gate on the new lock, which does not exist and is not
planned.

The reason this anti-pattern is worth naming: human operators seeing an
`informationally_preferred` verdict may be tempted to deploy the
ensemble on the grounds that "the protocol said it's preferred." The
protocol did **not** say that. The protocol said the ensemble meets a
stringent informational criterion on selection seeds that, if observed
reproducibly across future cohorts, would be evidence for promoting V6
to a locking gate in a future rulebook MAJOR version. One cohort's
`informationally_preferred` verdict is one data point toward that
promotion — it is not a deployment decision.

---

## 4. Fallbacks

V6 has no lock to fall back to — it is informational, so "fallback" means
reporting `inconclusive` when the comparison cannot be made, rather than
relaxing a lock criterion.

### 4.1 `inconclusive` report: insufficient ensemble inputs

If the V2 non-dominated set has fewer than 2 members (V2 locked by
Dominance, so there is only one surviving candidate model), V6 cannot
construct either ensemble. `decision.md` records:

- `ensemble_verdict: inconclusive`
- `inconclusive_reason: "v2_non_dominated_set has only {N} member(s); ensemble construction requires >= 2"`
- No $\Delta$ metrics reported (nothing to compare).

This is a valid, non-failing V6 outcome. The V5-locked pipeline is the
deployed pipeline, full stop.

### 4.2 `inconclusive` report: bootstrap CI construction fails

If fewer than 10 selection seeds have complete OOF predictions for all
base models (e.g., a crash truncated the replay), the 20-seed bootstrap
cannot be constructed stably. `decision.md` records:

- `ensemble_verdict: inconclusive`
- `inconclusive_reason: "complete OOF predictions available on {N} seeds; bootstrap requires full set of 20 selection seeds"`

The operator should re-run the V0→V4 pipeline replay on the missing
seeds before re-entering V6.

### 4.3 No "relax the informational bar" path

If `mean_proba` or `oof_stacking` achieves AUROC Direction **without**
REL Equivalence (e.g., AUROC gains of 0.03 but REL regression of 0.008),
the verdict is `baseline_retained`, NOT `informationally_preferred`.
V6 does **not** relax §3.1's joint criterion to Direction-only. The
joint criterion is the protocol's operational definition of "ensemble
preferred"; a Direction-only ensemble report is a cross-metric tradeoff
that belongs in the human operator's deployment conversation, not in
the protocol's verdict.

### 4.4 Forbidden: validation-seed peek to rescue a baseline_retained

If V6 reports `baseline_retained`, the temptation exists to peek at
validation seeds 120–129 to see whether the ensemble "would have won" on
those seeds. This is forbidden for the same reason V5 forbids re-tuning
on validation seeds ([[protocols/v5-confirmation]] §4.1): it breaks the
quarantine that makes any future generalization claim on this cohort
credible. A V6 `baseline_retained` verdict IS the deliverable on this
cohort; a validation-seed peek is a rule violation and produces a
fabricated verdict.

### 4.5 Forbidden: adding a strategy mid-gate to rescue a baseline_retained

If the operator observes `baseline_retained` under `mean_proba` and
`oof_stacking`, adding a third strategy (e.g., weighted average,
rank-average) to re-run V6 is a search-space expansion that makes V6 a
tuning surface. §2 restricts V6 to exactly three strategies for this
reason. Adding strategies requires a rulebook MINOR bump with an explicit
ledger entry, not an ad-hoc V6 retry.

---

## 5. Post-conditions

On V6 `decision.md` write:

1. **`ensemble_verdict`** — one of {`informationally_preferred`,
   `baseline_retained`, `inconclusive`}. This is the protocol's full
   verdict; there is no `PASS` / `FAIL` / `WARN` because V6 is not a
   locking gate.
2. **Comparison table.** Three rows (`none`, `mean_proba`,
   `oof_stacking`) × columns: point AUROC (mean over seeds 100–119),
   95% bootstrap CI on AUROC, point REL, 95% bootstrap CI on REL,
   $\Delta\mathrm{AUROC}$ vs `none`, $\Delta\mathrm{AUROC}$ 95% CI,
   $\Delta\mathrm{REL}$ vs `none`, $\Delta\mathrm{REL}$ 95% CI, §3.1
   joint-criterion pass/fail.
3. **Meta-learner weights (if `oof_stacking` was fit).** Report
   $\hat{\beta}$ per base model, with $\lambda$ tuning history. This is
   diagnostic — it reveals which base models the stack concentrated
   weight on, and whether the concentration matches the V2 non-dominated
   set's AUROC ordering or inverts it (a common diagnostic signal per
   [[condensates/stacking-requires-oof-not-infold]] is weight
   concentration on the model with the highest in-sample AUROC regardless
   of out-of-sample performance, which `oof_stacking` by construction
   should **not** produce).
4. **V5-locked pipeline reaffirmed as deployed.** The `decision.md`
   explicitly states: `deployed_pipeline: v5_locked` and `ensemble_lock:
   null`. These two fields are load-bearing — they are the mechanical
   enforcement that V6 did not lock anything.
5. **Tension filing (on `informationally_preferred` only).** If the
   verdict is `informationally_preferred`, a project-level tension is
   filed under `projects/<name>/tensions/gate-decisions/` naming
   "V6 ensemble informationally preferred on <cohort>: candidate for
   promotion to locking gate in future rulebook MAJOR version if
   reproduced across ≥2 additional cohorts." This is the promotion
   pathway — V6 does not promote itself on a single cohort, but the
   tension is the mechanism that accumulates evidence for the promotion
   proposal. A `baseline_retained` verdict files no tension (no
   rulebook-update signal). An `inconclusive` verdict files a tension
   naming the blocker (e.g., "V2 dominated to 1 model on celiac — V6 is
   unreachable under current factorial structure"), which may motivate a
   V2 protocol revision rather than a V6 promotion.
6. **Rulebook snapshot citation.** `rulebook_snapshot` bound at entry
   per §1.7 is cited verbatim so the CHANGELOG can link this verdict to
   the rulebook version at decision time.

What advances beyond V6: **nothing in this cohort's validation tree.**
V6 is the terminal gate of the cel-risk factorial. Its outputs are
informational inputs to future rulebook updates (via the tension
pathway), not to any downstream gate on this cohort.

What V6 does NOT produce:

- No new axis locks (V6 is not a selection gate).
- No changes to the V5-locked pipeline (deployment is unchanged).
- No deployment decision by itself (human operator deploys the
  V5-locked pipeline; the V6 verdict is metadata on that decision).

---

## Why V6 is informational, not locking

- The validation tree locks everything V0 through V5. V5 certified that
  the V0→V4 locks generalize from selection seeds (100–119) to
  validation seeds (120–129). **Changing the deployed pipeline at V6
  would invalidate V5's generalization verdict**, which was measured on
  the V5-locked pipeline, not on an ensemble. An ensemble's
  generalization would require its own V5-analog gate, which does not
  exist in this factorial's scope.
- Ensemble gains are often artifacts of small-sample noise on
  rare-event cohorts (celiac: 148 cases). A Direction-claim-plus-
  Equivalence-on-Brier criterion is stringent, and most ensembles on
  cohorts of this size will report `baseline_retained`. That is the
  expected-case V6 outcome; the protocol is designed to be hard to
  accidentally trigger.
- If ensemble is `informationally_preferred` on multiple cohorts
  (future datasets), that signal accumulates via the
  `projects/<name>/tensions/gate-decisions/` pathway and, at ≥ 3
  non-overlapping cohort confirmations, promotes to a rulebook update
  proposing V6 as a **locking** gate in a future MAJOR rulebook version
  (per SCHEMA.md §Versioning: MAJOR bumps break falsifier rubric or file
  schema — adding a locking gate is a rubric change). Until that
  accumulation occurs, V6 stays informational on first principles.

---

## Known tensions

### T-V6-01: ADR-007 vs V6 restructure — stacking's locking status

ADR-007 ("OOF Stacking Ensemble", 2026-01-22) is marked **Accepted**,
implying stacking is a locked architectural decision. The 2026-04-08
audit restructure (DECISION_TREE_AUDIT.md §Resolution log, "Implemented
as V6. Post-tree informational comparison. ... Human decides.") changed
the posture: stacking is now a **candidate** strategy under an
informational gate, not a locked one. The two documents are in tension.
This protocol resolves the tension by treating the current V6 framing as
authoritative — ADR-007's "Accepted" status applies to the **mechanics**
(OOF, L2 meta-learner, the `fit_from_oof` API) but not to the
**deployment status** (which is now "informational per V6, deploy only
on explicit human call after multi-cohort evidence accumulation"). The
parallel tension note
`projects/celiac/tensions/data-quality/adr007-vs-v6-restructure-stacking-status.md`
(being written separately; forward-reference acceptable) is the canonical
record; this protocol cites it as the tension of record.

### T-V6-02: SCHEMA.md does not define `informational: true` frontmatter

SCHEMA.md §Protocols defines the `gate`, `inputs`, `outputs`,
`axes_explored`, `axes_deferred`, `depends_on`, and `metric_overrides`
frontmatter fields, but does NOT define an `informational` field. V6 is
the first protocol to require one because it is the first non-locking
gate. The field is proposed here as a SCHEMA extension candidate:

- Field name: `informational: true | false` (default: `false` when
  omitted, preserving all current protocols' behavior).
- Semantics: when `true`, the protocol does NOT lock any axes; its
  `outputs` section describes a report, not locks; and gate tooling
  (cellml-reduce, validate_tree.R) routes the protocol through an
  informational pathway that skips lock propagation to downstream gates.
- SCHEMA patch: add "Informational protocols" subsection under
  §Protocols explaining the field and its constraints (e.g., an
  informational protocol's `outputs` list may not contain `locks:`
  entries; a fix that prevents downstream gates from reading V6 locks
  by convention since none exist).

Flag as candidate **rulebook PATCH** (field addition with wording, no
rubric or file-schema break), pending human review of this protocol.

### T-V6-03: V2 Dominance lock can orphan V6

If V2 locks by Dominance (a single model strictly dominates all others
on AUROC × REL), the V2 non-dominated set has one member and V6 cannot
run (§4.1). The V5-locked pipeline is the deployed pipeline in that
case, unambiguously — there is no ensemble to compare. The tension is
that V6 becomes a phantom gate on cohorts where V2 is decisive. This is
not a defect: V6's purpose is to ask "does ensembling help when multiple
models are plausible?"; if V2 rules out all but one, the question is
moot. The only change this tension implies is that the V6 ledger should
be allowed to record `skipped_reason` without penalty to the overall
factorial — which §4.1 already codifies.

### T-V6-04: Post-stack calibration is disabled by default

§2.3 disables post-stack isotonic calibration by default. The
alternative (enable by default) has one advantage — if the meta-learner's
raw output has sigmoid pathologies, post-stack isotonic repairs them —
and two disadvantages: (a) it adds parameters to `oof_stacking` and
makes the strategy more complex than `mean_proba` by more than a single
nested-family step, and (b) it requires a second CV loop that the
factorial's standard nested CV (5×10×5×200 per
[[equations/nested-cv]]) does not reserve wallclock for. The default is
disabled for parsimony. Enabling post-stack calibration is a ledger
declaration, not a silent default.

### T-V6-05: Validation-seed replay as a post-hoc confirmation

A plausible extension of V6 is to **also** compute the ensemble's
metrics on validation seeds 120–129 as a V5-style generalization check
for the ensemble. This is permitted but the number is strictly
informational (it does not change the V5-locked pipeline's deploy
status). If the validation-set ensemble AUROC drops markedly relative to
the selection-set ensemble AUROC — i.e., the ensemble suffers
selection-set optimism that the single model did not — that is evidence
**against** promoting V6 to a locking gate in future rulebook versions.
The §5.5 tension note should capture this diagnostic if computed. The
current protocol does not require the validation-seed replay because it
is not a lock input; future rulebook versions may mandate it.

---

## Sources

- `operations/cellml/DESIGN.md` §"Design Constraint: Single-Model
  Primary, Ensemble Comparison", §Validation Decision Tree (V6 block)
- `operations/cellml/DECISION_TREE_AUDIT.md` §3.4 (Ensemble Excluded),
  §Resolution log 2026-04-08 ("Implemented as V6. Post-tree
  informational comparison. ... Human decides.")
- ADR-007 (OOF Stacking Ensemble, 2026-01-22) — canonical source for
  stacking mechanics; deployment status superseded by V6 framing per
  T-V6-01
- `operations/cellml/rulebook/equations/oof-stacking.md` — stacking
  math and boundary conditions
- `operations/cellml/rulebook/condensates/stacking-requires-oof-not-infold.md`
  — the in-fold stacking failure mode V6 is designed to avoid
- `operations/cellml/rulebook/protocols/v5-confirmation.md` — prior
  gate whose PASS verdict V6 inherits as its pre-condition
- `operations/cellml/rulebook/protocols/v2-model.md` §2.3 — the
  Pareto non-dominated set construction that defines V6's candidate
  model pool
- `operations/cellml/rulebook/protocols/v4-calibration.md` §3.1 —
  source for the V6 REL Equivalence band (0.005 / CI $\subset [-0.01,
  0.01]$), inherited here for the cross-metric test
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric; §Protocols is the section T-V6-02 proposes extending
