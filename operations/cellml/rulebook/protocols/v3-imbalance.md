---
type: protocol
gate: v3-imbalance
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v2-model"
  - "v0_lock: imbalance_family"   # NEW (rb-v0.2.0) — V3 branches on this
  - "v0_lock: training_strategy"
outputs:
  - "locks: [imbalance_level]  # content depends on V0 family branch; see §5"
axes_explored:
  - "CONDITIONAL ON v0_lock.imbalance_family — see §2 for branch-specific axes"
axes_deferred:
  - "calibration: deferred to v4-calibration"
  - "threshold: deferred to fixed-spec gate"
  - "ensemble: deferred to v6-ensemble"
depends_on:
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/calib-per-fold-leakage]]"
  - "[[condensates/calib-parsimony-order]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[condensates/imbalance-two-stage-decision]]"   # NEW — authored in parallel; TODO verify slug on disk
  - "[[condensates/nested-downsampling-composition]]"   # applies now only in V0-Inconclusive fallback branch
  - "[[condensates/imbalance-utility-equal-weight]]"
  - "[[condensates/auprc-for-imbalance-handling]]"
  - "[[condensates/v3-utility-provenance-chain]]"
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[equations/brier-decomp]]"
---

# V3 Imbalance Gate — Refine the imbalance level within the family locked at V0

V3 is the third decision node of the cellml tree under the rb-v0.2.0
architecture. As of rb-v0.2.0, V0 locks a two-level pair:
`training_strategy` AND `imbalance_family ∈ {none, downsample, weight}`. V3
enters conditional on `imbalance_family` and refines the **level** within the
locked family — it does NOT revisit the family choice. See
[[condensates/imbalance-two-stage-decision]] for the two-stage decision logic
that this protocol implements. V4 (calibration) is NOT run until V3 closes,
because calibration is post-hoc on an already-imbalance-configured model
(per [[condensates/calib-per-fold-leakage]] and the ordering defended in
`DECISION_TREE_AUDIT.md` §2.2, resolution 2026-04-08).

**Semantic change from rb-v0.1.x**: V3 no longer runs the fixed 3×3 weighting
× downsampling grid as its default. V3's output LOCK **SUPERSEDES** V0's
probed level once a family is locked — this is not multiplicative composition
with V0's level. The only exception is the V0-Inconclusive fallback branch
(§2.4), which runs the legacy 3×3 grid; in that branch and only that branch,
control_ratio and class_weight compose multiplicatively per
[[condensates/nested-downsampling-composition]].

---

## 1. Pre-conditions

V3 is entered only when all of the following are finalized. Each pre-condition
is a structural invariant that V3 does not revisit.

**V0 gate locked** (REQUIRED — two V0 locks are branch inputs).
[[protocols/v0-strategy]] has emitted a `decision.md` with:

1. `training_strategy` locked.
2. `imbalance_family` locked to one of `{none, downsample, weight, Inconclusive}`.
   V3's branch selection (§2) reads this lock directly.

V0 may also have recorded a PROBED level (e.g., `downsample_5` as the
representative probe for the `downsample` family). V3's lock REPLACES that
probed level in any non-fallback branch — V3 is the authority on the
within-family level choice.

**V1 gate locked.** [[protocols/v1-recipe]] has emitted a `decision.md`
with `recipe_id`, `panel_composition`, `panel_size`, and `ordering_strategy`
locked. V3 operates on the single winning recipe; no recipe axis is re-opened.

**V2 gate locked.** The model-dominance tournament has emitted a `decision.md`
with a single winning model from {LR_EN, LinSVM_cal, RF, XGBoost}. V3 runs
its branch grid conditional on that locked (recipe, model) pair. If V2
returned a non-dominated Pareto frontier, V3 MUST be run independently on
each frontier member and the locks report per-model.

**Dataset fingerprint** (`projects/<name>/dataset/fingerprint.yaml`), frozen.
The hash binds V3 to the same 20 selection seeds (100–119) used upstream; the
10 confirmation seeds (120–129) remain reserved for V5.

**Environmental invariants**:

- Split strategy 50/25/25 stratified on the primary outcome label (inherited
  from V0, per [[condensates/three-way-split-prevents-threshold-leakage]]).
- Prevalence adjustment MUST be applied to all probability outputs before
  utility evaluation, per [[condensates/downsample-requires-prevalence-adjustment]].
  Raw downsampled scores are NEVER fed into the Brier-reliability proxy.
  Utility MUST be computed on the prevalence-adjusted OOF-posthoc substrate
  per [[condensates/v3-utility-provenance-chain]].
- Optuna budget: 200 trials per cell (full main-factorial budget).
- Code version pinned via git commit SHA in the gate ledger
  `rulebook_snapshot`.

---

## 2. Search space — BRANCHED on V0's `imbalance_family` lock

V3's search space is conditional on V0's `imbalance_family`. Each branch
below defines its own axes, its own level enumerations, its own parsimony
baseline, and its own cell count. V3 selects exactly one branch at ledger
entry based on the V0 decision; the other branches do not execute.

### 2.1 Branch A — `imbalance_family == none`: V3 is SKIPPED

**Trigger:** V0 locked `imbalance_family = none` via Direction or
Equivalence-parsimony. V0 has determined that no imbalance handling is
required at this cohort/recipe/model configuration.

**Action:** V3 does NOT run. The gate produces a `decision.md` noting that
V3 was skipped under `imbalance_family = none`, with the following locks
recorded:

- `imbalance_level = none` (inherited from V0's family lock)
- No further level refinement is meaningful when the family is `none`.

**Forward to V4:** V4 receives a (recipe, model, `imbalance = none`) tuple
plus the V3-skip notation. V4's calibration decision proceeds on an
un-rebalanced base model.

**Grounding:** [[condensates/imbalance-two-stage-decision]] — once the family
is locked to `none`, a level refinement within `none` has no admissible
levels and running V3 is a no-op. This branch is the canonical illustration
of the two-stage decision's short-circuit behavior.

**Cell count:** 0.

### 2.2 Branch B — `imbalance_family == downsample`: refine control ratio

**Trigger:** V0 locked `imbalance_family = downsample` (V0's probe at
`downsample_5` produced Direction against `none`, or Direction/Equivalence
within the downsample-vs-weight family comparison, sufficient to lock the
family). V3 now refines the level.

**Axis explored:**

- `control_ratio ∈ {2, 5}`

The `control_ratio = 1` level is excluded because `1:1` is operationally
equivalent to "no downsampling" — that state is covered by the
`imbalance_family = none` branch and would violate the family lock V0 has
already issued. Per
[[condensates/downsample-preserves-discrimination-cuts-compute]] boundary
condition ("Ratios below 2:1 are NOT permitted on the celiac cohort"), 2 is
the admissible lower bound and 5 is the ADR-003 default upper bound. V0's
probe at `downsample_5` is the evidence that promoted the family to locked;
V3 tests whether `downsample_2` is Direction-equivalent by parsimony and
thus preferable.

**Parsimony order:** `1 ≺ 2 ≺ 5` (per
[[condensates/parsimony-tiebreaker-when-equivalence]] V3-downsample
instantiation). Within this branch, the parsimony baseline is the lowest
admissible level: `control_ratio = 2`. V3 compares `control_ratio = 5`
against `control_ratio = 2`.

**Outputs:** `locks: [control_ratio]`. V3's lock REPLACES V0's probed level
(V0 probed `downsample_5`; V3 may lock `control_ratio = 2` if 2 is
Direction-equivalent by parsimony). This is NOT multiplicative composition.

**Grounding:** [[condensates/downsample-preserves-discrimination-cuts-compute]]
(family-level effect and ADR-003 default), [[condensates/downsample-requires-prevalence-adjustment]]
(prevalence-adjustment required when REL enters utility),
[[equations/case-control-ratio-downsampling]] (single-axis prevalence formula
— in this branch `r_V3` is the only active control-ratio axis; V0 does NOT
lock a separate upstream ratio in rb-v0.2.0).

**Cell count:** 2 cells × 200 Optuna trials × 20 selection seeds.

### 2.3 Branch C — `imbalance_family == weight`: refine class weight

**Trigger:** V0 locked `imbalance_family = weight` (V0's probe at
`class_weight = log` produced Direction against `none`, or Direction within
the weight-vs-downsample family comparison). V3 now refines the level.

**Axis explored:**

- `class_weight ∈ {sqrt, log, balanced}`

`class_weight = none` is excluded because V0's family lock rules it out by
construction (choosing `none` would invert the family lock). `sqrt`, `log`,
and `balanced` span the parsimony spectrum from softer to harder
up-weighting:

- **`sqrt`**: `w_case = sqrt(n_control / n_case)`. Softer than V0's probe
  (`log`). Tests whether a less aggressive weighting is Direction-equivalent
  and thus preferable by parsimony.
- **`log`**: `w_case = log(n_control / n_case) + 1`. V0's probe level —
  included so V3's comparison is against the same level V0 measured.
- **`balanced`**: `w_case = n_control / n_case` (full inverse-frequency,
  sklearn's `class_weight='balanced'`). Harder than V0's probe. Tests whether
  aggressive upweighting beats `log` by Direction; typically the
  parsimony-most-expensive level in this branch.

Weights enter via the model's native `class_weight` hook (sklearn-style for
LR_EN, LinSVM_cal, RF; `scale_pos_weight` for XGBoost). No custom loss
implementation is invoked.

**Parsimony order:** `none ≺ sqrt ≺ log ≺ balanced` (per
[[condensates/parsimony-tiebreaker-when-equivalence]] V3-weight
instantiation). Within this branch, the parsimony baseline is the lowest
admissible level **present in the branch's grid**: `class_weight = sqrt`
(since `none` is excluded by the family lock). V3 compares `log` and
`balanced` against `sqrt`.

**Outputs:** `locks: [class_weight]`. V3's lock REPLACES V0's probed level
(V0 probed `class_weight = log`; V3 may lock `sqrt` if `sqrt` is
Direction-equivalent by parsimony, or `balanced` if `balanced` beats `log`
by Direction).

**Grounding:** [[condensates/imbalance-two-stage-decision]] (the
family-then-level decomposition), [[condensates/auprc-for-imbalance-handling]]
(AUPRC is the primary discrimination metric for weight axis changes),
[[condensates/imbalance-utility-equal-weight]] (0.5/0.5 utility weighting).

**Cell count:** 3 cells × 200 Optuna trials × 20 selection seeds.

### 2.4 Branch D — `imbalance_family == Inconclusive`: FALLBACK to legacy 3×3 grid

**Trigger:** V0 returned `imbalance_family = Inconclusive` — V0 could not
separate the family choices by Direction or Equivalence. V3 falls back to
the legacy 3×3 composition behavior from rb-v0.1.x.

**Axes explored:**

- `control_ratio ∈ {1, 2, 5}`
- `class_weight ∈ {none, sqrt, log}`

Both axes are crossed as a full 3×3 grid. **In this branch AND ONLY this
branch, control_ratio and class_weight compose multiplicatively** per
[[condensates/nested-downsampling-composition]] — this condensate's claim
applies specifically to the V3-fallback boundary condition where both axes
are live simultaneously and a V0 upstream control ratio may or may not
compose. V0's Inconclusive outcome on the family level effectively removes
V0 as the authoritative level source; V3 becomes the authority and runs the
full factorial.

**Parsimony order (joint):** apply parsimony hierarchically —
`class_weight: none ≺ sqrt ≺ log`, then `control_ratio: 1 ≺ 2 ≺ 5`. The
parsimony baseline cell is `(class_weight = none, control_ratio = 1)`.

**Outputs:** `locks: [control_ratio, class_weight]` (joint lock). This is the
only branch in rb-v0.2.0 where V3 locks two axes simultaneously.

**Grounding:** [[condensates/nested-downsampling-composition]] (boundary
condition: multiplicative composition applies in the V3-fallback regime),
[[condensates/downsample-requires-prevalence-adjustment]],
[[equations/case-control-ratio-downsampling]].

**Cell count:** 9 cells × 200 Optuna trials × 20 selection seeds.

### 2.5 Utility function (all branches that run)

Per `DESIGN.md` §V3 and
[[condensates/imbalance-utility-equal-weight]], V3 scores each cell in the
active branch with a normalized utility:

$$U_{\text{cell}} = 0.5 \cdot \widetilde{\mathrm{AUPRC}}_{\text{cell}} + 0.5 \cdot \left(1 - \widetilde{\mathrm{REL}}_{\text{cell}}\right)$$

where

- $\widetilde{\mathrm{AUPRC}}$ is min-max-normalized AUPRC across the cells in
  the active branch (2 cells in Branch B, 3 cells in Branch C, 9 cells in
  Branch D).
- $\widetilde{\mathrm{REL}}$ is the analogous min-max normalization of the
  prevalence-adjusted Brier reliability component per
  [[equations/brier-decomp]].
- Both components are computed on the prevalence-adjusted OOF-posthoc
  substrate per [[condensates/v3-utility-provenance-chain]]. Raw
  within-training score substrate is forbidden.
- Weights fixed at 0.5/0.5 per
  [[condensates/imbalance-utility-equal-weight]]; any change requires a new
  ADR.

AUPRC is the discrimination metric rather than AUROC per
[[condensates/auprc-for-imbalance-handling]] (AUROC is prevalence-invariant
and would absorb exactly the variation V3's axes induce).

REL is computed via `oof_posthoc` per [[condensates/calib-per-fold-leakage]];
per-fold-calibrated VAL predictions are forbidden at V3 regardless of branch.

### 2.6 Axis-condensate mapping

| Branch | Axis | Justification condensate | Parsimony baseline |
|---|---|---|---|
| A (`none`) | — | [[condensates/imbalance-two-stage-decision]] (short-circuit) | N/A (V3 skipped) |
| B (`downsample`) | `control_ratio ∈ {2, 5}` | [[condensates/downsample-preserves-discrimination-cuts-compute]] | `control_ratio = 2` |
| C (`weight`) | `class_weight ∈ {sqrt, log, balanced}` | [[condensates/auprc-for-imbalance-handling]], [[condensates/imbalance-two-stage-decision]] | `class_weight = sqrt` |
| D (Inconclusive fallback) | `control_ratio ∈ {1,2,5}` × `class_weight ∈ {none,sqrt,log}` | [[condensates/nested-downsampling-composition]] (multiplicative composition, this boundary only) | `(none, 1)` |

---

## 3. Success criteria

All V3 decisions use the fixed falsifier rubric in `SCHEMA.md`. Within each
branch, the primary lock test is a **pairwise Direction** comparison of each
non-baseline cell against the branch's parsimony baseline, on $\Delta U$
with 95% bootstrap CI over 1000 seed-level paired resamples (20 selection
seeds are the paired units).

### 3.1 Cell-level claim types (applies per branch)

| Claim type | Criterion | V3 decision for this cell |
|---|---|---|
| **Direction** (cell beats baseline) | $\|\Delta U\| \ge 0.02$ AND 95% bootstrap CI excludes 0 | Cell is a lock candidate |
| **Equivalence** (cell ≈ baseline) | $\|\Delta U\| < 0.01$ AND 95% bootstrap CI $\subset [-0.02, 0.02]$ | Cell does NOT beat baseline; baseline retains lock by parsimony |
| **Inconclusive** | Neither met | Log tension; lock parsimony default (see §4) |

### 3.2 Branch-specific lock rule

**Branch A (`none`):** No cells, no comparison. Lock `imbalance_level = none`
by inheritance. No claim type recorded at V3 (claim type is inherited from
V0's family lock).

**Branch B (`downsample`):**

- If `control_ratio = 5` beats `control_ratio = 2` by Direction: lock
  `control_ratio = 5`. Claim type: Direction.
- If `control_ratio = 5` is Equivalent to `control_ratio = 2`: lock
  `control_ratio = 2` by parsimony. Claim type: Equivalence +
  parsimony-tiebreaker (cite
  [[condensates/parsimony-tiebreaker-when-equivalence]]).
- If Inconclusive: lock `control_ratio = 2` by parsimony default AND log
  candidate tension per §4.1. Claim type: Inconclusive.

**Branch C (`weight`):** Pairwise compare each of `log` and `balanced`
against the baseline `sqrt`.

- If `balanced` beats `sqrt` by Direction AND `balanced` beats `log` by
  Equivalence-or-better: lock `class_weight = balanced`. Claim type:
  Direction.
- If `log` beats `sqrt` by Direction AND `log` beats `balanced` by
  Equivalence-or-better: lock `class_weight = log`. Claim type: Direction.
- If both `log` and `balanced` are Equivalent to `sqrt`: lock `sqrt` by
  parsimony. Claim type: Equivalence + parsimony-tiebreaker.
- If both `log` and `balanced` beat `sqrt` by Direction but Directionally
  separate from each other: Direction-dispute — lock `sqrt` by parsimony
  default, log tension per §4.1. Claim type: Inconclusive (joint).
- If any Inconclusive across the pairwise tests: lock `sqrt` by parsimony
  default, log candidate tension per §4.1. Claim type: Inconclusive.

**Branch D (Inconclusive fallback, 3×3 grid):**

The lock rule for the 3×3 grid follows the legacy rb-v0.1.x logic adapted
for the dual-axis parsimony order. Each of the 8 non-baseline cells is
compared to the baseline `(class_weight = none, control_ratio = 1)`:

- If one or more cells satisfy Direction against the baseline, the cell
  with the largest $\Delta U$ with CI excluding 0 is the candidate, subject
  to the Equivalence-or-better pairwise check against every other
  Direction-passing cell. If one cell wins cleanly, lock that
  `(control_ratio, class_weight)` jointly. Claim type: Direction.
- If all 8 non-baseline cells return Equivalence, lock
  `(control_ratio = 1, class_weight = none)` by parsimony. Claim type:
  Equivalence + parsimony-tiebreaker.
- If two or more cells tie by Equivalence on $\Delta U$, apply parsimony
  hierarchically (weight first, then downsampling — per rb-v0.1.x
  precedence retained).
- If Direction-dispute (non-adjacent cells beat baseline with non-overlapping
  Directions): lock baseline by parsimony default, log tension per §4.1.
  Claim type: Inconclusive (joint).

### 3.3 Metric specification (all branches)

- **AUPRC**: computed on prevalence-adjusted, OOF-posthoc-calibrated
  predictions across the 20 selection seeds. 95% bootstrap CI over 1000
  paired seed resamples.
- **REL (Brier reliability component)**: per [[equations/brier-decomp]],
  computed with $K = 10$ quantile bins on the same prevalence-adjusted OOF
  predictions. 95% bootstrap CI over 1000 seed resamples.
- **$\Delta U$**: paired-bootstrap delta preserving seed pairing.
- Forbidden: per-fold-calibrated VAL predictions, raw (non-prevalence-
  adjusted) downsampled scores, cell-averaged SE.
- **Substrate invariant**: all metrics consume the prevalence-adjusted
  OOF-posthoc substrate per [[condensates/v3-utility-provenance-chain]];
  `observation.md` MUST name the substrate used.

### 3.4 V2 Pareto frontier handling

If V2 emitted $k > 1$ frontier models, V3 is run independently on each
(under the same V0 `imbalance_family` branch). Lock rule:

- If all $k$ V3 runs return the same `imbalance_level`, lock globally and
  carry $k$ candidate models into V4.
- If V3 runs disagree on the level, lock per-model and flag tension per
  `SCHEMA.md` §Rulebook updates.

---

## 4. Fallbacks

### 4.1 Within-branch Inconclusive → parsimony default + tension log

When the active branch's §3.2 pairwise comparison returns Inconclusive
(neither Direction nor Equivalence holds, or Direction-dispute), V3 locks
the **branch's parsimony default** and logs a candidate tension:

- Branch B default: `control_ratio = 2`.
- Branch C default: `class_weight = sqrt`.
- Branch D default: `(control_ratio = 1, class_weight = none)`.

Rationale: V3's job is to refine within a family that V0 has already
committed to. When V3 cannot separate levels by Direction or Equivalence,
the parsimony default is the safest lock and the Inconclusive outcome
becomes evidence for revisiting the branch specification — logged to
`projects/<name>/tensions/rule-vs-observation/` per
`SCHEMA.md` §Rulebook updates.

### 4.2 Widen Optuna budget before promoting

If Inconclusive is driven by wide CIs (bootstrap CI half-width $> 0.02$
despite no Direction), re-run only the contested cells at 400 Optuna trials
with the same 20 seeds as a power check before declaring the branch
Inconclusive. The axis under test is imbalance-level, not model
hyperparameters, so extra trials are not hyperparameter peeking.

### 4.3 V0-Inconclusive fallback branch (Branch D)

This is documented separately because it is a branch-selection outcome, not
a within-branch fallback. When V0 returns `imbalance_family = Inconclusive`,
V3 runs Branch D (the full 3×3 grid from rb-v0.1.x). Within Branch D, the
§4.1 Inconclusive fallback still applies — if the 3×3 itself cannot produce
a clean lock, the parsimony default `(control_ratio = 1, class_weight = none)`
is locked and tension is logged.

### 4.4 Forward to V4 without a clean lock

If §4.1 triggers in any branch and the parsimony default is the lock, V3
forwards the cell's prevalence-adjusted OOF-posthoc predictions to V4 along
with a list of **Inconclusive candidate cells** (those whose CI crossed 0
against the parsimony baseline). V4's calibration decision may partially
compensate for imbalance-handling choices V3 could not separate, per
[[condensates/calib-per-fold-leakage]] preserving OOF-posthoc substrate
forward.

---

## 5. Post-conditions

What V3 writes into `projects/<name>/gates/v3-imbalance/decision.md`:

**Per-branch lock content** (content of `locks: [imbalance_level]` depends
on which branch executed):

- **Branch A (`none`):** `imbalance_level = none` (V3 skipped). No
  within-V3 claim type.
- **Branch B (`downsample`):** `imbalance_level = {control_ratio: <2|5>}`
  with V3 claim type recorded (Direction, Equivalence-parsimony, or
  Inconclusive-parsimony-default).
- **Branch C (`weight`):** `imbalance_level = {class_weight: <sqrt|log|balanced>}`
  with V3 claim type recorded.
- **Branch D (Inconclusive fallback):** `imbalance_level = {control_ratio: <1|2|5>, class_weight: <none|sqrt|log>}`
  as a joint lock with claim type recorded.

**Other decision.md content:**

- Prevalence-adjusted metric table (AUPRC, REL, $U$) per cell in the
  active branch, with 95% bootstrap CIs. Raw metrics logged for provenance
  per [[condensates/downsample-requires-prevalence-adjustment]].
- Effective training prevalence per cell (Branch D only computes composite
  $\pi_\text{train, eff}$ per [[condensates/nested-downsampling-composition]];
  Branches B and C compute single-axis $\pi_\text{train}$).
- Substrate-provenance block per
  [[condensates/v3-utility-provenance-chain]]: names the utility substrate
  as `prevalence_adjusted_oof_posthoc`, logs `pi_train` (per cell) and
  `pi_population` (from fingerprint).
- V2 frontier disposition: per-model locks if $k > 1$ and any
  cross-model disagreement.
- Branch selection trace: the V0 `imbalance_family` value read, the branch
  executed, and the skip reason (Branch A only).
- Unlocked / forwarded Inconclusive cells (if §4.4 triggered).

**What advances to V4:**

- The locked `imbalance_level` (or branch-A skip notation).
- The V3 winner's (or parsimony-default's) OOF-posthoc prediction table
  (prevalence-adjusted) — V4's calibrators fit on this exact table, per
  [[condensates/calib-per-fold-leakage]].
- The dataset fingerprint hash.
- The V1–V2 locks (recipe, panel, model) — unchanged.
- The V0 locks (`training_strategy`, `imbalance_family`) — unchanged.

**What remains open:**

- Calibration strategy (V4).
- Threshold selection (post-factorial, fixed-spec gate).
- Seed-split confirmation (V5) — holds out seeds 120–129.

---

## Known tensions

### T-V3.1 — `nested-downsampling-composition` scope narrowed to V0-Inconclusive fallback

In rb-v0.1.x, V3 was assumed to compose multiplicatively with V0's
control-ratio lock on every run. The rb-v0.2.0 architecture removes that
coupling: in Branches A, B, and C, V0 does NOT carry a residual
control-ratio into V3 (V0's probe is at family level; V3 is the authority
on the level). Multiplicative composition between V0 and V3 applies ONLY in
Branch D (V0-Inconclusive fallback), because Branch D is the only branch
that runs the full 3×3 grid.

**Action:** [[condensates/nested-downsampling-composition]] should be
updated to declare this boundary condition explicitly. Its "applies to"
scope is no longer "gates that explore a within-training downsampling axis
on top of a split-generation control ratio locked upstream" in general —
it is "gates that fall back to the legacy 3×3 grid under V0 Inconclusive".
Flag as candidate rulebook PR.

### T-V3.2 — Branch A skip is new behavior not in prior validation tree

The `imbalance_family = none` short-circuit (V3 skipped) is new in
rb-v0.2.0. The prior validation tree always ran V3 regardless of V0's
conclusion. The two-stage decision rationale is grounded in
[[condensates/imbalance-two-stage-decision]]; the short-circuit follows
from the two-stage logic mechanically.

**Risk:** if V0's family lock at `none` is wrong (a Type II family error),
V3 never runs and the imbalance misconfiguration propagates to V4. Partial
mitigation: V5 seed-split confirmation can re-open V0 + V3 jointly if the
confirmation seeds contradict the V0 family lock. This is the same
V5-reopening mechanism that covers all V0 locks.

**Action:** Monitor for V5 contradictions of V0-`none` family locks across
the first 3 cohorts that use rb-v0.2.0; if the contradiction rate exceeds
the expected bootstrap-variance floor, the short-circuit needs revisiting.

### T-V3.3 — ADR-003 partially superseded by rb-v0.2.0

ADR-003 (cel-risk, 2026-01-20) originally specified
`train_control_per_case = 5` as the canonical default, locked pre-V0 and
inherited by the main factorial. Under rb-v0.2.0:

- V0 now PROBES `downsample_5` as the representative family probe (not a
  pre-gate lock).
- V3 may REFINE the control ratio to `2` within Branch B if `2` is
  Direction-equivalent by parsimony.

The ADR-003 default is therefore only canonical at the V0-probe layer, not
as a V3 lock. Flag ADR-003 as PARTIALLY SUPERSEDED: the compute-savings
arithmetic (~60× at 5:1) remains correct, but the lock authority has moved
from ADR-003 to the V0 family lock + V3 level refinement.

**Action:** Candidate ADR revision — ADR-003 should note that its
control-ratio default is now a V0 probe default, not a main-factorial lock,
and that V3 Branch B may refine it.

### T-V3.4 — Per-fold vs OOF-posthoc REL (leakage guard, inherited)

V3's REL MUST be OOF-posthoc per [[condensates/calib-per-fold-leakage]].
This is a structural precondition, not a new tension — but it remains a
frequent implementation failure mode because sklearn's default
`CalibratedClassifierCV` is per-fold. Any V3 `observation.md` that does
not cite `calibration.strategy: oof_posthoc` is a rulebook violation.

### T-V3.5 — V3 submitted with 20 selection seeds only (inherited from V0)

The 10-seed confirmation set (120–129) is reserved for V5. V3 locks are
therefore **provisional** pending V5 re-evaluation; V5 can re-open the V3
lock if the confirmation seeds contradict it by more than 1 SE. This
applies uniformly across all branches.

### T-V3.6 — V3/V4 ordering (training-time imbalance vs post-hoc calibration)

Resolved in favor of V3-before-V4 per `DECISION_TREE_AUDIT.md` §2.2 and
reaffirmed under rb-v0.2.0. Rationale unchanged:

1. Causal ordering — training-time imbalance decisions are upstream of
   post-hoc calibration.
2. Leakage control — V4's calibrator fits on V3-forwarded OOF-posthoc
   predictions keep [[condensates/calib-per-fold-leakage]] guarded.
3. Utility orthogonality — V3's REL uses OOF-posthoc baseline, defers
   calibrator-family selection to V4 per
   [[condensates/calib-parsimony-order]].

Logged as Known tension so the tension detector can re-open it if
cross-cohort evidence emerges.

---

## Sources

- `operations/cellml/DESIGN.md` — V3 specification (utility function, AUPRC
  + REL)
- `operations/cellml/MASTER_PLAN.md` — §V3 definition
- `operations/cellml/DECISION_TREE_AUDIT.md` — §2.3 utility resolution,
  §2.2 V3→V4 ordering resolution
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
- [[protocols/v0-strategy]] — locks `training_strategy` and
  `imbalance_family` (rb-v0.2.0); V3 branches on the `imbalance_family`
  lock
- [[protocols/v1-recipe]] — locks `recipe_id` and prescribes the paired-seed
  bootstrap structure V3 reuses
- [[protocols/v2-model]] — locks the single winning model; V3 runs
  conditional on it
- ADR-003 (cel-risk, 2026-01-20) — control-ratio axis source record;
  partially superseded by rb-v0.2.0 per T-V3.3
- rb-v0.2.0 — the rulebook version that introduces the V0 family lock and
  V3 branched refinement (this protocol rewrite is part of the v0.2.0 cut)
