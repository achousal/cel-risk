---
type: protocol
gate: v3-imbalance
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v2-model"
  - "v0_lock: train_control_per_case"
outputs:
  - "locks: [weighting, downsampling_training]"
axes_explored:
  - "weighting ∈ {none, sqrt, log}"
  - "downsampling ∈ {1.0, 2.0, 5.0}  # within-training resampling axis, distinct from V0's split-generation control_ratio — see Known tensions T-V3.1"
axes_deferred:
  - "calibration (locked at V4)"
  - "threshold (locked post-factorial at fixed-spec gate)"
  - "ensemble (informational at V6)"
  - "seed-split confirmation (deferred to V5)"
depends_on:
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/calib-per-fold-leakage]]"
  - "[[condensates/calib-parsimony-order]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[equations/brier-decomp]]"
---

# V3 Imbalance Gate — Lock training-time class-imbalance handling (weighting × downsampling) before committing to V4 calibration fits

V3 is the third decision node of the cellml tree. It answers a single coupled
methodological question — how should training-time class imbalance be handled
— by crossing two training-dynamics axes (loss-function weighting and
within-training downsampling) as a 3×3 grid evaluated under a joint utility
metric. V3 enters after V2 has locked a single winning (recipe, model) pair;
V4 (calibration) is NOT run until V3 closes, because calibration is post-hoc
on an already-imbalance-configured model (per
[[condensates/calib-per-fold-leakage]] and the ordering defended in
`DECISION_TREE_AUDIT.md` §2.2, resolution 2026-04-08).

---

## 1. Pre-conditions

V3 is entered only when all of the following are finalized. Each pre-condition
is a structural invariant that V3 does not revisit.

**V0 gate locked.** [[protocols/v0-strategy]] has emitted a `decision.md` with
`train_control_per_case` locked (from {1.0, 2.0, 5.0}). V3 inherits this
split-generation ratio as a fixed cell-level configuration — the V0 lock
determines how many controls reach the per-split TRAIN partition, and V3's
downsampling axis operates on top of that partition. The two axes compose
multiplicatively, not as substitutes (see Known tensions T-V3.1).

**V1 gate locked.** [[protocols/v1-recipe]] has emitted a `decision.md`
with `recipe_id`, `panel_composition`, `panel_size`, and `ordering_strategy`
locked. V3 operates on the single winning recipe; no recipe axis is re-opened.

**V2 gate locked.** The model-dominance tournament has emitted a `decision.md`
with a single winning model from {LR_EN, LinSVM_cal, RF, XGBoost}. V3 runs
the 3×3 imbalance grid conditional on that locked (recipe, model) pair. If V2
returned a non-dominated Pareto frontier, V3 MUST be run independently on
each frontier member — a stratified V3 — and the locks report per-model.

**Dataset fingerprint** (`projects/<name>/dataset/fingerprint.yaml`), frozen.
The hash binds V3 to the same 20 selection seeds (100–119) used upstream; the
10 confirmation seeds (120–129) remain reserved for V5.

**Environmental invariants**:

- Split strategy 50/25/25 stratified on the primary outcome label (inherited
  from V0, per [[condensates/three-way-split-prevents-threshold-leakage]]).
- Prevalence adjustment MUST be applied to all probability outputs before
  utility evaluation, per [[condensates/downsample-requires-prevalence-adjustment]].
  Raw downsampled scores are NEVER fed into the Brier-reliability proxy.
- Optuna budget: 200 trials per cell (full main-factorial budget, NOT V0's
  50-trial gate budget — V3 is locking a training-dynamics axis that
  interacts with hyperparameter selection, so under-tuning at V3 is a
  correctness hazard).
- Code version pinned via git commit SHA in the gate ledger
  `rulebook_snapshot`.

---

## 2. Search space

V3 crosses two training-time imbalance-handling axes under the fixed locks
from V0–V2. All 9 cells of the 3×3 grid are run at 200 Optuna trials per cell,
20 selection seeds per cell.

### 2.1 `weighting` ∈ {`none`, `sqrt`, `log`}

Class weighting operates on the loss function during model fit. Each level
defines a class-weight vector `w = [w_control, w_case]`:

- **`none`**: `w = [1.0, 1.0]`. No loss rebalancing. The baseline — vanillamax
  default per `DESIGN.md` §Vanillamax discovery principle. This is the
  parsimony floor: any non-`none` choice must earn its lock by Direction
  against `none`.
- **`sqrt`**: `w = [1.0, sqrt(n_control / n_case)]`. Inverse-square-root
  prevalence weighting. Less aggressive than full inverse-frequency —
  attenuates the case up-weighting so that the loss retains some sensitivity
  to control-side variance.
- **`log`**: `w = [1.0, log(n_control / n_case) + 1]`. Logarithmic
  prevalence weighting. Used by the Gen1 discovery phase (per `MASTER_PLAN.md`
  §Gen 1 caveat); included here both to test whether the discovery-era
  default holds under the locked (recipe, model) pair and because it
  represents the most aggressive upweighting in the grid.

Weights enter via the model's native `class_weight` hook (sklearn-style for
LR_EN, LinSVM_cal, RF; `scale_pos_weight` for XGBoost). No custom loss
implementation is invoked.

### 2.2 `downsampling` ∈ {1.0, 2.0, 5.0}

This is the **training-dynamics** downsampling axis: within each per-split
TRAIN partition that V0 has already generated at `train_control_per_case = r_V0`,
a second downsampling step is applied before model fit. The axis value is
the multiplicative ratio relative to V0's lock — i.e., `downsampling = 1.0`
means "no further downsampling, use V0's split as-is"; `downsampling = 5.0`
means "within the V0-locked TRAIN, sample controls down to a further 5:1
case:control ratio applied on top of V0's ratio".

- **1.0**: the parsimony floor. No further downsampling; TRAIN is used as V0
  locked it. This is the baseline against which higher ratios must earn
  their lock by Direction.
- **2.0**: a moderate interior point in the grid; tests whether a single
  halving of controls (relative to the V0 baseline) preserves utility.
- **5.0**: matches the ADR-003 compute-parsimony default per
  [[condensates/downsample-preserves-discrimination-cuts-compute]] actionable
  rule. At this level the effective TRAIN prevalence is
  $\pi_\text{train, V3} = 1 / (1 + r_{V0} \cdot 5)$ — shifted well above the
  population prevalence and requiring prevalence adjustment per
  [[condensates/downsample-requires-prevalence-adjustment]] before any Brier
  or reliability computation.

**Reiteration of axis semantics (see Known tensions T-V3.1):** V0's
`train_control_per_case` is a **split-generation** axis — it determines how
many controls survive stratified allocation into per-seed TRAIN/VAL/TEST
partitions. V3's `downsampling` is a **training-dynamics** axis — it operates
inside the TRAIN partition after split allocation and within-pipeline
prevalent injection have both happened. The two axes are nested (V3 on top
of V0), not alternative. Prevalence adjustment per
[[equations/case-control-ratio-downsampling]] must be applied using the
**composite** effective training prevalence
$\pi_\text{train, eff} = 1 / (1 + r_{V0} \cdot r_{V3})$, not either ratio
alone.

### 2.3 Utility function (joint axis evaluation)

Per `DESIGN.md` §V3 and `DECISION_TREE_AUDIT.md` §1.6 resolution (2026-04-08),
V3 scores each of the 9 cells with a normalized utility that combines
discrimination against calibration quality:

$$U_{(w, d)} = 0.5 \cdot \widetilde{\mathrm{AUPRC}}_{(w, d)} + 0.5 \cdot \left(1 - \widetilde{\mathrm{REL}}_{(w, d)}\right)$$

where

- $\widetilde{\mathrm{AUPRC}}_{(w, d)} = (\mathrm{AUPRC}_{(w,d)} - \mathrm{AUPRC}_\min) / (\mathrm{AUPRC}_\max - \mathrm{AUPRC}_\min)$
  is the min-max-normalized AUPRC over the 9 cells in the grid.
- $\widetilde{\mathrm{REL}}_{(w, d)}$ is the analogous min-max normalization of
  the **prevalence-adjusted** reliability component of Brier per
  [[equations/brier-decomp]] and [[condensates/downsample-requires-prevalence-adjustment]].
  Lower REL is better, so utility uses $(1 - \widetilde{\mathrm{REL}})$.
- Weights are fixed at 0.5/0.5 per the audit resolution. Any change to the
  weighting requires a new ADR.

**AUPRC choice rationale**: V3 metric is AUPRC rather than AUROC because
V3's axes directly change the empirical class frequency fed to the loss
function — AUROC is invariant to prevalence, AUPRC is not. Under extreme
imbalance (celiac cohort $\pi \approx 0.00338$) AUPRC is the more
discriminating metric for imbalance-handling choices. (This rationale is
stated in `DESIGN.md` §V3 but NOT grounded in a dedicated rulebook equation
— see gaps below.)

**REL choice rationale**: per [[condensates/calib-per-fold-leakage]], V3 MUST
use `oof_posthoc` reliability to avoid the optimism-bias leakage path that
per-fold calibration introduces. Specifically, V3's REL is computed on
prevalence-adjusted OOF predictions aggregated across the 20 selection seeds
— NOT on per-fold-calibrated VAL predictions. This keeps V3's utility
orthogonal to V4's calibration-strategy decision.

### 2.4 Cell count

$3 \text{ weightings} \times 3 \text{ downsamplings} = 9$ cells per
(recipe, model) winner from V2. If V2 returned $k$ non-dominated Pareto
frontier members, V3 runs $9k$ cells. Each cell consumes 20 selection seeds
(100–119) and 200 Optuna trials, matching the main-factorial budget.

### 2.5 Axis-condensate mapping

| Axis / element | Grounding condensate / equation | Load-bearing claim |
|---|---|---|
| `downsampling = 5.0` as compute-parsimony pivot | [[condensates/downsample-preserves-discrimination-cuts-compute]] | "5.0 is the ADR-003 default" plus the $\sim 60\times$ speedup argument; V3 re-tests whether Equivalence still holds on top of V0's lock |
| Prevalence adjustment before REL computation | [[condensates/downsample-requires-prevalence-adjustment]] | Any probability output reported into a Brier decomposition MUST be population-scaled via Bayes label-shift correction |
| Training prevalence math | [[equations/case-control-ratio-downsampling]] | Gives $\pi_\text{train, eff}$ under composite (V0, V3) ratios |
| OOF-posthoc reliability (not per-fold) | [[condensates/calib-per-fold-leakage]] | Per-fold calibration leaks hyperparameter-selection signal into the REL proxy — V3 MUST fit reliability on held-out OOF |
| REL as the calibration proxy component | [[equations/brier-decomp]] | Brier = REL − RES + UNC; V3 isolates REL to keep utility orthogonal to V4's calibration decision |
| Parsimony ordering (when Equivalence holds) | [[condensates/calib-parsimony-order]] pattern (extended) and `DESIGN.md` §Parsimony Ordering | logistic_intercept ≺ beta ≺ isotonic is one instance of a five-axis parsimony tiebreaker; the analogous V3 orderings are `none ≺ sqrt ≺ log` and `1.0 ≺ 2.0 ≺ 5.0` |

### 2.6 Axes flagged as lacking dedicated condensate support

Candidate additions to the rulebook; V3 runs against `DESIGN.md` and
`DECISION_TREE_AUDIT.md` for these axes, but the grounding is thin:

- **AUPRC as the discrimination metric at V3.** No condensate formalizes
  "under imbalance-handling axis changes, AUPRC is the load-bearing
  discrimination metric; AUROC is inappropriate here." TODO: add
  `condensates/auprc-for-imbalance-handling.md`.
- **0.5/0.5 utility weighting.** `DECISION_TREE_AUDIT.md` §2.3 resolution
  adopts 0.5/0.5 to correct the earlier raw-subtraction "net_benefit" scale
  mismatch, but there is no dedicated condensate formalizing the equal-weight
  claim. TODO: add `condensates/imbalance-utility-equal-weight.md`.
- **V3's nested (split-generation × training-dynamics) downsampling
  composition.** No condensate grounds the composition rule
  $\pi_\text{train, eff} = 1 / (1 + r_{V0} \cdot r_{V3})$ — the axes are
  silently assumed multiplicative. TODO: add
  `condensates/nested-downsampling-composition.md`, which should also speak
  to T-V3.1 below.
- **Weighting-level enumeration.** `DESIGN.md` §Factorial Factors lists
  `{none, sqrt, log}` but no condensate formalizes why these three are the
  relevant axis values (as opposed to, e.g., `{none, balanced, log}` where
  `balanced` is full inverse-frequency). This is inherited from ADR-era
  choices that predate the rulebook.

---

## 3. Success criteria

All V3 decisions use the fixed falsifier rubric in `SCHEMA.md`. The primary
lock test is a pair of **Direction** claims — one per axis — against the
parsimony-floor baseline cell `(weighting = none, downsampling = 1.0)`. Each
of the remaining 8 non-baseline cells is compared against the baseline by
$\Delta U$ (utility delta) with 95% bootstrap CI over 1000 seed-level paired
resamples (20 selection seeds are the paired units, per `SCHEMA.md` metric
rules for AUROC / PR-AUC / Brier).

### 3.1 Cell-level claim types

| Claim type | Criterion | V3 decision |
|---|---|---|
| **Direction** (cell beats baseline) | $\|\Delta U\| \ge 0.02$ AND 95% bootstrap CI excludes 0 | Cell is a lock candidate |
| **Equivalence** (cell ≈ baseline) | $\|\Delta U\| < 0.01$ AND 95% bootstrap CI $\subset [-0.02, 0.02]$ | Cell does NOT beat baseline; baseline retains lock by parsimony |
| **Inconclusive** | Neither met | Cell forwards to V4 as a candidate; V3 cannot lock against it |

### 3.2 Axis-level lock rule

V3 locks both axes jointly based on the claim pattern across the 9 cells:

**Case A — Direction from a non-baseline cell (A.k.a. "a grid cell earns its lock"):**

If one or more non-baseline cells $(w^\star, d^\star)$ satisfy the Direction
criterion against the baseline, the cell with the largest $\Delta U$ with CI
excluding 0 is the candidate. Secondary check: that candidate's $\Delta U$
must exceed every other non-baseline cell's $\Delta U$ by the Equivalence
criterion — i.e., the winner must be at least pairwise-Equivalent-or-better
against each runner-up. If yes: lock `weighting = w^\star` and
`downsampling = d^\star`. If two or more cells tie by Equivalence on
$\Delta U$, invoke the parsimony tiebreaker (§3.4).

**Case B — Equivalence across the grid (parsimony to baseline):**

If all 8 non-baseline cells return Equivalence against the baseline, lock
`weighting = none, downsampling = 1.0` by parsimony. This is the expected
outcome under the vanillamax discovery principle — no training-time
rebalancing is required — and is consistent with
[[condensates/downsample-preserves-discrimination-cuts-compute]] when the V0
split-generation ratio already carries the required imbalance handling.

**Case C — Mixed Equivalence / Inconclusive (no Direction):**

If no cell satisfies Direction against the baseline, but some cells return
Inconclusive rather than Equivalence, V3 cannot declare a clean parsimony
lock. Default: lock the baseline `(none, 1.0)` and forward the Inconclusive
cells to V4 as calibration-candidates (see §4). Rationale: V3's job is to
decide when to depart from vanillamax — absent positive evidence of a
departure, the default stands.

**Case D — Direction dispute (two non-adjacent cells beat baseline with
non-overlapping Directions):**

If the Direction-passing cells do not admit a single winner under the
pairwise-Equivalence check in Case A (e.g., $(sqrt, 1.0)$ beats baseline by
+0.03 AND $(none, 5.0)$ beats baseline by +0.04 AND they are Directionally
separated from each other), V3 returns Inconclusive on the joint axis and
forwards all Direction-passing cells to V4 per §4.

### 3.3 Metric specification

- **AUPRC**: computed on prevalence-adjusted, OOF-posthoc-calibrated
  predictions across the 20 selection seeds. 95% bootstrap CI over 1000
  resamples of seeds (paired by seed across cells, per `SCHEMA.md`
  metric rules).
- **REL (Brier reliability component)**: per [[equations/brier-decomp]],
  computed with $K = 10$ quantile bins on the same prevalence-adjusted OOF
  predictions. 95% bootstrap CI over 1000 seed resamples.
- **$\Delta U$**: paired-bootstrap delta (same-seed cell utilities
  differenced per resample), preserving the pairing structure as in
  [[protocols/v1-recipe]] §3.3.
- Forbidden: per-fold-calibrated VAL predictions, raw (non-prevalence-
  adjusted) downsampled scores, cell-averaged SE. These reproduce the
  V1-era statistical mistakes documented in `DECISION_TREE_AUDIT.md` §1.3
  and the leakage path formalized in [[condensates/calib-per-fold-leakage]].

### 3.4 Parsimony tiebreaker (Equivalence only)

When §3.2 Case A returns two or more candidate winners by Equivalence on
$\Delta U$, apply the parsimony ordering (per `DESIGN.md` §Parsimony Ordering):

- **Weighting**: `none` ≺ `sqrt` ≺ `log` (simplest → most complex loss
  rebalancing).
- **Downsampling**: `1.0` ≺ `2.0` ≺ `5.0` (smallest → largest within-TRAIN
  ratio; the parsimony move is toward no further downsampling on top of V0).

Apply weighting first, then downsampling. Rationale: weighting modifies the
loss surface (a modeling choice) while downsampling modifies the empirical
distribution (a data choice). Under Equivalence, the parsimony move is to
leave data alone first, then leave the loss alone — but the modeling choice
dominates the data choice in parsimony precedence because loss changes are
easier to audit than sampling changes.

Parsimony-tie resolution follows `[[condensates/parsimony-tiebreaker-when-equivalence]]`
with this axis's orders: **weighting `none` ≺ `sqrt` ≺ `log`** and
**downsampling `1.0` ≺ `2.0` ≺ `5.0`**. The meta-condensate is the parent
rule that governs Equivalence-triggered tiebreakers across the factorial;
the V3-specific orderings above are the local instantiations invoked here,
with the weighting-before-downsampling precedence retained as this axis's
internal composition rule.

### 3.5 Disagreement across V2 Pareto frontier (conditional on V2 returning non-dominated set)

If V2 emitted $k > 1$ frontier models, V3 is run independently on each. The
lock rule is:

- If all $k$ V3 runs return the same (weighting, downsampling), lock those
  values globally and carry $k$ candidate models into V4.
- If V3 runs disagree on the imbalance-lock across frontier models, lock
  per-model (i.e., each V2 frontier member carries its own V3 lock into V4)
  and flag the disagreement as a tension per `SCHEMA.md` §Rulebook updates.

---

## 4. Fallbacks

When the §3.2 axis-level lock rule returns Inconclusive (Case C or Case D),
V3 cannot lock the imbalance axes. The operational fallback moves are:

### 4.1 Widen the Optuna budget before promoting

If Inconclusive is driven by wide CIs (bootstrap CI half-width $> 0.02$
despite no Direction), the interpretation is that 200 trials × 20 seeds is
under-powered for this (recipe, model) pair at this utility function. Before
escalating, re-run only the contested cells (those with CI crossing 0) at
400 Optuna trials with the same 20 seeds. This is a power check, not
hyperparameter-peeking, because the axis under test is imbalance handling,
not model hyperparameters; the extra trials improve the estimate of the
cell's expected performance under its locked imbalance configuration.

### 4.2 Lock baseline, forward the 9 cells as candidates to V4

If widening does not resolve the Inconclusive, V3's default is to lock the
parsimony baseline `(none, 1.0)` and forward the full 9-cell grid to V4 as
candidates. V4 (calibration) will then fit calibrators on each of the 9
cells' OOF-posthoc predictions and re-evaluate the joint (imbalance ×
calibration) surface. This effectively promotes the (weighting, downsampling)
axis from V3-locked to V4-contested — with the understanding that V4's
post-hoc calibration can partially compensate for imbalance-handling choices
that V3 could not separate on discrimination grounds.

This is the explicit `DESIGN.md` §V3 fallback path and follows the ordering
principle in `DECISION_TREE_AUDIT.md` §2.2 resolution: training-dynamics
decisions are made first, but when they cannot be cleanly separated,
post-hoc calibration provides a second adjudication layer. Per
[[condensates/calib-per-fold-leakage]], this forward must preserve
OOF-posthoc predictions — V4 cannot fit per-fold calibrators on V3-forwarded
cells without re-introducing the leakage V3 took care to avoid.

### 4.3 Parsimony lock under Equivalence

When Equivalence holds across the grid (Case B, not Inconclusive), the
parsimony move is `(weighting = none, downsampling = 1.0)`. This is the
vanillamax baseline. Rationale stacks:

- No artificial loss rebalancing (per `DESIGN.md` §Vanillamax discovery).
- No composite downsampling beyond V0's locked split-generation ratio (no
  further departure from the population prevalence than V0 already commits).
- Fewer training-time operations to audit and reproduce.

### 4.4 V0 control-ratio sensitivity check (advisory, not a lock gate)

Because V3's `downsampling` axis operates on top of V0's
`train_control_per_case` lock, there is a latent sensitivity: if V0 locked a
very aggressive ratio (e.g., $r_{V0} = 5$), V3's `downsampling = 5.0` cell
takes training prevalence to $1/(1 + 25) = 0.038$ from the population's
0.00338 — that is an 11× shift, and AUPRC estimates on the resulting
very-few-control TRAIN partition become noisy. If the locked V0 ratio and
the candidate V3 downsampling jointly exceed an effective ratio $r_\text{eff}
= r_{V0} \cdot r_{V3} > 25$, V3 SHOULD flag the observation as an
"extreme-composite-imbalance cell" in `observation.md` and route the flag
to `tensions.md`. This check does NOT veto the V3 lock but should bias
downstream interpretation (and on a new cohort may argue for reopening V0).

---

## 5. Post-conditions

What V3 writes into `projects/<name>/gates/v3-imbalance/decision.md`:

- **Locks passed forward** (on the happy path):
  1. `weighting = <locked value>` — with the exact claim type (Direction,
     Equivalence-parsimony, or Inconclusive→baseline) that justified it.
  2. `downsampling_training = <locked value>` — claim type recorded.
- **Prevalence-adjusted metric table**: AUPRC, REL, and $U$ for each of the
  9 cells, with 95% bootstrap CIs. Raw (non-prevalence-adjusted) metrics are
  also logged per [[condensates/downsample-requires-prevalence-adjustment]]
  provenance requirement, but only the adjusted metrics enter the decision.
- **Composite effective training prevalence table**: $\pi_\text{train, eff}
  = 1 / (1 + r_{V0} \cdot r_{V3})$ per cell, so V4 can consume the exact
  prevalence at which each forwarded cell was fit.
- **V2 frontier disposition**: if V2 emitted $k > 1$ frontier models, records
  per-model V3 locks and flags any disagreement per §3.5.
- **Unlocked axes**: any axis promoted to V4 per Fallback §4.2 is recorded
  here with its forwarded-cell list.
- **Extreme-composite-imbalance flags**: any cell where
  $r_{V0} \cdot r_{V3} > 25$ per §4.4.

What advances to V4:

- The locked (weighting, downsampling) — or the full 9-cell grid if §4.2
  fallback triggered.
- The V3 winner's OOF-posthoc prediction table (prevalence-adjusted) — V4's
  calibrators fit on this exact table, NOT on per-fold VAL predictions, per
  [[condensates/calib-per-fold-leakage]].
- The dataset fingerprint hash (binds V4 to the same split realizations).
- The V1–V2 locks (recipe, panel, model) — unchanged.

What remains open:

- Calibration strategy (V4).
- Threshold selection (post-factorial).
- Seed-split confirmation (V5) — holds out seeds 120–129 and re-evaluates
  the V3 lock.

---

## Known tensions

Acknowledged per `DECISION_TREE_AUDIT.md` and the V0 protocol's Known
tensions section; surfaced here because V3 inherits them and several are
first-class tensions for V3 specifically.

### T-V3.1 — Control-ratio semantics double-use (inherited from V0; ADR-003 wording needs sharpening)

`train_control_per_case` appears twice in the cellml tree:

- **V0 layer (split-generation)**: determines how many controls survive
  stratified allocation into the per-seed TRAIN partition. Locked at V0 from
  {1.0, 2.0, 5.0} per [[condensates/downsample-preserves-discrimination-cuts-compute]].
- **V3 layer (training-dynamics, named `downsampling`)**: within-training
  resampling of TRAIN before model fit. Explored at V3 from {1.0, 2.0, 5.0}.

This protocol formalizes the intended relationship: **V3's downsampling
operates ON TOP OF V0's 5:1 split** (or whatever V0 locked). The two axes
compose multiplicatively — effective ratio is $r_\text{eff} = r_{V0} \cdot
r_{V3}$, effective prevalence is $\pi_\text{train, eff} = 1 / (1 + r_\text{eff})$.

However, ADR-003 (per its excerpt in
[[condensates/downsample-preserves-discrimination-cuts-compute]]) uses the
phrase "control downsampling" to cover BOTH operations without distinction.
This is a semantic double-use that `DECISION_TREE_AUDIT.md` flagged but did
not resolve.

**Resolution proposed here (non-binding until ADR-003 is revised):**

- Reserve the term `train_control_per_case` for the V0 split-generation axis.
- Reserve the term `downsampling` (or `downsampling_training`) for the V3
  training-dynamics axis.
- Update ADR-003 wording to make the distinction explicit, and add
  `condensates/nested-downsampling-composition.md` to formalize the
  multiplicative composition.
- Until ADR-003 is sharpened, every V3 `ledger.md` MUST state which ratio
  each axis value refers to.

This is a candidate rulebook PR: "ADR-003 language sharpening + new
`nested-downsampling-composition` condensate". Logged here pending resolution.

### T-V3.2 — V3/V4 ordering (training-time imbalance vs post-hoc calibration)

`DECISION_TREE_AUDIT.md` §2.2 questions whether training-time imbalance
handling (V3) should precede post-hoc calibration (V4), or the reverse. The
audit's resolution (2026-04-08) adopts V3-before-V4, on the principle that
training-time decisions are causally upstream of post-hoc corrections.

**This protocol's position: V3-before-V4 is correct, grounded in
[[condensates/calib-per-fold-leakage]].**

Rationale:

1. **Causal ordering**: V3 modifies the training distribution (via
   downsampling) and the loss surface (via weighting). V4 modifies the
   mapping from model output to probability. The model output is a
   function of V3's configuration; changing V4 after V3 affects only the
   output mapping, not the fit. Reversing the order would require V4 to be
   applied before a model has been fit under the imbalance configuration
   — which is not a well-defined operation.
2. **Leakage control**: [[condensates/calib-per-fold-leakage]] makes clear
   that calibrators fit inside the hyperparameter-selection loop inherit
   optimism bias. If V4 were run first (i.e., the calibration strategy were
   locked before V3), V4's calibrator fits would have to be re-done under
   each V3 cell's training distribution, and the leakage path would
   re-activate per-cell. Running V3 first and forwarding OOF-posthoc
   predictions to V4 is the only ordering that preserves the leakage-free
   property of `oof_posthoc`.
3. **Utility orthogonality**: V3's utility (AUPRC + REL) already carries a
   calibration proxy via REL, so V3 is not blind to calibration — it simply
   uses the OOF-posthoc baseline for REL and defers the *selection* of
   calibrator family to V4. If V4 preceded V3, V3's REL would either be
   raw (contradicting [[condensates/downsample-requires-prevalence-adjustment]])
   or would presuppose V4's family choice (contradicting the parsimony
   structure in [[condensates/calib-parsimony-order]]).

The counter-argument (V4-before-V3) would be that calibration can
"absorb" imbalance by re-shaping the probability output. But absorbing is
not the same as correctly handling — a poorly weighted loss produces a
model that *cannot* be calibrated to correct probabilities on a minority
class that the loss effectively ignored. Training-time imbalance is a
feature-space fix; calibration is an output-space fix; the feature-space fix
must come first.

Status: resolved in favor of V3-before-V4 per the audit and reaffirmed
here. Logged as a Known tension so that the tension detector can re-open it
if cross-cohort evidence emerges.

### T-V3.3 — AUPRC not formalized in a rulebook equation

V3's utility function uses AUPRC as the discrimination metric. AUPRC is
mentioned in `DESIGN.md` §V3 and in `SCHEMA.md` §Fixed falsifier rubric
(as a bootstrap-CI metric), but no `equations/` file defines AUPRC or
bounds its bootstrap behavior under low prevalence. The celiac cohort has
$n_\text{case} = 148$, of which ~74 reach TRAIN and ~37 reach VAL/TEST per
seed — bootstrapping AUPRC on 37-case slices has non-trivial tail behavior
that a dedicated equation should document. TODO:
`equations/auprc-definition-and-bootstrap.md`. Until then, V3 operates
against the implicit standard sklearn `average_precision_score` over
prevalence-adjusted predictions and the 1000-resample rule from `SCHEMA.md`.

### T-V3.4 — 0.5/0.5 utility weighting is a choice, not a derived rule

`DECISION_TREE_AUDIT.md` §2.3 resolution adopts equal 0.5/0.5 weighting for
AUPRC-normalized and REL-normalized terms. This choice is defended in the
audit ("normalized to respective ranges; equal weight") but is not grounded
in a condensate. A plausible failure mode is that on a cohort where AUPRC
improvements are small but REL improvements are large (or vice versa), the
equal-weight rule picks a utility-dominant cell whose AUPRC gain is
practically nil. Flagged as TODO: emit
`condensates/imbalance-utility-equal-weight.md` so the choice is falsifiable.
Until then, V3 `observation.md` MUST also log the un-weighted $\Delta \mathrm{AUPRC}$
and $\Delta \mathrm{REL}$ per cell so a retrospective re-weighting is
reconstructable.

### T-V3.5 — V3 submitted with 20 selection seeds only (inherited from V0)

Same structure as V0's T-V0.3: the 10-seed confirmation set (120–129) is
reserved for V5, so all V3 Direction / Equivalence claims are
within-selection-seed bootstraps. V3 locks are therefore **provisional**
pending V5 re-evaluation; V5 can re-open the V3 lock if the confirmation
seeds contradict it by more than 1 SE.

### T-V3.6 — Per-fold vs OOF-posthoc REL (leakage guard)

V3's REL MUST be OOF-posthoc per [[condensates/calib-per-fold-leakage]].
This is a structural precondition, not a Known tension in the "unresolved"
sense — but it is a frequent implementation failure mode because the
sklearn default `CalibratedClassifierCV` is per-fold. Any V3 implementation
that silently uses per-fold REL reproduces the `DECISION_TREE_AUDIT.md`
§1.2 "inconsistent uncertainty handling" pattern and is a rulebook
violation. The tension detector should flag any V3 `observation.md` that
does not cite `calibration.strategy: oof_posthoc` in its provenance.

---

## Sources

- `operations/cellml/DESIGN.md` — V3 specification (3×3 grid, normalized
  utility, AUPRC + REL)
- `operations/cellml/MASTER_PLAN.md` — §V3 definition and decision tree
- `operations/cellml/DECISION_TREE_AUDIT.md` — §2.3 V4–V5 joint resolution
  (now V3), §1.6 net_benefit scale fix, §2.2 V3→V4 ordering resolution,
  §3.5 V0/V3 control-ratio double-use tension
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
- [[protocols/v0-strategy]] — locks `train_control_per_case`, the
  split-generation axis that V3's downsampling composes with
- [[protocols/v1-recipe]] — locks `recipe_id` and prescribes the paired-seed
  bootstrap structure V3 reuses
- ADR-003 (cel-risk, 2026-01-20) — source decision record for the
  control-ratio axis; language needs sharpening per T-V3.1
