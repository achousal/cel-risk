---
type: protocol
gate: v0-strategy
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: null"
outputs:
  - "locks: [training_strategy, prevalent_train_frac, train_control_per_case]"
axes_explored:
  - "training_strategy: [incident_only, incident_plus_prevalent, prevalent_only]"
  - "prevalent_train_frac: [0.25, 0.5, 1.0]  # conditional on training_strategy == incident_plus_prevalent"
  - "train_control_per_case: [1.0, 2.0, 5.0]"
axes_deferred:
  - "model (crossed in V0 as a robustness check, not locked; actual model-axis lock happens at V2)"
  - "recipe composition beyond R1_sig (4p) and R1_plateau (8p) (locked at V1)"
  - "calibration (locked at V4)"
  - "weighting (locked at V3, joint with downsampling)"
  - "downsampling (V3 training-dynamics axis; distinct from the V0 split-generation control_ratio — see Known tensions)"
  - "panel size beyond the two representative recipes (locked at V1)"
  - "seed-split confirmation (deferred to V5)"
depends_on:
  - "[[condensates/three-way-split-prevents-threshold-leakage]]"
  - "[[condensates/prevalent-restricted-to-train]]"
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/perm-validity-full-pipeline]]"
  - "[[condensates/feature-selection-needs-model-gate]]"
  - "[[condensates/calib-parsimony-order]]"
  - "[[equations/perm-test-pvalue]]"
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[equations/stratified-split-proportions]]"
---

# V0 Strategy Gate — Lock training strategy and control ratio before committing main-factorial compute

The V0 gate is the first decision node of the cellml tree. It answers two
coupled methodological questions — case partitioning and control ratio — using
four models and two representative recipes as a robustness check, so that the
main factorial (1,566 cells at the baseline scope) does not inherit an
unvalidated partitioning prior.

---

## 1. Pre-conditions

V0 is entered only when all of the following are finalized. Each pre-condition
is a structural invariant that V0 does not revisit; changing any of them
requires a new project slug.

**Dataset fingerprint** (`projects/<name>/dataset/fingerprint.yaml`), frozen:

- `n_cases`, `n_controls`, `prevalence` (population rate, used by the Bayes
  prevalence correction; see [[condensates/downsample-requires-prevalence-adjustment]]).
- `n_features` (full assayed protein set — V0 operates on the full universe
  per the vanillamax discovery principle; feature pre-filtering is a V1 axis).
- `cohort` and `platform` strings.
- `hash` (sha256 over `splits/seed_*.csv`) — binds the gate to the specific
  split realizations.

**Discovery-phase outputs** (produced before V0, consumed as priors):

- RRA significance table (trunk T1 source) — used to derive R1_sig at 4p
  and R1_plateau at 8p under the size rules declared in `manifest.yaml`.
- Compiled sweep (30-seed AUROC averages across panel sizes 4–25) — used by
  the 3-criterion rule to set the 8p plateau for R1_plateau. On a new cohort
  where the discovery sweep has not run, V0 cannot proceed until R1_sig and
  R1_plateau are derivable from `ced derive-recipes`.
- Consensus panels for R1_sig and R1_plateau (the two V0 recipes), derived
  deterministically from the manifest — no hand-picking.

**Environmental invariants**:

- Split strategy locked to 50/25/25 stratified on the primary outcome label,
  per [[condensates/three-way-split-prevents-threshold-leakage]] and the
  allocation math in [[equations/stratified-split-proportions]]. V0 does not
  revisit the split ratio; it operates within it.
- Split seeds locked to 100–119 (20 selection seeds). The 10-seed confirmation
  set (120–129) is reserved for V5 and is NOT used at V0.
- Code version pinned via git commit SHA recorded in the gate ledger
  `rulebook_snapshot`.
- Optuna budget for V0 cells: 50 trials (gate budget; main factorial uses 200
  trials per cell).
- Permutation-test $B$ at 200 (V0 gate budget per [[equations/perm-test-pvalue]]
  boundary condition `B \ge 200` for $\alpha=0.05$).

---

## 2. Search space

V0 crosses three axes under the fixed structural invariants above. Each axis
lists allowed values with a citation to the condensate that justifies its
inclusion in the grid.

### 2.1 `training_strategy` ∈ {`incident_only`, `incident_plus_prevalent`, `prevalent_only`}

- **`incident_only`**: TRAIN, VAL, and TEST all drawn from incident cases
  only. Per [[condensates/prevalent-restricted-to-train]], incident cases are
  the prospective-screening reference population — this level is the
  "aligned" baseline against which augmentation is tested.
- **`incident_plus_prevalent`**: prevalent cases injected into TRAIN at
  `prevalent_train_frac` (see 2.2). VAL and TEST remain incident-only —
  hook-enforced per [[condensates/prevalent-restricted-to-train]]. Included
  because the condensate explicitly flags the 0.5 fraction as "a prior, not a
  lock — the factorial should test it".
- **`prevalent_only`**: TRAIN drawn from prevalent cases only. VAL/TEST
  remain incident-only. Included as an upper-bound on the distribution-shift
  hypothesis: if prevalent training beats incident training on an
  incident-only TEST, the "prospective alignment" argument is weakened.
  <!-- TODO: no direct condensate explicitly enumerates prevalent_only as an allowed level; the level extends [[condensates/prevalent-restricted-to-train]]'s prevalent/incident distinction to its endpoint but is not named there. Candidate tension: either extend the condensate to enumerate this third level, or drop it from the V0 grid. -->

### 2.2 `prevalent_train_frac` ∈ {0.25, 0.5, 1.0}

Conditional on `training_strategy == incident_plus_prevalent`.

- **0.5**: the ADR-002 default fraction; [[condensates/prevalent-restricted-to-train]]
  flags it explicitly as "untested prior. It trades signal gain (more
  positives) against distribution shift (more retrospective bias in TRAIN).
  The factorial V0 axis should measure AUROC at `prevalent_train_frac` ∈
  {0.0, 0.5, 1.0}".
- **1.0**: upper bound per the same condensate ("∈ {0.0, 0.5, 1.0}").
- **0.25**: added per `DECISION_TREE_AUDIT.md` §3.5 action P3 to bracket the
  space below 0.5 (the condensate enumeration lists 0.0 as the lower bound;
  0.25 is a grid interior point introduced to catch a low-fraction optimum).
  <!-- TODO: the condensate enumerates {0.0, 0.5, 1.0}; the V0 grid uses {0.25, 0.5, 1.0} instead of {0.0, 0.5, 1.0}. The `training_strategy = incident_only` level already covers the 0.0 case (no prevalent injection), so {0.25, 0.5, 1.0} is not strictly a condensate-enumerated grid but is the operational equivalent. Flag for condensate revision. -->

### 2.3 `train_control_per_case` ∈ {1.0, 2.0, 5.0}

- **5.0**: the ADR-003 default per
  [[condensates/downsample-preserves-discrimination-cuts-compute]], where the
  actionable rule states "The V0 gate locks `train_control_per_case` from
  {1.0, 2.0, 5.0} based on measured AUROC equivalence. 5.0 is the ADR-003
  default but the factorial should test all three."
- **2.0** and **1.0**: lower bounds from the same condensate. The condensate's
  boundary condition "Ratios below 2:1 are NOT permitted on the celiac cohort
  — the factorial grid explicitly bounds the axis at {1, 2, 5}" caps the
  lower end, making 1.0 the minimum permitted grid point (boundary value,
  not a test below it).
- Prevalence adjustment under each ratio follows the Bayes correction in
  [[equations/case-control-ratio-downsampling]] (π_train = 1 / (1 + r)),
  required per [[condensates/downsample-requires-prevalence-adjustment]]
  whenever the observation table reports Brier / ECE / reliability.

### 2.4 Model axis (robustness check, not an exploration axis)

- Levels: `LR_EN`, `LinSVM_cal`, `RF`, `XGBoost`.
- Crossed under all (strategy × control_ratio) combinations so the lock
  test can demand Dominance across all four models (see 3.3). The model
  itself is NOT locked at V0; that decision belongs to V2.

### 2.5 Recipe axis (representative subset, not an exploration axis)

- Levels: `R1_sig` (4p, significance_count size rule) and `R1_plateau` (8p,
  3-criterion unanimous). Two recipes chosen to deconfound strategy choice
  from panel size. Consensus ordering is fixed across both, so this axis does
  not test ordering — that is V1.

### 2.6 Model-gate permutation test (pre-gate filter, not an explored axis)

Before any (strategy, control_ratio) cell is considered eligible for the lock
test, its model must pass the Stage-1 permutation gate at $p < 0.05$ per
[[condensates/feature-selection-needs-model-gate]] and
[[condensates/perm-validity-full-pipeline]]. The permutation test MUST re-run
the full inner pipeline per permutation (not partial permutation after feature
selection), with $B = 200$ per [[equations/perm-test-pvalue]] boundary
conditions. Cells whose model fails the gate are marked `excluded_from_lock`
in the observation table and do not contribute to the Dominance claim.

### 2.7 Cell count

$5 \text{ strategies} \times 3 \text{ control ratios} \times 4 \text{ models}
\times 2 \text{ recipes} = 120$ cells, at 20 selection seeds (100–119) per
cell. Strategy expansion: 1 (`incident_only`) + 3 (`incident_plus_prevalent`
× 3 fractions) + 1 (`prevalent_only`) = 5 strategy levels. Matches the count
in `MASTER_PLAN.md` §V0.

---

## 3. Success criteria

All claims use the fixed falsifier rubric in `SCHEMA.md`. The primary lock
test is a **Dominance** claim: the winning (strategy, control_ratio) pair
must beat its alternatives by the Direction criterion **independently on each
of the four model axes**. Metric: AUROC on the incident-only TEST split, 95%
bootstrap CI over 1000 resamples of seeds 100–119.

### 3.1 `training_strategy` lock

**Primary Direction claim (lock path):**

- Lock `training_strategy = incident_only` if, for every model $m \in$
  {LR_EN, LinSVM_cal, RF, XGBoost}:
  $\Delta\text{AUROC}_m = \text{AUROC}(\text{incident\_only}) -
  \max(\text{AUROC}(\text{incident\_plus\_prevalent at best } f),
  \text{AUROC}(\text{prevalent\_only})) \ge 0.02$
  AND the 95% bootstrap CI over seed resamples excludes 0.

**Alternative Direction claim:**

- Lock `training_strategy = incident_plus_prevalent` (with its best
  `prevalent_train_frac`, see 3.2) if the analogous Direction holds with
  incident_plus_prevalent replacing incident_only as the winner, on all four
  models.

**Equivalence fallthrough:**

- If Direction fails but Equivalence holds on the incident_only vs
  incident_plus_prevalent pair ($|\Delta\text{AUROC}| < 0.01$ AND 95% CI
  $\subset [-0.02, 0.02]$), on all four models, lock the simpler partition:
  `training_strategy = incident_only` (parsimony per
  [[condensates/prevalent-restricted-to-train]] "If a new cohort lacks a
  meaningful incident/prevalent distinction... the axis should be dropped...
  rather than defaulted") — i.e., the simpler partitioning is the default when
  augmentation does not improve the Direction claim.

**Inconclusive:**

- If neither Direction nor Equivalence holds on all four models (e.g.,
  Direction on two and Equivalence on two, or any Inconclusive), fall through
  to the fallback path (§4.1).

### 3.2 `prevalent_train_frac` lock (conditional on 3.1 picking incident_plus_prevalent)

Only evaluated if 3.1 locks `training_strategy = incident_plus_prevalent`.

- **Direction claim**: lock the fraction $f^\star$ if
  $\text{AUROC}(f^\star) - \text{AUROC}(f')  \ge 0.02$ with 95% CI excluding
  0 for every $f' \ne f^\star$ in {0.25, 0.5, 1.0}, on all four models.
- **Equivalence fallthrough**: if Equivalence holds across all three
  fractions on all four models, lock $f = 0.5$ (ADR-002 default — falling
  back to the documented prior per [[condensates/prevalent-restricted-to-train]]
  evidence gap "ADR-002 asserts a 50% sampling fraction... provides no
  empirical justification").
- **Inconclusive**: defer the fraction to V1 (promote to V1 axis per §4.2).

### 3.3 `train_control_per_case` lock

- **Direction claim**: lock the ratio $r^\star$ if
  $\text{AUROC}(r^\star) - \text{AUROC}(r') \ge 0.02$ with 95% CI excluding 0
  for every $r' \ne r^\star$ in {1, 2, 5}, on all four models.
- **Equivalence fallthrough**: if the pairwise Equivalence criterion holds
  across all three ratios on all four models, lock $r = 5$ per
  [[condensates/downsample-preserves-discrimination-cuts-compute]] ("5.0 is
  the ADR-003 default" plus the ~60× compute savings argument in
  [[equations/case-control-ratio-downsampling]]). The condensate's falsifier
  is explicit that Equivalence confirms the condensate; the parsimony move is
  then to the compute-cheapest ratio under the Equivalence, which on this
  cohort is $r = 5$.
- **Inconclusive**: defer to V1 (§4.2).

### 3.4 Dominance composition

The gate's **aggregate lock** is a Dominance claim per the SCHEMA rubric:
Direction holds **independently on each axis** (training_strategy AND
control_ratio). V0 locks both axes only if the Direction or
Direction+Equivalence criteria above hold simultaneously. If only one axis
meets the criterion, lock that axis and promote the other to V1.

### 3.5 Brier / calibration cross-check (advisory, not a lock gate)

Per [[condensates/downsample-requires-prevalence-adjustment]], the observation
table MUST log both raw and prevalence-adjusted Brier. If the Dominance claim
holds on AUROC but a model's prevalence-adjusted Brier at the locked
(strategy, ratio) degrades by $|\Delta\text{Brier}| \ge 0.02$ with CI
excluding 0 relative to an unlocked alternative, flag a cross-metric tension
(Fallback C in the retrospective ledger) and escalate to V4 (calibration).
This check does NOT veto the V0 lock — calibration parsimony at V4 is the
adjudicator per [[condensates/calib-parsimony-order]].

---

## 4. Fallbacks

When the lock test returns Inconclusive (neither Direction nor Equivalence
holds across all four models), V0 cannot lock the contested axis. The
operational fallback moves are:

### 4.1 Widen the search before promoting

If Inconclusive is driven by a **single** dissenting model (3 out of 4 meet
Direction, 1 does not), the interpretation is that the dissenting model may
have a fragile fit at V0's 50-trial budget. Re-run the dissenting model's
cells at 200 trials (the main-factorial budget) before declaring
Inconclusive. This is not a hyperparameter-peeking violation because the
axis under test is partition, not hyperparameters; the 200-trial re-run is
a power check.

### 4.2 Promote contested axes to V1

If widening does not resolve the Inconclusive, promote the contested axis to
V1 as a full factorial factor:

- Training strategy promotion: cell count grows from 1,566 to 4,698 (3×
  multiplier, with prevalent_train_frac absorbed into V1).
- Control ratio promotion: cell count grows from 1,566 to 4,698 (3×
  multiplier).

This is the explicit `MASTER_PLAN.md` §V0 fallback. The cost is
main-factorial wallclock; the benefit is a principled V1 resolution under
the full tuning budget.

### 4.3 Parsimony lock under Equivalence

Where Equivalence holds (not Inconclusive), the parsimony moves are
axis-specific:

- **Strategy**: prefer `incident_only` per §3.1 Equivalence fallthrough
  (fewer augmentation operations, no distribution shift, prospective
  alignment by construction — three parsimony arguments stack).
- **Prevalent fraction**: prefer $f = 0.5$ (documented ADR-002 default; the
  condensate treats 0.5 as the prior to fall back to).
- **Control ratio**: prefer $r = 5$ (compute-parsimony wins under
  AUROC-Equivalence; calibration invariance is guaranteed as long as the
  prevalence correction is applied per §3.5).
- **Calibration-tied parsimony**: if the cross-check in §3.5 surfaces a
  Brier tension, the parsimony ordering on the calibration axis is
  logistic_intercept ≺ beta ≺ isotonic per [[condensates/calib-parsimony-order]],
  but the call is deferred to V4.

---

## 5. Post-conditions

What V0 writes into `projects/<name>/gates/v0-strategy/decision.md`:

- **Locks passed forward** (on the happy path):
  1. `training_strategy = <locked value>` — with the exact claim type
     (Direction or Equivalence-parsimony) that justified it.
  2. `prevalent_train_frac = <locked value>` — only present if
     `training_strategy == incident_plus_prevalent`. Otherwise recorded as
     N/A.
  3. `train_control_per_case = <locked value>` — claim type recorded.
- **Permutation-test pass list**: the set of (model × recipe) combinations
  that passed the Stage-1 model gate at $p < 0.05$. This list is the
  V1 `axes_deferred.model` input per [[condensates/feature-selection-needs-model-gate]]
  — V1 consensus panel construction MUST consume only models from this list.
- **Prevalence-adjusted Brier table**: raw and adjusted Brier per cell, per
  [[condensates/downsample-requires-prevalence-adjustment]] provenance
  requirement.
- **Unlocked axes**: any axis that fell through to Fallback 4.2 is recorded
  here with its promotion target (V1 factor).
- **Cross-metric tensions**: if §3.5 flagged a Brier-vs-AUROC tension, it is
  written here and routed to `projects/<name>/tensions/rule-vs-observation/`
  per SCHEMA §Rulebook updates.

What advances to V1:

- The locked data partitioning — every V1 cell inherits `training_strategy`,
  `prevalent_train_frac`, and `train_control_per_case` from the V0 decision.
- The Stage-1 pass-list of models.
- The dataset fingerprint hash (binds V1 to the same split realizations).
- Any contested-axis promotion (§4.2) is added to the V1 factor list.

What remains open:

- Model selection (V2).
- Panel composition and ordering beyond R1_sig / R1_plateau (V1).
- Calibration (V4).
- Weighting × downsampling joint grid (V3 — distinct axis from V0's
  control_ratio; see Known tensions).
- Seed-split confirmation (V5), which holds out seeds 120–129.

---

## Known tensions

Acknowledged per `DECISION_TREE_AUDIT.md`; surfaced here because the V0
protocol inherits them but cannot resolve them without a main-factorial run.

### T-V0.1 — Control-ratio semantics double-use

`train_control_per_case` appears twice in the cellml tree:

- At V0 as a **split-generation** axis — it determines how many controls
  are retained after stratified allocation, fixing the training prevalence
  $\pi_\text{train}$ per [[equations/case-control-ratio-downsampling]].
- At V3 as a **training-dynamics** axis named `downsampling` — levels
  {1.0, 2.0, 5.0} per `DESIGN.md` §Factorial Factors. V3's downsampling is
  applied **after** V0's control_ratio split is generated, so the two axes
  compose multiplicatively, not as substitutes.

`DECISION_TREE_AUDIT.md` does not resolve whether V3's downsampling is
redundant under a V0 lock or intentionally nested. This protocol locks the
V0 split-generation ratio only; the V3 axis is explicitly deferred to the
V3 protocol for joint (weighting × downsampling) resolution.

### T-V0.2 — ADR-002 vs V0 factorial spec mismatch on `prevalent_train_frac`

[[condensates/prevalent-restricted-to-train]] enumerates
`prevalent_train_frac ∈ {0.0, 0.5, 1.0}`. The V0 factorial grid expands to
{0.25, 0.5, 1.0} per `DECISION_TREE_AUDIT.md` §3.5 P3 action. The grids are
not identical: V0 does not test 0.0 explicitly (it is implicit in
`training_strategy = incident_only`), and V0 adds 0.25 (which the condensate
does not enumerate). This is a candidate condensate revision — either the
condensate's actionable rule should be updated to match the expanded grid, or
the V0 grid should be restricted. Logged here pending resolution.

### T-V0.3 — V0 submitted with 20 selection seeds only

The 10-seed confirmation set (120–129) is reserved for V5 per the
`DECISION_TREE_AUDIT.md` §3.7 action. All V0-era Dominance claims rely on
within-selection-seed bootstraps, not held-out seed confirmation. V0 locks
are therefore **provisional** pending V5 re-evaluation; V5 can re-open any
V0 lock if the confirmation seeds contradict it by more than 1 SE.

### T-V0.4 — Vanillamax vs Gen1 discovery settings

Per `DESIGN.md` §Vanillamax and `MASTER_PLAN.md` §Discovery Methodology,
Gen1 discovery ran under restricted settings (`prevalent_train_frac = 0.5`,
`train_control_per_case = 5`, log class weights) — not vanillamax. The BH
core (4p) is robust to this, but the 8p boundary used for R1_plateau may
reflect restricted settings. V0 compensates by testing across strategies
and control ratios, but on a new cohort the discovery sweep should run
vanillamax first. This does not block V0 on the celiac cohort; it is a
note for downstream replication.
