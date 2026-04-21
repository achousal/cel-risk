---
type: protocol
gate: v0-strategy
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: null"
  - "discovery: consensus_panel.yaml"
  - "discovery: sweep_compiled.csv"
outputs:
  - "locks: [training_strategy, imbalance_family]"
axes_explored:
  - "training_strategy ∈ {IncidentOnly, IncidentPlusPrevalent(0.25), IncidentPlusPrevalent(0.5), IncidentPlusPrevalent(1.0), PrevalentOnly}"
  - "imbalance_probe ∈ {none, downsample_5, cw_log}   # probes {none, downsample, weight} families at representative levels; V3 refines the level within locked family"
axes_deferred:
  - "imbalance_level: deferred to v3-imbalance (refined within locked family)"
  - "recipe: deferred to v1-recipe"
  - "model: deferred to v2-model"
  - "calibration: deferred to v4-calibration"
  - "ensemble: deferred to v6-ensemble"
depends_on:
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/perm-validity-full-pipeline]]"
  - "[[condensates/feature-selection-needs-model-gate]]"
  - "[[condensates/three-way-split-prevents-threshold-leakage]]"
  - "[[condensates/prevalent-restricted-to-train]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[condensates/imbalance-two-stage-decision]]"
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[equations/perm-test-pvalue]]"
  - "[[equations/stratified-split-proportions]]"
---

# V0 Strategy Gate — Lock training strategy and imbalance-handling family before committing main-factorial compute

The V0 gate is the first decision node of the cellml tree. It answers two
coupled methodological questions — case partitioning and imbalance-handling
family — using four models and two representative recipes as a robustness
check, so that the main factorial does not inherit an unvalidated
partitioning prior or an unvalidated imbalance-family prior. V0 does NOT
fix the imbalance *level* within a family — that refinement is V3's job.

This protocol supersedes the earlier V0 specification that locked
`train_control_per_case` directly. The `control_ratio` axis has been retired
from V0 and replaced with `imbalance_probe`, a three-level axis that probes
the three imbalance-handling families (none, downsample, weight) at a
representative level each. The family is locked at V0; the within-family
level is refined at V3.

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

**Splits generated** (incident + prevalent seeds 100–119):

- Split strategy locked to 50/25/25 stratified on the primary outcome label,
  per [[condensates/three-way-split-prevents-threshold-leakage]] and the
  allocation math in [[equations/stratified-split-proportions]]. V0 does not
  revisit the split ratio; it operates within it.
- Split seeds locked to 100–119 (20 selection seeds). The 10-seed confirmation
  set (120–129) is reserved for V5 and is NOT used at V0.
- Prevalent-case IDs are available in the per-seed manifest so that
  `incident_plus_prevalent` strategies can inject prevalent rows into TRAIN
  while VAL and TEST remain incident-only (hook-enforced per
  [[condensates/prevalent-restricted-to-train]]).
- **TEST quarantine invariant**: no V0 operation touches TEST. Only TRAIN is
  modified by the imbalance_probe (downsample_5 applies random control
  subsampling to TRAIN only; cw_log re-weights the TRAIN loss only; none
  leaves TRAIN unchanged). TEST retains the full stratified control
  population at population prevalence.

**Optuna infrastructure ready**:

- Optuna TPE sampler at a fixed seed, MedianPruner enabled, 50-trial budget
  per cell (V0 gate budget; main factorial uses 200 trials per cell).
- Code version pinned via git commit SHA recorded in the gate ledger
  `rulebook_snapshot`.
- Permutation-test $B$ at 200 (V0 gate budget per [[equations/perm-test-pvalue]]
  boundary condition $B \ge 200$ for $\alpha=0.05$).

---

## 2. Search space

V0 crosses four axes under the fixed structural invariants above. Each axis
value is cited to the condensate that justifies its inclusion.

### 2.1 `training_strategy` ∈ {`IncidentOnly`, `IncidentPlusPrevalent(0.25)`, `IncidentPlusPrevalent(0.5)`, `IncidentPlusPrevalent(1.0)`, `PrevalentOnly`}

Five levels, unchanged from the prior V0 spec (the training-strategy axis is
not restructured by this revision).

- **`IncidentOnly`**: TRAIN, VAL, and TEST all drawn from incident cases
  only. Per [[condensates/prevalent-restricted-to-train]], incident cases are
  the prospective-screening reference population — this level is the
  "aligned" baseline against which augmentation is tested.
- **`IncidentPlusPrevalent(f)`** for $f \in \{0.25, 0.5, 1.0\}$: prevalent
  cases injected into TRAIN at sampling fraction $f$. VAL and TEST remain
  incident-only — hook-enforced per [[condensates/prevalent-restricted-to-train]].
  The condensate explicitly enumerates $f \in \{0.0, 0.5, 1.0\}$; $0.0$ is
  covered by `IncidentOnly`, and $0.25$ is added as a grid interior point to
  catch a low-fraction optimum that the condensate's three-point enumeration
  would miss.
- **`PrevalentOnly`**: TRAIN drawn from prevalent cases only. VAL/TEST
  remain incident-only. Included as an upper-bound on the distribution-shift
  hypothesis: if prevalent training beats incident training on an
  incident-only TEST, the "prospective alignment" argument is weakened.

### 2.2 `imbalance_probe` ∈ {`none`, `downsample_5`, `cw_log`} — NEW axis, replaces `control_ratio`

This axis probes the three imbalance-handling families (none, downsample,
weight) at a single representative level each. V0 locks the FAMILY; V3
refines the level within the locked family. The three values are not
alternatives on a ratio scale — they are categorical representatives of
three distinct families of imbalance handling.

- **`none`** (baseline, family `none`): `class_weight=None`,
  `train_control_per_case=1`. No imbalance handling. This is the parsimony
  floor per [[condensates/parsimony-tiebreaker-when-equivalence]]
  (`none` ≺ `weight` ≺ `downsample` across families) — any non-baseline
  family must earn its lock by Direction against `none`. Cited by
  [[condensates/imbalance-two-stage-decision]] as the family-level anchor.

- **`downsample_5`** (family `downsample`): `class_weight=None`,
  `train_control_per_case=5`. Aggressive downsampling representative.
  Justified by [[condensates/downsample-preserves-discrimination-cuts-compute]]
  ("5.0 is the ADR-003 default" and the ~60× speedup argument) and by the
  prevalence math in [[equations/case-control-ratio-downsampling]]
  ($\pi_\text{train} = 1/(1+5) = 0.1667$ on the celiac cohort). This single
  level stands in for the downsample family at V0; the refinement across
  $\{1, 2, 5\}$ is V3's job when the family is locked. Paired with
  [[condensates/downsample-requires-prevalence-adjustment]]: whenever
  `observation.md` reports Brier / ECE / reliability under `downsample_5`,
  the prevalence-adjusted metric MUST be logged alongside the raw metric.

- **`cw_log`** (family `weight`): `class_weight = log(n_control / n_case) + 1`,
  `train_control_per_case=1`. Logarithmic prevalence weighting, the Gen 1
  default. Cited by [[condensates/imbalance-two-stage-decision]] as the
  representative for the weight family at V0. No data is discarded (all
  controls remain in TRAIN), so there is no prevalence shift and no Bayes
  correction is required under this level. V3 refines across
  $\{none, sqrt, log\}$ if the weight family is locked.

**Probe semantics (critical).** `imbalance_probe` is NOT a ratio-scaled axis
and is NOT a superset of the old `control_ratio` axis. Its purpose is
two-stage family identification per [[condensates/imbalance-two-stage-decision]]:
V0 lock on family, V3 refine on level. The three probe values are the
minimum viable family-discriminating set; they are not a grid on a common
numerical scale.

### 2.3 Model axis (robustness check, not an exploration axis)

- Levels: `LR_EN`, `LinSVM_cal`, `RF`, `XGBoost`.
- Crossed under all (training_strategy × imbalance_probe) combinations so
  the lock test can demand Dominance across all four models per
  [[condensates/feature-selection-needs-model-gate]] (§3). The model itself
  is NOT locked at V0; that decision belongs to V2.

### 2.4 Recipe axis (representative subset, not an exploration axis)

- Levels: `R1_sig` (4p, significance_count size rule) and `R1_plateau` (8p,
  3-criterion unanimous). Two recipes chosen to deconfound strategy/family
  choice from panel size. Consensus ordering is fixed across both, so this
  axis does not test ordering — that is V1.

### 2.5 Model-gate permutation test (pre-gate filter, not an explored axis)

Before any (training_strategy, imbalance_probe) cell is considered eligible
for the lock test, its model must pass the Stage-1 permutation gate at
$p < 0.05$ per [[condensates/feature-selection-needs-model-gate]] and
[[condensates/perm-validity-full-pipeline]]. The permutation test MUST
re-run the full inner pipeline per permutation (not partial permutation
after feature selection), with $B = 200$ per [[equations/perm-test-pvalue]]
boundary conditions. Cells whose model fails the gate are marked
`excluded_from_lock` in the observation table and do not contribute to the
Dominance claim.

### 2.6 Cell count

$5 \text{ strategies} \times 3 \text{ imbalance probes} \times 4 \text{ models}
\times 2 \text{ recipes} = 120$ cells, at 20 selection seeds (100–119) per
cell. Cell count is preserved at 120 (same compute budget as the prior V0
spec); the `control_ratio` axis's three levels have been replaced by the
three categorical `imbalance_probe` levels.

---

## 3. Success criteria

All claims use the fixed falsifier rubric in `SCHEMA.md`. The primary lock
tests are two independent **Dominance** claims — one for `training_strategy`,
one for `imbalance_family` — against a parsimony baseline. Metric: AUROC on
the incident-only TEST split, 95% bootstrap CI over 1000 paired resamples
of outer-fold seeds (seeds 100–119). Standard error is defined as the SD of
the bootstrap delta distribution; cell-averaged SE is explicitly forbidden
(same SE-bug rule as V1 §3.3, inherited here).

### 3.1 `training_strategy` lock

Unchanged from the prior V0 spec (this revision does not restructure the
training-strategy axis). The lock rule is a Dominance claim across the four
models: Direction (AUROC margin $\ge 0.02$, 95% CI excludes 0) must hold
independently on each of {LR_EN, LinSVM_cal, RF, XGBoost}.

**Primary Direction claim (lock path):**

- Lock `training_strategy = IncidentOnly` if, for every model $m \in$
  {LR_EN, LinSVM_cal, RF, XGBoost}:
  $\Delta\text{AUROC}_m = \text{AUROC}(\text{IncidentOnly}) -
  \max(\text{AUROC}(\text{IncidentPlusPrevalent at best } f),
  \text{AUROC}(\text{PrevalentOnly})) \ge 0.02$
  AND the 95% bootstrap CI over seed resamples excludes 0.

**Alternative Direction claim:**

- Lock `training_strategy = IncidentPlusPrevalent(f^\star)` (best fraction)
  if the analogous Direction holds with IncidentPlusPrevalent replacing
  IncidentOnly as the winner, on all four models.

**Equivalence fallthrough:**

- If Direction fails but Equivalence holds across candidate strategies on
  all four models ($|\Delta\text{AUROC}| < 0.01$ AND 95% CI $\subset [-0.02, 0.02]$),
  lock the simpler partition: `training_strategy = IncidentOnly` (parsimony
  per [[condensates/prevalent-restricted-to-train]] and the parent meta-rule
  [[condensates/parsimony-tiebreaker-when-equivalence]]).

**Inconclusive:**

- If neither Direction nor Equivalence holds on all four models, fall
  through to the fallback path (§4.1).

### 3.2 `imbalance_family` lock (NEW — replaces prior `train_control_per_case` lock)

V0 locks the FAMILY, not the level. The claim is Dominance against the
parsimony-baseline family `none` per
[[condensates/parsimony-tiebreaker-when-equivalence]] (cross-family order:
`none ≺ weight ≺ downsample`, least invasive first — weighting keeps all
data, downsampling discards data).

**Primary Direction claims (family lock path):**

For each candidate family $F \in \{\text{weight}, \text{downsample}\}$,
compute per-model $\Delta\text{AUROC}_m^F$ = AUROC at the probe
representative for family $F$ minus AUROC at `none`, on each model $m$.

- Lock `imbalance_family = downsample` if the Dominance criterion holds
  for the `downsample_5` probe relative to `none` (Direction on every model
  axis: $\Delta\text{AUROC}_m^\text{downsample} \ge 0.02$ with 95% CI
  excluding 0 for all four models) AND the analogous Direction does NOT
  hold for `weight` against `none`, OR the Direction margin for
  `downsample` exceeds the Direction margin for `weight` by Equivalence
  on the inter-family delta.
- Lock `imbalance_family = weight` if the Dominance criterion holds for
  `cw_log` relative to `none` on all four models AND the analogous
  Direction does not hold (or is Equivalence-weaker than weight) for
  `downsample_5`.
- Lock `imbalance_family = none` if neither non-baseline probe satisfies
  Dominance against `none`, AND at least one of them falls inside
  Equivalence (|Δ| < 0.01, 95% CI $\subset$ [−0.02, 0.02]) on all four
  models — i.e., `none` is not dominated.

**Equivalence fallthrough (parsimony to `none`):**

- If every non-baseline probe returns Equivalence against `none` on all
  four models, lock `imbalance_family = none` per
  [[condensates/parsimony-tiebreaker-when-equivalence]] cross-family order.
  Rationale: neither downsampling nor weighting earned its departure from
  the vanillamax baseline.

**Inconclusive:**

- If the claim across the four models is neither Direction nor Equivalence
  (e.g., Direction on two models and Inconclusive on two; or one family
  beats on three models but fails on the fourth), fall through to the
  fallback path (§4.2).

**Anti-pattern (critical, rubric violation):**

- V0 MUST NOT lock `imbalance_level` (e.g., `train_control_per_case = 5`
  or `class_weight_scheme = log`) as a result of the V0 decision. The
  probe level (`downsample_5`, `cw_log`) is a family representative, NOT
  the final level. Locking the level at V0 confuses the family-probe with
  a level-search and is a rubric violation. The level is V3's decision
  (conditional refinement within the locked family, per
  [[condensates/imbalance-two-stage-decision]]).

### 3.3 Dominance composition

The two axis locks (training_strategy and imbalance_family) are independent
Dominance claims. V0 can lock one without the other; if only one clears
the rubric, that axis locks and the other falls through to its
axis-specific fallback (§4). Joint Dominance is NOT required.

### 3.4 Standard-error definition (CRITICAL — SE bug rule inherited from V1)

V0 MUST use paired bootstrap over outer-fold seeds, NOT cell-averaged SE.
This is the same rule as [[protocols/v1-recipe]] §3.3. Specifically:

    auroc_ci = BOOTSTRAP(
        unit = seed,                   # outer-fold seed as paired unit
        n_boot = 1000,
        stat = "delta_AUROC",          # paired per-seed delta across probe levels
    )

Key rules:

1. **Bootstrap unit is the seed (outer fold), not the cell.** 20 selection
   seeds (100–119) are the paired units. Seeds are shared across cells.
2. **Per-seed aggregation, then delta.** For each seed $s$, compute AUROC
   for the compared probe/strategy configuration; delta is taken per seed,
   then resampled.
3. **No per-fold (inner) SE.** Per-inner-fold AUROC from Optuna is noisier
   than seed-level; V0 uses only seed-level aggregation.
4. **No cell-averaged SE.** A cell-averaged SE (SD of cell means divided
   by $\sqrt{n\_\text{cells}}$) is explicitly forbidden per the V1 SE-bug
   resolution (DECISION_TREE_AUDIT.md §1.3, reproduced here as a V0 rule).

### 3.5 Brier / calibration cross-check (advisory, not a lock gate)

Per [[condensates/downsample-requires-prevalence-adjustment]], the
observation table MUST log both raw and prevalence-adjusted Brier for any
cell where `imbalance_probe = downsample_5` (the probe that shifts training
prevalence). The `cw_log` probe does not shift training prevalence (no
controls discarded), so Bayes correction is a no-op there. If the Dominance
claim holds on AUROC but a model's prevalence-adjusted Brier at the locked
(strategy, family) degrades by $|\Delta\text{Brier}| \ge 0.02$ with CI
excluding 0 relative to an unlocked alternative, flag a cross-metric tension
and escalate to V4 (calibration). This check does NOT veto the V0 lock —
calibration parsimony at V4 is the adjudicator per [[condensates/calib-parsimony-order]].

---

## 4. Fallbacks

When a lock test returns Inconclusive, V0 cannot lock the contested axis.
The operational fallback moves are axis-specific.

### 4.1 `training_strategy` Inconclusive

Unchanged from the prior V0 spec:

- **Widen before promoting.** If Inconclusive is driven by a single
  dissenting model (3 out of 4 meet Direction, 1 does not), re-run the
  dissenting model's cells at 200 Optuna trials (the main-factorial budget)
  before declaring Inconclusive. This is not a hyperparameter-peeking
  violation because the axis under test is partition, not hyperparameters.
- **Promote contested axis to V1.** If widening does not resolve,
  `training_strategy` is added as a V1 axis and carried forward unlocked.

### 4.2 `imbalance_family` Inconclusive (NEW — two-branch fallback)

When the family-lock rubric returns Inconclusive:

**Branch A — Parsimony default to `none` AND flag V3.**

1. Lock `imbalance_family = none` as the parsimony default per
   [[condensates/parsimony-tiebreaker-when-equivalence]] cross-family order
   (least invasive first, under uncertainty).
2. **Flag V3 to run the OLD 3×3 weight × downsample grid** (see
   `v3-imbalance.md` fallback branch — V3 expands its search space from the
   locked-family refinement to the full 9-cell grid because V0 could not
   separate families). The V0 decision.md MUST set
   `v3_expansion_required: true` and list the per-model per-family deltas
   and CIs that motivated the flag.

**Branch B — Widen before promoting (optional, precedes Branch A).**

If Inconclusive is driven by wide CIs (bootstrap CI half-width $> 0.02$
despite no Direction on any probe pair), re-run the contested cells at
200 Optuna trials. Only if this widening fails to separate families does
the gate fall to Branch A.

### 4.3 Parsimony lock under Equivalence (not Inconclusive)

Where Equivalence holds cleanly (not Inconclusive), the parsimony moves
are axis-specific and follow
[[condensates/parsimony-tiebreaker-when-equivalence]]:

- **Strategy**: prefer `IncidentOnly` (fewer augmentation operations, no
  distribution shift, prospective alignment by construction — three
  parsimony arguments stack, per §3.1).
- **Imbalance family**: prefer `none` over `weight` over `downsample`
  (least invasive first — weighting keeps all data but modifies loss;
  downsampling discards data). Every non-baseline family must earn
  Direction to win; Equivalence never locks `weight` or `downsample`.

---

## 5. Post-conditions

What V0 writes into `projects/<name>/gates/v0-strategy/decision.md`:

- **Locks passed forward** (on the happy path):
  1. `training_strategy = <locked value>` — with the exact claim type
     (Direction or Equivalence-parsimony) that justified it.
  2. `imbalance_family = <none | downsample | weight>` — NEW. Claim type
     recorded. This is a categorical family label, NOT a level value.
- **Permutation-test pass list**: the set of (model × recipe) combinations
  that passed the Stage-1 model gate at $p < 0.05$. This list is the V1
  `axes_deferred.model` input per
  [[condensates/feature-selection-needs-model-gate]] — V1 consensus-panel
  construction MUST consume only models from this list.
- **Prevalence-adjusted Brier table**: raw and adjusted Brier per cell
  under `downsample_5` (for `none` and `cw_log`, Bayes adjustment is a
  no-op and the raw metric is reported as-is), per
  [[condensates/downsample-requires-prevalence-adjustment]] provenance
  requirement.
- **V3 expansion flag**: if §4.2 Branch A triggered, `v3_expansion_required: true`
  with the family-delta table that motivated the flag.
- **Unlocked axes**: any axis that fell through to §4.1 promotion is
  recorded with its promotion target (V1 factor).
- **Cross-metric tensions**: if §3.5 flagged a Brier-vs-AUROC tension, it
  is written here and routed to `projects/<name>/tensions/rule-vs-observation/`
  per SCHEMA §Rulebook updates.

What advances to V1:

- The locked data partitioning — every V1 cell inherits `training_strategy`
  from the V0 decision. Prevalent-fraction $f$ is carried if the locked
  strategy is `IncidentPlusPrevalent(f)`.
- The locked imbalance family — V1 cells are configured under a single
  family-representative level at V1 (the V0 probe level for the locked
  family: `downsample_5` if family=downsample, `cw_log` if family=weight,
  `none` if family=none). V1 does NOT re-search the level; that is V3.
- The Stage-1 pass-list of models.
- The dataset fingerprint hash (binds V1 to the same split realizations).
- Any contested-axis promotion (§4.1) is added to the V1 factor list.

What advances to V3:

- The locked imbalance family. V3's search space is **conditional on the
  locked family**:
  - If `imbalance_family = none`: V3 runs a minimal sensitivity check at
    the baseline and typically re-confirms `none` by parsimony.
  - If `imbalance_family = weight`: V3 refines across
    $\{none, sqrt, log\}$ within the weight family.
  - If `imbalance_family = downsample`: V3 refines across
    $\{1, 2, 5\}$ within the downsample family.
- If `v3_expansion_required: true` (§4.2 Branch A), V3 expands to the full
  3×3 weight × downsample grid as a fallback search.

What remains open:

- Model selection (V2).
- Panel composition and ordering beyond R1_sig / R1_plateau (V1).
- Calibration (V4).
- Imbalance LEVEL within the locked family (V3).
- Ensemble (V6, informational).
- Seed-split confirmation (V5), which holds out seeds 120–129.

---

## Known tensions

Acknowledged here because the V0 protocol inherits them but cannot resolve
them without a main-factorial run or a concurrent rulebook update.

### T-V0.1 — `imbalance_probe` replaces `control_ratio`; ADR-003 partially superseded

Prior V0 locked `train_control_per_case` directly from $\{1, 2, 5\}$ per
ADR-003. This revision retires that axis and replaces it with
`imbalance_probe`. ADR-003's specification of `train_control_per_case = 5`
as the default is no longer a V0 lock — it is now a V3 within-family lock,
conditional on V0 locking `imbalance_family = downsample`. ADR-003 is
therefore **partially superseded**: the compute-savings argument remains
valid at V3, but V0 no longer locks the ratio.

Flag as a **rule-vs-observation tension**: the rulebook condensate
[[condensates/downsample-preserves-discrimination-cuts-compute]] still
carries an actionable rule phrased as "The V0 gate locks
`train_control_per_case` from {1.0, 2.0, 5.0}". That phrasing is now
out-of-date — at the next rulebook MINOR bump the condensate's actionable
rule should be restated as "The V0 gate locks `imbalance_family`; V3 locks
`train_control_per_case` within the downsample family." Logged here as a
candidate rulebook PR pending the concurrent rewrite of
[[condensates/imbalance-two-stage-decision]] (which formalizes the
two-stage family/level decision structure).

### T-V0.2 — The V0 probe level is NOT the final lock

The probe levels `downsample_5` and `cw_log` are *representatives* of their
families, chosen to be informative discriminators at V0's 50-trial budget.
V3 may lock a different level within the same family (e.g., V0 locks
family=downsample via the `downsample_5` probe, V3 then refines to
`train_control_per_case = 2` because V3's 200-trial budget reveals a
lower-ratio optimum). This is NOT a contradiction — V0 and V3 answer
different questions (family vs level). The tension detector should NOT
flag a V3 refinement that deviates from the V0 probe level.

The risk is that V0's probe-level choice is not representative of the
family's optimum. Mitigation: the probe levels are chosen as "aggressive
representative" for downsample (`r=5`, the compute-parsimony default) and
"Gen 1 default" for weight (`cw_log`). If the family optimum is far from
the probe level, V3's 3-level within-family search catches it. Cross-
cohort evidence that V0 probe-family locks are later inverted by V3 would
trigger a revision of the probe choices and is logged here as a
candidate tension.

### T-V0.3 — `nested-downsampling-composition` and `v3-utility-provenance-chain` are being concurrently rewritten

The prior V0 spec's Known tensions cited
[[condensates/nested-downsampling-composition]] as the formalization of
multiplicative V0/V3 composition of the downsample axis. Under this
revision, V0 no longer locks a downsample level, so multiplicative
composition across V0/V3 for the downsample axis is retired; composition
becomes **within-family only** at V3 (V3's within-TRAIN downsampling is
the sole downsampling operation; V0's representative-probe sampling is a
family-identification step, not a composable level).

Consequence: `nested-downsampling-composition` is being rewritten in
parallel to remove the multiplicative-across-V0-V3 claim. Until that
rewrite lands, V0 `ledger.md` citations SHOULD NOT cite
`nested-downsampling-composition` as a load-bearing dependency, and V0
protocols SHOULD flag this citation as "concurrently rewritten" in
tension notes. The same applies to `v3-utility-provenance-chain`, which is
also touched by the V3 conditional-refinement restructuring.

### T-V0.4 — V0 submitted with 20 selection seeds only

The 10-seed confirmation set (120–129) is reserved for V5. All V0-era
Dominance claims rely on within-selection-seed bootstraps, not held-out
seed confirmation. V0 locks are therefore **provisional** pending V5
re-evaluation; V5 can re-open any V0 lock if the confirmation seeds
contradict it by more than 1 SE. This is inherited from the prior V0 spec
(T-V0.3 in that version); unchanged here.

### T-V0.5 — Vanillamax vs Gen1 discovery settings

Per `DESIGN.md` §Vanillamax and `MASTER_PLAN.md` §Discovery Methodology,
Gen1 discovery ran under restricted settings (log class weights, `r=5`
downsampling, `prevalent_train_frac=0.5`) — not vanillamax. The BH core
(4p) is robust to this, but the 8p boundary used for R1_plateau may
reflect restricted settings. V0's `imbalance_probe` axis partially
addresses this — the `none` and `downsample_5` and `cw_log` probes span
the relevant imbalance-handling space — but the discovery sweep itself
should run vanillamax on a new cohort. This does not block V0 on the
celiac cohort; it is a note for downstream replication. Inherited from
the prior V0 spec (T-V0.4 there); unchanged.

---

## Sources

- `operations/cellml/DESIGN.md` — cellml tree specification and factorial
  factors
- `operations/cellml/MASTER_PLAN.md` — V0 definition and decision tree
- `operations/cellml/DECISION_TREE_AUDIT.md` — SE bug, cell-count
  asymmetry, audit resolutions
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
- [[condensates/imbalance-two-stage-decision]] — the formalization of the
  family-then-level decomposition (authored in parallel with this
  revision; verify slug resolves after merge)
- ADR-001, ADR-002, ADR-003, ADR-004, ADR-011 — source decision records
  (ADR-003 partially superseded per T-V0.1)
