---
type: protocol
gate: v1-recipe
inputs:
  - "dataset/fingerprint.yaml"
  - "prior_gate: v0-strategy"
  - "discovery: consensus_panel.yaml"
  - "discovery: sweep_compiled.csv"
outputs:
  - "locks: [recipe, panel_composition, panel_size, ordering]"
axes_explored:
  - "recipe_family: {shared_panel, model_specific}"
  - "trunk: {T1_consensus, T2_incident_sparse}"
  - "ordering: {consensus_desc, stream_balanced, oof_importance, rfe_elimination, abs_coefficient_desc, stability_freq_desc}"
  - "panel_size_rule: {significance_count, three_criterion, three_criterion_unanimous, stability}"
  - "panel_size_p: {4, 5, 6, 7, 8, 9, 19}"
axes_deferred:
  - "model: deferred to v2-model (single-model dominance within winning recipe family)"
  - "calibration: deferred to v4-calibration"
  - "weighting: deferred to v3-imbalance"
  - "downsampling: deferred to v3-imbalance"
  - "ensemble: deferred to v6-ensemble (informational)"
depends_on:
  - "[[condensates/feature-selection-needs-model-gate]]"
  - "[[condensates/stability-as-hard-filter]]"
  - "[[condensates/nested-cv-prevents-tuning-optimism]]"
  - "[[condensates/optuna-beats-random-search]]"
  - "[[condensates/optuna-warm-start-couples-gates]]"
  - "[[condensates/perm-validity-full-pipeline]]"
  - "[[condensates/three-way-split-prevents-threshold-leakage]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[equations/consensus-rank-aggregation]]"
  - "[[equations/nested-cv]]"
  - "[[equations/optuna-tpe]]"
  - "[[equations/perm-test-pvalue]]"
  - "[[equations/stratified-split-proportions]]"
  - "[[equations/brier-decomp]]"
---

# V1 Recipe Protocol

Factorial gate that compares derived recipes (panel composition + ordering +
size) under the strategy locked at [[protocols/v0-strategy]]. V1 locks the
recipe family and panel before [[protocols/v2-model]] opens model dominance
testing.

V1 is structured as a within-family pairwise tournament: shared-panel recipes
compete only against other shared-panel recipes (108 cells each), and
model-specific recipes compete only against other model-specific recipes (27
cells each). Cross-family comparison is handled through a model-matched bridge
and deferred to v2-model when not decisive. This structure is a direct response
to the cell-count asymmetry documented in `DECISION_TREE_AUDIT.md` §1.4.

---

## 1. Pre-conditions

Before V1 ledger can be written:

1. **V0 must be locked.** [[protocols/v0-strategy]] must have emitted a
   `decision.md` with `training_strategy` and `control_ratio` locked under the
   fixed falsifier rubric. V1 inherits those locks as fixed cell-level
   configuration. If V0 returned Inconclusive on any model, V1 cannot run on
   that model until the tension is resolved.
2. **Discovery artifacts exist.** The discovery-phase sweep (30-seed averaged
   AUROC at panel sizes 4–25) must be available at
   `results/compiled_results_aggregated.csv`, and the RRA consensus ranking and
   T2 incident-sparse feature list must be present. Missing discovery artifacts
   block the 3-criterion size rule (see DESIGN.md §Size rules) and the
   per-model plateau derivation.
3. **All recipes derived from the manifest.** Every recipe in the V1 search
   space MUST be produced by `ced derive-recipes --manifest
   configs/manifest.yaml --data-path <parquet>`. Hand-picked panels are
   rejected: DESIGN.md §Objective mandates "every panel is a generated artifact
   of declared rules — no hand-picking." The ledger must record the manifest
   hash and the derivation audit logs
   (`configs/recipes/{recipe_id}/size_derivation.json` and
   `ordering_derivation.json`) that certify provenance.
4. **Stage-1 model gate has been applied.** Per
   [[condensates/feature-selection-needs-model-gate]], recipes that consume
   cross-model consensus (`T1` trunk via [[equations/consensus-rank-aggregation]])
   MUST restrict the aggregation to models that pass the permutation test at
   alpha = 0.05. Stage-1 p-values come from the V0 gate; protocols that skip
   this check produce a consensus ranking polluted by noise models and the
   recipe family is invalid.
5. **Stability hard filter has been applied where required.** Per
   [[condensates/stability-as-hard-filter]], trunk-T1 recipes feed per-model
   ranked lists through the hard selection-frequency filter (threshold
   declared in the manifest; default 0.90) BEFORE the RRA aggregation. Recipes
   that conflate stability into a composite weight instead of a hard filter
   are rejected at pre-condition check.
6. **Rulebook snapshot bound at gate entry.** Per SCHEMA.md §Versioning, V1
   ledger frontmatter must cite `rulebook_snapshot: "rb-v{x.y.z}"` at entry
   and remain immutable through decision.

---

## 2. Search space

V1 spans the 34 recipes declared in DESIGN.md §Recipes, all derived from the
T1 and T2 trunks via the manifest's `size_rules` and `ordering_rules`
dispatchers. The factorial axes are comparison-level (which pair is being
tested), not configuration-level (cell configuration is inherited from V0
locks and cross-axis factors are deferred).

### 2.1 Recipe families (8 + 8 + 18)

- **Shared-panel family (8 recipes)** — panels that all four models share.
  Each recipe expands to 108 cells (4 models × 3 calibrations × 3 weightings
  × 3 downsamplings). Tests ordering and size independent of model choice.
    - T1 trunk × consensus-descending ordering × {significance_count=4,
      three_criterion=5, three_criterion_unanimous=8}: R1_sig, R1_criterion,
      R1_plateau.
    - T1 trunk × stream-balanced ordering × same three size rules:
      R2_sig, R2_criterion, R2_plateau.
    - T2 trunk: R3_incident_sparse (|coef| descending, stability=19 proteins),
      R3_consensus (stability_freq descending, stability=19 proteins). R3 pair
      deconfounds ordering from trunk choice, addressing
      `DECISION_TREE_AUDIT.md` §3.2.

- **Model-specific base family (8 recipes)** — each recipe pinned to one of
  {LR_EN, LinSVM_cal, RF, XGBoost} and one of two orderings
  ({oof_importance, rfe_elimination}). Size = per-model plateau from the
  unanimous 3-criterion rule applied to the model-filtered sweep (DESIGN.md §
  Per-model size derivation). 27 cells each.

- **Nested expansion (18 recipes)** — RF and XGBoost recipes (both orderings)
  auto-expand from their plateau (8 or 9) down to the significance core (4)
  in single-protein steps. Each sub-recipe reuses the parent ordering
  truncated to top-N. LR_EN and LinSVM_cal plateau at 4 and do not expand.

### 2.2 Within-family pairwise tournament

Within each family, every pair of recipes is a comparison. Shared × shared
gives 28 pairs (C(8,2)); model-specific (base + nested expansion) gives up to
C(26,2) = 325 pairs but tournament is restricted to pairs that vary exactly
one axis (ordering-matched size ladders, size-matched ordering comparisons,
and trunk pairs), following DESIGN.md §V1 axes:

| Axis | Pairs to run | Rationale |
|---|---|---|
| Size (within ordering) | R1_sig vs R1_criterion vs R1_plateau; R2_sig vs R2_criterion vs R2_plateau; size-ladder pairs within each MS_{oof,rfe}_{RF,XGBoost} expansion | Isolate panel-size effect |
| Ordering (within size) | R1_* vs R2_* at each size; MS_oof_* vs MS_rfe_* per model | Isolate ordering effect |
| Trunk (within ordering type) | R3_incident_sparse vs R1_plateau at 8p; R3_consensus vs R1_plateau | Deconfound T1 vs T2 |
| Ordering-on-T2 | R3_incident_sparse vs R3_consensus | Isolate ordering effect on incident trunk |

Pairs that vary more than one axis are forbidden at V1 — they are reserved
for V2's model-conditional view.

### 2.3 Axis-condensate mapping

| Axis | Grounding condensate / equation | Load-bearing claim |
|---|---|---|
| Recipe family composition (shared vs model-specific) | [[condensates/feature-selection-needs-model-gate]] | Shared-panel consensus is valid only over Stage-1-gated models; without the gate, shared recipes are measuring noise aggregation |
| Shared-panel ordering via consensus | [[equations/consensus-rank-aggregation]] | Geometric-mean reciprocal-rank aggregator; only meaningful after the model gate and the stability pre-filter |
| Per-model ranked lists fed to consensus | [[condensates/stability-as-hard-filter]] | Hard stability filter (default 0.90) applied before aggregation; no composite weighting |
| Inner-loop tuning inside each cell | [[equations/nested-cv]], [[condensates/nested-cv-prevents-tuning-optimism]] | 5×10×5×200 nested CV per cell; outer-fold isolation mandatory |
| Hyperparameter search inside inner loop | [[equations/optuna-tpe]], [[condensates/optuna-beats-random-search]] | TPE + MedianPruner default at T >= 50; sampler seed explicit |
| Warm-start from V0 scout | [[condensates/optuna-warm-start-couples-gates]] | Declared explicitly in ledger; top-k bounded at k/T <= 0.05; source gate fingerprint recorded |
| Permutation validity for Stage-1 gate | [[equations/perm-test-pvalue]], [[condensates/perm-validity-full-pipeline]] | Full-pipeline permutation; partial-permutation p-values are anti-conservative |
| Cell-level TRAIN/VAL/TEST allocation | [[equations/stratified-split-proportions]], [[condensates/three-way-split-prevents-threshold-leakage]] | 50/25/25 stratified on incident_CeD; locked at V0 and inherited |
| V1 calibration-component-free comparison | [[equations/brier-decomp]] | V1 reports AUROC + REL separately so Brier's calibration term does not confound the ordering/size comparison reserved for V4 |

Axes flagged as lacking dedicated condensate support (candidate additions to
the rulebook; V1 still runs, but the grounding is thin):

- **3-criterion size rule itself** — DESIGN.md §Size rules codifies it but
  there is no condensate that states the claim "smallest p passing >=2 of {
  non-inferiority z-test, within-1-SE, marginal-gain z-test } is the
  parsimonious plateau." TODO: add `condensates/three-criterion-size-rule.md`
  so the rule is falsifiable.
- **Pooling sweep across orderings before size decision** — DESIGN.md §Size
  rules pools sweep data across {importance, pathway, rra} orderings before
  applying the 3-criterion rule. Treated as a methodological choice in this
  protocol but lacks a condensate; flag as TODO and see Known tensions §4.
- **Nested expansion step-down logic** — DESIGN.md §Nested expansion claims
  single-protein truncation maps each model's performance floor "without RFE
  retraining." No condensate grounds this as vs. retraining-per-size. TODO:
  `condensates/nested-expansion-vs-refit.md`.

---

## 3. Success criteria

All V1 decisions use the fixed falsifier rubric from SCHEMA.md §Fixed
falsifier rubric. A recipe "wins" over its pair only when the criterion below
is met on AUROC as the primary discrimination metric; Brier REL is reported
as a secondary axis and carried forward for V4 but does NOT participate in
the V1 decision.

### 3.1 Claim types

| Claim type | Criterion | V1 decision |
|---|---|---|
| **Direction** (A > B) | `|ΔAUROC| >= 0.02` AND 95% bootstrap CI (1000 outer-fold resamples) excludes 0 | Winner locked; loser retired from the family |
| **Equivalence** (A ≈ B) | `|ΔAUROC| < 0.01` AND 95% bootstrap CI ⊂ [-0.02, 0.02] | Parsimony tiebreaker applies — see §3.2 |
| **Inconclusive** | Neither Direction nor Equivalence met | Both recipes forwarded to V2; lock deferred (see §4) |

Counts (panel size `p`, cell count) are compared exactly without CIs per
SCHEMA.md metric-specific rules.

### 3.2 Parsimony tiebreaker (Equivalence only)

Per DESIGN.md §Parsimony Ordering: when the Equivalence criterion holds,
prefer the recipe with fewer proteins. If panel sizes tie, prefer the
recipe with the simpler ordering strategy (consensus-descending <
stream-balanced < model-specific). If ordering ties too, prefer the T1
(consensus) trunk over T2 (incident-sparse) — T1 was locked earlier in the
discovery chain and has more confirmations.

Parsimony-tie resolution follows `[[condensates/parsimony-tiebreaker-when-equivalence]]`
with this axis's order: **smaller panel size `p` wins** when the rubric
returns Equivalence on AUROC, with ordering-strategy and trunk choice as
secondary tiebreakers as above.

### 3.3 SE definition (CRITICAL — fixes V1 SE bug)

`DECISION_TREE_AUDIT.md` §1.3 documents that the original V1 SE was computed
as:

    auroc_se = sd(summary_auroc_mean) / sqrt(n_cells)

Each `summary_auroc_mean` is a 30-seed average for one cell, and cells
within a shared recipe differ along four nuisance axes (model × calibration
× weighting × downsampling). That quantity is downstream-factor variance,
not measurement uncertainty, and it is systematically inflated for
shared-panel recipes relative to model-specific ones.

V1 MUST use the following SE definition:

    auroc_ci = BOOTSTRAP(
        unit = seed,                   # outer-fold / seed-level paired resample
        n_boot = 1000,
        aggregation = "recipe_mean",   # mean AUROC across cells in the recipe, per seed
        stat = "delta_AUROC",          # ΔAUROC between the two recipes in the pair
    )
    auroc_se = sd(bootstrap_deltas) / sqrt(1)   # CI is reported directly

Key rules:

1. **Bootstrap unit is the seed (outer fold), not the cell.** 30 seeds
   (100–129) are the paired units. Seeds are shared across cells, so delta is
   computed seed-by-seed and the bootstrap resamples seeds with replacement.
2. **Per-seed cell aggregation, then delta.** For recipe A, compute mean
   AUROC across A's cells at seed s (shared: 108 cells / 4 for same-model
   match, or 27 cells if stratified by model; model-specific: 27 cells
   directly). Repeat for recipe B. Delta at seed s is A_s − B_s.
3. **No per-fold SE.** Per-fold AUROC from Optuna (`fold_aurocs` trial attr,
   MASTER_PLAN.md Optuna Enhancement #4) is noisier than seed-level and
   entangles inner-CV variance with outer-CV variance. V1 uses only
   seed-level aggregation.
4. **Paired bootstrap.** Resample seeds, not (seed, recipe) pairs. The same
   seed appearing for both A and B in the resample preserves the pairing
   structure. This is a standard paired-design bootstrap.

The old cell-averaged SE is explicitly forbidden. Any V1 `observation.md`
that reports a cell-averaged SE is a rulebook violation and must be rejected
by the tension detector.

### 3.4 Cell-count asymmetry (shared 108 vs model-specific 27)

Per `DECISION_TREE_AUDIT.md` §1.4, the variance components embedded in a
shared recipe (marginal over 4 models × 3 cal × 3 weight × 3 ds) differ from
those in a model-specific recipe (marginal over 3 cal × 3 weight × 3 ds
only). Seed-level bootstrap from §3.3 absorbs this correctly WITHIN a family
but does NOT make shared and model-specific directly comparable.

V1 resolves this by stratifying the tournament:

- **Shared vs shared** pairs run on the recipe-mean from all 108 cells.
- **Model-specific vs model-specific** pairs run on the recipe-mean from
  all 27 cells (same model across both, since model is pinned).
- **Shared vs model-specific** (cross-family) is NOT run at V1 using the
  all-cells mean. Instead, a model-matched bridge is run: for each model m
  ∈ {LR_EN, LinSVM_cal, RF, XGBoost}, restrict the shared recipe to the 27
  cells with that model and compare against the model-specific recipe pinned
  to m. The bridge produces four per-model deltas; V1 locks only if all
  four yield the same claim type under the rubric. Otherwise defer to V2
  (see §4).

This matches DESIGN.md §Validation Decision Tree: "V1 uses a stratified
comparison to fix cell-count asymmetry."

### 3.5 Calibration REL is reported, not adjudicated

Per [[equations/brier-decomp]] and the V3/V4 positioning in DESIGN.md, V1
compares recipes on AUROC (discrimination) but reports REL as a diagnostic
axis so that V4 can inherit the information. A recipe that wins on AUROC but
inflates REL beyond a documented equivalence band
(|ΔREL| >= 0.005) is flagged in `tensions.md` but is NOT disqualified at V1.
Calibration is V4's decision, not V1's.

---

## 4. Fallbacks

When a pair returns Inconclusive under §3.1:

1. **Both recipes forward to V2.** Neither is locked; both enter the V2
   model-dominance tournament. V2 may resolve the ordering choice by finding
   a model × recipe interaction.
2. **If all pairs in a family return Inconclusive,** the entire family
   forwards as a set. This is a common expected outcome for the nested
   expansion ladders, where neighbour sizes differ by exactly one protein and
   rarely clear the 0.02 ΔAUROC threshold.
3. **Dominating recipe family still checks against single-model plateau
   estimates.** Even when shared-vs-shared yields a clear winner, the
   winning recipe must pass the ADR-004 consensus-gate sanity check
   (per-model permutation p < 0.05 on the locked panel — uses
   [[equations/perm-test-pvalue]] at B >= 1000). A recipe family that
   dominates on discrimination but whose consensus fails the model gate on
   any member model is flagged and held at V1 pending investigation.
4. **Missing artefacts escalate, not relax.** If the sweep
   (`compiled_results_aggregated.csv`) is unavailable for any of the size
   rules, the affected recipes are dropped from the tournament and
   `tensions.md` is emitted. V1 does NOT substitute a default size.
5. **Stage-1 failure is not recoverable by weakening alpha.** Per
   [[condensates/feature-selection-needs-model-gate]], if zero models pass
   the permutation gate at alpha = 0.05, V1 does not run consensus recipes.
   The correct response is to revisit cohort definition or feature
   engineering, not to lower the threshold.
6. **Cross-family bridge disagreement -> defer.** If the four per-model
   bridges in §3.4 yield different claim types, V1 locks nothing on the
   cross-family axis and records the bridge disagreement as a V2 input.

---

## 5. Post-conditions

On V1 `decision.md` write:

1. **Locks (on Direction).** If shared-vs-shared and model-specific-vs-
   model-specific each produce a within-family winner AND the cross-family
   bridge produces a consistent direction on all four models, V1 locks:
   - `recipe_id` — the single winning recipe
   - `panel_composition` — the derived protein list from the manifest
   - `panel_size` — the integer `p`
   - `ordering_strategy` — the ordering rule (consensus / stream / OOF / RFE
     / |coef| / stability_freq)
2. **Locks (on Equivalence via parsimony).** When the rubric returns
   Equivalence, apply the §3.2 tiebreaker and still lock the four fields
   above. `decision.md` explicitly names the claim type as `Equivalence`
   and cites the parsimony rule invoked.
3. **Forward (on Inconclusive).** No locks. `decision.md` records all
   non-dominated recipes and forwards them to V2. The `locks_passed_forward`
   field is empty; the `predictions_for_v2` field lists the candidate
   recipes with their V1 point estimates and bootstrap CIs.
4. **Claim type MUST be declared explicitly.** Per SCHEMA.md §Fixed
   falsifier rubric, `decision.md` under `## Actual claim type (per rubric)`
   states exactly one of {Direction, Equivalence, Inconclusive} for each
   axis (size, ordering, trunk, tailoring). The ledger's `##
   Predictions with criteria` section must be matched one-to-one against
   these claim types.
5. **Tensions auto-populate.** Any pair where the ledger predicted
   Direction but observation returned Equivalence or Inconclusive (or
   vice versa) writes a tension to
   `projects/<name>/tensions/rule-vs-observation/`, keyed on the
   condensate whose prediction failed. The tension detector does this; V1
   does not write tensions manually.
6. **Provenance for downstream gates.** V2 consumes V1's `panel_composition`
   list as the fixed feature set. V3 (imbalance) and V4 (calibration) each
   consume V1's `recipe_id`. V2 re-tests model dominance on the locked
   recipe only — it does not re-open the V1 axis space.

---

## Known tensions

These are first-class tensions for this protocol, documented here so the
tension detector can link observations back to them.

### T-V1-01: V1 SE definition (resolved by mandate)

`DECISION_TREE_AUDIT.md` §1.3 identifies the cell-averaged SE as
measurement-incorrect. §3.3 of this protocol mandates bootstrap-over-outer-
folds (seed-level paired resample) and forbids the cell-averaged SE. The
audit log entry (2026-04-08) records the fix as implemented in
`validate_tree.R` (`SE = mean(auroc_std)/sqrt(n_seeds)` under stratified
comparison). This protocol adopts that implementation.

### T-V1-02: Cell-count asymmetry (resolved by stratification)

`DECISION_TREE_AUDIT.md` §1.4 identifies the sqrt(108) vs sqrt(27) mismatch.
§3.4 resolves this by stratifying shared-vs-shared and MS-vs-MS
tournaments and using a model-matched bridge for cross-family. The 2026-04-
08 audit resolution adopted this structure. The residual risk is that the
bridge aggregates four per-model claims into a single cross-family claim;
on a sensitivity run, V1 should also report the four per-model deltas
un-aggregated.

### T-V1-03: Pooling sweep across orderings before size decision

DESIGN.md §Size rules pools the sweep across {importance, pathway, rra}
orderings before the 3-criterion rule runs; the size decision is therefore
ordering-agnostic by construction. This is a methodological choice, not a
derived rule. If a new dataset shows that size × ordering interaction is
non-negligible (|ΔAUROC| at fixed size varies by > 0.02 across orderings
for the best-performing model), pooling should be revisited. Flagged as
TODO: emit a dedicated `condensates/size-ordering-pooling.md` so this
choice is falsifiable.

### T-V1-04: Cross-family bridge aggregation

§3.4's requirement that all four per-model bridges agree on claim type is
strict. A plausible failure mode is three-out-of-four agreement with one
model disagreeing. The current rule forwards this as Inconclusive. A
looser variant (majority agreement with the dissenting model flagged)
could lock earlier but would trade off against V1's "unanimous
model-matched bridge" principle. Document as open design question; track
in the resolution log.

### T-V1-05: Missing condensate support (TODOs)

Three rulebook gaps are flagged in §2.3:
- `condensates/three-criterion-size-rule.md` (TODO)
- `condensates/size-ordering-pooling.md` (TODO)
- `condensates/nested-expansion-vs-refit.md` (TODO)

V1 runs against DESIGN.md for these axes, but the rulebook does not yet
provide a falsifier. Promote these to condensates when V1's first
observation on a second dataset provides the evidence.

---

## Sources

- `operations/cellml/DESIGN.md` — recipes, size rules, factorial structure
- `operations/cellml/MASTER_PLAN.md` — V1 definition and decision tree
- `operations/cellml/DECISION_TREE_AUDIT.md` — V1 SE bug, cell-count
  asymmetry, T2-trunk-ordering confound (§1.3, §1.4, §3.2)
- `operations/cellml/rulebook/SCHEMA.md` — protocol format and fixed
  falsifier rubric
- `validate_tree.R` — reference V1 implementation with stratified
  comparison (ADR-004/ADR-006 per resolution log 2026-04-08)
